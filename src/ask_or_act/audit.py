"""``ask-or-act audit-redaction``: check that no Aegis prompt text is on disk outside data/raw.

How it works:

1. Index every Aegis prompt in the local raw files (test, validation and, if present,
   train) as overlapping **word n-grams** (default 8 words, lower-cased, punctuation
   ignored). Prompts shorter than n words but at least ``min_words`` long are indexed
   whole. Shorter prompts are too generic to match safely and are skipped (counted).
2. Scan files under the given paths (default: ``runs/``, ``.cache/`` and ``logs/``)
   with every string decoded first, so JSON escapes don't hide a match:
   - JSON / JSON Lines, string by string;
   - SQLite databases, value by value (each value also decoded as JSON);
   - any other text file, as text (``\\n``-style escapes are undone).
3. Any shared n-gram is a match. A match prints the file, the location, the Aegis row
   id and how many n-grams matched, **never the text**. Exit code 1 if anything
   matched.

What it can't see: paraphrases. The 2026-11-20 leak was model reasoning restating the
prompt in its own words, which a verbatim check misses. Its job is to catch input text
copied into outputs; the redaction rules (deciders/prompts.py ``SENSITIVE_TASKS``) are
what stop restated text.
"""

import contextlib
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ask_or_act.data.hub import read_json_rows
from ask_or_act.data.sources import AEGIS, AEGIS_TRAIN

DEFAULT_TARGETS = (Path("runs"), Path(".cache"), Path("logs"))
_WORD = re.compile(r"[^\W_]+")
_ESCAPE = re.compile(r"\\[nrtu\"\\/]")
MAX_FILE_BYTES = 512 * 1024 * 1024


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Index:
    n: int
    min_words: int
    grams: dict[tuple[str, ...], str] = field(default_factory=dict)  # n-gram -> row id
    lengths: set[int] = field(default_factory=set)
    prompts: int = 0
    skipped_short: int = 0

    def add(self, row_id: str, text: str) -> None:
        w = words(text)
        if len(w) < self.min_words:
            self.skipped_short += 1
            return
        self.prompts += 1
        size = min(self.n, len(w))
        self.lengths.add(size)
        for i in range(len(w) - size + 1):
            self.grams.setdefault(tuple(w[i : i + size]), row_id)

    def matches(self, text: str) -> dict[str, int]:
        """Aegis row id -> number of matching n-grams in ``text``."""
        w = words(text)
        hits: dict[str, int] = {}
        for size in self.lengths:
            for i in range(len(w) - size + 1):
                row_id = self.grams.get(tuple(w[i : i + size]))
                if row_id is not None:
                    hits[row_id] = hits.get(row_id, 0) + 1
        return hits


def aegis_files(raw_dir: Path) -> list[Path]:
    root = raw_dir / AEGIS.repo
    return [root / f for f in (*AEGIS.files, *AEGIS_TRAIN.files) if (root / f).exists()]


def build_index(paths: Sequence[Path], *, n: int = 8, min_words: int = 5) -> Index:
    index = Index(n, min_words)
    for p in paths:
        for r in read_json_rows(p):
            prompt = str(r.get("prompt") or "")
            if prompt and prompt != "REDACTED":
                index.add(str(r.get("id", "?")), prompt)
    return index


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
        if value[:1] in "[{":
            with contextlib.suppress(json.JSONDecodeError):
                yield from _strings(json.loads(value))
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _sqlite_texts(path: Path) -> Iterator[tuple[str, str]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            quoted = '"' + t.replace('"', '""') + '"'
            for rowid, *values in con.execute(f"SELECT rowid, * FROM {quoted}"):
                for v in values:
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", "replace")
                    if isinstance(v, str):
                        for s in _strings(v):
                            yield f"{t} rowid {rowid}", s
    finally:
        con.close()


def file_texts(path: Path) -> Iterator[tuple[str, str]]:
    """(location, text) pairs for one file, strings decoded."""
    with path.open("rb") as f:
        head = f.read(16)
    if head.startswith(b"SQLite format 3\x00"):
        yield from _sqlite_texts(path)
        return
    data = path.read_bytes()
    if b"\x00" in data[:4096]:
        return  # binary, not SQLite: nothing we write
    text = data.decode("utf-8", "replace")
    if path.suffix in (".json", ".jsonl"):
        try:
            for s in _strings(json.loads(text)):
                yield "json", s
            return
        except json.JSONDecodeError:
            pass
        for i, line in enumerate(text.splitlines(), 1):
            try:
                for s in _strings(json.loads(line)):
                    yield f"line {i}", s
            except json.JSONDecodeError:
                yield f"line {i}", _ESCAPE.sub(" ", line)
        return
    for i, line in enumerate(text.splitlines(), 1):
        yield f"line {i}", _ESCAPE.sub(" ", line)


def iter_files(targets: Iterable[Path], exclude: Sequence[Path]) -> Iterator[Path]:
    skip = [e.resolve() for e in exclude]
    for t in targets:
        candidates = [t] if t.is_file() else sorted(p for p in t.rglob("*") if p.is_file())
        for p in candidates:
            rp = p.resolve()
            if any(rp == s or s in rp.parents for s in skip):
                continue
            if p.stat().st_size <= MAX_FILE_BYTES:
                yield p


@dataclass(frozen=True)
class Match:
    file: Path
    location: str
    row_id: str
    ngrams: int


@dataclass
class AuditResult:
    files: int = 0
    strings: int = 0
    matches: list[Match] = field(default_factory=list)


def audit(targets: Iterable[Path], index: Index, exclude: Sequence[Path] = ()) -> AuditResult:
    result = AuditResult()
    for p in iter_files(targets, exclude):
        result.files += 1
        for location, text in file_texts(p):
            result.strings += 1
            for row_id, count in index.matches(text).items():
                result.matches.append(Match(p, location, row_id, count))
    return result


def run_audit(
    targets: Sequence[Path],
    raw_dir: Path,
    *,
    n: int = 8,
    min_words: int = 5,
    out: Any = print,
) -> int:
    sources = aegis_files(raw_dir)
    if not sources:
        out(f"error: no Aegis raw files under {raw_dir / AEGIS.repo}; nothing to check against")
        return 2
    index = build_index(sources, n=n, min_words=min_words)
    existing = [t for t in targets if t.exists()]
    for t in targets:
        if not t.exists():
            out(f"note: {t} does not exist, skipped")
    result = audit(existing, index, exclude=[raw_dir])
    out(
        f"indexed {index.prompts:,} Aegis prompts from {len(sources)} file(s) as {n}-word "
        f"n-grams ({index.skipped_short:,} prompts under {min_words} words skipped)"
    )
    out(f"scanned {result.files:,} file(s), {result.strings:,} string(s)")
    if not result.matches:
        out("OK: no Aegis prompt text found")
        return 0
    seen: set[tuple[Path, str]] = set()
    for m in result.matches:
        if (m.file, m.row_id) in seen:
            continue
        seen.add((m.file, m.row_id))
        out(f"MATCH {m.file} ({m.location}): Aegis row {m.row_id}, {m.ngrams} n-gram(s)")
    out(f"FAILED: {len(seen)} file/row match(es); text not shown")
    return 1

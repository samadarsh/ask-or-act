"""`ask-or-act audit-redaction`: finds Aegis prompt text however it was stored."""

import json
import sqlite3
from pathlib import Path

import pytest

from ask_or_act.audit import build_index, run_audit
from ask_or_act.cli import main
from ask_or_act.data.sources import AEGIS

PROMPT = "Placeholder prompt number one, standing in for an unsafe request text here."
SHORT = "tiny placeholder"


def raw_dir(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    d = root / AEGIS.repo
    d.mkdir(parents=True)
    rows = [
        {"id": "row-1", "prompt": PROMPT},
        {"id": "row-2", "prompt": SHORT},
        {"id": "row-3", "prompt": "REDACTED"},
    ]
    (d / "test.json").write_text(json.dumps(rows))
    return root


def audit(tmp_path: Path, *targets: Path) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = run_audit(list(targets), raw_dir(tmp_path), out=lines.append)
    return code, lines


def test_clean_files_pass(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "decisions.jsonl").write_text(json.dumps({"example_id": "row-1", "choice": "safe"}))
    code, lines = audit(tmp_path, runs)
    assert code == 0
    assert lines[-1] == "OK: no Aegis prompt text found"
    assert "1 Aegis prompts" in lines[0]
    assert "1 distinct prompts under 5 words by exact match" in lines[0]


@pytest.mark.parametrize("kind", ["jsonl-escaped", "nested-json-string", "sqlite", "log"])
def test_planted_text_is_found_and_never_printed(tmp_path: Path, kind: str) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    # A 9-word excerpt, with case and punctuation changed and a JSON escape inside.
    excerpt = "NUMBER one, standing in for\nan unsafe REQUEST text"
    if kind == "jsonl-escaped":
        (runs / "d.jsonl").write_text(json.dumps({"raw": {"error": f"x {excerpt} y"}}) + "\n")
    elif kind == "nested-json-string":
        inner = json.dumps({"content": excerpt})
        (runs / "d.jsonl").write_text(json.dumps({"raw": inner}) + "\n")
    elif kind == "sqlite":
        con = sqlite3.connect(runs / "cache.sqlite")
        con.execute("CREATE TABLE t (k TEXT, v TEXT)")
        con.execute("INSERT INTO t VALUES (?, ?)", ("key", json.dumps({"e": excerpt})))
        con.commit()
        con.close()
    else:
        (runs / "smoke.log").write_text("first error: " + json.dumps(excerpt)[1:-1] + "\n")
    code, lines = audit(tmp_path, runs)
    assert code == 1
    assert any(ln.startswith("MATCH") and "Aegis row row-1" in ln for ln in lines)
    assert lines[-1].startswith("FAILED: 1 file/row match")
    joined = "\n".join(lines).lower()
    assert "standing in" not in joined
    assert "unsafe request" not in joined


def test_short_and_generic_text_does_not_match(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "a.log").write_text(f"{SHORT}\nplaceholder prompt number one, standing in\n")
    code, _ = audit(tmp_path, runs)
    assert code == 0  # 7 consecutive words < 8, and SHORT isn't indexed


def test_raw_dir_itself_is_excluded(tmp_path: Path) -> None:
    root = raw_dir(tmp_path)
    (tmp_path / "clean.log").write_text("nothing to see\n")
    lines: list[str] = []
    assert run_audit([tmp_path], root, out=lines.append) == 0
    assert "scanned 1 file(s)" in lines[1]  # raw_dir's Aegis file was not scanned


def test_index_counts_ngrams() -> None:
    index = build_index([], n=3, min_words=2)
    index.add("a", "one two three four")
    assert index.matches("x ONE two three y") == {"a": 1}
    assert index.matches("one two four") == {}


def test_cli_exit_codes(tmp_path: Path) -> None:
    root = raw_dir(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    assert main(["audit-redaction", str(runs), "--raw-dir", str(root)]) == 2  # empty
    (runs / "ok.log").write_text("clean\n")
    assert main(["audit-redaction", str(runs), "--raw-dir", str(root)]) == 0
    (runs / "x.log").write_text(PROMPT)
    assert main(["audit-redaction", str(runs), "--raw-dir", str(root)]) == 1
    assert main(["audit-redaction", str(runs), "--raw-dir", str(tmp_path / "none")]) == 2


# ------------------------------------------------- short prompts (exact match)


def test_short_prompt_stored_as_a_value_is_found(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    # SHORT ("tiny placeholder"), with case and punctuation changed, as a JSON value.
    (runs / "d.jsonl").write_text(json.dumps({"raw": {"error": "Tiny, PLACEHOLDER!"}}) + "\n")
    code, lines = audit(tmp_path, runs)
    assert code == 1
    assert any("Aegis row row-2" in ln for ln in lines)
    assert "placeholder" not in "\n".join(lines).lower()  # text never printed


def test_short_prompt_inside_longer_text_or_a_plain_line_is_not_flagged(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "d.jsonl").write_text(json.dumps({"note": "a tiny placeholder value"}) + "\n")
    (runs / "x.log").write_text("tiny placeholder\n")  # whole line, but not a value
    code, _ = audit(tmp_path, runs)
    assert code == 0


def test_short_prompts_equal_to_our_own_vocabulary_are_not_checked(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    d = root / AEGIS.repo
    d.mkdir(parents=True)
    (d / "test.json").write_text(
        json.dumps([{"id": "v", "prompt": "test"}, {"id": "c", "prompt": "?x"}])
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run.json").write_text(json.dumps({"dataset": {"split": "test"}, "a": "x"}))
    lines: list[str] = []
    assert run_audit([runs], root, out=lines.append) == 0
    assert "2 not checked" in lines[0]


def test_scanning_nothing_is_an_error_not_ok(tmp_path: Path) -> None:
    lines: list[str] = []
    code = run_audit([tmp_path / "no such dir"], raw_dir(tmp_path), out=lines.append)
    assert code == 2
    assert "no files were scanned" in lines[-1]
    assert not any(ln.startswith("OK") for ln in lines)

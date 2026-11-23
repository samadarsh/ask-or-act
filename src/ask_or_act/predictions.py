"""Read a user's predictions file into a DecisionTable, and check it (docs/cli-design.md §4).

Input: CSV, TSV or JSONL with the columns ``id``, ``prediction``, ``confidence`` and
``human_label``, plus optional ``options``, ``split`` and ``group``. Any other source
that is a mapping of columns works too (e.g. ``df.to_dict("list")``).

**One pass, every problem at once.** ``check`` validates everything before computing
anything and returns grouped problems: one per kind, with a count, the first few line
numbers and a hint. ``read_predictions`` returns the table, or raises ``InputError``
carrying those problems.

**Privacy.** Messages quote at most a short snippet of a bad cell, and never a
prediction or a label in full (the file may hold unsafe text).

**Split.** Without a ``split`` column, each id (or its ``group``) goes to calibration
or test by hashing it with the seed: the same id always lands in the same half, so
adding, removing or reordering rows never moves the others (a seeded shuffle would).
"""

import csv
import difflib
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal

from ask_or_act.core.types import JSON
from ask_or_act.guarantees.fixed_sequence import DEFAULT_STARTS, grid_for
from ask_or_act.guarantees.recommend import GridChoice
from ask_or_act.guarantees.sample_size import min_rows
from ask_or_act.report.table import DecisionRow, DecisionTable, continuous_grid

REQUIRED = ("id", "prediction", "confidence", "human_label")
OPTIONAL = ("options", "split", "group")
KNOWN = REQUIRED + OPTIONAL
# Common names for our columns, used only to suggest a --columns mapping.
ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("example_id", "uid", "row_id", "item_id", "key", "idx"),
    "prediction": ("pred", "predicted", "predicted_label", "y_pred", "output", "answer"),
    "confidence": ("conf", "score", "prob", "probability", "p", "confidence_score"),
    "human_label": ("label", "gold", "truth", "ground_truth", "target", "y_true", "reference"),
    "options": ("choices", "labels", "classes"),
    "split": ("fold", "subset", "partition"),
    "group": ("group_id", "cluster", "conversation_id", "user_id"),
}
FORMATS = {".csv": "csv", ".tsv": "tsv", ".jsonl": "jsonl", ".ndjson": "jsonl"}
SPLITS = ("calibration", "test")
BENCHMARKS = ("banking77", "aegis", "mt_bench")  # dataset names with special handling
SHOW_LINES = 5  # line numbers shown per problem
SNIPPET = 20  # characters of a bad cell shown in a message
SUSPICIOUS_SHARE = 0.05  # share of answered rows below 1/k that triggers the warning
SUSPICIOUS_MIN = 10  # ... and at least this many rows (a couple of rows is noise)
FEW_DISTINCT = 20
SMALL_TEST = 100

type Severity = Literal["error", "warning"]
type ConfidenceOf = Literal["predicted", "positive"]
type Source = str | Path | IO[str] | Mapping[str, Sequence[object]]


@dataclass(frozen=True)
class Problem:
    code: str
    severity: Severity
    message: str
    hint: str = ""
    lines: tuple[int, ...] = ()  # the first few, 1-based
    count: int = 0  # rows affected (0: not about rows)

    def to_json(self) -> dict[str, JSON]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "hint": self.hint,
            "lines": list(self.lines),
            "count": self.count,
        }


class InputError(ValueError):
    """The input failed validation. ``problems`` holds every error and warning."""

    def __init__(self, problems: Sequence[Problem]) -> None:
        self.problems = tuple(problems)
        errors = [p.message for p in self.problems if p.severity == "error"]
        super().__init__("; ".join(errors) or "invalid input")


# --------------------------------------------------------------------------- scales


@dataclass(frozen=True)
class Scale:
    """How the confidence column is written. Each scale maps to a pre-registered,
    frozen threshold grid (guarantees/fixed_sequence.py), never chosen from data."""

    kind: Literal["probability", "percent", "votes", "logprob"]
    k: int = 0  # votes only

    @classmethod
    def parse(cls, text: str) -> "Scale":
        t = text.strip().lower()
        if t in ("probability", "percent", "logprob"):
            return cls(t)  # type: ignore[arg-type]
        if t.startswith("votes:"):
            try:
                k = int(t.split(":", 1)[1])
            except ValueError:
                k = 0
            if k >= 1:
                return cls("votes", k)
        raise ValueError(
            f"unknown confidence scale {text!r}: use probability, percent, votes:K or logprob"
        )

    def __str__(self) -> str:
        return f"votes:{self.k}" if self.kind == "votes" else self.kind

    def range(self) -> str:
        return {
            "probability": "[0, 1]",
            "percent": "0–100",
            "votes": f"whole numbers 0–{self.k}",
            "logprob": "≤ 0",
        }[self.kind]

    def describe(self) -> str:
        return {
            "probability": "probabilities in [0, 1]",
            "percent": "percentages 0–100",
            "votes": f"votes out of {self.k}",
            "logprob": "log-probabilities",
        }[self.kind]

    def convert(self, v: float) -> float | None:
        """The value as a probability in [0, 1], or None if it doesn't fit the scale."""
        if self.kind == "probability":
            return v if 0.0 <= v <= 1.0 else None
        if self.kind == "percent":
            return v / 100.0 if 0.0 <= v <= 100.0 else None
        if self.kind == "votes":
            whole = abs(v - round(v)) < 1e-9
            return round(v) / self.k if whole and 0 <= round(v) <= self.k else None
        return math.exp(min(v, 0.0)) if v <= 1e-9 else None  # logprob

    def grid(self) -> GridChoice:
        if self.kind == "percent":
            return GridChoice(grid_for("verbalized"), DEFAULT_STARTS["verbalized"], "percent steps")
        if self.kind == "votes":
            return GridChoice(
                grid_for("self_consistency", samples=self.k),
                DEFAULT_STARTS["self_consistency"],
                f"votes, k = {self.k}: {self.k + 1} steps",
            )
        return continuous_grid()


def suggest_scale(values: Sequence[float]) -> tuple[str, str]:
    """(scale, hint) for values that don't look like probabilities; ("", "") if they do.
    Only a suggestion: the scale is never switched silently (it picks the grid)."""
    if not values:
        return "", ""
    lo, hi = min(values), max(values)
    if lo >= 0.0 and hi <= 1.0:
        return "probability", ""
    if hi <= 0.0:
        return "logprob", "they look like log-probabilities: pass --confidence-scale logprob"
    if lo < 0.0 or hi > 100.0:
        return "", (
            "they look like unbounded scores (logits or margins), not probabilities: "
            "calibrate them first (e.g. a sigmoid fitted on held-out data)"
        )
    if all(abs(v - round(v)) < 1e-9 for v in values) and hi <= 20:
        k = round(hi)
        return f"votes:{k}", (
            f"they are whole numbers up to {k}: if they count agreeing samples out of K, pass "
            f"--confidence-scale votes:K (K = the number of samples, at least {k}); if they "
            "are percentages, pass --confidence-scale percent"
        )
    return "percent", "they look like percentages: pass --confidence-scale percent"


def _num(v: float) -> str:
    return f"{v:g}"


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class CheckResult:
    """Everything ``check`` found. ``table`` is None when there are errors."""

    source: str
    format: str
    n_rows: int
    columns: tuple[str, ...]  # the file's own column names
    mapping: Mapping[str, str]  # our name → the file's column
    ignored: tuple[str, ...]
    problems: tuple[Problem, ...]
    table: DecisionTable | None
    scale: Scale
    confidence_of: ConfidenceOf
    positive: str | None
    split_source: Literal["column", "hash"]
    split_seed: int
    cal_fraction: float
    alpha: float
    delta: float
    n_distinct: int = 0
    n_failed: int = 0
    n_unlabelled: int = 0
    labelled: Mapping[str, int] = field(default_factory=dict)  # per split
    min_cal_rows: int | None = None  # zero errors
    min_cal_rows_typical: int | None = None  # errors at α/2

    @property
    def errors(self) -> tuple[Problem, ...]:
        return tuple(p for p in self.problems if p.severity == "error")

    @property
    def warnings(self) -> tuple[Problem, ...]:
        return tuple(p for p in self.problems if p.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_json(self) -> dict[str, JSON]:
        return {
            "ok": self.ok,
            "rows": self.n_rows,
            "columns": dict(self.mapping),
            "ignored_columns": list(self.ignored),
            "confidence_scale": str(self.scale),
            "confidence_of": self.confidence_of,
            "positive": self.positive,
            "distinct_confidences": self.n_distinct,
            "failed_predictions": self.n_failed,
            "unlabelled": self.n_unlabelled,
            "labelled": dict(self.labelled),
            "split": {
                "source": self.split_source,
                "seed": self.split_seed if self.split_source == "hash" else None,
                "cal_fraction": self.cal_fraction if self.split_source == "hash" else None,
            },
            "min_calibration_rows": {
                "alpha": self.alpha,
                "delta": self.delta,
                "no_errors": self.min_cal_rows,
                "errors_at_half_alpha": self.min_cal_rows_typical,
            },
            "errors": [p.to_json() for p in self.errors],
        }


# --------------------------------------------------------------------------- reading


@dataclass(frozen=True)
class _Raw:
    line: int
    values: Mapping[str, object]


class _StopError(Exception):
    def __init__(self, problem: Problem) -> None:
        self.problem = problem


def _fail(code: str, message: str, hint: str = "") -> _StopError:
    return _StopError(Problem(code, "error", message, hint))


def _format_of(source: Source, fmt: str | None) -> str:
    if fmt is not None:
        if fmt not in ("csv", "tsv", "jsonl"):
            raise _fail("format", f"unknown format {fmt!r}", "use csv, tsv or jsonl")
        return fmt
    if isinstance(source, Mapping):
        return "columns"
    if isinstance(source, str | Path):
        suffix = Path(source).suffix.lower()
        if suffix in FORMATS:
            return FORMATS[suffix]
        if suffix == ".json":
            raise _fail(
                "format",
                f"{Path(source).name} is a .json file; this needs JSONL (one JSON object per line)",
                "convert it, e.g. jq -c '.[]' file.json > file.jsonl",
            )
        raise _fail(
            "format",
            f"can't tell the format of {Path(source).name} from its extension",
            "pass --format csv, tsv or jsonl",
        )
    raise _fail("format", "reading from a stream needs a format", "pass --format csv, tsv or jsonl")


def _text_rows(stream: IO[str], fmt: str) -> tuple[list[str], list[_Raw]]:
    if fmt == "jsonl":
        rows: list[_Raw] = []
        seen: dict[str, None] = {}
        for n, text in enumerate(stream, start=1):
            if not text.strip():
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise _fail("jsonl", f"line {n} is not valid JSON ({exc.msg})") from None
            if not isinstance(obj, dict):
                raise _fail("jsonl", f"line {n} is not a JSON object")
            seen.update(dict.fromkeys(obj))
            rows.append(_Raw(n, obj))
        return list(seen), rows
    reader = csv.reader(stream, delimiter="\t" if fmt == "tsv" else ",")
    try:
        header = next(reader)
    except StopIteration:
        raise _fail("empty", "the file is empty") from None
    header = [h.strip() for h in header]
    dupes = sorted({h for h in header if header.count(h) > 1})
    if dupes:
        raise _fail("header", f"the header repeats column {', '.join(map(_q, dupes))}")
    out: list[_Raw] = []
    too_long: list[int] = []
    for cells in reader:
        if not cells or all(not c.strip() for c in cells):
            continue
        if len(cells) > len(header):
            too_long.append(reader.line_num)
            continue
        out.append(_Raw(reader.line_num, dict(zip(header, cells, strict=False))))
    if too_long:
        raise _fail(
            "fields",
            f"{_rows(len(too_long))} have more fields than the header "
            f"({_lines(too_long, len(too_long))})",
            "quote cells that contain the delimiter",
        )
    return header, out


def _raw_rows(source: Source, fmt: str) -> tuple[list[str], list[_Raw]]:
    if isinstance(source, Mapping):
        cols = [str(c) for c in source]
        lengths = {len(v) for v in source.values()}
        if len(lengths) > 1:
            raise _fail("columns", "the columns have different lengths")
        n = lengths.pop() if lengths else 0
        values = list(source.values())
        return cols, [_Raw(i + 1, {c: values[j][i] for j, c in enumerate(cols)}) for i in range(n)]
    if isinstance(source, str | Path):
        path = Path(source)
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                return _text_rows(f, fmt)
        except FileNotFoundError:
            raise _fail("unreadable", f"cannot read {path}: no such file") from None
        except IsADirectoryError:
            raise _fail("unreadable", f"{path} is a directory, not a predictions file") from None
        except UnicodeDecodeError as exc:
            raise _fail(
                "encoding",
                f"{path.name} is not UTF-8 text (bad byte at offset {exc.start})",
                "save it as UTF-8",
            ) from None
    return _text_rows(source, fmt)


def _text(v: object) -> str | None:
    """A cell as text: None for empty. JSON numbers and booleans keep their JSON form."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, list | dict):
        raise TypeError
    s = str(v).strip()
    return s or None


def _number(v: object) -> float | None:
    if isinstance(v, bool):
        raise ValueError
    if isinstance(v, int | float):
        f = float(v)
    else:
        s = str(v).strip()
        if not s:
            return None
        f = float(s)
    if not math.isfinite(f):
        raise ValueError
    return f


def _options(v: object) -> tuple[str, ...]:
    if v is None:
        return ()
    if isinstance(v, list):
        items = [_text(x) for x in v]
    else:
        items = [x.strip() or None for x in str(v).split("|")]
    return tuple(x for x in items if x is not None)


# --------------------------------------------------------------------------- formatting


def _q(s: str) -> str:
    s = s if len(s) <= SNIPPET else s[: SNIPPET - 1] + "…"
    return f'"{s}"'


def _rows(n: int) -> str:
    return f"{n:,} row" if n == 1 else f"{n:,} rows"


def _lines(lines: Sequence[int], count: int, unit: str = "line") -> str:
    shown = ", ".join(str(n) for n in lines[:SHOW_LINES])
    more = ", …" if count > SHOW_LINES else ""
    return f"{unit}{'s' if count > 1 else ''} {shown}{more}"


class _Issues:
    """Collects row-level issues by code, keeping a count and the first few lines."""

    def __init__(self, unit: str) -> None:
        self.unit = unit
        self.lines: dict[str, list[int]] = defaultdict(list)
        self.samples: dict[str, list[str]] = defaultdict(list)

    def add(self, code: str, line: int, sample: str | None = None) -> None:
        self.lines[code].append(line)
        if sample is not None and len(self.samples[code]) < 3:
            self.samples[code].append(sample)

    def count(self, code: str) -> int:
        return len(self.lines.get(code, ()))

    def where(self, code: str) -> str:
        lines = self.lines.get(code, [])
        return _lines(lines, len(lines), self.unit)

    def first(self, code: str) -> tuple[int, ...]:
        return tuple(self.lines.get(code, [])[:SHOW_LINES])

    def examples(self, code: str) -> str:
        return ", ".join(_q(s) for s in self.samples.get(code, []))


# --------------------------------------------------------------------------- check


def hashed_split(key: str, seed: int, cal_fraction: float) -> str:
    """calibration or test, from sha256(f"{seed}:{key}") read as a number in [0, 1)."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    return "calibration" if u < cal_fraction else "test"


def _guess(name: str, header: Sequence[str], taken: set[str]) -> str | None:
    free = [h for h in header if h not in taken]
    lower = {h.lower(): h for h in free}
    for alias in (name, *ALIASES.get(name, ())):
        if alias in lower:
            return lower[alias]
    close = difflib.get_close_matches(name, free, n=1, cutoff=0.75)
    return close[0] if close else None


def _resolve_columns(
    header: Sequence[str], columns: Mapping[str, str] | None
) -> tuple[dict[str, str], list[Problem]]:
    """Our name → the file's column. A required column that is missing but has an
    obvious match is still mapped (so the rest can be checked), with an error."""
    problems: list[Problem] = []
    mapping: dict[str, str] = {}
    for ours, theirs in (columns or {}).items():
        if ours not in KNOWN:
            problems.append(
                Problem(
                    "columns",
                    "error",
                    f"--columns names unknown column {_q(ours)}",
                    f"known names: {', '.join(KNOWN)}",
                )
            )
        elif theirs not in header:
            problems.append(
                Problem(
                    "columns",
                    "error",
                    f"--columns maps {ours} to {_q(theirs)}, but the file has no such column",
                    f"the file's columns: {', '.join(header)}",
                )
            )
        else:
            mapping[ours] = theirs
    for name in KNOWN:
        if name not in mapping and name in header and name not in mapping.values():
            mapping[name] = name
    missing = [n for n in REQUIRED if n not in mapping]
    if missing:
        taken = set(mapping.values())
        guesses: dict[str, str] = {}
        for n in missing:
            g = _guess(n, header, taken)
            if g is not None:
                guesses[n] = g
                taken.add(g)
        names = ", ".join(f'"{n}"' for n in missing)
        word = "column" if len(missing) == 1 else "columns"
        if len(guesses) == len(missing):
            hint = "rename them, or map them: --columns " + ",".join(
                f"{n}={g}" for n, g in guesses.items()
            )
        elif guesses:
            hint = "map the ones that exist under another name, e.g. --columns " + ",".join(
                f"{n}={g}" for n, g in guesses.items()
            )
        else:
            hint = "add them, or map existing columns with --columns name=column"
        problems.append(
            Problem(
                "missing_column",
                "error",
                f"missing required {word} {names}. Found: {', '.join(header) or '(none)'}",
                hint,
            )
        )
        if len(guesses) == len(missing):
            mapping.update(guesses)  # check the rest under the obvious mapping
    return mapping, problems


def check(
    source: Source,
    *,
    format: str | None = None,
    columns: Mapping[str, str] | None = None,
    confidence_scale: str | Scale = "probability",
    confidence_of: ConfidenceOf = "predicted",
    positive: str | None = None,
    split_seed: int = 0,
    cal_fraction: float = 0.5,
    split_map: Mapping[str, str] | None = None,
    alpha: float = 0.1,
    delta: float = 0.1,
    name: str | None = None,
) -> CheckResult:
    """Validate a predictions source. Never raises for bad data: see ``.problems``."""
    scale = (
        confidence_scale if isinstance(confidence_scale, Scale) else (Scale.parse(confidence_scale))
    )
    if not 0.0 < cal_fraction < 1.0:
        raise ValueError("cal_fraction must be between 0 and 1")
    label = name or _source_name(source)
    base: dict[str, Any] = {
        "source": label,
        "scale": scale,
        "confidence_of": confidence_of,
        "positive": positive,
        "split_seed": split_seed,
        "cal_fraction": cal_fraction,
        "alpha": alpha,
        "delta": delta,
    }
    try:
        fmt = _format_of(source, format)
        header, raws = _raw_rows(source, fmt)
    except _StopError as f:
        return CheckResult(
            format=format or "?",
            n_rows=0,
            columns=(),
            mapping={},
            ignored=(),
            problems=(f.problem,),
            table=None,
            split_source="hash",
            **base,
        )
    mapping, problems = _resolve_columns(header, columns)
    ignored = tuple(h for h in header if h not in mapping.values())
    result = _check_rows(
        raws,
        fmt,
        mapping,
        problems,
        scale,
        confidence_of,
        positive,
        split_seed,
        cal_fraction,
        split_map or {},
        alpha,
        delta,
        label,
    )
    return CheckResult(
        format=fmt,
        n_rows=len(raws),
        columns=tuple(header),
        mapping=mapping,
        ignored=ignored,
        **(base | result),
    )


def _source_name(source: Source) -> str:
    if isinstance(source, str | Path):
        return Path(source).name
    if isinstance(source, Mapping):
        return "columns"
    return "stdin"


def _check_rows(
    raws: Sequence[_Raw],
    fmt: str,
    mapping: Mapping[str, str],
    problems: list[Problem],
    scale: Scale,
    confidence_of: ConfidenceOf,
    positive: str | None,
    split_seed: int,
    cal_fraction: float,
    split_map: Mapping[str, str],
    alpha: float,
    delta: float,
    label: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {"split_source": "column" if "split" in mapping else "hash"}
    if not raws:
        problems.append(Problem("empty", "error", "the file has a header but no rows"))
    if any(n not in mapping for n in REQUIRED) or not raws:
        out.update(problems=tuple(problems), table=None)
        return out

    unit = "row" if fmt == "columns" else "line"
    bad = _Issues(unit)
    ids: list[str] = []
    preds: list[str | None] = []
    labels: list[str | None] = []
    raw_conf: list[float | None] = []
    opts: list[tuple[str, ...]] = []
    splits: list[str | None] = []
    groups: list[str | None] = []
    lines: list[int] = []

    def cell(r: _Raw, name: str) -> object:
        col = mapping.get(name)
        return None if col is None else r.values.get(col)

    for r in raws:
        lines.append(r.line)
        try:
            rid = _text(cell(r, "id"))
            pred = _text(cell(r, "prediction"))
            gold = _text(cell(r, "human_label"))
            grp = _text(cell(r, "group"))
        except TypeError:
            bad.add("not_text", r.line)
            rid = pred = gold = grp = None
        if rid is None:
            bad.add("empty_id", r.line)
        ids.append(rid or "")
        preds.append(pred)
        labels.append(gold)
        groups.append(grp)
        conf_cell = cell(r, "confidence")
        try:
            c = _number(conf_cell)
            if c is None and pred is not None:
                bad.add("conf_empty", r.line)
        except (TypeError, ValueError):
            bad.add("conf_not_number", r.line, str(conf_cell))
            c = None
        raw_conf.append(c)
        opts.append(_options(cell(r, "options")))
        if "split" in mapping:
            s = _text(cell(r, "split"))
            s = split_map.get(s, s) if s is not None else None
            s = s.lower() if s is not None else None
            if s is None:
                bad.add("split_empty", r.line)
            elif s not in SPLITS:
                bad.add("split_unknown", r.line, s)
            splits.append(s if s in SPLITS else None)
        else:
            splits.append(None)

    # --- ids
    if bad.count("not_text"):
        problems.append(
            Problem(
                "not_text",
                "error",
                f"a text column holds a list or object in {_rows(bad.count('not_text'))} "
                f"({bad.where('not_text')})",
                lines=bad.first("not_text"),
                count=bad.count("not_text"),
            )
        )
    if bad.count("empty_id"):
        problems.append(
            Problem(
                "empty_id",
                "error",
                f"id is empty in {_rows(bad.count('empty_id'))} ({bad.where('empty_id')})",
                "every row needs a unique id",
                bad.first("empty_id"),
                bad.count("empty_id"),
            )
        )
    where_id: dict[str, list[int]] = defaultdict(list)
    for rid, line in zip(ids, lines, strict=True):
        if rid:
            where_id[rid].append(line)
    dupes = {k: v for k, v in where_id.items() if len(v) > 1}
    if dupes:
        first_id, first_lines = next(iter(dupes.items()))
        extra = f" (and {len(dupes) - 1:,} more ids)" if len(dupes) > 1 else ""
        problems.append(
            Problem(
                "duplicate_id",
                "error",
                f"id {_q(first_id)} appears {len(first_lines)} times "
                f"({_lines(first_lines, len(first_lines), unit)}){extra}",
                "ids must be unique: one row per decision",
                tuple(first_lines[:SHOW_LINES]),
                sum(len(v) for v in dupes.values()),
            )
        )

    # --- confidence: numbers, then scale
    if bad.count("conf_not_number"):
        problems.append(
            Problem(
                "conf_not_number",
                "error",
                f"confidence is not a finite number in {_rows(bad.count('conf_not_number'))} "
                f"({bad.where('conf_not_number')}): {bad.examples('conf_not_number')}",
                lines=bad.first("conf_not_number"),
                count=bad.count("conf_not_number"),
            )
        )
    if bad.count("conf_empty"):
        problems.append(
            Problem(
                "conf_empty",
                "error",
                f"confidence is empty in {_rows(bad.count('conf_empty'))} that have a "
                f"prediction ({bad.where('conf_empty')})",
                "leave both empty for a failed decision; otherwise give a confidence",
                bad.first("conf_empty"),
                bad.count("conf_empty"),
            )
        )
    present = [v for v in raw_conf if v is not None]
    conf: list[float | None] = []
    for v, line in zip(raw_conf, lines, strict=True):
        if v is None:
            conf.append(None)
            continue
        prob = scale.convert(v)
        if prob is None:
            bad.add("conf_range", line)
        conf.append(prob)
    if bad.count("conf_range"):
        suggested, hint = suggest_scale(present)
        if scale.kind != "probability" and suggested in ("", str(scale)):
            hint = hint or f"with --confidence-scale {scale}, values must be {scale.range()}"
        elif not hint:
            hint = f"with --confidence-scale {scale}, values must be {scale.range()}"
        lo, hi = min(present), max(present)
        problems.append(
            Problem(
                "conf_range",
                "error",
                f"confidence is outside {scale.range()} in "
                f"{_rows(bad.count('conf_range'))} ({bad.where('conf_range')}); "
                f"values run {_num(lo)} to {_num(hi)}",
                hint,
                bad.first("conf_range"),
                bad.count("conf_range"),
            )
        )

    # --- labels: the label set, positive class, options
    label_set = sorted(
        {x for x in preds if x is not None}
        | {x for x in labels if x is not None}
        | {o for os_ in opts for o in os_}
    )
    if confidence_of == "positive":
        positive = _positive(label_set, positive, problems)
        if positive is not None:
            conf = [
                None if c is None or p is None else (c if p == positive else 1.0 - c)
                for c, p in zip(conf, preds, strict=True)
            ]
    out["positive"] = positive
    for o, p, g, line in zip(opts, preds, labels, lines, strict=True):
        if not o:
            continue
        if p is not None and p not in o:
            bad.add("pred_not_option", line, p)
        if g is not None and g not in o:
            bad.add("label_not_option", line, g)
    for code, what in (("pred_not_option", "prediction"), ("label_not_option", "human_label")):
        if bad.count(code):
            problems.append(
                Problem(
                    code,
                    "error",
                    f"{what} is not one of the row's options in {_rows(bad.count(code))} "
                    f"({bad.where(code)}): {bad.examples(code)}",
                    "options are separated by | in CSV, or a list in JSONL",
                    bad.first(code),
                    bad.count(code),
                )
            )

    # --- split
    if bad.count("split_empty"):
        problems.append(
            Problem(
                "split_empty",
                "error",
                f"split is empty in {_rows(bad.count('split_empty'))} ({bad.where('split_empty')})",
                "fill it in, or drop the split column to split automatically",
                bad.first("split_empty"),
                bad.count("split_empty"),
            )
        )
    if bad.count("split_unknown"):
        seen = sorted(set(bad.samples["split_unknown"]))
        problems.append(
            Problem(
                "split_unknown",
                "error",
                f"split must be calibration or test; found {', '.join(map(_q, seen))} in "
                f"{_rows(bad.count('split_unknown'))} ({bad.where('split_unknown')})",
                f"rename the values, or map them: --split-map {seen[0]}=calibration",
                bad.first("split_unknown"),
                bad.count("split_unknown"),
            )
        )
    if out["split_source"] == "hash":
        splits = [
            hashed_split(g or i, split_seed, cal_fraction) for i, g in zip(ids, groups, strict=True)
        ]
    else:
        by_group: dict[str, set[str]] = defaultdict(set)
        group_lines: dict[str, list[int]] = defaultdict(list)
        for g, s, line in zip(groups, splits, lines, strict=True):
            if g is not None and s is not None:
                by_group[g].add(s)
                group_lines[g].append(line)
        leaks = [g for g, ss in by_group.items() if len(ss) > 1]
        if leaks:
            g0 = leaks[0]
            more = f" (and {len(leaks) - 1:,} more groups)" if len(leaks) > 1 else ""
            problems.append(
                Problem(
                    "group_leak",
                    "error",
                    f"group {_q(g0)} has rows in both calibration and test "
                    f"({_lines(group_lines[g0], len(group_lines[g0]), unit)}){more}: "
                    "that leaks information between the halves",
                    "keep each group in one split, or drop the split column to split by group",
                    tuple(group_lines[g0][:SHOW_LINES]),
                    sum(len(group_lines[g]) for g in leaks),
                )
            )

    # --- warnings
    answered = [c for c, p in zip(conf, preds, strict=True) if p is not None and c is not None]
    n_failed = sum(p is None for p in preds)
    n_unlabelled = sum(g is None for g in labels)
    labelled = Counter(s for s, g in zip(splits, labels, strict=True) if g is not None and s)
    n_labelled = sum(labelled.values())
    failed_lines = [line for p, line in zip(preds, lines, strict=True) if p is None]
    if n_failed:
        problems.append(
            Problem(
                "failed",
                "warning",
                f"{_rows(n_failed)} {'has' if n_failed == 1 else 'have'} an empty prediction "
                f"({_lines(failed_lines, n_failed, unit)}): counted as wrong with confidence 0",
                lines=tuple(failed_lines[:SHOW_LINES]),
                count=n_failed,
            )
        )
    if n_labelled == 0:
        problems.append(
            Problem(
                "no_labels",
                "warning",
                "no row has a human_label: metrics and thresholds need labelled rows",
                "label some rows; unlabelled rows only estimate coverage",
            )
        )
    elif n_unlabelled:
        problems.append(
            Problem(
                "unlabelled",
                "warning",
                f"{_rows(n_unlabelled)} {'has' if n_unlabelled == 1 else 'have'} no "
                "human_label: used to estimate coverage only",
                count=n_unlabelled,
            )
        )
    distinct = len(set(answered))
    if answered and distinct == 1:
        problems.append(
            Problem(
                "constant_conf",
                "warning",
                f"every confidence is {_num(answered[0])}: no threshold can separate anything",
                "check that the confidence column is the model's score",
            )
        )
    elif answered and distinct < FEW_DISTINCT:
        problems.append(
            Problem(
                "few_conf",
                "warning",
                f"confidence has only {distinct} distinct values: thresholds will be coarse",
                "common for verbalized confidence; a finer signal gives finer thresholds",
            )
        )
    case_lines = [
        line
        for p, g, line in zip(preds, labels, lines, strict=True)
        if p is not None and g is not None and p != g and p.casefold() == g.casefold()
    ]
    if case_lines:
        problems.append(
            Problem(
                "case_mismatch",
                "warning",
                f"prediction and human_label differ only by upper/lower case in "
                f"{_rows(len(case_lines))} ({_lines(case_lines, len(case_lines), unit)}): "
                "they count as wrong",
                "probably a formatting mismatch; make the spellings match",
                tuple(case_lines[:SHOW_LINES]),
                len(case_lines),
            )
        )
    if confidence_of == "predicted" and len(label_set) >= 2 and answered:
        _confidence_meaning(label_set, preds, conf, problems)
    if n_labelled and labelled.get("test", 0) < SMALL_TEST:
        problems.append(
            Problem(
                "small_test",
                "warning",
                f"only {labelled.get('test', 0):,} labelled rows in the test half: its risk "
                "estimate will have a wide confidence interval",
                "label more rows if you can",
            )
        )
    grid = scale.grid()
    need = min_rows(alpha, delta, starts=len(grid.starts))
    typical = min_rows(alpha, delta, starts=len(grid.starts), error_rate=alpha / 2)
    have = labelled.get("calibration", 0)
    if n_labelled and need is not None and have < need:
        problems.append(
            Problem(
                "few_calibration",
                "warning",
                f"only {have:,} labelled calibration {'row' if have == 1 else 'rows'}: "
                f"a guarantee at α = {alpha:g}, "
                f"δ = {delta:g} needs at least {need:,}, so thresholds will refuse",
                _loosen_hint(need, have, delta, len(grid.starts), alpha),
            )
        )

    out.update(
        n_distinct=distinct,
        n_failed=n_failed,
        n_unlabelled=n_unlabelled,
        labelled={s: labelled.get(s, 0) for s in SPLITS},
        min_cal_rows=need,
        min_cal_rows_typical=typical,
    )
    has_error = any(p.severity == "error" for p in problems)
    out["problems"] = tuple(problems)
    out["table"] = None
    if not has_error:
        rows = tuple(
            DecisionRow(
                id=i,
                prediction=p,
                confidence=c,
                label=g,
                split=s,
                group=grp,
                options=o,
            )
            for i, p, c, g, s, grp, o in zip(
                ids, preds, conf, labels, splits, groups, opts, strict=True
            )
        )
        stem = Path(label).stem if label not in ("stdin", "columns") else label
        # A user file named like a benchmark must not get that benchmark's handling.
        dataset = f"{stem} (file)" if stem in BENCHMARKS else stem
        out["table"] = DecisionTable(
            name=stem,
            dataset=dataset,
            rows=rows,
            source=label,
            grid=grid,
            split_source=out["split_source"],
        )
    return out


def _loosen_hint(need: int, have: int, delta: float, starts: int, alpha: float) -> str:
    """Label more, or the smallest common α these rows can support."""
    for a in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        if a <= alpha:
            continue
        n = min_rows(a, delta, starts=starts)
        if n is not None and n <= have:
            return (
                f"label at least {need - have:,} more rows, "
                f"or loosen the target: --alpha {a:g} needs {n:,}"
            )
    return f"label at least {need - have:,} more rows"


def _positive(
    label_set: Sequence[str], positive: str | None, problems: list[Problem]
) -> str | None:
    if len(label_set) > 2:
        problems.append(
            Problem(
                "positive_not_binary",
                "error",
                f"--confidence-of positive needs exactly two labels; found {len(label_set)}: "
                f"{', '.join(map(_q, label_set[:5]))}{', …' if len(label_set) > 5 else ''}",
                "P(positive class) only makes sense for two classes; drop --confidence-of",
            )
        )
        return None
    if positive is not None:
        if label_set and positive not in label_set:
            problems.append(
                Problem(
                    "positive_unknown",
                    "error",
                    f"--positive {_q(positive)} is not one of the labels: "
                    f"{', '.join(map(_q, label_set))}",
                )
            )
            return None
        return positive
    lowered = {x.casefold(): x for x in label_set}
    if set(lowered) <= {"0", "1"} and "1" in lowered:
        return lowered["1"]
    if set(lowered) <= {"false", "true"} and "true" in lowered:
        return lowered["true"]
    problems.append(
        Problem(
            "positive_missing",
            "error",
            "--confidence-of positive needs to know which label is positive",
            f"pass --positive with one of: {', '.join(map(_q, label_set))}",
        )
    )
    return None


def _confidence_meaning(
    label_set: Sequence[str],
    preds: Sequence[str | None],
    conf: Sequence[float | None],
    problems: list[Problem],
) -> None:
    """Warn when the column looks like P(one class) rather than confidence in the
    predicted answer: an answer chosen among k can't be less than 1/k likely."""
    pairs = [(p, c) for p, c in zip(preds, conf, strict=True) if p is not None and c is not None]
    k = len(label_set)
    floor = 1.0 / k
    below = sum(c < floor - 1e-12 for _, c in pairs)
    share = below / len(pairs)
    if share < SUSPICIOUS_SHARE or below < SUSPICIOUS_MIN:
        return
    if k == 2:
        means = {
            lab: sum(c for p, c in pairs if p == lab) / max(1, sum(p == lab for p, _ in pairs))
            for lab in label_set
        }
        pos = max(label_set, key=lambda lab: means[lab])
        problems.append(
            Problem(
                "looks_like_positive",
                "warning",
                f"{share:.0%} of answered rows have confidence below 0.5. With two labels "
                f"({', '.join(label_set)}), that looks like P({pos}), not confidence in the "
                "predicted answer",
                f"if it is, pass --confidence-of positive --positive {pos}",
                count=below,
            )
        )
    else:
        problems.append(
            Problem(
                "below_chance",
                "warning",
                f"{share:.0%} of answered rows have confidence below 1/{k} = {floor:.2f}, "
                f"the least a model can have in the answer it picked among {k} labels",
                "confidence should be the probability that the predicted answer is right; "
                "check what the column measures",
                count=below,
            )
        )


def read_predictions(source: Source, **options: Any) -> DecisionTable:
    """Read and validate a predictions source. Raises ``InputError`` on any error;
    warnings don't stop it (see ``check`` for them). Options as for ``check``."""
    result = check(source, **options)
    if result.table is None:
        raise InputError(result.problems)
    return result.table


__all__ = [
    "CheckResult",
    "InputError",
    "Problem",
    "Scale",
    "check",
    "hashed_split",
    "read_predictions",
    "suggest_scale",
]

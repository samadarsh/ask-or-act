"""The decision changelog: one JSON line per gate run, append-only.

Inspired by Self-Improvement Memory (arXiv:2611.18856): every change records the
**expected effect before the run**, then the observed metrics, so predicted vs
observed can be checked later. Entries are never rewritten. Two kinds:

* ``expectation``: written by ``ask-or-act changelog expect`` *before* the new
  predictions exist. The next gate run picks up the newest open one.
* ``gate``: one per gate run: the change, its reason, the expectation (its own
  ``--expect``, or the open entry's id and time), the files' hashes, the policy, every
  check's observed values, and whether each prediction held.

**Never backfill a prediction.** A tool can record times but can't prove them, so the
log makes timing visible instead:

* ``expected_recorded`` says when the prediction was written. For ``--expect`` that is
  the gate run itself.
* A gate run on files it has already seen (same new, baseline and policy hashes) is
  marked ``rerun_of`` the first one. That first run's expectation is the one that
  counts, so running, looking, and re-running with a better guess shows up.

The log holds ids, hashes, numbers and the text people write in ``--change``/``--why``,
never input rows.
"""

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ask_or_act import __version__
from ask_or_act.core.types import JSON
from ask_or_act.gate import Expectation, GateResult

SCHEMA = "ask-or-act/changelog/v1"
DEFAULT_PATH = Path("ask-or-act-changelog.jsonl")


class ChangelogError(ValueError):
    """An unreadable changelog (CLI exit 6) or a conflicting request (exit 2)."""


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _id(time: str, body: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return f"{time}-{digest[:4]}"


def file_info(path: Path | None) -> dict[str, JSON] | None:
    if path is None or not path.is_file():
        return None
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def git_info(cwd: Path | None = None) -> dict[str, JSON]:
    """The commit: ``GITHUB_SHA`` in GitHub Actions, else local ``git`` (no network)."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return {"commit": sha[:12], "dirty": None}

    def git(*args: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    commit = git("rev-parse", "--short=12", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


# --------------------------------------------------------------------------- reading


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ChangelogError(f"{path}:{i}: not a JSON line ({exc.msg})") from exc
        if not isinstance(entry, dict) or entry.get("schema") != SCHEMA:
            raise ChangelogError(f"{path}:{i}: not an {SCHEMA} entry")
        entries.append(entry)
    return entries


def append(path: Path, entry: Mapping[str, Any]) -> None:
    if entry.get("schema") != SCHEMA or entry.get("kind") not in ("expectation", "gate"):
        raise ChangelogError(f"not an {SCHEMA} entry")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=False) + "\n")


def open_expectation(entries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The newest ``expectation`` entry that no gate entry has used yet."""
    used = {e.get("expectation_id") for e in entries if e.get("kind") == "gate"}
    for e in reversed(entries):
        if e.get("kind") == "expectation" and e["id"] not in used:
            found: Mapping[str, Any] = e
            return found
    return None


def _key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    f, pol = entry.get("files") or {}, entry.get("policy") or {}
    return (
        (f.get("new") or {}).get("sha256"),
        (f.get("baseline") or {}).get("sha256"),
        json.dumps(pol, sort_keys=True),
    )


# --------------------------------------------------------------------------- writing


def expectation_entry(
    *, change: str, why: str, expectations: Sequence[Expectation], time: str | None = None
) -> dict[str, JSON]:
    if not expectations:
        raise ChangelogError("give at least one --expect")
    t = time or now()
    body: dict[str, JSON] = {
        "schema": SCHEMA,
        "kind": "expectation",
        "time": t,
        "version": __version__,
        **git_info(),
        "change": change,
        "why": why,
        "expected": [e.to_json() for e in expectations],
    }
    return {"id": _id(t, body), **body}


def gate_entry(
    result: GateResult,
    *,
    entries: Sequence[Mapping[str, Any]],
    new_path: Path | None,
    baseline_path: Path | None,
    change: str = "",
    why: str = "",
    context: Mapping[str, JSON] | None = None,
    expectation: Mapping[str, Any] | None = None,
    time: str | None = None,
) -> dict[str, JSON]:
    """The entry for one gate run. ``expectation``: the open entry it closes, if the
    expected effect came from ``changelog expect`` rather than ``--expect``."""
    t = time or now()
    res = result.to_json()
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "gate",
        "time": t,
        "version": __version__,
        **git_info(),
        "change": change or (expectation or {}).get("change", ""),
        "why": why or (expectation or {}).get("why", ""),
        "context": dict(context or {}),
        "expected": [v.expectation.to_json() for v in result.verdicts],
        "expected_recorded": (expectation or {}).get("time", t),
        "expectation_id": (expectation or {}).get("id"),
        "files": {"new": file_info(new_path), "baseline": file_info(baseline_path)},
        "policy": res["policy"],
        "observed": {
            c.name: {
                "status": c.status,
                "new": c.new,
                "baseline": c.baseline,
                "diff": c.diff,
                "low": c.low,
                "high": c.high,
                "p_holm": c.p_holm,
            }
            for c in result.checks
        },
        "verdicts": [
            {
                "text": v.expectation.text,
                "observed": v.observed,
                "held": v.held,
                "beyond_noise": v.beyond_noise,
            }
            for v in result.verdicts
        ],
        "gate": "pass" if result.passed else "fail",
        "rerun_of": None,
    }
    first = next((e for e in entries if e.get("kind") == "gate" and _key(e) == _key(body)), None)
    if first is not None:
        body["rerun_of"] = first.get("rerun_of") or first["id"]
    from ask_or_act.metrics.report import to_jsonable

    plain: dict[str, JSON] = to_jsonable(body)
    return {"id": _id(t, plain), **plain}


# --------------------------------------------------------------------------- showing


def _when(entry: Mapping[str, Any]) -> str:
    """When the expectation was written, relative to the gate run."""
    if not entry.get("expectation_id"):
        return "given with the gate run"
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        gap = datetime.strptime(str(entry["time"]), fmt) - datetime.strptime(
            str(entry.get("expected_recorded")), fmt
        )
    except ValueError:
        return "recorded before the run"
    secs = int(gap.total_seconds())
    if secs < 120:
        return f"recorded {secs} s before the run"
    if secs < 7200:
        return f"recorded {secs // 60} min before the run"
    return f"recorded {secs // 3600} h before the run"


def render(entries: Sequence[Mapping[str, Any]], last: int | None = None) -> str:
    """Newest last, one block per entry, predicted vs observed."""
    by_id = {e["id"]: e for e in entries}
    shown = list(entries[-last:] if last else entries)
    if not shown:
        return "The changelog is empty. ask-or-act gate appends an entry on every run."
    blocks = []
    for e in shown:
        commit = e.get("commit") or "no commit"
        dirty = " + uncommitted changes" if e.get("dirty") else ""
        if e.get("kind") == "expectation":
            head = (
                f"{e['time']}  EXPECTED  {e.get('change') or '(no description)'}  ({commit}{dirty})"
            )
            lines = [head]
            if e.get("why"):
                lines.append(f"  why: {e['why']}")
            lines += [f"  expects {x['text']}" for x in e.get("expected", [])]
            blocks.append("\n".join(lines))
            continue
        verdict = str(e.get("gate", "?")).upper()
        head = (
            f"{e['time']}  {verdict:<4}  {e.get('change') or '(no description)'}  ({commit}{dirty})"
        )
        lines = [head]
        if e.get("why"):
            lines.append(f"  why: {e['why']}")
        failed = [k for k, v in (e.get("observed") or {}).items() if v.get("status") == "fail"]
        unsure = [
            k for k, v in (e.get("observed") or {}).items() if v.get("status") == "inconclusive"
        ]
        if failed:
            lines.append(f"  failed: {', '.join(failed)}")
        if unsure:
            lines.append(f"  inconclusive: {', '.join(unsure)}")
        first = by_id.get(e["rerun_of"]) if e.get("rerun_of") else None
        if e.get("rerun_of"):
            lines.append(
                f"  rerun of {e['rerun_of']}: that run's expectation is the one that counts"
            )
            if e.get("expected") and first is not None and first.get("expected") != e["expected"]:
                lines.append("  ! this rerun's expectation differs from the first run's")
        if e.get("verdicts"):
            lines.append(f"  expected ({_when(e)}):")
            for v in e["verdicts"]:
                metric = str(v["text"]).split()[0].lower()
                obs = v.get("observed")
                shown_obs = "not computable" if obs is None else _obs(metric, float(obs))
                word = "held" if v.get("held") else "missed"
                noise = ", beyond noise" if v.get("beyond_noise") else ""
                lines.append(f"    {v['text']} → {shown_obs}: {word}{noise}")
        else:
            lines.append("  expected: (none recorded)")
        blocks.append("\n".join(lines))
    gates = [e for e in entries if e.get("kind") == "gate" and not e.get("rerun_of")]
    scored = [v for e in gates for v in e.get("verdicts", []) if v.get("held") is not None]
    held = sum(bool(v["held"]) for v in scored)
    tail = (
        f"{len(gates)} gate run(s) (reruns not counted); predictions held {held} of {len(scored)}."
        if scored
        else f"{len(gates)} gate run(s) (reruns not counted); no predictions recorded yet."
    )
    return "\n\n".join(blocks) + "\n\n" + tail


def _obs(metric: str, x: float) -> str:
    return f"{x * 100:+.1f} points" if metric in ("accuracy", "coverage") else f"{x:+.3f}"

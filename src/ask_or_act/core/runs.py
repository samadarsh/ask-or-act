"""Run storage: every run is saved so any number can be traced back and reproduced.

Layout of ``<runs_dir>/<run_id>/``:

* ``run.json``: metadata, including the git commit and whether the tree was dirty,
  the full config and its hash, the dataset manifest and its hash, the seeds,
  timestamps, versions, and the cost summary.
* ``decisions.jsonl``: one Decision per line. No input text.

A run is written to a temporary directory and then renamed into place. A run directory
therefore either exists completely or not at all, so a crash can't leave a half-written
run that looks real.
"""

import json
import os
import platform
import secrets
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ask_or_act.core.types import JSON, Decision, Example


@dataclass(frozen=True)
class GoldRecord:
    """What metrics need to know about an example: everything except its input text."""

    id: str
    gold: str
    options: tuple[str, ...]
    meta: Mapping[str, JSON] = field(default_factory=dict)

    @classmethod
    def of(cls, ex: Example) -> "GoldRecord":
        return cls(ex.id, ex.gold, ex.options, ex.meta)

    def to_json(self) -> dict[str, JSON]:
        return {
            "id": self.id,
            "gold": self.gold,
            "options": list(self.options),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "GoldRecord":
        return cls(data["id"], data["gold"], tuple(data["options"]), data.get("meta", {}))


@dataclass(frozen=True)
class RunRecord:
    path: Path
    metadata: dict[str, JSON]
    decisions: list[Decision]
    examples: list[GoldRecord]  # same order as decisions


def git_info(cwd: Path | None = None) -> dict[str, JSON]:
    """Current commit and whether there are uncommitted changes. Missing git -> nulls."""

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if commit else None
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def environment_info() -> dict[str, JSON]:
    from ask_or_act import __version__

    return {
        "ask_or_act": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def make_run_id(name: str, started_at: datetime) -> str:
    """Sortable by start time; the random suffix keeps two runs in the same second apart."""
    return f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{name}-{secrets.token_hex(3)}"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, JSON]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_run(
    runs_dir: Path,
    run_id: str,
    metadata: Mapping[str, JSON],
    decisions: Sequence[Decision],
    examples: Sequence[Example | GoldRecord],
) -> Path:
    """``examples`` are stored as GoldRecords (id, gold, options, meta), never inputs."""
    if [d.example_id for d in decisions] != [e.id for e in examples]:
        raise ValueError("decisions and examples must be aligned by id")
    runs_dir.mkdir(parents=True, exist_ok=True)
    final = runs_dir / run_id
    if final.exists():
        raise FileExistsError(final)
    tmp = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_dir))
    _write_jsonl(tmp / "decisions.jsonl", (d.to_json() for d in decisions))
    _write_jsonl(
        tmp / "examples.jsonl",
        ((e if isinstance(e, GoldRecord) else GoldRecord.of(e)).to_json() for e in examples),
    )
    (tmp / "run.json").write_text(
        json.dumps({"run_id": run_id, **metadata}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, final)
    return final


def load_run(path: Path) -> RunRecord:
    metadata = json.loads((path / "run.json").read_text(encoding="utf-8"))
    decisions = [Decision.from_json(r) for r in _read_jsonl(path / "decisions.jsonl")]
    examples = [GoldRecord.from_json(r) for r in _read_jsonl(path / "examples.jsonl")]
    return RunRecord(path, metadata, decisions, examples)


# --------------------------------------------------------------------------- partial runs

STOP_REASONS = {"quota_stopped": "daily limit", "budget_stopped": "budget"}


def partial_text(decided: int, total: int, skipped: int, status: str) -> str:
    """ "partial: 21/77 decided (56 skipped: daily limit)". Shared by `run`'s quick check
    and the analysis refusal, so the two always say the same thing."""
    reason = STOP_REASONS.get(status, status)
    other = total - decided - skipped
    extra = f"; {other} other error(s)" if other else ""
    return f"partial: {decided}/{total} decided ({skipped} skipped: {reason}{extra})"


class PartialRunError(ValueError):
    """The run stopped early (daily limit or budget). Its skipped examples are stored
    as errors, and errors score as wrong with confidence 0, so analysing it would mix
    "not asked yet" into accuracy, calibration and thresholds."""


def require_completed(record: RunRecord) -> None:
    """Analysis (metrics, thresholds, judge, report) reads completed runs only. The fix
    for a partial run is to re-run its config after the reset: everything already
    decided comes from the cache, and that new run is saved as completed."""
    status = str(record.metadata.get("status", "unknown"))
    if status == "completed":
        return
    from ask_or_act.core.runner import SKIPPED_PREFIXES

    decided = sum(d.ok for d in record.decisions)
    skipped = sum(bool(d.error and d.error.startswith(SKIPPED_PREFIXES)) for d in record.decisions)
    cfg = record.metadata.get("config")
    config = cfg.get("path", "<config>") if isinstance(cfg, dict) else "<config>"
    raise PartialRunError(
        f"{record.path.name} is not a completed run "
        f"({partial_text(decided, len(record.decisions), skipped, status)}). "
        f"Only completed runs are analysed: re-run `ask-or-act run {config} --yes` after "
        "the reset; cached decisions are reused and the new run is complete."
    )

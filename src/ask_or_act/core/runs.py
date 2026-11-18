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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ask_or_act.core.types import JSON, Decision


@dataclass(frozen=True)
class RunRecord:
    path: Path
    metadata: dict[str, JSON]
    decisions: list[Decision]


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


def save_run(
    runs_dir: Path, run_id: str, metadata: Mapping[str, JSON], decisions: Sequence[Decision]
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    final = runs_dir / run_id
    if final.exists():
        raise FileExistsError(final)
    tmp = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_dir))
    with (tmp / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d.to_json(), ensure_ascii=False) + "\n")
    (tmp / "run.json").write_text(
        json.dumps({"run_id": run_id, **metadata}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, final)
    return final


def load_run(path: Path) -> RunRecord:
    metadata = json.loads((path / "run.json").read_text(encoding="utf-8"))
    with (path / "decisions.jsonl").open(encoding="utf-8") as f:
        decisions = [Decision.from_json(json.loads(line)) for line in f if line.strip()]
    return RunRecord(path, metadata, decisions)

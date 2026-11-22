"""scripts/study-job.sh: when a scheduled run is due, the plist, and a dry run.

macOS only (the script uses BSD ``date``, launchd and caffeinate). Nothing is
installed and nothing is called: ``due`` reads log names only, and the dry run uses
``study status`` instead of ``study run``.

Times are in Asia/Kolkata (UTC+5:30), so 10:00 local is 04:30 UTC, and a new UTC day
starts at 05:30 local. That separates "the 10:00 slot" from "the UTC day".
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS launchd job")

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "study-job.sh"
TZ = "Asia/Kolkata"


def _epoch(local: str) -> int:
    return int(datetime.fromisoformat(local).replace(tzinfo=ZoneInfo(TZ)).timestamp())


def _due(tmp_path: Path, now: str, logs: list[str]) -> tuple[bool, str]:
    for name in logs:
        (tmp_path / name).write_text("")
    env = {
        **os.environ,
        "TZ": TZ,
        "ASK_OR_ACT_LOGS": str(tmp_path),
        "ASK_OR_ACT_NOW": str(_epoch(now)),
    }
    r = subprocess.run(
        ["bash", str(SCRIPT), "due"], env=env, capture_output=True, text=True, check=False
    )
    return r.returncode == 0, r.stdout.strip()


CASES = [
    # (case, now (local), existing logs, due?)
    ("first ever, at 10:00", "2026-11-23T10:00:05", [], True),
    ("first ever, before 10:00: yesterday's slot was missed", "2026-11-23T09:00:00", [], True),
    (
        "ran at 10:00 today; later the same day",
        "2026-11-23T15:00:00",
        ["20261123T043000Z-schedule.log"],
        False,
    ),
    (
        "asleep at 10:00, wakes at 14:00",
        "2026-11-23T14:00:00",
        ["20261122T043000Z-schedule.log"],
        True,
    ),
    (
        "off yesterday at 10:00; boots at 09:00 (a new UTC day): catch up once",
        "2026-11-24T09:00:00",
        ["20261122T043000Z-schedule.log"],
        True,
    ),
    (
        "...and at 10:00 that day: same UTC day, no second run",
        "2026-11-24T10:00:00",
        ["20261122T043000Z-schedule.log", "20261124T033000Z-schedule.log"],
        False,
    ),
    (
        "manual run late last night; 08:00 is not a new slot",
        "2026-11-24T08:00:00",
        ["20261123T173000Z-schedule.log"],
        False,
    ),
    (
        "manual run late last night; 10:00 is the new slot",
        "2026-11-24T10:00:00",
        ["20261123T173000Z-schedule.log"],
        True,
    ),
    (
        "dry runs and launchd.log never count",
        "2026-11-23T15:00:00",
        ["20261123T043000Z-dryrun.log", "launchd.log"],
        True,
    ),
]


@pytest.mark.parametrize(("case", "now", "logs", "expected"), CASES, ids=[c[0] for c in CASES])
def test_due_rule(tmp_path: Path, case: str, now: str, logs: list[str], expected: bool) -> None:
    due, reason = _due(tmp_path, now, logs)
    assert due is expected, (case, reason)
    assert reason.startswith("due:" if expected else "not due:")


def test_plist_is_valid_and_holds_no_secret(tmp_path: Path) -> None:
    out = subprocess.run(
        ["bash", str(SCRIPT), "plist"], capture_output=True, text=True, check=True
    ).stdout
    path = tmp_path / "job.plist"
    path.write_text(out)
    subprocess.run(["plutil", "-lint", str(path)], check=True, capture_output=True)
    assert "GROQ" not in out
    assert ".env" not in out.replace("the key is read from .env", "")
    for key in ("StartCalendarInterval", "RunAtLoad", "StartInterval"):
        assert f"<key>{key}</key>" in out


def test_the_script_never_prints_the_key() -> None:
    text = SCRIPT.read_text()
    assert "GROQ_API_KEY=" not in text.replace("'^GROQ_API_KEY='", "")
    for leak in ("cat .env", "source .env", ". .env", "echo $GROQ", "${GROQ"):
        assert leak not in text


def test_a_run_from_the_main_folder_is_refused_before_any_call(tmp_path: Path) -> None:
    """Only the pinned study worktree may call the API (docs/study.md)."""
    env = {
        **os.environ,
        "ASK_OR_ACT_LOGS": str(tmp_path),
        "ASK_OR_ACT_JOB_DRY": "1",
        "ASK_OR_ACT_NOTIFY": "0",
    }
    r = subprocess.run(
        ["bash", str(SCRIPT), "run"], env=env, capture_output=True, text=True, check=True
    )
    assert "notification suppressed: preflight failed" in r.stdout
    (log,) = tmp_path.glob("*-dryrun.log")
    text = log.read_text()
    assert "not the study worktree" in text
    assert "result: ATTENTION: preflight failed; nothing was called" in text
    assert "=== study run" not in text


def test_the_plist_runs_the_study_worktree_and_logs_to_the_main_folder() -> None:
    out = subprocess.run(
        ["bash", str(SCRIPT), "plist"], capture_output=True, text=True, check=True
    ).stdout
    assert f"<string>{ROOT}-study/scripts/study-job.sh</string>" in out
    assert f"<string>{ROOT}/logs/launchd.log</string>" in out

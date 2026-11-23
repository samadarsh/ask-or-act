"""scripts/study-job.sh: when a scheduled run is due, the plist, and a dry run.

macOS only (the script uses BSD ``date``, launchd and caffeinate). Nothing is
installed and nothing is called: ``due`` reads log names only, and the dry run uses
``study status`` instead of ``study run``.

Times are in Asia/Kolkata (UTC+5:30), so 10:00 local is 04:30 UTC, and a new UTC day
starts at 05:30 local. The UTC date plays no part in the rule (study-v3): Groq's
allowance refills gradually, so a run is due whenever none has started since the most
recent 10:00 slot. The replay tests below drive the real ``due`` through whole weeks.
"""

import itertools
import os
import subprocess
import sys
from datetime import datetime, timedelta
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
        "...and at 10:00 that day: the normal run, back on the slot",
        "2026-11-24T10:00:00",
        ["20261122T043000Z-schedule.log", "20261124T033000Z-schedule.log"],
        True,
    ),
    (
        "catch-up at 06:00 (after the UTC day changed); 10:00 still runs",
        "2026-11-24T10:00:00",
        ["20261122T043000Z-schedule.log", "20261124T003000Z-schedule.log"],
        True,
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


# --------------------------------------------------------------------------- replays

DAY = timedelta(days=1)
HOUR = timedelta(hours=1)


def _replay(
    tmp_path: Path,
    on: list[tuple[str, str]],
    manual: tuple[str, ...] = (),
    before: tuple[str, ...] = ("2026-11-22T10:00:00",),
) -> list[datetime]:
    """Replay launchd over a stretch of days and return every run's start (local).

    While the Mac is on (each ``on`` interval, boot to shutdown) launchd checks at
    boot (RunAtLoad), every hour after boot (StartInterval 3600) and at 10:00
    (StartCalendarInterval). Each check calls the real ``due``; when it says due, a
    run starts and its log appears, as in ``run``. ``manual`` are scripts/daily.sh
    runs, ``before`` runs that happened before the replay starts. Sleep is the same
    as off here: launchd skips the checks and fires one on wake."""
    tz = ZoneInfo(TZ)

    def log(at: datetime) -> None:
        name = at.astimezone(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ-schedule.log")
        (tmp_path / name).write_text("")

    runs = [datetime.fromisoformat(t).replace(tzinfo=tz) for t in before]
    for at in runs:
        log(at)
    events: list[tuple[datetime, str]] = []
    for start, end in on:
        boot = datetime.fromisoformat(start).replace(tzinfo=tz)
        stop = datetime.fromisoformat(end).replace(tzinfo=tz)
        t = boot
        while t < stop:
            events.append((t, "check"))  # RunAtLoad at boot, then hourly
            t += HOUR
        ten = boot.replace(hour=10, minute=0, second=0)
        ten = ten if ten >= boot else ten + DAY
        while ten < stop:
            events.append((ten, "check"))
            ten += DAY
    events += [(datetime.fromisoformat(t).replace(tzinfo=tz), "manual") for t in manual]
    for at, kind in sorted(events):
        if kind == "check":
            due, _ = _due(tmp_path, at.isoformat(timespec="seconds")[:19], [])
            if not due:
                continue
        log(at)
        runs.append(at)
    return sorted(runs)


def _assert_gaps(runs: list[datetime], on: list[tuple[str, str]]) -> None:
    """While the Mac is on, never more than 24 h without a run: within each on-stretch,
    from boot (or the last run) to the next run, and from the last run to shutdown."""
    tz = ZoneInfo(TZ)
    for start, end in on:
        a = datetime.fromisoformat(start).replace(tzinfo=tz)
        b = datetime.fromisoformat(end).replace(tzinfo=tz)
        inside = [r for r in runs if a <= r < b]
        for r1, r2 in itertools.pairwise([a, *inside, b]):
            assert r2 - r1 <= DAY, f"{r1:%d %H:%M} -> {r2:%d %H:%M} while on"


def _hm(runs: list[datetime]) -> list[str]:
    return [f"{r:%d %H:%M}" for r in runs]


def test_replay_normal_days_run_at_10(tmp_path: Path) -> None:
    on = [("2026-11-23T00:00:00", "2026-11-27T00:00:00")]
    runs = _replay(tmp_path, on)
    assert _hm(runs) == ["22 10:00", "23 10:00", "24 10:00", "25 10:00", "26 10:00"]
    _assert_gaps(runs, on)


@pytest.mark.parametrize("boot", ["06:00", "08:00", "09:00"])
def test_replay_missed_day_catches_up_then_10_00(tmp_path: Path, boot: str) -> None:
    # Off over the 23rd's 10:00; boots on the 24th before 10:00 (06:00 is after the
    # UTC day changed at 05:30, the case the old rule got wrong).
    on = [
        ("2026-11-22T10:30:00", "2026-11-23T09:00:00"),
        (f"2026-11-24T{boot}:00", "2026-11-27T00:00:00"),
    ]
    runs = _replay(tmp_path, on)
    assert _hm(runs) == ["22 10:00", f"24 {boot}", "24 10:00", "25 10:00", "26 10:00"]
    _assert_gaps(runs, on)


def test_replay_missed_day_boot_after_10_00(tmp_path: Path) -> None:
    on = [
        ("2026-11-22T10:30:00", "2026-11-23T09:00:00"),
        ("2026-11-24T14:00:00", "2026-11-27T00:00:00"),
    ]
    runs = _replay(tmp_path, on)
    assert _hm(runs) == ["22 10:00", "24 14:00", "25 10:00", "26 10:00"]
    _assert_gaps(runs, on)


def test_replay_manual_run_late_at_night(tmp_path: Path) -> None:
    on = [("2026-11-23T00:00:00", "2026-11-26T00:00:00")]
    runs = _replay(tmp_path, on, manual=("2026-11-23T23:30:00",))
    assert _hm(runs) == ["22 10:00", "23 10:00", "23 23:30", "24 10:00", "25 10:00"]
    _assert_gaps(runs, on)


def test_replay_manual_run_early_morning_still_keeps_10_00(tmp_path: Path) -> None:
    on = [("2026-11-23T00:00:00", "2026-11-26T00:00:00")]
    runs = _replay(tmp_path, on, manual=("2026-11-24T07:00:00",))
    assert _hm(runs) == ["22 10:00", "23 10:00", "24 07:00", "24 10:00", "25 10:00"]
    _assert_gaps(runs, on)


def test_replay_mac_off_for_two_days(tmp_path: Path) -> None:
    on = [
        ("2026-11-22T10:30:00", "2026-11-23T12:00:00"),
        ("2026-11-25T19:00:00", "2026-11-29T00:00:00"),
    ]
    runs = _replay(tmp_path, on)
    # One catch-up on boot (not one per missed day), then back on the 10:00 slot.
    assert _hm(runs) == ["22 10:00", "23 10:00", "25 19:00", "26 10:00", "27 10:00", "28 10:00"]
    _assert_gaps(runs, on)


def test_due_sees_runs_through_a_linked_logs_folder(tmp_path: Path) -> None:
    """In the study worktree ``logs`` is a link to the main folder's. BSD ``find``
    doesn't follow a linked start folder without -H; then no run is ever seen and the
    hourly check would start a run every hour (found 2026-11-22)."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "20261123T043000Z-schedule.log").write_text("")
    link = tmp_path / "logs"
    link.symlink_to(real)
    env = {
        **os.environ,
        "TZ": TZ,
        "ASK_OR_ACT_LOGS": str(link),
        "ASK_OR_ACT_NOW": str(_epoch("2026-11-23T15:00:00")),
    }
    r = subprocess.run(
        ["bash", str(SCRIPT), "due"], env=env, capture_output=True, text=True, check=False
    )
    assert r.returncode == 1, r.stdout
    assert r.stdout.startswith("not due: the last run")
    assert "20261123T043000Z" in r.stdout


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

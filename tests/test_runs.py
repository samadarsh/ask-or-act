import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ask_or_act.core.runs import (
    GoldRecord,
    environment_info,
    git_info,
    load_run,
    make_run_id,
    save_run,
)
from ask_or_act.core.types import Decision, Example


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    decisions = [Decision("a", "x", 0.9, cost_usd=0.1), Decision.failed("b", "boom")]
    examples = [
        Example("a", "SECRET INPUT", ("x", "y"), "x", meta={"judge": "j1"}),
        Example("b", "SECRET INPUT", ("x", "y"), "y"),
    ]
    path = save_run(tmp_path, "run-1", {"name": "t", "seed": 7}, decisions, examples)
    record = load_run(path)
    assert record.metadata == {"run_id": "run-1", "name": "t", "seed": 7}
    assert record.decisions == decisions
    assert record.examples == [GoldRecord.of(e) for e in examples]
    # Gold labels and metadata are saved; input text never is.
    assert "SECRET" not in (path / "examples.jsonl").read_text()
    # Atomic write: no temp directories left behind.
    assert [p.name for p in tmp_path.iterdir()] == ["run-1"]


def test_never_overwrites_a_run(tmp_path: Path) -> None:
    save_run(tmp_path, "run-1", {}, [], [])
    with pytest.raises(FileExistsError):
        save_run(tmp_path, "run-1", {}, [], [])


def test_decisions_and_examples_must_align(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aligned"):
        save_run(tmp_path, "r", {}, [Decision("a", "x", 0.5)], [GoldRecord("b", "x", ("x", "y"))])


def test_run_ids_are_timestamped_and_unique_within_a_second() -> None:
    t = datetime(2026, 11, 18, 9, 30, tzinfo=UTC)
    ids = {make_run_id("demo", t) for _ in range(50)}
    assert len(ids) == 50
    assert all(i.startswith("20261118T093000Z-demo-") for i in ids)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_git_info_records_commit_and_dirty_flag(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("1")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    info = git_info(tmp_path)
    assert isinstance(info["commit"], str)
    assert len(info["commit"]) == 40
    assert info["dirty"] is False
    (tmp_path / "f.txt").write_text("2")
    assert git_info(tmp_path)["dirty"] is True


def test_git_info_outside_a_repo(tmp_path: Path) -> None:
    assert git_info(tmp_path) == {"commit": None, "dirty": None}


def test_environment_info() -> None:
    info = environment_info()
    assert set(info) == {"ask_or_act", "python", "platform"}

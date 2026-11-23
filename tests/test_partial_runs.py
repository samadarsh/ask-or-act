"""Partial runs (stopped at the daily limit or the budget) are never analysed. No network.

A partial run stores its skipped examples as errors, and errors score as wrong with
confidence 0. Analysing one would therefore read "not asked yet" as "answered wrong".
Every analysis entry point (metrics, thresholds, judge and report, through both run
converters) must refuse it and accept the completed re-run that the cache makes cheap.
The `run` quick check must say "partial" and score the decided examples only.
"""

from pathlib import Path
from typing import Any

import pytest

from ask_or_act.cli import main
from ask_or_act.core.runs import PartialRunError, load_run
from ask_or_act.metrics.data import from_run as scored_from_run
from ask_or_act.report.table import from_run as table_from_run
from ask_or_act.run import EXIT_OK, EXIT_QUOTA, run_from_config
from tests.provider_fixtures import Recorder, groq_429, groq_ok
from tests.test_groq import KEY, groq_config
from tests.test_run_end_to_end import write_world


@pytest.fixture
def two_days(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, list[str]]:
    """Day 1 stops at the daily limit after 2 of 6 examples; day 2 completes from the
    cache. Returns (partial run, completed run, day 1's output lines)."""
    manifest_path = write_world(tmp_path, monkeypatch)  # 6 test examples
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    cfg = groq_config(tmp_path, manifest_path)
    day1_lines: list[str] = []

    def call(transport: Recorder, lines: list[str]) -> Any:
        return run_from_config(
            cfg,
            raw_dir=tmp_path,
            out=lines.append,
            transport=transport.transport,
            approve_spend=True,
        )

    day1 = Recorder(
        [groq_ok("card_arrival", 90)] * 4 + [groq_429("requests per day (RPD)", retry_after="3600")]
    )
    first = call(day1, day1_lines)
    assert first.exit_code == EXIT_QUOTA
    second = call(Recorder([groq_ok("top_up", 70)] * 8), [])
    assert second.exit_code == EXIT_OK
    return first.run_dir, second.run_dir, day1_lines


def test_the_quick_check_says_partial_and_scores_decided_only(
    two_days: tuple[Path, Path, list[str]],
) -> None:
    partial, _, lines = two_days
    text = "\n".join(lines)
    assert "partial: 2/6 decided (4 skipped: daily limit)" in text
    assert "accuracy on the 2 decided only (quick check; partial run, not analysed):" in text
    assert "done: " not in text  # "done" is for completed runs
    record = load_run(partial)
    golds = {e.id: e.gold for e in record.examples}
    decided = [d for d in record.decisions if d.ok]
    expected = sum(d.choice == golds[d.example_id] for d in decided) / len(decided)
    assert f"not analysed): {expected:.3f}" in text  # out of 2, not out of 6


def test_both_run_converters_refuse_a_partial_run(
    two_days: tuple[Path, Path, list[str]],
) -> None:
    partial, complete, _ = two_days
    with pytest.raises(PartialRunError, match=r"partial: 2/6 decided \(4 skipped: daily limit\)"):
        scored_from_run(load_run(partial))
    with pytest.raises(PartialRunError, match="re-run `ask-or-act run"):
        table_from_run(partial)
    # The completed re-run (2 decisions from the cache, 4 new) is accepted by both.
    assert len(scored_from_run(load_run(complete))) == 6
    assert len(table_from_run(complete).rows) == 6


@pytest.mark.parametrize(
    "argv",
    [
        ["metrics", "{partial}"],
        ["metrics", "{complete}", "{partial}"],
        ["thresholds", "--cal", "{partial}"],
        ["judge", "{partial}"],
        ["report", "{partial}", "-o", "{tmp}/r.html"],
    ],
    ids=["metrics", "metrics-paired", "thresholds", "judge", "report"],
)
def test_every_analysis_command_refuses_a_partial_run(
    argv: list[str],
    two_days: tuple[Path, Path, list[str]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    partial, complete, _ = two_days
    args = [a.format(partial=partial, complete=complete, tmp=tmp_path) for a in argv]
    assert main(args) == 2
    err = capsys.readouterr().err
    assert f"{partial.name} is not a completed run" in err
    assert "Only completed runs are analysed" in err
    assert not (tmp_path / "r.html").exists()


def test_the_completed_rerun_is_analysed(
    two_days: tuple[Path, Path, list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    _, complete, _ = two_days
    assert main(["metrics", str(complete), "--n-boot", "50"]) == 0
    assert "not a completed run" not in capsys.readouterr().err

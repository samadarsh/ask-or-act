"""The report's input boundary: a generic table of decisions. Fakes only, no network."""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from ask_or_act.core.runs import load_run, save_run
from ask_or_act.core.types import JSON, Decision
from ask_or_act.data.sampling import Split
from ask_or_act.metrics.data import from_run as scored_from_run
from ask_or_act.metrics.report import Settings
from ask_or_act.report import build as report
from ask_or_act.report.table import DecisionRow, DecisionTable, from_run
from tests.helpers import fake
from tests.test_report import SECRET, _examples, _save

FAST = Settings(n_boot=30)


def test_a_saved_run_converts_to_the_same_arrays_as_the_metrics_loader(tmp_path: Path) -> None:
    examples = _examples(Split.TEST, 50)
    decisions = asyncio.run(fake("overconfident").decide_batch(examples))
    decisions[3] = Decision.failed(examples[3].id, "timeout")  # errors: wrong, confidence 0
    meta: dict[str, JSON] = {
        "name": "x",
        "dataset": {"name": "banking77", "split": "test", "limit": 50},
    }
    path = save_run(tmp_path, "2026-x", meta, decisions, examples)

    table = from_run(path)
    a, b = table.scored, scored_from_run(load_run(path))
    for name in ("ids", "gold", "pred", "probabilities", "options", "groups", "meta", "dataset"):
        assert getattr(a, name) == getattr(b, name), name
    np.testing.assert_array_equal(a.confidence, b.confidence)
    np.testing.assert_array_equal(a.correct, b.correct)
    assert a.n_errors == b.n_errors == 1
    assert table.split == "test"
    assert all(SECRET not in str(r) for r in table.rows)  # input text never enters the table


def _user_table(n: int = 400, seed: int = 0) -> DecisionTable:
    """A table as a user's CSV would give it: both splits in one file, no run, no config."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        conf = float(rng.uniform(0.5, 1.0))
        right = bool(rng.uniform() < conf)
        rows.append(
            DecisionRow(
                id=f"row{k}",
                prediction=None if k == 7 else ("yes" if right else "no"),
                confidence=conf,
                label="yes",
                split="calibration" if k % 2 else "test",
                options=("yes", "no"),
            )
        )
    return DecisionTable(name="my-model", dataset="my_data", rows=tuple(rows), source="mine.csv")


def test_a_user_table_with_both_splits_gets_metrics_and_exploratory_thresholds() -> None:
    table = _user_table()
    halves = table.by_split()
    assert [h.split for h in halves] == ["calibration", "test"]
    assert [len(h) for h in halves] == [200, 200]

    html = report.build([table], seen_path=None, settings=FAST)
    assert "<code>mine.csv</code>" in html
    assert "my_data" in html
    # No registered target for user data: every threshold row is EXPLORATORY.
    assert html.count('badge pre">PRE-REGISTERED') == 1  # the legend only
    assert html.count('badge expl">EXPLORATORY') >= len(report.EXPLORATORY_ALPHAS)


def test_runs_and_tables_mix_in_one_report(tmp_path: Path) -> None:
    run = _save(tmp_path / "runs", "cheap-test", "calibrated", Split.TEST)
    html = report.build([run, _user_table()], seen_path=None, settings=FAST)
    assert "cheap-test" in html
    assert "my-model" in html


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ((), "at least one row"),
        ((DecisionRow("a", "x", 0.5, "x"), DecisionRow("a", "x", 0.5, "x")), "duplicate ids"),
        ((DecisionRow("a", "x", 1.5, "x"),), r"outside \[0, 1\]"),
        ((DecisionRow("a", "x", 0.5, "x", split="dev"),), "unknown split"),
    ],
)
def test_bad_tables_are_refused(rows: tuple[DecisionRow, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DecisionTable(name="t", dataset="d", rows=rows)


def test_two_tables_with_the_same_name_and_split_are_refused() -> None:
    t = _user_table()
    with pytest.raises(ValueError, match="share a key"):
        report.build([t, t], seen_path=None, settings=FAST)

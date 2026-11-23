"""The public API: find_thresholds, evaluate, compare (docs/cli-design.md §5.3). No network.

The statistics underneath (guarantees/, metrics/) have their own tests; these check
what the API adds: loading any source, halves, refusals, unlabelled coverage,
cascades, the summaries and the JSON.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ask_or_act import api
from ask_or_act.predictions import InputError, read_predictions
from ask_or_act.report.table import DecisionRow, DecisionTable

FIXTURES = Path(__file__).parent / "fixtures" / "predictions"
SMALL = FIXTURES / "support-bot.csv"
LARGE = FIXTURES / "support-bot-large.csv"
N_BOOT = 100


def _calibrated(n: int, seed: int = 0, unlabelled: int = 0) -> DecisionTable:
    """A well-calibrated model: P(correct) = confidence."""
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.5, 1.0, n).round(3)
    right = rng.uniform(size=n) < conf
    rows = tuple(
        DecisionRow(
            id=f"r{i}",
            prediction="a" if right[i] else "b",
            confidence=float(conf[i]),
            label=None if i < unlabelled else "a",
            split="calibration" if i % 2 else "test",
        )
        for i in range(n)
    )
    return DecisionTable(name="m", dataset="d", rows=rows, split_source="column")


def test_a_certified_threshold_holds_on_the_test_half() -> None:
    th = api.find_thresholds(_calibrated(4000))
    assert th.certified
    assert th.threshold is not None
    assert 0.5 < th.threshold < 1.0
    assert th.test_risk is not None
    assert th.test_risk <= 0.1 + 0.03  # the guarantee holds w.p. ≥ 0.9; margin for noise
    assert th.summary().startswith(f"Act when confidence ≥ {th.threshold:.3g}")
    assert set(th.to_json()) >= {"certified", "threshold", "coverage", "calibration", "test"}


def test_too_few_labelled_rows_raise_with_the_numbers() -> None:
    with pytest.raises(api.NotEnoughData) as info:
        api.find_thresholds(FIXTURES / "pilot.csv")
    e = info.value
    assert (e.have, e.need, e.typical) == (19, 36, 114)
    assert "--alpha 0.2 needs 17" in e.hint


def test_nothing_certified_is_an_answer_with_the_closest_attempt() -> None:
    th = api.find_thresholds(SMALL)
    assert not th.certified
    assert th.closest is not None
    assert (th.closest.threshold, th.closest.n, th.closest.wrong) == (0.999, 130, 8)
    assert "Act on nothing" in th.summary()
    assert "Closest attempt: 130 calibration rows" in th.summary()


def test_unlabelled_rows_get_a_coverage_estimate() -> None:
    t = _calibrated(4000, unlabelled=500)
    th = api.find_thresholds(t)
    assert th.unlabelled == 500
    lam = th.threshold
    assert lam is not None
    expected = sum(
        r.label is None and r.confidence is not None and r.confidence >= lam - 1e-9 for r in t.rows
    )
    assert th.unlabelled_acted == expected


def test_a_cascade_routes_to_the_second_model() -> None:
    th = api.find_thresholds([SMALL, LARGE], alpha=0.2)
    assert [s.name for s in th.stages] == ["support-bot", "support-bot-large"]
    assert all(s.threshold is not None for s in th.stages)
    text = th.summary()
    assert "otherwise ask support-bot-large" in text
    assert "escalate the remaining" in text


def test_explicit_halves_use_whole_files(tmp_path: Path) -> None:
    t = _calibrated(4000)
    cal = DecisionTable(name="cal", dataset="d", rows=t.rows[:2000])
    test = DecisionTable(name="test", dataset="d", rows=t.rows[2000:])
    th = api.find_thresholds(cal, test=test)
    assert (th.n_cal, th.n_test) == (2000, 2000)


def test_files_with_different_ids_are_an_input_error() -> None:
    a = _calibrated(400)
    b = DecisionTable(name="b", dataset="d", rows=a.rows[:300])
    with pytest.raises(InputError, match="different examples"):
        api.compare(a, b, n_boot=N_BOOT)


def test_evaluate_scores_the_test_half_unless_asked_for_all() -> None:
    test_half = api.evaluate(SMALL, n_boot=N_BOOT)
    everything = api.evaluate(SMALL, rows="all", n_boot=N_BOOT)
    assert (test_half.n, everything.n) == (1000, 2000)
    assert "overstates accuracy" in test_half.summary()  # the fixture is overconfident
    json.dumps(test_half.to_json(), default=float)


def test_evaluate_on_a_well_calibrated_model_says_so() -> None:
    ev = api.evaluate(_calibrated(4000), n_boot=N_BOOT)
    assert "no clear miscalibration" in ev.summary()


def test_compare_finds_the_better_model_and_reports_coverage_at_alpha() -> None:
    c = api.compare(SMALL, LARGE, n_boot=N_BOOT)
    assert c.report["tests"]["accuracy"]["better"] == "support-bot-large"
    text = c.summary()
    assert text.startswith("support-bot-large is more accurate than support-bot")
    assert "support-bot acts on nothing, support-bot-large acts on" in text


def test_identical_files_show_no_difference() -> None:
    t = _calibrated(600)
    c = api.compare(t, t, n_boot=N_BOOT)
    assert c.names == ("m (A)", "m (B)")
    assert "no clear difference" in c.summary()


def test_every_source_kind_loads() -> None:
    table = read_predictions(SMALL)
    cols = {
        "id": [r.id for r in table.rows],
        "prediction": [r.prediction or "" for r in table.rows],
        "confidence": [r.confidence if r.confidence is not None else "" for r in table.rows],
        "human_label": [r.label or "" for r in table.rows],
        "split": [r.split for r in table.rows],
    }
    sources: list[api.Data] = [SMALL, str(SMALL), table, cols]  # type: ignore[list-item]
    for source in sources:
        assert len(api.load(source).table) == 2400


def test_a_folder_without_run_json_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match=r"without a run\.json"):
        api.load(tmp_path)


def test_public_names_are_importable_from_the_package() -> None:
    import ask_or_act as aoa

    assert aoa.find_thresholds is api.find_thresholds
    assert aoa.evaluate is api.evaluate
    assert aoa.compare is api.compare
    assert aoa.read_predictions is read_predictions

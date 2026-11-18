"""Fakes on the real committed samples (ids + gold labels; no network, no raw text).

These checks matter most before spending money. The metrics must:

* call the calibrated fake calibrated;
* catch the overconfident fake's ~0.125 gap;
* find a real difference when there is one;
* not invent one when there isn't.
"""

import asyncio

import pytest

from ask_or_act.metrics.calibration import ece
from ask_or_act.metrics.report import Settings, compare_report, single_report
from ask_or_act.metrics.stats import mcnemar
from tests.helpers import fake
from tests.metrics_helpers import run_fake, sample_examples

FAST = Settings(n_boot=300)


@pytest.mark.parametrize(("name", "n"), [("banking77", 770), ("aegis", 500), ("mt_bench", 500)])
def test_calibrated_vs_overconfident(name: str, n: int) -> None:
    ex = sample_examples(name)
    assert len(ex) == n
    cal = run_fake(fake("calibrated", seed=7), ex, name)
    over = run_fake(fake("overconfident", seed=7), ex, name)

    assert ece(cal.confidence, cal.correct) < 0.06  # ~0 plus finite-sample ECE bias
    gap = float(over.confidence.mean() - over.correct.mean())
    assert gap == pytest.approx(0.125, abs=0.03)
    assert ece(over.confidence, over.correct) == pytest.approx(gap, abs=0.02)

    if name == "mt_bench":
        return  # mt_bench rows have no model names in the manifest; see test below
    r = compare_report(cal, over, FAST)
    # Same correctness by construction: no accuracy difference may be "found"...
    assert r["mcnemar"]["p_value"] == 1.0
    assert r["paired"]["accuracy"]["diff"] == 0
    # ...but the calibration difference must be.
    assert r["tests"]["ece_width"]["p_holm"] < 0.05
    assert r["tests"]["ece_width"]["better"] == "A"


def test_detects_real_accuracy_difference_on_banking77() -> None:
    ex = sample_examples("banking77")
    weak = run_fake(fake("calibrated", seed=1, low=0.5), ex, "banking77")
    strong = run_fake(fake("calibrated", seed=2, low=0.75), ex, "banking77")
    r = compare_report(strong, weak, FAST)
    assert r["tests"]["accuracy"]["p_holm"] < 0.01
    assert r["tests"]["accuracy"]["better"] == "A"
    assert r["paired"]["accuracy"]["low"] > 0


def test_false_positive_rate_on_banking77() -> None:
    """20 pairs of same-quality fakes with different seeds: expect ~1 rejection at 0.05;
    allow at most 4 (P[≥5 | Binomial(20, 0.05)] ≈ 0.003)."""
    ex = sample_examples("banking77")
    rejections = sum(
        mcnemar(
            run_fake(fake("calibrated", seed=s), ex).correct,
            run_fake(fake("calibrated", seed=s + 1000), ex).correct,
        ).p_value
        < 0.05
        for s in range(20)
    )
    assert rejections <= 4


def test_single_report_shape_on_aegis() -> None:
    ex = sample_examples("aegis")
    r = single_report(run_fake(fake("calibrated", seed=3), ex, "aegis"), FAST)
    assert r["n"] == 500
    assert set(r["metrics"]) >= {
        "accuracy",
        "macro_f1",
        "cohen_kappa",
        "ece_width",
        "ece_mass",
        "brier",
        "aurc",
        "auroc",
        "brier_multiclass",
    }
    for m in r["metrics"].values():
        assert m["low"] <= m["estimate"] <= m["high"] or m["estimate"] is None
    assert [f["alpha"] for f in r["feasibility"]] == [0.01, 0.05, 0.1, 0.2]


def test_noise_floor_on_banking77() -> None:
    ex = sample_examples("banking77")
    cal = single_report(run_fake(fake("calibrated", seed=7), ex, "banking77"), FAST)
    over = single_report(run_fake(fake("overconfident", seed=7), ex, "banking77"), FAST)
    for key in ("ece_width", "ece_mass"):
        assert cal["metrics"][key]["estimate"] <= cal["noise_floor"][key]["p95"]
        assert over["metrics"][key]["estimate"] > over["noise_floor"][key]["p95"]


def test_errors_reported_with_and_without() -> None:
    from dataclasses import replace

    from ask_or_act.core.types import Decision
    from ask_or_act.metrics.report import render_single
    from tests.metrics_helpers import scored

    ex = sample_examples("aegis")
    decisions = asyncio.run(fake("overconfident", seed=2).decide_batch(ex))
    # Every 4th call failed: as "wrong, confidence 0" they'd look perfectly calibrated.
    decisions = [
        Decision.failed(d.example_id, "timeout") if i % 4 == 0 else d
        for i, d in enumerate(decisions)
    ]
    r = single_report(scored(decisions, ex, "aegis"), FAST)
    assert r["n_errors"] == 125
    ex_err = r["excluding_errors"]
    assert ex_err["n"] == 375
    # Errors flatter ECE: it's lower with them than without.
    assert r["metrics"]["ece_width"]["estimate"] < ex_err["metrics"]["ece_width"]["estimate"]
    assert "accuracy" not in ex_err["metrics"]  # only confidence-based metrics are repeated
    assert "excluding 125 errored" in render_single(r, "run")
    no_errors = single_report(
        scored([replace(d) for d in asyncio.run(fake("calibrated").decide_batch(ex))], ex, "aegis"),
        FAST,
    )
    assert "excluding_errors" not in no_errors

"""Bootstrap, paired tests, McNemar and Holm: correctness, power, and false positives."""

from collections import Counter

import numpy as np
import pytest
from scipy.stats import binomtest

from ask_or_act.metrics.stats import (
    bootstrap_ci,
    holm,
    mcnemar,
    paired_bootstrap,
    resamples,
)
from tests.helpers import fake, make_examples
from tests.metrics_helpers import run_fake


@pytest.mark.parametrize(("b", "c"), [(0, 0), (3, 0), (10, 4), (25, 40), (100, 130)])
def test_mcnemar_matches_scipy_exact_binomial(b: int, c: int) -> None:
    a_right = np.array([True] * b + [False] * c + [True] * 50)
    b_right = np.array([False] * b + [True] * c + [True] * 50)
    result = mcnemar(a_right, b_right)
    assert (result.only_a_right, result.only_b_right) == (b, c)
    expected = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    assert result.p_value == pytest.approx(expected)


def test_holm_hand_computed() -> None:
    # sorted: .005×4=.02, .01×3=.03, .03×2=.06, .04×1=.04 → running max keeps .06
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm([]) == []


def test_bootstrap_is_seeded_and_covers_the_mean() -> None:
    x = np.random.default_rng(0).normal(10.0, 2.0, 400)

    def mean(i: np.ndarray) -> float:
        return float(x[i].mean())

    ci = bootstrap_ci(mean, 400, n_boot=1000, seed=3)
    assert ci == bootstrap_ci(mean, 400, n_boot=1000, seed=3)
    assert ci.low is not None
    assert ci.high is not None
    assert ci.low < 10.0 < ci.high
    assert ci.high - ci.low == pytest.approx(2 * 1.96 * 2.0 / 20, rel=0.2)


def test_group_resampling_keeps_clusters_whole() -> None:
    groups = ["q1"] * 3 + ["q2"] * 2 + ["q3"] * 4
    for idx in resamples(9, n_boot=50, seed=1, groups=groups):
        per_group = Counter(groups[i] for i in idx)
        counts = Counter(int(i) for i in idx)
        for i, g in enumerate(groups):
            # every member of a group appears exactly as often as the group was drawn
            assert counts[i] * {"q1": 3, "q2": 2, "q3": 4}[g] == per_group[g]


def test_undefined_resamples_are_skipped() -> None:
    def sometimes(i: np.ndarray) -> float | None:
        return None if i[0] % 2 else 1.0

    ci = bootstrap_ci(sometimes, 10, n_boot=100, seed=0)
    assert 0 < ci.n_boot < 100


def test_identical_deciders_give_p_one() -> None:
    s = run_fake(fake("calibrated", seed=1), make_examples(300))
    r = paired_bootstrap(
        lambda i: float(s.correct[i].mean()), lambda i: float(s.correct[i].mean()), 300, n_boot=200
    )
    assert r.diff == 0
    assert r.p_value == 1.0


def test_detects_a_real_accuracy_difference() -> None:
    ex = make_examples(770, k=4)
    weak = run_fake(fake("calibrated", seed=1, low=0.5), ex)  # ~75% accurate
    strong = run_fake(fake("calibrated", seed=2, low=0.7), ex)  # ~85% accurate
    assert mcnemar(strong.correct, weak.correct).p_value < 1e-4
    r = paired_bootstrap(
        lambda i: float(strong.correct[i].mean()),
        lambda i: float(weak.correct[i].mean()),
        770,
        n_boot=500,
    )
    assert r.p_value is not None
    assert r.p_value < 0.01
    assert r.low is not None
    assert r.low > 0


def test_does_not_invent_differences() -> None:
    """Same decider type, different seeds: true accuracy is equal, so ~5% of tests at
    α = 0.05 reject by chance. Over 40 independent pairs we allow at most 6 rejections
    (P[≥7 | Binomial(40, 0.05)] ≈ 0.003)."""
    ex = make_examples(500, k=4)
    rejections = 0
    for k in range(40):
        a = run_fake(fake("calibrated", seed=100 + 2 * k), ex)
        b = run_fake(fake("calibrated", seed=101 + 2 * k), ex)
        rejections += mcnemar(a.correct, b.correct).p_value < 0.05
    assert rejections <= 6

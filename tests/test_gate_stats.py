"""The gate's statistics: Tango's score test, the vectorised threshold refit, the
shifted bootstrap and the --expect grammar. No network."""

import itertools
import math

import numpy as np
import pytest

from ask_or_act.gate import Expectation, _boot
from ask_or_act.guarantees.binomial import p_value
from ask_or_act.guarantees.fixed_sequence import (
    DEFAULT_STARTS,
    FastFit,
    grid_for,
    max_errors,
    multi_start,
)
from ask_or_act.metrics.tango import interval, paired_accuracy, q21_hat, z_score

# Tango (1998), Table I: b = 0, Δ = 0.1; Z and the 90% interval (Eq. (24), (33)).
TABLE_I = [
    (30, 0, 1.83, (-0.083, 0.083)),
    (30, 1, 1.22, (-0.136, 0.052)),
    (30, 2, 0.61, (-0.183, 0.022)),
    (50, 0, 2.36, (-0.051, 0.051)),
    (50, 1, 1.89, (-0.085, 0.032)),
    (50, 2, 1.41, (-0.114, 0.013)),
    (80, 0, 2.98, (-0.033, 0.033)),
    (80, 1, 2.61, (-0.054, 0.021)),
    (80, 2, 2.24, (-0.073, 0.009)),
]


@pytest.mark.parametrize(("n", "c", "z", "ci"), TABLE_I)
def test_tango_reproduces_table_1(n: int, c: int, z: float, ci: tuple[float, float]) -> None:
    assert round(z_score(0, c, n, 0.1), 2) == z
    lo, hi = interval(0, c, n, 0.90)
    assert (round(lo, 3), round(hi, 3)) == ci


@pytest.mark.parametrize(("b", "c"), [(10, 4), (3, 9), (1, 0), (25, 25)])
def test_with_delta_zero_it_is_mcnemar(b: int, c: int) -> None:
    """Eq. (31): q̂21 = (b + c)/2n and Z = (b − c)/√(b + c)."""
    assert q21_hat(b, c, 200, 0.0) == pytest.approx((b + c) / 400)
    assert z_score(b, c, 200, 0.0) == pytest.approx((b - c) / math.sqrt(b + c))


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_zero_off_diagonal_interval_is_eq_34(n: int) -> None:
    z2 = 1.6448536269514722**2
    lo, hi = interval(0, 0, n, 0.90)
    assert lo == pytest.approx(-z2 / (n + z2), abs=1e-9)
    assert hi == pytest.approx(z2 / (n + z2), abs=1e-9)


def test_boundary_mle_of_eq_27() -> None:
    """b = 0 and c < 2nΔ/(1 + Δ): the MLE is on the boundary, q̂21 = Δ (Eq. (27))."""
    assert q21_hat(0, 3, 100, 0.1) == pytest.approx(0.1)


@pytest.mark.parametrize(("b", "c", "n"), [(0, 0, 50), (5, 12, 300), (40, 3, 100), (0, 9, 60)])
def test_z_decreases_in_lambda(b: int, c: int, n: int) -> None:
    """What the interval's bisection relies on."""
    lams = np.linspace(-0.99, 0.99, 397)
    zs = [z_score(b, c, n, -lam) for lam in lams]
    assert all(z1 >= z2 for z1, z2 in itertools.pairwise(zs))


def test_paired_accuracy_direction_and_counts() -> None:
    base = np.array([True] * 80 + [False] * 20)
    new = base.copy()
    new[:15] = False  # 15 the baseline got right, the new one doesn't
    new[90:95] = True  # 5 the other way
    r = paired_accuracy(new, base, tolerance=0.01, level=0.05)
    assert (r.b, r.c, r.n) == (5, 15, 100)
    assert r.diff == pytest.approx(-0.10)
    assert r.low < r.diff < r.high
    assert r.p_worse == pytest.approx(1 - r.p_within)
    # identical files: no evidence either way at Δ = 0, but well within Δ = 0.05
    same = paired_accuracy(base, base, tolerance=0.0, level=0.05)
    assert same.p_worse == 0.5
    assert paired_accuracy(base, base, tolerance=0.05, level=0.05).p_within < 0.05


def test_max_errors_is_the_p_value_test() -> None:
    k = max_errors(300, 0.1, 0.025)
    for n in (1, 7, 30, 120, 300):
        for e in range(n + 1):
            assert (e <= k[n]) == (p_value(e, n, 0.1) <= 0.025), (n, e)
    assert k[0] == -1
    assert all(a <= b for a, b in itertools.pairwise(k))


@pytest.mark.parametrize("kind", ["continuous", "verbalized", "self_consistency"])
def test_fast_fit_equals_multi_start(kind: str) -> None:
    rng = np.random.default_rng(7)
    for _ in range(60):
        n = int(rng.integers(15, 700))
        if kind == "continuous":
            grid, conf = grid_for(None), rng.beta(5, 1, n)
        elif kind == "verbalized":
            grid, conf = grid_for("verbalized"), np.round(rng.beta(6, 1, n) * 100) / 100
        else:
            grid, conf = grid_for("self_consistency", samples=5), rng.integers(0, 6, n) / 5
        starts = DEFAULT_STARTS[kind]
        correct = rng.uniform(size=n) < np.clip(conf * rng.uniform(0.8, 1.05), 0, 1)
        alpha = float(rng.choice([0.03, 0.1, 0.25]))
        want = multi_start(
            conf, correct, alpha=alpha, delta=0.1, grid=grid, starts=starts
        ).threshold
        fit = FastFit(alpha=alpha, delta=0.1, grid=grid, starts=starts, n=n // 2)  # grows
        assert fit.threshold(conf, correct) == want


def test_shifted_bootstrap_p_values_and_interval() -> None:
    draws = np.random.default_rng(0).normal(0.0, 0.01, 4000)  # spread of an estimate
    # an improvement of −5 points with a 1-point tolerance: clearly worse
    pw, pi, lo, hi = _boot(-0.05, draws - 0.05, 0.01, 0.05)
    assert pw < 0.01 < pi
    assert hi < -0.01
    # an estimate of 0: within a 3-point tolerance, not shown worse
    pw, pi, lo, hi = _boot(0.0, draws, 0.03, 0.05)
    assert pi < 0.05 < pw
    assert lo == pytest.approx(-0.0165, abs=0.002)


@pytest.mark.parametrize(
    ("text", "metric", "low", "high"),
    [
        ("accuracy +0 to +2 points", "accuracy", 0.0, 0.02),
        ("accuracy −1 to 1", "accuracy", -0.01, 0.01),
        ("Coverage >= -5 points", "coverage", -0.05, math.inf),
        ("calibration <= 0", "calibration", -math.inf, 0.0),
        ("ece -0.02 to 0.01 abs", "ece", -0.02, 0.01),
        ("calibration ≤ .01", "calibration", -math.inf, 0.01),
    ],
)
def test_expectations_parse(text: str, metric: str, low: float, high: float) -> None:
    e = Expectation.parse(text)
    assert e.metric == metric
    assert e.low == pytest.approx(low)
    assert e.high == pytest.approx(high)
    assert Expectation.from_json(e.to_json()) == e


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("accuracy bogus", "can't read"),
        ("speed +0 to +2", "unknown metric 'speed'"),
        ("accuracy 2 to 0 points", "high to low"),
        ("calibration -0.02 to 0 points", "measured in abs"),
        ("coverage 0 to 1 abs", "measured in points"),
    ],
)
def test_bad_expectations_say_why(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Expectation.parse(text)

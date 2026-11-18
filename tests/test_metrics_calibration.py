"""Calibration metrics against scikit-learn, brute-force references and hand values."""

import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from ask_or_act.metrics.calibration import (
    bin_ids,
    brier_multiclass,
    brier_top_label,
    ece,
    reliability,
)

rng = np.random.default_rng(7)


def random_scores(n: int, decimals: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    c = rng.uniform(0.0, 1.0, n)
    if decimals is not None:
        c = np.round(c, decimals)
    z = rng.random(n) < c**1.5  # deliberately miscalibrated
    return c, z


def ece_reference(c: np.ndarray, z: np.ndarray, n_bins: int) -> float:
    """Independent, loop-based equal-width ECE with sklearn's (lo, hi] edge rule."""
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (c > lo) & (c <= hi) if b > 0 else (c >= lo) & (c <= hi)
        if mask.any():
            total += mask.sum() / len(c) * abs(z[mask].mean() - c[mask].mean())
    return float(total)


@pytest.mark.parametrize("n_bins", [5, 10, 15])
def test_reliability_matches_sklearn_calibration_curve(n_bins: int) -> None:
    c, z = random_scores(2000)
    prob_true, prob_pred = calibration_curve(z, c, n_bins=n_bins, strategy="uniform")
    bins = reliability(c, z, n_bins=n_bins, strategy="width")
    assert [b.accuracy for b in bins] == pytest.approx(list(prob_true))
    assert [b.mean_confidence for b in bins] == pytest.approx(list(prob_pred))


@pytest.mark.parametrize("decimals", [None, 1, 2])
def test_equal_width_ece_matches_reference(decimals: int | None) -> None:
    c, z = random_scores(1500, decimals)  # rounding puts many values exactly on edges
    assert ece(c, z, n_bins=10) == pytest.approx(ece_reference(c, z, 10))


def test_ece_hand_computed() -> None:
    c = np.array([0.9, 0.9, 0.9, 0.9, 0.3, 0.3])
    z = np.array([True, True, True, False, False, True])
    # bin(0.9): acc 0.75, conf 0.9, weight 4/6; bin(0.3): acc 0.5, conf 0.3, weight 2/6
    assert ece(c, z, n_bins=10) == pytest.approx(4 / 6 * 0.15 + 2 / 6 * 0.2)


def test_perfect_calibration_scores_zero() -> None:
    c = np.array([0.25] * 4 + [0.75] * 4)
    z = np.array([True, False, False, False, True, True, True, False])
    assert ece(c, z) == pytest.approx(0.0)
    assert ece(c, z, strategy="mass") == pytest.approx(0.0)


def test_equal_mass_bins_have_equal_size_without_ties() -> None:
    c = rng.permutation(np.linspace(0.01, 0.99, 300))
    sizes = np.bincount(bin_ids(c, 15, "mass"))
    assert len(sizes) == 15
    assert set(sizes) == {20}


def test_equal_mass_keeps_ties_together_and_is_order_independent() -> None:
    c = np.array([0.5] * 50 + [0.6] * 7 + [0.9] * 43)
    ids = bin_ids(c, 4, "mass")
    for value in (0.5, 0.6, 0.9):
        assert len(set(ids[c == value])) == 1, "tied values split across bins"
    z = rng.random(100) < c
    perm = rng.permutation(100)
    assert ece(c, z, strategy="mass") == pytest.approx(ece(c[perm], z[perm], strategy="mass"))


def test_weights_equal_duplicating_rows() -> None:
    c, z = random_scores(300, decimals=2)
    w = rng.integers(1, 4, size=300).astype(np.float64)
    rep = np.repeat(np.arange(300), w.astype(int))
    assert ece(c, z, weights=w) == pytest.approx(ece(c[rep], z[rep]))
    assert brier_top_label(c, z, weights=w) == pytest.approx(brier_top_label(c[rep], z[rep]))


def test_brier_matches_sklearn() -> None:
    c, z = random_scores(500)
    assert brier_top_label(c, z) == pytest.approx(brier_score_loss(z, c))
    w = rng.uniform(0.1, 3.0, 500)
    assert brier_top_label(c, z, weights=w) == pytest.approx(
        brier_score_loss(z, c, sample_weight=w)
    )


def test_brier_multiclass_hand_computed() -> None:
    probs = [{"a": 0.7, "b": 0.2, "c": 0.1}, {"a": 0.5, "b": 0.5}]
    # (0.3² + 0.2² + 0.1²) = 0.14 ; (0.5² + 0.5² + 0²) = 0.5 → mean 0.32
    assert brier_multiclass(probs, ["a", "b"], ["a", "b", "c"]) == pytest.approx(0.32)
    assert brier_multiclass([{}, {"a": 1.0}], ["a", "a"], ["a", "b"]) is None
    assert brier_multiclass([{"zzz": 1.0}], ["a"], ["a", "b"]) is None


def test_invalid_binning() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        bin_ids(np.array([0.5]), 0, "width")
    with pytest.raises(ValueError, match="unknown binning"):
        bin_ids(np.array([0.5]), 5, "log")  # type: ignore[arg-type]


def test_noise_floor_separates_noise_from_miscalibration() -> None:
    from ask_or_act.metrics.calibration import ece_noise_floor

    c = rng.uniform(0.5, 1.0, 770)
    calibrated = rng.random(770) < c
    overconfident_c = c + 0.5 * (1 - c)  # same outcomes, inflated confidence
    floor = ece_noise_floor(c, n_bins=15, seed=1)
    assert 0.0 < floor.mean < floor.p95 < 0.08
    assert ece(c, calibrated) <= floor.p95
    over_floor = ece_noise_floor(overconfident_c, n_bins=15, seed=1)
    assert ece(overconfident_c, calibrated) > over_floor.p95
    # Seeded and reproducible; the mass strategy works too.
    assert floor == ece_noise_floor(c, n_bins=15, seed=1)
    assert ece_noise_floor(c, strategy="mass", seed=1).mean > 0


def test_noise_floor_matches_brute_force_simulation() -> None:
    from ask_or_act.metrics.calibration import ece_noise_floor

    c = np.round(rng.uniform(0, 1, 300), 2)
    sim_rng = np.random.default_rng(5)
    z = sim_rng.random((200, 300)) < c
    brute = np.array([ece(c, row, n_bins=10) for row in z])
    floor = ece_noise_floor(c, n_bins=10, n_sim=200, seed=5)
    assert floor.mean == pytest.approx(float(brute.mean()))
    assert floor.p95 == pytest.approx(float(np.quantile(brute, 0.95)))


def test_noise_floor_empty() -> None:
    from ask_or_act.metrics.calibration import ece_noise_floor

    assert np.isnan(ece_noise_floor(np.zeros(0)).mean)

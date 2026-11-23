"""The fast bootstrap paths give the same numbers as the reference implementations.

Speed-ups in metrics/ (streamed resamples, one pass over all metrics, integer-coded
labels, vectorized ECE) must not change any result. Each is checked against the
original function on random data and random resamples.
"""

import numpy as np
import pytest

from ask_or_act.metrics.calibration import ece, ece_fast
from ask_or_act.metrics.classification import LabelCodes, cohen_kappa, macro_f1
from ask_or_act.metrics.report import Settings, single_report, stats_for
from ask_or_act.metrics.stats import bootstrap_ci, bootstrap_cis, iter_resamples, resamples

RNG = np.random.default_rng(7)


@pytest.mark.parametrize("n", [1, 2, 3, 770, 1001, 50_000])
def test_streamed_resamples_equal_one_bulk_draw(n: int) -> None:
    """Pins numpy's behaviour: n_boot draws of size n == one (n_boot, n) draw."""
    bulk = np.random.default_rng(0).integers(0, n, size=(20, n))
    streamed = np.stack(list(iter_resamples(n, n_boot=20, seed=0)))
    assert np.array_equal(bulk, streamed)
    assert all(
        np.array_equal(a, b) for a, b in zip(resamples(n, n_boot=20, seed=0), bulk, strict=True)
    )


def _labels(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    names = np.array([f"c{j}" for j in range(k)])
    gold = names[RNG.integers(0, k, n)]
    pred = np.where(RNG.uniform(size=n) < 0.7, gold, names[RNG.integers(0, k, n)])
    return gold, pred


@pytest.mark.parametrize(("n", "k"), [(50, 2), (300, 7), (40, 30)])
def test_label_codes_equal_macro_f1_and_kappa_on_resamples(n: int, k: int) -> None:
    gold, pred = _labels(n, k)
    codes = LabelCodes(gold, pred)
    for idx in iter_resamples(n, n_boot=200, seed=1):
        assert codes.macro_f1(idx) == pytest.approx(macro_f1(gold[idx], pred[idx]), abs=1e-12)
        ref = cohen_kappa(gold[idx], pred[idx])
        got = codes.cohen_kappa(idx)
        assert (got is None) == (ref is None)
        if ref is not None:
            assert got == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("strategy", ["width", "mass"])
def test_ece_fast_equals_ece(strategy: str) -> None:
    for n in (1, 5, 200, 3000):
        c = RNG.uniform(size=n).round(RNG.choice([2, 3, 6]))
        z = RNG.uniform(size=n) < c
        for bins in (1, 10, 15):
            fast = ece_fast(c, z, n_bins=bins, strategy=strategy)  # type: ignore[arg-type]
            assert fast == pytest.approx(ece(c, z, n_bins=bins, strategy=strategy), abs=1e-12)  # type: ignore[arg-type]


def test_one_pass_cis_equal_separate_cis() -> None:
    n = 400
    c = RNG.uniform(0.3, 1.0, n)
    z = RNG.uniform(size=n) < c
    stats = {
        "acc": lambda i: float(z[i].mean()),
        "conf": lambda i: float(c[i].mean()),
    }
    together = bootstrap_cis(stats, n, n_boot=300, seed=3)
    for name, fn in stats.items():
        assert together[name] == bootstrap_ci(fn, n, n_boot=300, seed=3)


def test_single_report_cis_equal_the_reference_functions() -> None:
    """The full report's CIs, recomputed with the original (slow) functions."""
    from ask_or_act.metrics.data import Scored

    n = 300
    gold, pred = _labels(n, 5)
    conf = RNG.uniform(0.2, 1.0, n).round(3)
    s = Scored(
        ids=tuple(f"r{i}" for i in range(n)),
        gold=tuple(gold),
        pred=tuple(pred),
        confidence=conf,
        correct=gold == pred,
        probabilities=tuple({} for _ in range(n)),
        options=(),
        groups=tuple(f"r{i}" for i in range(n)),
        meta=tuple({} for _ in range(n)),
        dataset="x",
        n_errors=0,
    )
    settings = Settings(n_boot=200)
    got = single_report(s, settings)["metrics"]
    z = s.correct
    reference = {
        "macro_f1": lambda i: macro_f1(gold[i], pred[i]),
        "cohen_kappa": lambda i: cohen_kappa(gold[i], pred[i]),
        "ece_width": lambda i: ece(conf[i], z[i], n_bins=15, strategy="width"),
        "ece_mass": lambda i: ece(conf[i], z[i], n_bins=15, strategy="mass"),
    }
    for name, fn in reference.items():
        ci = bootstrap_ci(fn, n, n_boot=200, seed=settings.seed)
        for key, value in (("estimate", ci.estimate), ("low", ci.low), ("high", ci.high)):
            assert got[name][key] == pytest.approx(value, abs=1e-12), (name, key)
    assert set(stats_for(s, 15)) >= set(reference)


@pytest.mark.parametrize("decimals", [1, 2, 4])  # coarse values: many ties
def test_confidence_levels_equal_aurc_excess_aurc_and_auroc(decimals: int) -> None:
    from ask_or_act.metrics.selective import ConfidenceLevels, aurc, auroc, excess_aurc

    n = 500
    c = RNG.uniform(size=n).round(decimals)
    z = RNG.uniform(size=n) < c
    levels = ConfidenceLevels(c, z)
    for idx in iter_resamples(n, n_boot=200, seed=2):
        assert levels.aurc(idx) == pytest.approx(aurc(c[idx], z[idx]), abs=1e-12)
        assert levels.excess_aurc(idx) == pytest.approx(excess_aurc(c[idx], z[idx]), abs=1e-12)
        ref, got = auroc(c[idx], z[idx]), levels.auroc(idx)
        assert (got is None) == (ref is None)
        if ref is not None:
            assert got == pytest.approx(ref, abs=1e-12)
    every = np.zeros(n, dtype=bool)  # all wrong: AUROC undefined
    assert ConfidenceLevels(c, every).auroc(np.arange(n)) is None


def test_chunked_noise_floor_equals_one_big_draw() -> None:
    """The noise floor simulates in chunks; the numbers equal the all-at-once version."""
    from ask_or_act.metrics.calibration import bin_ids, ece_noise_floor

    for n in (7, 1000, 30_000):  # 30,000 rows → several chunks of 66 simulations
        c = RNG.uniform(size=n).round(3)
        ids = bin_ids(c, 15, "width")
        onehot = np.zeros((n, int(ids.max()) + 1))
        onehot[np.arange(n), ids] = 1.0
        z = np.random.default_rng(0).random((500, n)) < c
        sims = np.abs(z.astype(np.float64) @ onehot - c @ onehot).sum(axis=1) / n
        got = ece_noise_floor(c, n_bins=15, strategy="width", seed=0)
        assert got.mean == pytest.approx(float(sims.mean()), abs=1e-15)
        assert got.p95 == pytest.approx(float(np.quantile(sims, 0.95)), abs=1e-15)

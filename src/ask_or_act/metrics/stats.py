"""Uncertainty and significance: bootstrap CIs, paired comparisons, McNemar, Holm.

**Bootstrap.** Resample the examples with replacement many times, recompute the metric
each time, and read the spread off the percentiles. Everything is seeded, so a CI is
exactly reproducible.

When examples come in clusters that aren't independent, such as several MT-Bench votes
on the same question, the *clusters* are resampled instead (``groups=``). Resampling
single rows would treat correlated votes as independent evidence and give CIs that are
too narrow.

**Paired.** Two deciders on the same examples are resampled with the *same* indices, so
per-example difficulty cancels out. This is much more sensitive than comparing two
independent CIs.
"""

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ask_or_act.metrics.data import BoolArray, IntArray

type Stat = Callable[[IntArray], float | None]

DEFAULT_N_BOOT = 2000  # as in arXiv 2606.30814, Appendix E.6
DEFAULT_SEED = 0


def iter_resamples(
    n: int, *, n_boot: int, seed: int, groups: Sequence[str] | None = None
) -> Iterator[IntArray]:
    """``n_boot`` index arrays, one at a time (memory O(n), not O(n_boot·n)). With
    ``groups``, whole groups are drawn with replacement.

    Drawing row by row gives exactly the indices that one (n_boot, n) draw gives, so
    every CI is unchanged; tests/test_metrics_speed.py pins that."""
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_boot):
            yield rng.integers(0, n, size=n)
        return
    if len(groups) != n:
        raise ValueError("groups must have one entry per example")
    members: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    blocks = [np.array(v, dtype=np.intp) for v in members.values()]
    for _ in range(n_boot):
        picks = rng.integers(0, len(blocks), size=len(blocks))
        yield np.concatenate([blocks[k] for k in picks])


def resamples(
    n: int, *, n_boot: int, seed: int, groups: Sequence[str] | None = None
) -> list[IntArray]:
    """All of ``iter_resamples`` as a list (small n only: it holds n_boot·n indices)."""
    return list(iter_resamples(n, n_boot=n_boot, seed=seed, groups=groups))


def _valid(values: list[float | None]) -> np.ndarray:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    return arr[np.isfinite(arr)]


@dataclass(frozen=True)
class CI:
    estimate: float | None
    low: float | None
    high: float | None
    level: float
    n_boot: int  # resamples where the metric was defined


def bootstrap_ci(
    stat: Stat,
    n: int,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
    groups: Sequence[str] | None = None,
) -> CI:
    """Percentile bootstrap CI for ``stat`` (a function of example indices)."""
    return bootstrap_cis({"x": stat}, n, n_boot=n_boot, seed=seed, level=level, groups=groups)["x"]


def _ci(point: float | None, values: list[float | None], level: float) -> CI:
    vals = _valid(values)
    if point is None or len(vals) == 0:
        return CI(point, None, None, level, len(vals))
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return CI(float(point), float(lo), float(hi), level, len(vals))


def bootstrap_cis(
    stats: Mapping[str, Stat],
    n: int,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
    groups: Sequence[str] | None = None,
) -> dict[str, CI]:
    """``bootstrap_ci`` for several statistics over the *same* resamples, drawn once.
    Each result equals a separate ``bootstrap_ci`` call with the same seed."""
    full = np.arange(n, dtype=np.intp)
    points = {k: f(full) for k, f in stats.items()}
    values: dict[str, list[float | None]] = {k: [] for k in stats}
    for idx in iter_resamples(n, n_boot=n_boot, seed=seed, groups=groups):
        for k, f in stats.items():
            values[k].append(f(idx))
    return {k: _ci(points[k], values[k], level) for k in stats}


@dataclass(frozen=True)
class PairedResult:
    a: float | None
    b: float | None
    diff: float | None  # a − b
    low: float | None
    high: float | None
    p_value: float | None
    n_boot: int


def paired_bootstrap(
    stat_a: Stat,
    stat_b: Stat,
    n: int,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
    groups: Sequence[str] | None = None,
) -> PairedResult:
    """CI for the difference a − b, and a two-sided p-value for "no difference".

    The p-value uses the *centered* bootstrap: under the null, the observed difference d
    is as extreme as the resampled ones shifted to mean zero, so
    p = (1 + #{|d*_b − d| ≥ |d|}) / (B + 1). The +1 keeps p > 0 with finitely many
    resamples. Identical deciders give d = 0 and p = 1.
    """
    return paired_bootstraps(
        {"x": (stat_a, stat_b)}, n, n_boot=n_boot, seed=seed, level=level, groups=groups
    )["x"]


def paired_bootstraps(
    pairs: Mapping[str, tuple[Stat, Stat]],
    n: int,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
    groups: Sequence[str] | None = None,
) -> dict[str, PairedResult]:
    """``paired_bootstrap`` for several metrics over the same resamples, drawn once."""
    full = np.arange(n, dtype=np.intp)
    points = {k: (fa(full), fb(full)) for k, (fa, fb) in pairs.items()}
    live = {k for k, (a, b) in points.items() if a is not None and b is not None}
    diffs: dict[str, list[float | None]] = {k: [] for k in live}
    for idx in iter_resamples(n, n_boot=n_boot, seed=seed, groups=groups):
        for k in live:
            fa, fb = pairs[k]
            ra, rb = fa(idx), fb(idx)
            diffs[k].append(None if ra is None or rb is None else ra - rb)
    out = {}
    for k, (a, b) in points.items():
        out[k] = (
            _paired(a, b, diffs[k], level)
            if k in live
            else PairedResult(a, b, None, None, None, None, 0)
        )
    return out


def _paired(
    a: float | None, b: float | None, diffs: list[float | None], level: float
) -> PairedResult:
    assert a is not None
    assert b is not None
    d = a - b
    vals = _valid(diffs)
    if len(vals) == 0:
        return PairedResult(a, b, d, None, None, None, 0)
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    extreme = int(np.sum(np.abs(vals - d) >= abs(d) - 1e-12))
    p = (1 + extreme) / (len(vals) + 1)
    return PairedResult(a, b, d, float(lo), float(hi), min(1.0, p), len(vals))


@dataclass(frozen=True)
class McNemar:
    only_a_right: int  # b in the usual 2×2 notation
    only_b_right: int  # c
    p_value: float


def mcnemar(correct_a: BoolArray, correct_b: BoolArray) -> McNemar:
    """Exact McNemar test: are the disagreements split evenly between the two deciders?

    Only examples where exactly one decider is right carry information. Under the null
    each of those is a fair coin flip, so the p-value is a two-sided binomial test with
    p = ½ on the discordant pairs. The exact test is used because it stays valid when
    there are few disagreements, unlike the χ² approximation.
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return McNemar(b, c, 1.0)
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return McNemar(b, c, min(1.0, 2.0 * tail))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (controls the family-wise error rate).

    Sort ascending and multiply the k-th smallest (0-based) by (m − k). A running
    maximum keeps the adjusted values in order. Compare each adjusted p to α directly.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted

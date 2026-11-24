"""Act-or-escalate threshold with a high-probability risk guarantee (fixed-sequence test).

**What is guaranteed.** Given calibration examples with confidences and correctness,
pick λ̂ so that, with probability ≥ 1 − δ over the calibration sample,

    P(wrong | confidence ≥ λ̂) ≤ α        on new data from the same distribution.

So "act when confidence ≥ λ̂, otherwise escalate" has selective risk ≤ α (human
agreement ≥ 1 − α for a judge).

**Sources (read in full for this code, 2026-11-20).**

* Procedure: Jung, Brahman, Choi, *Trust or Escalate*, arXiv:2407.18370 **v1**, §2.1,
  **Eq. (4)** (exact binomial upper bound, see binomial.py), **Eq. (5)**
  λ̂ = inf{λ : R̂⁺(λ′) ≤ α for all λ′ ≥ λ}, and **Theorem 1** (Eq. (2) holds with
  probability ≥ 1 − δ). Thresholds are tested from the largest down, and the scan
  stops at the first failure (fixed-sequence testing, Bauer 1991).
* Proof of validity for the *selected* λ̂: Kotte, arXiv:2606.29054 **v2**, Appendix
  E.2, **Lemma E.1** (descending fixed-sequence scan). The certified thresholds form a
  prefix of a pre-registered order, so a false certification requires certifying the
  first true null λ† in that order, which a level-δ test does with probability ≤ δ.
  No correction over the number of thresholds and no monotonicity of risk in λ is
  needed. (ToE's own Appendix A.1 shows the bound is valid for a *fixed* λ and then
  applies it at λ̂; Lemma E.1 is the argument that licenses that step.)

**Our choices, and why.**

* **The grid is fixed before seeing data** and depends only on the signal's scale
  (``grid_for``): verbalized confidence is an integer percentage, so {1.00, 0.99, …,
  0}; self-consistency with k samples is {k/k, …, 0/k}; anything else a 0.001 grid.
  Lemma E.1 needs λ† to be a fixed point of a fixed order. Kotte's implementation
  restricts candidates to n(λ) ≥ 20, which makes the order depend on the calibration
  scores. Its proof handles that by conditioning on the scores, but then the risk
  being controlled is the average over those scores, not the population
  P(wrong | confidence ≥ λ). We avoid the issue: no data-dependent candidate filter.
* **The cost of that choice:** an emit set that is too small to pass at the top of the
  grid (e.g. few answers at confidence 100) fails and stops the scan. Grid points
  with the same emit set as the previous one give the same test, so sparse
  confidences cost nothing extra. ``start`` lets you pre-register a lower top
  (e.g. 0.99). It must be chosen without looking at calibration labels, or the
  guarantee is void.
* **Ties** are handled as blocks: an emit set is "confidence ≥ λ", never a cut inside
  a group of equal confidences (arXiv:2608.04899 §3).
* **Errors** (failed calls) count as wrong with confidence 0, as in metrics/data.py.
  They are emitted only at λ = 0.

**Assumption:** calibration and future examples are i.i.d. (exchangeable). MT-Bench
votes on the same question are not independent: there the effective sample size is
closer to the number of questions, and the guarantee is optimistic. See ``recommend``.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ask_or_act.guarantees.binomial import p_value, upper_bound
from ask_or_act.metrics.data import BoolArray, FloatArray

_EPS = 1e-9  # confidences like 0.29 are stored as floats; compare with a margin


def grid_for(signal: str | None, *, samples: int | None = None) -> tuple[float, ...]:
    """The pre-registered descending threshold grid for a confidence signal."""
    if signal == "verbalized":
        return tuple(j / 100 for j in range(100, -1, -1))
    if signal == "self_consistency":
        if not samples or samples < 1:
            raise ValueError("self_consistency grid needs the number of samples k")
        return tuple(j / samples for j in range(samples, -1, -1))
    return tuple(j / 1000 for j in range(1000, -1, -1))


@dataclass(frozen=True)
class Step:
    threshold: float
    n: int  # calibration examples emitted (confidence >= threshold)
    errors: int
    p_value: float
    certified: bool  # p_value <= δ: the test; the upper bound below is for reporting
    delta: float

    @property
    def upper_bound(self) -> float:
        """ToE Eq. (4) at level δ. Computed on first use only: it's for reports, and a
        bisection over a binomial sum is slow for large emit sets. R̂⁺ ≤ α ⇔ p ≤ δ, so
        it never changes a decision (binomial.py)."""
        return upper_bound(self.errors, self.n, self.delta)


@dataclass(frozen=True)
class FixedSequenceResult:
    alpha: float
    delta: float
    n: int
    threshold: float | None  # λ̂; None: nothing certified, escalate everything
    coverage: float  # share of calibration examples acted on at λ̂
    risk: float | None  # empirical risk of those
    steps: tuple[Step, ...]  # distinct emit sets tested, in scan order, until the stop
    grid_size: int

    @property
    def certified(self) -> bool:
        return self.threshold is not None


def emit_counts(confidence: FloatArray, correct: BoolArray, threshold: float) -> tuple[int, int]:
    """(n emitted, errors among them) at ``confidence >= threshold``."""
    mask = confidence >= threshold - _EPS
    return int(mask.sum()), int((mask & ~correct).sum())


def fixed_sequence(
    confidence: FloatArray,
    correct: BoolArray,
    *,
    alpha: float,
    delta: float,
    grid: Sequence[float],
    start: float | None = None,
) -> FixedSequenceResult:
    """ToE §2.1 Eq. (5): scan ``grid`` downwards from ``start`` (default: its top),
    certify while the exact binomial test passes at level δ, stop at the first failure.
    λ̂ is the smallest certified threshold."""
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("need 0 < alpha < 1 and 0 < delta < 1")
    if len(confidence) != len(correct):
        raise ValueError("confidence and correct must have the same length")
    order = [float(x) for x in grid]
    if any(b >= a for a, b in itertools.pairwise(order)):
        raise ValueError("grid must be strictly decreasing (scan order)")
    if start is not None:
        order = [x for x in order if x <= start + _EPS]
    conf = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.bool_)
    steps: list[Step] = []
    last: tuple[int, int] | None = None
    threshold: float | None = None
    for lam in order:
        n, errors = emit_counts(conf, ok, lam)
        if (n, errors) == last:  # same emit set as the previous grid point: same test
            threshold = lam if steps and steps[-1].certified else threshold
            continue
        last = (n, errors)
        p = p_value(errors, n, alpha)
        passed = p <= delta
        steps.append(Step(lam, n, errors, p, passed, delta))
        if not passed:
            break
        threshold = lam
    n_total = len(conf)
    if threshold is None:
        return FixedSequenceResult(alpha, delta, n_total, None, 0.0, None, tuple(steps), len(order))
    n_emit, n_err = emit_counts(conf, ok, threshold)
    return FixedSequenceResult(
        alpha,
        delta,
        n_total,
        threshold,
        n_emit / max(1, n_total),
        n_err / n_emit if n_emit else None,
        tuple(steps),
        len(order),
    )


# FROZEN 2026-11-21, before any LLM results existed; chosen using only the free
# baselines and fake deciders (docs/guarantees.md, "Pre-registration"). Do not change
# after seeing LLM results: that would void the guarantee's pre-registration.
# tests/test_guarantees.py pins these values.
# Pre-registered scan starts per grid kind (chosen from the scale, before any data).
# One start is exactly ToE §2.1; several guard against an empty or tiny emit set at
# the top of the scale stopping the only scan at once (e.g. continuous probabilities
# never reach 1.000; a model may rarely say "100").
DEFAULT_STARTS = {
    "verbalized": (1.0, 0.99, 0.95, 0.9),
    "self_consistency": (1.0,),
    "continuous": (0.999, 0.99, 0.95, 0.9),
}


@dataclass(frozen=True)
class MultiStartResult:
    """Several fixed-sequence scans, each from a pre-registered start at level δ/S.

    Validity: each scan is a fixed-sequence test (Kotte v2 Lemma E.1), so it falsely
    certifies with probability ≤ δ/S; a union bound over the S scans gives ≤ δ for
    the union of their certified thresholds. (ToE A.2 splits δ over cascade stages
    the same way.) λ̂ is the smallest threshold any scan certified."""

    alpha: float
    delta: float
    scans: tuple[FixedSequenceResult, ...]
    starts: tuple[float, ...]

    @property
    def threshold(self) -> float | None:
        certified = [s.threshold for s in self.scans if s.threshold is not None]
        return min(certified) if certified else None

    @property
    def certified(self) -> bool:
        return self.threshold is not None

    @property
    def steps(self) -> tuple[Step, ...]:
        return tuple(step for s in self.scans for step in s.steps)

    def coverage_and_risk(
        self, confidence: FloatArray, correct: BoolArray
    ) -> tuple[float, float | None]:
        lam = self.threshold
        if lam is None:
            return 0.0, None
        n, errors = emit_counts(confidence, correct, lam)
        return n / max(1, len(confidence)), (errors / n if n else None)


def multi_start(
    confidence: FloatArray,
    correct: BoolArray,
    *,
    alpha: float,
    delta: float,
    grid: Sequence[float],
    starts: Sequence[float],
) -> MultiStartResult:
    if not starts:
        raise ValueError("need at least one start")
    if len(set(starts)) != len(starts):
        raise ValueError("starts must be distinct")
    level = delta / len(starts)
    scans = tuple(
        fixed_sequence(confidence, correct, alpha=alpha, delta=level, grid=grid, start=s)
        for s in starts
    )
    return MultiStartResult(alpha, delta, scans, tuple(starts))


def max_errors(n_max: int, alpha: float, level: float) -> list[int]:
    """``k[n]``: the most errors among n emitted examples that still pass the test
    P(Bin(n, α) ≤ k) ≤ level (ToE Eq. (4), p-value form; binomial.py), or −1 when none
    does. Index 0 is −1: an empty emit set never passes.

    The p-value increases in k, so "passes" is exactly k ≤ k[n]; and it decreases in n
    at fixed k, so k[n] never goes down, which lets the table be filled in one pass."""
    out = [-1] * (n_max + 1)
    k = -1
    for n in range(1, n_max + 1):
        while k + 1 < n and p_value(k + 1, n, alpha) <= level:
            k += 1
        out[n] = k
    return out


class FastFit:
    """``multi_start(...).threshold`` for many resamples of one calibration set: the
    same tests, vectorised over the grid. Used by the gate's bootstrap, which refits a
    threshold on every resample (gate.py).

    Why it is the same: at one level δ/S every grid point's test depends only on its
    emit counts (n, errors), so a point passes or fails the same way in every scan, and
    duplicates of a tested emit set inherit its result, as in ``fixed_sequence``. A
    scan from start s certifies the run of consecutive passing points that begins at
    the first grid point ≤ s, stopping before the first failure; λ̂ is the smallest
    threshold any scan certified. tests/test_gate_stats.py checks equality with
    ``multi_start`` on random data and every grid kind."""

    def __init__(
        self, *, alpha: float, delta: float, grid: Sequence[float], starts: Sequence[float], n: int
    ) -> None:
        if not starts or len(set(starts)) != len(starts):
            raise ValueError("need distinct starts")
        self.grid = np.asarray(grid, dtype=np.float64)
        if np.any(np.diff(self.grid) >= 0):
            raise ValueError("grid must be strictly decreasing (scan order)")
        self.alpha, self.level = alpha, delta / len(starts)
        self.kmax = np.asarray(max_errors(n, alpha, self.level), dtype=np.int64)
        self.first = [int(np.searchsorted(-self.grid, -(s + _EPS), side="left")) for s in starts]

    def threshold(self, confidence: FloatArray, correct: BoolArray) -> float | None:
        order = np.argsort(confidence, kind="stable")
        conf = confidence[order]
        wrong = ~correct[order]
        n_all = len(conf)
        if n_all >= len(self.kmax):  # a resample of whole groups can be longer than n
            self.kmax = np.asarray(max_errors(2 * n_all, self.alpha, self.level), dtype=np.int64)
        # emitted = confidence ≥ λ − ε: everything from the first such position on
        pos = np.searchsorted(conf, self.grid - _EPS, side="left")
        n = n_all - pos
        suffix = np.concatenate([np.cumsum(wrong[::-1])[::-1], [0]])
        errors = suffix[pos]
        passed = (n > 0) & (errors <= self.kmax[n])
        fails = np.flatnonzero(~passed)
        best: float | None = None
        for j0 in self.first:
            if j0 >= len(self.grid) or not passed[j0]:
                continue
            k = int(np.searchsorted(fails, j0))
            end = int(fails[k]) if k < len(fails) else len(self.grid)
            lam = float(self.grid[end - 1])
            best = lam if best is None else min(best, lam)
        return best

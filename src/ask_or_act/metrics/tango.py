"""Tango's score test and score interval for a difference of paired proportions.

**Source (read in full, all 18 pages, 2026-11-24):** T. Tango, *Equivalence test and
confidence interval for the difference in proportions for the paired-sample design*,
Statistics in Medicine **17**, 891–908 (1998).

**Notation (§3, Eq. (15)).** Paired 2×2 counts, rows the new treatment N and columns
the standard S, with n pairs:

    a = both respond     b = N responds, S doesn't   (cell 12)
    c = S responds, N doesn't (cell 21)              d = neither

Here "responds" is "is correct", N is the new predictions and S the baseline. The
difference is λ = π_N − π_S = q12 − q21 (Eq. (32)).

**Statistic (Eq. (24)–(26)).** For the null H0: π_N = π_S − Δ (Eq. (17)),

    Z(b, c; n, Δ) = (b − c + nΔ) / √( n (2 q̂21 − Δ(Δ + 1)) )                  (24)
    q̂21 = ( √(B² − 4AC) − B ) / (2A)                                            (25)
    A = 2n,   B = −b − c − (2n − b + c) Δ,   C = c Δ (Δ + 1)                      (26)

q̂21 is the restricted maximum-likelihood estimate of q21 under H0. Appendix II shows the
larger root is the right one for either sign of Δ and every (b, c), including zero
off-diagonal cells and the boundary case of Eq. (27). With Δ = 0, Z is McNemar's
(b − c)/√(b + c) (Eq. (31)).

**Interval (§4, Eq. (33)).** The limits of the 1 − 2α interval for λ are the two
solutions of Z(b, c; n, −λ) = ±z_α. The plus sign gives the lower limit. Tango uses the
secant method; we bisect, because Z(b, c; n, −λ) decreases in λ, which the tests check.

**How the gate uses it (our adaptation, not the paper's test).** Tango tests
non-inferiority: H1: π_N > π_S − Δ, rejected for large Z (§5.1 compares Z with the
upper 5% point). The gate also asks the other question: is the new model worse by
*more* than Δ? That is H0': π_N − π_S ≥ −Δ against H1': π_N − π_S < −Δ.

* At the boundary π_N − π_S = −Δ, Z is asymptotically standard normal (§3, after
  Eq. (22)). So we reject H0' when Z is small, with p = Φ(Z).
* Equivalently, the upper limit of the 1 − 2·level interval (33) lies below −Δ. That
  is the same duality the paper states at the start of §4, applied to the other limit.
* Tango's simulations (Table II: Δ = 0.1, n = 30–80) show an empirical size at or
  below the nominal 5% for the upper tail. The lower tail isn't simulated there, so our
  own simulation checks it (docs/gate-simulation.md).

**Assumption:** the pairs are independent (a multinomial sample, §3). Clustered rows,
such as several MT-Bench votes on one question, make the test optimistic.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist

_N = NormalDist()


def q21_hat(b: int, c: int, n: int, delta: float) -> float:
    """Eq. (25)–(26): the restricted MLE of q21 under π_N − π_S = −Δ (the larger root)."""
    a_ = 2.0 * n
    b_ = -b - c - (2.0 * n - b + c) * delta
    c_ = c * delta * (delta + 1.0)
    disc = max(0.0, b_ * b_ - 4.0 * a_ * c_)
    return (math.sqrt(disc) - b_) / (2.0 * a_)


def z_score(b: int, c: int, n: int, delta: float) -> float:
    """Eq. (24). With no information (zero variance and zero numerator) returns 0."""
    if n <= 0:
        raise ValueError("need n > 0 pairs")
    if b < 0 or c < 0 or b + c > n:
        raise ValueError("need 0 ≤ b, c and b + c ≤ n")
    var = n * (2.0 * q21_hat(b, c, n, delta) - delta * (delta + 1.0))
    num = b - c + n * delta
    if var <= 1e-12:
        if abs(num) < 1e-9:
            return 0.0
        return math.copysign(math.inf, num)
    return num / math.sqrt(var)


def interval(b: int, c: int, n: int, level: float) -> tuple[float, float]:
    """Eq. (33): the score interval for λ = π_N − π_S at confidence ``level``
    (Tango's 1 − 2α, so α = (1 − level)/2 in each tail)."""
    z = _N.inv_cdf(1.0 - (1.0 - level) / 2.0)

    def solve(target: float) -> float:
        # Z(b, c; n, −λ) − target decreases in λ: find its root in [−1, 1].
        lo, hi = -1.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if z_score(b, c, n, -mid) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        return (lo + hi) / 2.0

    return solve(z), solve(-z)


@dataclass(frozen=True)
class PairedAccuracy:
    """New vs baseline accuracy on the same rows, with Tango's test at a tolerance."""

    n: int
    b: int  # new right, baseline wrong (Tango's cell 12)
    c: int  # new wrong, baseline right (cell 21)
    new: float
    baseline: float
    tolerance: float  # Δ: how much worse the new model may be
    z: float  # Z(b, c; n, Δ), Eq. (24)
    low: float  # interval (33) at 1 − 2·level
    high: float

    @property
    def diff(self) -> float:
        return self.new - self.baseline

    @property
    def p_worse(self) -> float:
        """p-value for "worse by more than the tolerance" (H0': λ ≥ −Δ): Φ(Z)."""
        return _N.cdf(self.z) if math.isfinite(self.z) else (0.0 if self.z < 0 else 1.0)

    @property
    def p_within(self) -> float:
        """Tango's own test: p-value for "within the tolerance" (H0: λ ≤ −Δ): 1 − Φ(Z)."""
        return 1.0 - self.p_worse


def paired_accuracy(
    new_correct: object, base_correct: object, *, tolerance: float, level: float
) -> PairedAccuracy:
    """``new_correct``/``base_correct``: aligned boolean arrays (same rows, same order).
    ``level`` is the one-sided test level; the interval is at 1 − 2·level."""
    import numpy as np

    nc = np.asarray(new_correct, dtype=np.bool_)
    bc = np.asarray(base_correct, dtype=np.bool_)
    if nc.shape != bc.shape or nc.ndim != 1 or len(nc) == 0:
        raise ValueError("need two aligned, non-empty boolean arrays")
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("need 0 ≤ tolerance < 1")
    n = len(nc)
    b = int(np.sum(nc & ~bc))
    c = int(np.sum(~nc & bc))
    lo, hi = interval(b, c, n, 1.0 - 2.0 * level)
    return PairedAccuracy(
        n, b, c, float(nc.mean()), float(bc.mean()), tolerance, z_score(b, c, n, tolerance), lo, hi
    )

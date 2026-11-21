"""Exact binomial tail and upper confidence bound, in pure Python (no SciPy).

Source: Jung, Brahman, Choi, *Trust or Escalate: LLM Judges with Provable Guarantees for
Human Agreement*, arXiv:2407.18370 **v1** (2024-07-25; the only version), §2.1,
**Equation (4)**. For a threshold λ with n(λ) emitted calibration examples, of which
k(λ) = n(λ)·R̂(λ) are wrong, the exact (1 − δ) upper confidence bound on the risk is

    R̂⁺(λ) = sup{ R : P(Bin(n(λ), R) ≤ k(λ)) ≥ δ }.

Why it is valid (ToE Appendix A.1): for a *fixed* λ, conditional on n(λ) the emitted
examples are i.i.d. draws from the law given "confidence ≥ λ", so k(λ) ~ Bin(n(λ), R(λ))
and P(R(λ) > R̂⁺(λ)) ≤ δ. That is a one-sided Clopper–Pearson bound.

The test used by the threshold search is the equivalent p-value form:
R̂⁺(λ) ≤ α  ⇔  P(Bin(n, α) ≤ k) ≤ δ, because P(Bin(n, R) ≤ k) decreases in R.
With n = 0 the p-value is 1, so an empty emit set never passes.
"""

import math


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) ≤ k), summed in log space (stable for n in the thousands)."""
    if n < 0 or not 0.0 <= p <= 1.0:
        raise ValueError("need n >= 0 and 0 <= p <= 1")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    lg_n1 = math.lgamma(n + 1)
    terms = [
        lg_n1 - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
        for i in range(k + 1)
    ]
    top = max(terms)
    return min(1.0, math.exp(top) * math.fsum(math.exp(t - top) for t in terms))


def p_value(errors: int, n: int, alpha: float) -> float:
    """p-value of H0: risk > α, given ``errors`` wrong out of ``n`` (ToE Eq. 4)."""
    return binom_cdf(errors, n, alpha) if n > 0 else 1.0


def upper_bound(errors: int, n: int, delta: float, *, tol: float = 1e-10) -> float:
    """ToE Eq. (4): sup{R : P(Bin(n, R) ≤ errors) ≥ δ}, by bisection (the CDF is
    continuous and decreasing in R). 1.0 when n = 0 or every example is wrong."""
    if not 0.0 < delta < 1.0:
        raise ValueError("need 0 < delta < 1")
    if n == 0 or errors >= n:
        return 1.0
    lo, hi = errors / n, 1.0  # the CDF at R = errors/n is >= 1/2 > δ for δ < 1/2
    if binom_cdf(errors, n, lo) < delta:  # only for large δ; then search below
        lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if binom_cdf(errors, n, mid) >= delta:
            lo = mid
        else:
            hi = mid
    return lo


def two_sided_interval(errors: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper–Pearson interval for reporting observed test-half risk (not a guarantee)."""
    if n == 0:
        return 0.0, 1.0
    tail = (1.0 - level) / 2
    hi = upper_bound(errors, n, tail)
    lo = 0.0 if errors == 0 else 1.0 - upper_bound(n - errors, n, tail)
    return lo, hi

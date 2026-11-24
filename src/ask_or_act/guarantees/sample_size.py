"""How many labelled calibration rows a guarantee needs, before computing anything.

The fixed-sequence threshold (fixed_sequence.py) certifies a threshold only if the
rows it acts on pass the exact binomial test of Trust or Escalate (Jung, Brahman,
Choi, arXiv:2407.18370 **v1**, §2.1, **Eq. (4)**, used in **Theorem 1**):

    P(Bin(n, α) ≤ errors) ≤ level,

where n is the number of rows acted on and ``errors`` how many of them are wrong. With
S pre-registered scan starts, each scan runs at level δ/S (union bound, see
``MultiStartResult``; validity of each scan's selected threshold: Kotte,
arXiv:2606.29054 **v2**, Appendix E.2, **Lemma E.1**).

Since n can't exceed the labelled calibration rows, the smallest n that passes with
**zero** errors is a hard floor. Below it no threshold can be certified, whatever the
model does: ``thresholds`` refuses, and ``check`` says how many rows are needed. With
zero errors the test is (1 − α)ⁿ ≤ δ/S, so n ≥ log(δ/S) / log(1 − α).

The "realistic" count assumes the acted-on rows are wrong at a given rate (for
example α/2). It's a planning number for the message, not a guarantee.
"""

import math

from ask_or_act.guarantees.binomial import p_value

MAX_ROWS = 200_000  # beyond this we report "more than 200,000"


def min_rows(alpha: float, delta: float, *, starts: int = 1, error_rate: float = 0.0) -> int | None:
    """Smallest n whose exact binomial test at level δ/starts passes with
    ⌊error_rate·n⌋ errors. None if no n ≤ MAX_ROWS passes (e.g. error_rate ≥ α)."""
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0 or starts < 1:
        raise ValueError("need 0 < alpha < 1, 0 < delta < 1 and starts >= 1")
    if not 0.0 <= error_rate < 1.0:
        raise ValueError("need 0 <= error_rate < 1")
    level = delta / starts
    if error_rate == 0.0:
        # Closed form: (1 - α)^n ≤ level. Checked with p_value against float rounding.
        n = max(1, math.ceil(math.log(level) / math.log1p(-alpha)))
        while n > 1 and p_value(0, n - 1, alpha) <= level:
            n -= 1
        while p_value(0, n, alpha) > level:
            n += 1
        return n
    if error_rate >= alpha:
        return None
    return _scan(alpha, level, error_rate)


def _scan(alpha: float, level: float, rate: float) -> int | None:
    """The first n ≤ MAX_ROWS with p_value(⌊rate·n⌋, n, α) ≤ level.

    Recomputing that p-value for every n costs O(n) each, so a rate close to α (tens of
    thousands of rows) took minutes. Instead F = P(Bin(n, α) ≤ k) is carried along
    with the standard recurrences, O(1) per step:

        F_{n+1}(k)     = F_n(k) − α·f_n(k)        (one more trial, same k)
        F_{n+1}(k + 1) = F_{n+1}(k) + f_{n+1}(k + 1)

    with f the binomial pmf, from lgamma. The carried value is accurate to far better
    than the 1e-6 relative margin used here, and every candidate it flags is confirmed
    with the exact ``p_value``. So the answer is exactly the old linear scan's, which
    tests/test_sample_size.py checks."""
    log_a, log_q = math.log(alpha), math.log1p(-alpha)

    def pmf(k: int, n: int) -> float:
        if k < 0 or k > n:
            return 0.0
        return math.exp(
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * log_a
            + (n - k) * log_q
        )

    k = math.floor(rate)  # n = 1
    cdf = p_value(k, 1, alpha)
    for n in range(1, MAX_ROWS + 1):
        if cdf <= level * (1 + 1e-6) and p_value(k, n, alpha) <= level:
            return n
        nk = math.floor(rate * (n + 1))
        cdf = cdf - alpha * pmf(k, n)  # n + 1 trials, still ≤ k errors
        while k < nk:  # ⌊rate·n⌋ rises by at most one per step, but be general
            k += 1
            cdf += pmf(k, n + 1)
        cdf = min(1.0, max(0.0, cdf))
    return None

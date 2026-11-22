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
    for n in range(1, MAX_ROWS + 1):
        if p_value(math.floor(error_rate * n), n, alpha) <= level:
            return n
    return None

"""``min_rows``' O(1)-per-step scan returns exactly what the plain linear scan did."""

import math
import time

import pytest

from ask_or_act.guarantees.binomial import p_value
from ask_or_act.guarantees.sample_size import min_rows


def linear_scan(alpha: float, delta: float, starts: int, rate: float, cap: int) -> int | None:
    """The previous implementation: the exact p-value at every n."""
    for n in range(1, cap + 1):
        if p_value(math.floor(rate * n), n, alpha) <= delta / starts:
            return n
    return None


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.25])
@pytest.mark.parametrize("frac", [0.05, 0.4, 0.75])
@pytest.mark.parametrize("starts", [1, 4])
def test_same_answer_as_the_linear_scan(alpha: float, frac: float, starts: int) -> None:
    want = linear_scan(alpha, 0.1, starts, alpha * frac, cap=8000)
    assert want is not None
    assert min_rows(alpha, 0.1, starts=starts, error_rate=alpha * frac) == want


def test_a_rate_close_to_alpha_is_fast() -> None:
    """It took minutes before (the gate called it on every uncertified file)."""
    start = time.perf_counter()
    n = min_rows(0.1, 0.1, starts=4, error_rate=0.095)
    assert n is not None
    assert n > 10_000
    assert time.perf_counter() - start < 2.0

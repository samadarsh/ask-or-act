import asyncio

import pytest

from ask_or_act.core.limits import RateLimiter, RetryPolicy
from tests.helpers import FakeClock


def test_rate_limiter_spaces_requests() -> None:
    clock = FakeClock()
    limiter = RateLimiter(60, burst=1, clock=clock, sleep=clock.sleep)  # 1 per second

    async def go() -> None:
        for _ in range(5):
            await limiter.acquire()

    asyncio.run(go())
    assert clock.now == pytest.approx(4.0)  # first is free, then one per second


def test_rate_limiter_allows_a_burst_then_throttles() -> None:
    clock = FakeClock()
    limiter = RateLimiter(600, clock=clock, sleep=clock.sleep)  # 10/s, burst 10

    async def go() -> None:
        for _ in range(15):
            await limiter.acquire()

    asyncio.run(go())
    assert clock.now == pytest.approx(0.5)  # 10 free, then 5 more at 0.1 s each


def test_token_limiter_charges_by_amount() -> None:
    clock = FakeClock()
    limiter = RateLimiter(6000, burst=100, clock=clock, sleep=clock.sleep)  # 100 tokens/s

    async def go() -> None:
        await limiter.acquire(100)
        await limiter.acquire(50)

    asyncio.run(go())
    assert clock.now == pytest.approx(0.5)


def test_rate_limiter_validates() -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(0)


def test_backoff_is_exponential_capped_and_deterministic() -> None:
    policy = RetryPolicy(base_s=1.0, max_s=8.0)
    waits = [policy.backoff(a, "ex1") for a in range(6)]
    assert waits == [policy.backoff(a, "ex1") for a in range(6)]
    for attempt, w in enumerate(waits):
        nominal = min(8.0, 2.0**attempt)
        assert 0.5 * nominal <= w < 1.5 * nominal
    assert policy.backoff(0, "ex1") != policy.backoff(0, "ex2")  # jitter differs per key

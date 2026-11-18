"""Rate limiting and retry backoff.

The clock and sleep functions can be injected, so tests run against a fake clock
instantly instead of waiting for real time to pass.
"""

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

type Clock = Callable[[], float]
type Sleep = Callable[[float], Awaitable[None]]


class RateLimiter:
    """Token bucket: holds up to ``burst`` tokens, refilled at ``per_minute / 60`` per second.

    ``acquire(n)`` waits until n tokens are available. Use one limiter for requests per
    minute (n=1) and another for tokens per minute (n = the request's token count).
    Waiters are served first-come, first-served because the lock is held while waiting.
    """

    def __init__(
        self,
        per_minute: float,
        *,
        burst: float | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self._rate = per_minute / 60.0
        self._capacity = burst if burst is not None else max(1.0, self._rate)
        self._tokens = self._capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
        self._last = now

    async def acquire(self, amount: float = 1.0) -> None:
        # A request bigger than the bucket would wait forever; let it through when the
        # bucket is full instead.
        amount = min(amount, self._capacity)
        async with self._lock:
            self._refill()
            while self._tokens < amount:
                await self._sleep((amount - self._tokens) / self._rate)
                self._refill()
            self._tokens -= amount


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_s: float = 1.0
    max_s: float = 60.0

    def backoff(self, attempt: int, key: str) -> float:
        """Wait before retry number ``attempt`` (0-based).

        Exponential (base * 2^attempt), capped, with jitter in [0.5, 1.5). The jitter
        stops many concurrent calls from retrying in lockstep. It is derived from a hash
        of ``key``, so it is deterministic and reproducible.
        """
        digest = hashlib.sha256(f"{key}:{attempt}".encode()).digest()
        jitter = 0.5 + int.from_bytes(digest[:8]) / 2**64
        return float(min(self.max_s, self.base_s * 2**attempt) * jitter)

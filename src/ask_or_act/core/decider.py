"""The Decider interface.

Python's ``Protocol`` is the closest thing to a Go interface:

* **Structural, like Go.** A class satisfies ``Decider`` by having the right methods
  and properties. There is no ``implements`` keyword and no base class to inherit.
* **Checked by mypy, not by the runtime.** Go rejects a non-conforming type at compile
  time. Python only finds out when mypy runs (or when a missing method gets called).
  Assigning ``d: Decider = MyDecider()`` makes mypy check it, much like Go's
  ``var _ Decider = (*MyDecider)(nil)`` idiom.
* **Can require properties as well as methods,** which Go interfaces can't.

Concurrency is in the interface from the start. ``decide_batch`` is ``async`` (it
returns a coroutine, roughly a Go function you'd start with ``go``). It is a batch call
because some providers answer many questions per request; the Decisions API takes up to
1,000. The runner handles rate limits, retries, timeouts and concurrency, so a decider
implementation only makes the call.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ask_or_act.core.types import DeciderConfig, Decision, PublicExample, canonical_json


class RetryableError(Exception):
    """A transient failure worth retrying: rate limited (429), server error (5xx),
    dropped connection. Any other exception from a decider is treated as permanent.

    ``retry_after_s`` is the provider's own hint (HTTP ``retry-after``); the runner waits
    at least that long before retrying."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class QuotaExhaustedError(Exception):
    """A per-day (or other long-window) quota is used up, e.g. a free tier's requests
    per day. Retrying within this run can't help, so the runner stops cleanly: nothing
    new is started, finished decisions stay cached, and re-running the same config
    after the quota resets continues where this run stopped."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


@dataclass(frozen=True)
class CostEstimate:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0  # what this run would be charged; the budget is enforced on it
    # The same tokens at the published paid price. Differs from ``usd`` on a free tier
    # ($0 charged), so studies can still report cost per 1,000 decisions. None = usd.
    list_usd: float | None = None

    @property
    def at_list_price(self) -> float:
        return self.usd if self.list_usd is None else self.list_usd

    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        both_none = self.list_usd is None and other.list_usd is None
        return CostEstimate(
            self.calls + other.calls,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.usd + other.usd,
            None if both_none else self.at_list_price + other.at_list_price,
        )


@dataclass(frozen=True)
class Pricing:
    """USD per 1M tokens. Copy it from the provider's price sheet and note the date."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


def rough_token_count(example: PublicExample) -> int:
    """About 4 characters per token. Good enough for budgeting. A real decider should
    count with its provider's tokenizer, or use the provider's count-tokens endpoint."""
    text = canonical_json(example.input) + "".join(example.options)
    return math.ceil(len(text) / 4)


class Decider(Protocol):
    @property
    def config(self) -> DeciderConfig:
        """Identity of this decider; part of every cache key."""
        ...

    @property
    def max_batch_size(self) -> int:
        """Most examples one ``decide_batch`` call accepts (1 = no batching)."""
        ...

    @property
    def needs_gold(self) -> bool:
        """True only for test fakes that simulate a known accuracy. The runner gives
        every other decider ``PublicExample`` objects, which have no gold label."""
        ...

    def estimate(self, examples: Sequence[PublicExample]) -> CostEstimate:
        """Predicted cost of one ``decide_batch`` call on these examples. Must not do I/O."""
        ...

    async def decide_batch(self, examples: Sequence[PublicExample]) -> list[Decision]:
        """One decision per example, in the same order. Raise ``RetryableError`` for
        transient failures. Set ``Decision.charged_usd`` only if part of the work came
        from a cache (e.g. self-consistency samples); otherwise the runner sets it."""
        ...

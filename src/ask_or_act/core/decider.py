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

from ask_or_act.core.types import DeciderConfig, Decision, Example, canonical_json


class RetryableError(Exception):
    """A transient failure worth retrying: rate limited (429), server error (5xx),
    dropped connection. Any other exception from a decider is treated as permanent."""


@dataclass(frozen=True)
class CostEstimate:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        return CostEstimate(
            self.calls + other.calls,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.usd + other.usd,
        )


@dataclass(frozen=True)
class Pricing:
    """USD per 1M tokens. Copy it from the provider's price sheet and note the date."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


def rough_token_count(example: Example) -> int:
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

    def estimate(self, examples: Sequence[Example]) -> CostEstimate:
        """Predicted cost of one ``decide_batch`` call on these examples. Must not do I/O."""
        ...

    async def decide_batch(self, examples: Sequence[Example]) -> list[Decision]:
        """One decision per example, in the same order. Raise ``RetryableError`` for
        transient failures. Must not read ``Example.gold``."""
        ...

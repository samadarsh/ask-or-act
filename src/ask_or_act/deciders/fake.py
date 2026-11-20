"""Fake deciders with known behaviour, for testing metrics before spending money.

Every random draw is ``hash(seed, example id, purpose)`` mapped to [0, 1). Outputs depend
only on the seed and the example, not on call order. Results are therefore identical
regardless of concurrency or batching, and across Python versions.

Kinds:

* ``perfect``: always right, with confidence 1.
* ``random``: a uniform choice, with confidence 1/k. Calibrated, but useless.
* ``constant``: always ``params.choice`` (default: the first option). This is the
  "always-same" baseline.
* ``calibrated``: draws a confidence c uniformly from [low, high], then is right with
  probability exactly c. So P(correct | confidence = c) = c: perfectly calibrated by
  construction.
* ``overconfident``: the same process, but it *reports* c + inflation * (1 - c). Its
  accuracy matches ``calibrated``, but its confidence is too high by
  inflation * E[1 - c]. With the defaults (c ~ U[0.5, 1], inflation 0.5) that is a
  gap of about 0.125, which a correct ECE implementation must detect.

Params also accept ``input_price_per_mtok``, ``output_price_per_mtok``, ``latency_s``
and ``max_batch_size``, to exercise budgets, timeouts and batching.
"""

import asyncio
import hashlib
from collections.abc import Sequence
from enum import StrEnum

from ask_or_act.core.decider import CostEstimate, Pricing, rough_token_count
from ask_or_act.core.types import JSON, DeciderConfig, Decision, Example, PublicExample


class FakeKind(StrEnum):
    PERFECT = "perfect"
    RANDOM = "random"
    CONSTANT = "constant"
    CALIBRATED = "calibrated"
    OVERCONFIDENT = "overconfident"


KNOWN_PARAMS = frozenset(
    {
        "kind",
        "seed",
        "choice",
        "confidence",
        "low",
        "high",
        "inflation",
        "input_price_per_mtok",
        "output_price_per_mtok",
        "latency_s",
        "max_batch_size",
    }
)


def _num(params: dict[str, JSON], key: str, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"fake decider param {key!r} must be a number, got {value!r}")
    return float(value)


class FakeDecider:
    """Satisfies the ``Decider`` protocol without inheriting from it."""

    def __init__(self, config: DeciderConfig) -> None:
        params = dict(config.params)
        unknown = set(params) - KNOWN_PARAMS
        if unknown:
            raise ValueError(f"unknown fake decider params: {sorted(unknown)}")
        self._config = config
        self.kind = FakeKind(str(params.get("kind", "calibrated")))
        self.seed = int(_num(params, "seed", 0))
        choice = params.get("choice")
        self.choice = None if choice is None else str(choice)
        self.fixed_confidence = _num(params, "confidence", 0.0) if "confidence" in params else None
        if self.fixed_confidence is not None and not 0.0 <= self.fixed_confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.low = _num(params, "low", 0.5)
        self.high = _num(params, "high", 1.0)
        self.inflation = _num(params, "inflation", 0.5)
        self.latency_s = _num(params, "latency_s", 0.0)
        self._max_batch = int(_num(params, "max_batch_size", 1))
        self.pricing = Pricing(
            _num(params, "input_price_per_mtok", 0.0), _num(params, "output_price_per_mtok", 0.0)
        )
        if not 0.0 <= self.low <= self.high <= 1.0:
            raise ValueError("need 0 <= low <= high <= 1")
        if not 0.0 <= self.inflation <= 1.0:
            raise ValueError("inflation must be in [0, 1]")

    @property
    def config(self) -> DeciderConfig:
        return self._config

    @property
    def max_batch_size(self) -> int:
        return self._max_batch

    @property
    def needs_gold(self) -> bool:
        return True  # simulates a known accuracy, so it must know the answer

    def estimate(self, examples: Sequence[PublicExample]) -> CostEstimate:
        tokens_in = sum(rough_token_count(ex) for ex in examples)
        tokens_out = len(examples)
        return CostEstimate(1, tokens_in, tokens_out, self.pricing.cost(tokens_in, tokens_out))

    async def decide_batch(self, examples: Sequence[PublicExample]) -> list[Decision]:
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        out = []
        for ex in examples:
            if not isinstance(ex, Example):
                raise TypeError("fake deciders need gold labels (needs_gold=True)")
            out.append(self._decide(ex))
        return out

    # ------------------------------------------------------------------ internals

    def _uniform(self, example_id: str, purpose: str) -> float:
        digest = hashlib.sha256(f"{self.seed}:{example_id}:{purpose}".encode()).digest()
        return int.from_bytes(digest[:8]) / 2**64

    def _pick(self, options: Sequence[str], example_id: str, purpose: str) -> str:
        return options[int(self._uniform(example_id, purpose) * len(options))]

    def _decide(self, ex: Example) -> Decision:
        # Only the fake deciders read ``gold``: they simulate a known accuracy.
        k = len(ex.options)
        if self.kind is FakeKind.PERFECT:
            choice, true_c, reported = ex.gold, 1.0, 1.0
        elif self.kind is FakeKind.RANDOM:
            choice = self._pick(ex.options, ex.id, "choice")
            true_c = reported = 1.0 / k
        elif self.kind is FakeKind.CONSTANT:
            choice = self.choice if self.choice is not None else ex.options[0]
            if choice not in ex.options:
                raise ValueError(f"constant choice {choice!r} is not an option of {ex.id}")
            fixed = self.fixed_confidence
            true_c = reported = 1.0 / k if fixed is None else fixed
        else:
            true_c = self.low + (self.high - self.low) * self._uniform(ex.id, "confidence")
            if self._uniform(ex.id, "correct") < true_c:
                choice = ex.gold
            else:
                choice = self._pick([o for o in ex.options if o != ex.gold], ex.id, "wrong")
            reported = true_c
            if self.kind is FakeKind.OVERCONFIDENT:
                reported = true_c + self.inflation * (1.0 - true_c)

        rest = (1.0 - reported) / (k - 1)
        probabilities = {o: (reported if o == choice else rest) for o in ex.options}
        tokens_in = rough_token_count(ex)
        return Decision(
            example_id=ex.id,
            choice=choice,
            confidence=reported,
            probabilities=probabilities,
            signals={"native": reported},
            model=self._config.model,
            input_tokens=tokens_in,
            output_tokens=1,
            cost_usd=self.pricing.cost(tokens_in, 1),
            raw={"true_confidence": true_c},
        )

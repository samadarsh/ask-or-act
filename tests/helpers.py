"""Shared test doubles. Nothing here touches the network."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ask_or_act.core.decider import CostEstimate
from ask_or_act.core.types import DeciderConfig, Decision, Example
from ask_or_act.deciders.fake import FakeDecider


def make_examples(n: int, k: int = 4, prefix: str = "ex") -> list[Example]:
    options = tuple(f"o{j}" for j in range(k))
    return [Example(f"{prefix}{i}", f"input {i}", options, options[i % k]) for i in range(n)]


def fake(kind: str = "calibrated", **params: float | str) -> FakeDecider:
    return FakeDecider(DeciderConfig("fake", f"fake-{kind}", params={"kind": kind, **params}))


@dataclass
class FakeClock:
    """Injected in place of time.monotonic + asyncio.sleep: sleeping advances time instantly."""

    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)  # still yield to the event loop, like a real sleep


class ScriptedDecider:
    """A decider whose behaviour each test scripts: delays, failures, costs."""

    def __init__(
        self,
        *,
        batch_size: int = 1,
        delay_s: float = 0.0,
        failures: Mapping[str, Sequence[Exception]] | None = None,
        estimate_usd: float = 0.0,
        actual_usd: float = 0.0,
        prompt_version: str = "v1",
        wrong_ids: bool = False,
    ) -> None:
        self._config = DeciderConfig("scripted", "scripted-model", prompt_version)
        self._batch_size = batch_size
        self.delay_s = delay_s
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.estimate_usd = estimate_usd
        self.actual_usd = actual_usd
        self.wrong_ids = wrong_ids
        self.calls: list[list[str]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    @property
    def config(self) -> DeciderConfig:
        return self._config

    @property
    def max_batch_size(self) -> int:
        return self._batch_size

    def estimate(self, examples: Sequence[Example]) -> CostEstimate:
        return CostEstimate(1, 10 * len(examples), len(examples), self.estimate_usd * len(examples))

    async def decide_batch(self, examples: Sequence[Example]) -> list[Decision]:
        self.calls.append([ex.id for ex in examples])
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay_s)
            queued = self.failures.get(examples[0].id)
            if queued:
                raise queued.pop(0)
        finally:
            self.in_flight -= 1
        return [
            Decision(
                example_id="wrong" if self.wrong_ids else ex.id,
                choice=ex.options[0],
                confidence=0.5,
                model="scripted-model",
                cost_usd=self.actual_usd,
            )
            for ex in examples
        ]

    @property
    def n_examples_called(self) -> int:
        return sum(len(c) for c in self.calls)

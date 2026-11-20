"""Simple baselines that every real decider must beat.

* ``MajorityDecider`` (type ``majority``): always answers the most common *training*
  label. Its confidence is that label's training frequency, and its probabilities are
  the training label distribution. Free.
* ``random_at_equal_cost``: for a router that sends some traffic to a cheap decider
  and some to a strong one, the fair baseline is to route *randomly* at the same
  overall cost. It is built from two saved runs, so it costs nothing extra.
  ``always-cheap`` and ``always-strong`` are just those two runs on their own.
"""

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

from ask_or_act.core.decider import CostEstimate
from ask_or_act.core.types import DeciderConfig, Decision, PublicExample
from ask_or_act.deciders import DeciderContext


class MajorityDecider:
    def __init__(self, config: DeciderConfig, ctx: DeciderContext) -> None:
        if set(config.params) - {"labels"}:
            raise ValueError("majority accepts only the optional 'labels' param (for tests)")
        self._config = config
        self._ctx = ctx
        labels = config.params.get("labels")
        self._counts: Counter[str] | None = (
            Counter(str(x) for x in labels) if isinstance(labels, list) else None
        )

    @property
    def config(self) -> DeciderConfig:
        return self._config

    @property
    def max_batch_size(self) -> int:
        return 10_000

    @property
    def needs_gold(self) -> bool:
        return False

    def estimate(self, examples: Sequence[PublicExample]) -> CostEstimate:
        return CostEstimate(1, 0, 0, 0.0)

    def _training_counts(self) -> Counter[str]:
        if self._counts is None:
            from ask_or_act.data import training

            self._counts = Counter(training.load(self._ctx.dataset, self._ctx.raw_dir).labels)
        return self._counts

    async def decide_batch(self, examples: Sequence[PublicExample]) -> list[Decision]:
        counts = self._training_counts()
        total = sum(counts.values())
        out = []
        for ex in examples:
            dist = {o: counts.get(o, 0) / total for o in ex.options}
            # Ties broken by option order, so the result is deterministic.
            choice = max(ex.options, key=lambda o: (dist[o], -ex.options.index(o)))
            out.append(
                Decision(
                    ex.id,
                    choice,
                    dist[choice],
                    probabilities=dist,
                    signals={"native": dist[choice]},
                    model=self._config.model,
                )
            )
        return out


def _coin(seed: int, example_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8]) / 2**64


def equal_cost_fraction(cheap_cost: float, strong_cost: float, target_cost: float) -> float:
    """Share of traffic for the strong decider so that the expected cost is ``target_cost``."""
    if strong_cost <= cheap_cost:
        raise ValueError("strong decider must cost more than the cheap one")
    return min(1.0, max(0.0, (target_cost - cheap_cost) / (strong_cost - cheap_cost)))


def random_at_equal_cost(
    cheap: Sequence[Decision], strong: Sequence[Decision], strong_fraction: float, seed: int = 0
) -> list[Decision]:
    """Per example, take the strong decision with probability ``strong_fraction``.

    Uses a hash of the seed and example id, so it doesn't depend on order. Decisions
    are tagged in ``raw["routed_to"]``.
    """
    if [d.example_id for d in cheap] != [d.example_id for d in strong]:
        raise ValueError("cheap and strong runs must cover the same examples in order")
    out = []
    for c, s in zip(cheap, strong, strict=True):
        pick_strong = _coin(seed, c.example_id) < strong_fraction
        chosen = s if pick_strong else c
        out.append(
            replace(chosen, raw={**chosen.raw, "routed_to": "strong" if pick_strong else "cheap"})
        )
    return out

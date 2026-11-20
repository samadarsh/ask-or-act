"""Builders for metric tests: Scored objects from fakes, without the network."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

from ask_or_act.core.runs import GoldRecord, RunRecord
from ask_or_act.core.types import Decision, Example
from ask_or_act.data import manifest
from ask_or_act.data.sampling import Split
from ask_or_act.deciders.fake import FakeDecider
from ask_or_act.metrics.data import Scored, from_run

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def scored(
    decisions: Sequence[Decision], examples: Sequence[Example], dataset: str = "toy"
) -> Scored:
    record = RunRecord(
        path=Path("memory"),
        metadata={"dataset": {"name": dataset}},
        decisions=list(decisions),
        examples=[GoldRecord.of(e) for e in examples],
    )
    return from_run(record)


def run_fake(decider: FakeDecider, examples: Sequence[Example], dataset: str = "toy") -> Scored:
    return scored(asyncio.run(decider.decide_batch(examples)), examples, dataset)


def sample_examples(name: str, split: Split = Split.TEST) -> list[Example]:
    """Real committed sample (ids, gold labels) with a placeholder input. Offline.

    The input is only used by the fakes' token estimate; fakes never read it.
    """
    m = manifest.read(SAMPLES / f"{name}.json")
    options = tuple(sorted({e.label for e in m.examples}))
    return [
        Example(e.id, e.id, options, e.label, meta=dict(e.meta))
        for e in m.examples
        if e.split == split
    ]

"""Turn a saved run into aligned arrays, the common input of every metric.

**How errors are handled (one policy, everywhere):** a decision with an error counts as a
wrong answer with confidence 0. Concretely:

* Accuracy, F1 and kappa penalize failed calls instead of quietly ignoring them.
* Selective prediction ranks errors last, as if the decider abstained on them first.

Every report also states ``n_errors``, so a run with many failures is visible rather
than just scoring a bit lower.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ask_or_act.core.runs import RunRecord
from ask_or_act.core.types import JSON

ERROR_LABEL = "<error>"

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.intp]


@dataclass(frozen=True)
class Scored:
    """One decider's decisions on one sample, aligned with the gold labels.

    ``groups`` are the bootstrap resampling units: the example id itself, except for
    MT-Bench, where it is the question. Several votes on one question aren't
    independent, so they're resampled together.
    """

    ids: tuple[str, ...]
    gold: tuple[str, ...]
    pred: tuple[str, ...]  # ERROR_LABEL for failed decisions
    confidence: FloatArray
    correct: BoolArray
    probabilities: tuple[Mapping[str, float], ...]
    options: tuple[str, ...]
    groups: tuple[str, ...]
    meta: tuple[Mapping[str, JSON], ...]
    dataset: str
    n_errors: int

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, idx: IntArray) -> "Scored":
        """Subset (or bootstrap resample, with repeats) by position."""
        pick = [int(i) for i in idx]
        return Scored(
            ids=tuple(self.ids[i] for i in pick),
            gold=tuple(self.gold[i] for i in pick),
            pred=tuple(self.pred[i] for i in pick),
            confidence=self.confidence[idx],
            correct=self.correct[idx],
            probabilities=tuple(self.probabilities[i] for i in pick),
            options=self.options,
            groups=tuple(self.groups[i] for i in pick),
            meta=tuple(self.meta[i] for i in pick),
            dataset=self.dataset,
            n_errors=sum(self.pred[i] == ERROR_LABEL for i in pick),
        )


def from_run(record: RunRecord) -> Scored:
    if [d.example_id for d in record.decisions] != [e.id for e in record.examples]:
        raise ValueError(f"{record.path}: decisions and examples are not aligned")
    dataset_info = record.metadata.get("dataset")
    dataset = str(dataset_info.get("name", "")) if isinstance(dataset_info, dict) else ""

    pred = tuple(
        d.choice if d.ok and d.choice is not None else ERROR_LABEL for d in record.decisions
    )
    gold = tuple(e.gold for e in record.examples)
    confidence = np.array(
        [d.confidence if d.ok and d.confidence is not None else 0.0 for d in record.decisions],
        dtype=np.float64,
    )
    options = tuple(sorted({o for e in record.examples for o in e.options}))
    groups = tuple(
        f"q{e.meta['question_id']}" if dataset == "mt_bench" and "question_id" in e.meta else e.id
        for e in record.examples
    )
    return Scored(
        ids=tuple(e.id for e in record.examples),
        gold=gold,
        pred=pred,
        confidence=confidence,
        correct=np.array([p == g for p, g in zip(pred, gold, strict=True)], dtype=np.bool_),
        probabilities=tuple(dict(d.probabilities) for d in record.decisions),
        options=options,
        groups=groups,
        meta=tuple(e.meta for e in record.examples),
        dataset=dataset,
        n_errors=sum(p == ERROR_LABEL for p in pred),
    )


def align(a: Scored, b: Scored) -> tuple[Scored, Scored]:
    """Put two runs on the same examples in the same order (required for paired tests)."""
    if set(a.ids) != set(b.ids):
        only_a, only_b = set(a.ids) - set(b.ids), set(b.ids) - set(a.ids)
        raise ValueError(
            f"runs cover different examples ({len(only_a)} only in A, {len(only_b)} only "
            f"in B); paired comparison needs the same sample"
        )
    position = {i: k for k, i in enumerate(b.ids)}
    b = b.take(np.array([position[i] for i in a.ids], dtype=np.intp))
    if a.gold != b.gold:
        raise ValueError("runs disagree on gold labels for the same example ids")
    return a, b

"""MT-Bench: agreement with human experts, and human–human agreement as the ceiling.

**Display order.** A vote says ``model_a``, ``model_b`` or ``tie``, where A and B are
whatever order that voter happened to see. Two humans may have seen the same pair
swapped. Labels are therefore converted to model names (or "tie") before *any*
agreement calculation. For a decider compared with the human vote on the same row, the
conversion changes nothing, because they saw the same order; it is applied anyway so
every comparison goes through one code path.

**Ties.** We follow the two setups of Zheng et al. 2023 (arXiv:2306.05685 v4, Table 5):

* **S1, with ties.** "tie" is a third label, and agreement means identical labels.
  Random chance is 1/3.
* **S2, without ties.** Only pairs where neither vote is a tie count. Random chance is
  1/2.

Zheng et al. also count position-inconsistent LLM verdicts as ties in S1. That only
applies once a judge is run in both orders, which is milestone 4.

**Human–human.** Agreement between two different experts on the same comparison: every
pair of votes from different judges, pooled. Zheng et al. define agreement as two
random individuals, "but not identical", agreeing on a random question. They report
63% (S1) / 81% (S2) for turn 1 and 67% / 82% for turn 2 (Table 5). Our sample is a
subset, so expect similar but not identical values.

Everything is computed per question and then summed. A question is the bootstrap unit,
because votes on the same question are correlated.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ask_or_act.metrics.data import ERROR_LABEL, FloatArray, IntArray, Scored

TIE = "tie"
PUBLISHED_HUMAN_HUMAN = {  # Zheng et al. 2023, arXiv:2306.05685 v4, Table 5 (S1, S2)
    "turn_1": {"with_ties": 0.63, "without_ties": 0.81},
    "turn_2": {"with_ties": 0.67, "without_ties": 0.82},
}


def to_model_name(label: str, model_a: str, model_b: str) -> str | None:
    """'model_a' / 'model_b' / 'tie' → the winning model's name or 'tie'. None if invalid."""
    return {"model_a": model_a, "model_b": model_b, TIE: TIE}.get(label)


def normalized(s: Scored, labels: tuple[str, ...]) -> list[str | None]:
    out: list[str | None] = []
    for label, meta in zip(labels, s.meta, strict=True):
        if label == ERROR_LABEL:
            out.append(None)
        else:
            out.append(to_model_name(label, str(meta["model_a"]), str(meta["model_b"])))
    return out


@dataclass(frozen=True)
class AgreementCounts:
    """Per-question sums: agreements and pair counts under S1 and S2."""

    groups: tuple[str, ...]
    agree_s1: FloatArray
    pairs_s1: FloatArray
    agree_s2: FloatArray
    pairs_s2: FloatArray

    def rate(self, setup: str, idx: IntArray | None = None) -> float | None:
        sel = np.arange(len(self.groups)) if idx is None else idx
        agree = self.agree_s1 if setup == "S1" else self.agree_s2
        pairs = self.pairs_s1 if setup == "S1" else self.pairs_s2
        n = float(pairs[sel].sum())
        return None if n == 0 else float(agree[sel].sum()) / n

    def n_pairs(self, setup: str) -> int:
        return int((self.pairs_s1 if setup == "S1" else self.pairs_s2).sum())


def _count(pairs_by_group: dict[str, list[tuple[str | None, str | None]]]) -> AgreementCounts:
    groups = tuple(sorted(pairs_by_group))
    a1, p1, a2, p2 = (np.zeros(len(groups)) for _ in range(4))
    for k, g in enumerate(groups):
        for x, y in pairs_by_group[g]:
            if x is None or y is None:
                continue  # a failed decision contributes no pair
            p1[k] += 1
            a1[k] += x == y
            if TIE not in (x, y):
                p2[k] += 1
                a2[k] += x == y
    return AgreementCounts(groups, a1, p1, a2, p2)


def human_human(s: Scored) -> AgreementCounts:
    """Pairs of votes by *different* judges on the same comparison (the ceiling)."""
    gold = normalized(s, s.gold)
    rows_by_comparison: dict[str, list[int]] = defaultdict(list)
    for i, meta in enumerate(s.meta):
        rows_by_comparison[str(meta["comparison"])].append(i)
    pairs: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    for rows in rows_by_comparison.values():
        for i, j in combinations(rows, 2):
            if s.meta[i]["judge"] != s.meta[j]["judge"]:
                pairs[s.groups[i]].append((gold[i], gold[j]))
    for g in set(s.groups):
        pairs.setdefault(g, [])
    return _count(pairs)


def decider_human(s: Scored) -> AgreementCounts:
    """The decider's verdict against the human vote on the same row."""
    pred, gold = normalized(s, s.pred), normalized(s, s.gold)
    pairs: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    for i, g in enumerate(s.groups):
        pairs[g].append((pred[i], gold[i]))
    return _count(pairs)

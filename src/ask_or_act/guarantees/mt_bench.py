"""MT-Bench for guaranteed thresholds: one example per comparison, and the human ceiling.

**Pre-registered 2026-11-21, before any LLM data on MT-Bench existed.** Must not change
after seeing LLM results.

Why: the binomial test assumes independent examples, but MT-Bench has several votes per
comparison (and several comparisons per question). Trust or Escalate (arXiv:2407.18370
v1, §2.1, footnote 1) uses one example per input with the majority human label:
"When we have multiple human annotations per input x, we define
y_human := argmax_y Σᵢ 1{yᵢ = y}". We do the same, per comparison.

The rule:

* **Unit = comparison** (question, turn, unordered model pair), i.e. the manifest's
  ``comparison`` id. Labels are mapped from display order (model_a / model_b) to model
  names; "tie" stays "tie". So votes cast on either display order count together.
* **Gold = plurality of all the comparison's votes.** If the top count is shared (e.g.
  one vote for each model, or 1-1-1), the comparison has no majority. It is **excluded**
  and counted in the report. We don't map such splits to "tie": a split between humans
  is not a vote for "tie".
* **The decider's answer for the unit** is its decision on the display order with the
  most votes, which is what humans saw most. Ties go to the order whose model_a
  name sorts first. Failed decisions count as wrong with confidence 0, as everywhere.
* **Groups** (bootstrap units) stay the question.

The vote-level analysis (every vote an example) is reported next to it as a sensitivity
check; its guarantee is optimistic because votes are not independent.

**Human ceiling** (``human_agreement``): how often a typical human agrees with the
majority of the *other* humans who voted on the same comparison (leave-one-out; votes
whose others have no strict majority, or who voted alone, are skipped). The target
"agree with the majority at least as often as a typical human does" is then
α = 1 − h. h is estimated from the calibration half's votes, which the threshold test
also uses, so this α depends slightly on the calibration labels; for a strictly
pre-registered target, fix the printed α and pass it as ``--alpha`` later.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from ask_or_act.metrics.data import ERROR_LABEL, Scored


def _named(label: str, meta: dict[str, object]) -> str:
    if label in ("model_a", "model_b"):
        return str(meta[label])
    return label  # "tie", or ERROR_LABEL for a failed decision


def _comparison(meta: dict[str, object]) -> str:
    c = meta.get("comparison")
    if not isinstance(c, str):
        raise ValueError("MT-Bench example without a 'comparison' id in its meta")
    return c


def _plurality(votes: list[str]) -> str | None:
    top = Counter(votes).most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


@dataclass(frozen=True)
class Units:
    scored: Scored  # one row per comparison with a majority
    n_votes: int
    n_comparisons: int
    n_no_majority: int  # excluded: the top count was shared


def by_comparison(s: Scored) -> Units:
    """Collapse a vote-level MT-Bench run to one example per comparison (see module)."""
    if s.dataset != "mt_bench":
        raise ValueError("by_comparison is for MT-Bench runs")
    votes: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(s.meta):
        votes[_comparison(dict(m))].append(i)
    keep: list[tuple[str, int, str]] = []  # (comparison, index of the decision used, gold)
    no_majority = 0
    for comp in sorted(votes):
        idx = votes[comp]
        gold = _plurality([_named(s.gold[i], dict(s.meta[i])) for i in idx])
        if gold is None:
            no_majority += 1
            continue
        orders = Counter(str(s.meta[i]["model_a"]) for i in idx)
        shown = min(orders, key=lambda a: (-orders[a], a))
        chosen = next(i for i in idx if str(s.meta[i]["model_a"]) == shown)
        keep.append((comp, chosen, gold))
    pick = np.array([i for _, i, _ in keep], dtype=np.intp)
    pred = tuple(_named(s.pred[i], dict(s.meta[i])) for _, i, _ in keep)
    gold_named = tuple(g for _, _, g in keep)
    collapsed = Scored(
        ids=tuple(c for c, _, _ in keep),
        gold=gold_named,
        pred=pred,
        confidence=s.confidence[pick] if len(pick) else np.zeros(0),
        correct=np.array(
            [p == g and p != ERROR_LABEL for p, g in zip(pred, gold_named, strict=True)],
            dtype=np.bool_,
        ),
        probabilities=tuple({} for _ in keep),
        options=s.options,
        groups=tuple(s.groups[i] for _, i, _ in keep),
        meta=tuple(s.meta[i] for _, i, _ in keep),
        dataset=s.dataset,
        n_errors=sum(p == ERROR_LABEL for p in pred),
        signal=s.signal,
        n_signal_fallback=s.n_signal_fallback,
    )
    return Units(collapsed, len(s), len(votes), no_majority)


@dataclass(frozen=True)
class HumanAgreement:
    rate: float  # h: share of votes that agree with the other voters' majority
    n_votes_used: int
    n_votes: int

    @property
    def alpha(self) -> float:
        """The matching risk target: disagree with the majority no more than humans do."""
        return 1.0 - self.rate


def human_agreement(s: Scored) -> HumanAgreement:
    """Leave-one-out agreement of each human vote with the others' plurality."""
    votes: dict[str, list[str]] = defaultdict(list)
    for g, m in zip(s.gold, s.meta, strict=True):
        votes[_comparison(dict(m))].append(_named(g, dict(m)))
    agree = used = 0
    for vs in votes.values():
        for k, v in enumerate(vs):
            others = vs[:k] + vs[k + 1 :]
            if not others:
                continue
            majority = _plurality(others)
            if majority is None:
                continue
            used += 1
            agree += v == majority
    if used == 0:
        raise ValueError("no comparison has two or more votes; human ceiling undefined")
    return HumanAgreement(agree / used, used, len(s))

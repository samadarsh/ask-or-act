"""Judge mode (milestone 4): an LLM judge's MT-Bench verdicts against the human experts.

Everything here is a pure function of a saved vote-level MT-Bench run (``Scored``), plus
answer lengths for the length-bias check. Nothing makes an API call.

Four questions, following the judge analyses of Zheng et al. 2023 ("Judging
LLM-as-a-judge", arXiv:2306.05685 v4: §3.3 position and verbosity bias, §4.2 / Table 5
agreement):

1. **Agreement.** With the human majority (one unit per comparison, the
   pre-registered unit of guarantees/mt_bench.py), with individual humans (every vote,
   setups S1 with ties and S2 without), and against the **human ceiling**:
   human–human agreement, and the paired leave-one-out view. For every vote with a
   strict majority among the *other* votes, the human scores 1 if they agree with that
   majority, and the judge scores 1 if its verdict on the same row does. Same
   reference, same rows, so judge − human is a paired difference.
2. **Position bias.** A comparison judged in both display orders is *consistent* if
   the named verdict is the same in both. Inconsistent pairs are split as in Zheng et
   al.: "always the first answer" (``model_a`` both times), "always the second"
   (``model_b`` both times), or other (a tie in one order). Comparisons judged in one
   order only are counted: each needs one extra call with the answers swapped. A
   cheaper one-order signal is also reported: how often the judge's non-tie verdict
   picks the first-shown answer, against the humans on the same rows.
3. **Length (verbosity) bias.** On rows where the two answers differ in length and
   both the judge and the human gave a non-tie verdict: how often each picks the
   longer answer. Length is the number of characters in the judged turn's answer.
4. **Ties.** The judge's tie rate against the humans' on the same rows, a 3 × 3
   table of verdicts, and the judge's tie rate where the humans split.

**CIs:** the same bootstrap as the metrics (2,000 resamples, seeded), resampled by
question, since votes on one question are correlated. Paired differences use the
centered paired bootstrap (metrics/stats.py).

These are descriptive analyses, not statistical guarantees: none of them sets a
threshold.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ask_or_act.guarantees.mt_bench import _plurality, by_comparison
from ask_or_act.metrics import mt_bench
from ask_or_act.metrics.data import ERROR_LABEL, BoolArray, IntArray, Scored
from ask_or_act.metrics.stats import (
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    bootstrap_ci,
    paired_bootstrap,
)

POSITIONS = ("model_a", "model_b", "tie")
type Lengths = Mapping[str, tuple[int, int]]  # example id → (answer A, answer B), as shown


@dataclass(frozen=True)
class Boot:
    n_boot: int = DEFAULT_N_BOOT
    seed: int = DEFAULT_SEED
    level: float = 0.95


def _named(label: str, meta: Mapping[str, Any]) -> str | None:
    if label == ERROR_LABEL:
        return None
    return mt_bench.to_model_name(label, str(meta["model_a"]), str(meta["model_b"]))


def _rate(values: BoolArray, groups: tuple[str, ...], boot: Boot) -> dict[str, Any]:
    def stat(idx: IntArray) -> float | None:
        return float(values[idx].mean()) if len(idx) else None

    ci = bootstrap_ci(
        stat, len(values), groups=groups, n_boot=boot.n_boot, seed=boot.seed, level=boot.level
    )
    return {"estimate": ci.estimate, "low": ci.low, "high": ci.high, "n": len(values)}


def _paired(
    judge: BoolArray, human: BoolArray, groups: tuple[str, ...], boot: Boot
) -> dict[str, Any]:
    def mean_of(x: BoolArray) -> Callable[[IntArray], float | None]:
        return lambda idx: float(x[idx].mean()) if len(idx) else None

    n = len(judge)
    if n == 0:
        return {"n": 0, "judge": None, "human": None, "diff": None, "low": None, "high": None}
    r = paired_bootstrap(
        mean_of(judge),
        mean_of(human),
        n,
        groups=groups,
        n_boot=boot.n_boot,
        seed=boot.seed,
        level=boot.level,
    )
    return {
        "n": n,
        "judge": r.a,
        "human": r.b,
        "diff": r.diff,
        "low": r.low,
        "high": r.high,
        "p_value": r.p_value,
    }


def _ci(counts: mt_bench.AgreementCounts, setup: str, boot: Boot) -> dict[str, Any]:
    ci = bootstrap_ci(
        lambda idx: counts.rate(setup, idx),
        len(counts.groups),
        n_boot=boot.n_boot,
        seed=boot.seed,
        level=boot.level,
    )
    return {"estimate": ci.estimate, "low": ci.low, "high": ci.high, "n": counts.n_pairs(setup)}


# ----------------------------------------------------------------------------- agreement


def agreement(s: Scored, boot: Boot = Boot()) -> dict[str, Any]:
    _check(s)
    units = by_comparison(s)
    u = units.scored
    out: dict[str, Any] = {
        "with_majority": {
            **_rate(u.correct, u.groups, boot),
            "n_votes": units.n_votes,
            "n_comparisons": units.n_comparisons,
            "n_no_majority": units.n_no_majority,
        },
        "published_human_human": mt_bench.PUBLISHED_HUMAN_HUMAN,
    }
    judge_human, human_human = mt_bench.decider_human(s), mt_bench.human_human(s)
    for setup in ("S1", "S2"):
        out[f"with_individuals_{setup}"] = _ci(judge_human, setup, boot)
        out[f"human_human_{setup}"] = _ci(human_human, setup, boot)
    out["vs_others_majority"] = _leave_one_out(s, boot)
    return out


def _leave_one_out(s: Scored, boot: Boot) -> dict[str, Any]:
    by_comp: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(s.meta):
        by_comp[str(m["comparison"])].append(i)
    judge_ok, human_ok, groups = [], [], []
    for rows in by_comp.values():
        named_gold = [_named(s.gold[i], s.meta[i]) for i in rows]
        for k, i in enumerate(rows):
            others = [g for j, g in enumerate(named_gold) if j != k and g is not None]
            majority = _plurality(others) if others else None
            if majority is None:
                continue
            human_ok.append(named_gold[k] == majority)
            judge_ok.append(_named(s.pred[i], s.meta[i]) == majority)
            groups.append(s.groups[i])
    return _paired(
        np.array(judge_ok, dtype=np.bool_), np.array(human_ok, dtype=np.bool_), tuple(groups), boot
    )


# ----------------------------------------------------------------------------- position


def position(s: Scored, boot: Boot = Boot()) -> dict[str, Any]:
    _check(s)
    # comparison → display order (model shown first) → the judge's positional verdict.
    seen: dict[str, dict[str, str]] = defaultdict(dict)
    group_of: dict[str, str] = {}
    for i, m in enumerate(s.meta):
        comp = str(m["comparison"])
        seen[comp].setdefault(str(m["model_a"]), s.pred[i])
        group_of[comp] = s.groups[i]
    kinds: Counter[str] = Counter()
    consistent, groups = [], []
    one_order = errors = 0
    for comp, orders in sorted(seen.items()):
        if len(orders) < 2:
            one_order += 1
            continue
        (first, v1), (second, v2) = sorted(orders.items())
        if ERROR_LABEL in (v1, v2):
            errors += 1
            continue
        named1 = mt_bench.to_model_name(v1, first, second)
        named2 = mt_bench.to_model_name(v2, second, first)
        if named1 == named2:
            kinds["consistent"] += 1
        elif v1 == v2 == "model_a":
            kinds["always_first"] += 1
        elif v1 == v2 == "model_b":
            kinds["always_second"] += 1
        else:
            kinds["other"] += 1
        consistent.append(named1 == named2)
        groups.append(group_of[comp])
    pairs = len(consistent)
    return {
        "n_comparisons": len(seen),
        "both_orders": pairs + errors,
        "both_orders_with_error": errors,
        "one_order_only": one_order,
        "extra_calls_for_full_coverage": one_order,
        "counts": {k: kinds[k] for k in ("consistent", "always_first", "always_second", "other")},
        "consistency": _rate(np.array(consistent, dtype=np.bool_), tuple(groups), boot)
        if pairs
        else {"estimate": None, "low": None, "high": None, "n": 0},
        "first_shown": _first_shown(s, boot),
    }


def _first_shown(s: Scored, boot: Boot) -> dict[str, Any]:
    """Rows where judge and human both picked a side: share picking the first answer."""
    rows = [
        i
        for i in range(len(s))
        if s.pred[i] in ("model_a", "model_b") and s.gold[i] in ("model_a", "model_b")
    ]
    return _paired(
        np.array([s.pred[i] == "model_a" for i in rows], dtype=np.bool_),
        np.array([s.gold[i] == "model_a" for i in rows], dtype=np.bool_),
        tuple(s.groups[i] for i in rows),
        boot,
    )


# ----------------------------------------------------------------------------- length


def length(s: Scored, lengths: Lengths | None, boot: Boot = Boot()) -> dict[str, Any]:
    _check(s)
    if lengths is None:
        return {"available": False, "note": "answer lengths need the local MT-Bench raw data"}
    sides = ("model_a", "model_b")
    rows, judge, human = [], [], []
    for i, ex_id in enumerate(s.ids):
        if ex_id not in lengths:
            continue
        la, lb = lengths[ex_id]
        if la == lb or s.pred[i] not in sides or s.gold[i] not in sides:
            continue
        longer = "model_a" if la > lb else "model_b"
        rows.append(i)
        judge.append(s.pred[i] == longer)
        human.append(s.gold[i] == longer)
    missing = sum(1 for i in s.ids if i not in lengths)
    return {
        "available": True,
        "measure": "characters in the judged turn's answer",
        "n_missing_lengths": missing,
        "prefers_longer": _paired(
            np.array(judge, dtype=np.bool_),
            np.array(human, dtype=np.bool_),
            tuple(s.groups[i] for i in rows),
            boot,
        ),
    }


def answer_lengths(examples: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """example id → (length of answer A, of answer B) for the judged turn, from MT-Bench
    inputs ``{"turn", "conversation_a", "conversation_b"}``. Only lengths leave here."""
    out = {}
    for ex_id, data in examples.items():
        turn = int(data["turn"])
        a = data["conversation_a"][2 * turn - 1]["content"]
        b = data["conversation_b"][2 * turn - 1]["content"]
        out[ex_id] = (len(str(a)), len(str(b)))
    return out


def load_lengths(
    ids: list[str] | tuple[str, ...], raw_dir: Path
) -> dict[str, tuple[int, int]] | None:
    """Answer lengths for MT-Bench example ids, from the raw parquet already on disk.
    Never downloads: returns None when the file isn't there (e.g. in CI)."""
    from ask_or_act.data.sources import MT_BENCH

    paths = [raw_dir / MT_BENCH.repo / f for f in MT_BENCH.files]
    if not all(p.is_file() for p in paths):
        return None
    from ask_or_act.data.hub import rows_by_id  # needs the `data` extra (pyarrow)

    rows, _ = rows_by_id("mt_bench", paths)
    wanted = {i: rows[i] for i in ids if i in rows}
    return answer_lengths(wanted)


# ----------------------------------------------------------------------------- ties


def ties(s: Scored, boot: Boot = Boot()) -> dict[str, Any]:
    _check(s)
    ok = [i for i in range(len(s)) if s.pred[i] != ERROR_LABEL]
    table = {h: dict.fromkeys(POSITIONS, 0) for h in POSITIONS}
    for i in ok:
        table[s.gold[i]][s.pred[i]] += 1
    # Comparisons where the humans split (no strict majority) vs. where they agreed.
    by_comp: dict[str, list[int]] = defaultdict(list)
    for i in ok:
        by_comp[str(s.meta[i]["comparison"])].append(i)
    split_rows: list[int] = []
    agreed_rows: list[int] = []
    for rows in by_comp.values():
        named = [_named(s.gold[i], s.meta[i]) for i in rows]
        target = split_rows if _plurality([n for n in named if n]) is None else agreed_rows
        target.extend(rows)

    def tie_rate(rows: list[int]) -> dict[str, Any]:
        if not rows:
            return {"estimate": None, "low": None, "high": None, "n": 0}
        return _rate(
            np.array([s.pred[i] == "tie" for i in rows], dtype=np.bool_),
            tuple(s.groups[i] for i in rows),
            boot,
        )

    return {
        "n": len(ok),
        "n_errors": len(s) - len(ok),
        "tie_rate": _paired(
            np.array([s.pred[i] == "tie" for i in ok], dtype=np.bool_),
            np.array([s.gold[i] == "tie" for i in ok], dtype=np.bool_),
            tuple(s.groups[i] for i in ok),
            boot,
        ),
        "table": table,  # human (rows) × judge (columns), positional labels
        "judge_tie_rate_where_humans_split": tie_rate(split_rows),
        "judge_tie_rate_where_humans_agree": tie_rate(agreed_rows),
    }


# ----------------------------------------------------------------------------- all


def judge_report(s: Scored, lengths: Lengths | None = None, boot: Boot = Boot()) -> dict[str, Any]:
    return {
        "n_votes": len(s),
        "n_errors": s.n_errors,
        "agreement": agreement(s, boot),
        "position": position(s, boot),
        "length": length(s, lengths, boot),
        "ties": ties(s, boot),
    }


def _check(s: Scored) -> None:
    if s.dataset != "mt_bench":
        raise ValueError("judge mode needs an MT-Bench run (vote level)")


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def _ci_text(r: Mapping[str, Any]) -> str:
    if r.get("estimate") is None:
        return "—"
    if r.get("low") is None:
        return _pct(r["estimate"])
    return f"{_pct(r['estimate'])} [{_pct(r['low'])}, {_pct(r['high'])}]"


def _diff_text(r: Mapping[str, Any]) -> str:
    if r.get("diff") is None:
        return f"n = {r.get('n', 0)}: —"
    ci = "" if r.get("low") is None else f" [{r['low']:+.1%}, {r['high']:+.1%}]"
    return (
        f"judge {_pct(r['judge'])} vs humans {_pct(r['human'])}: "
        f"{r['diff']:+.1%}{ci} (n = {r['n']})"
    )


def describe(r: Mapping[str, Any]) -> list[str]:
    a, p, ln, t = r["agreement"], r["position"], r["length"], r["ties"]
    m = a["with_majority"]
    lines = [
        f"MT-Bench judge: {r['n_votes']} votes, {r['n_errors']} failed",
        f"agreement with the human majority: {_ci_text(m)} on {m['n']} comparisons "
        f"({m['n_no_majority']} without a strict majority excluded)",
        f"with individual humans: S1 {_ci_text(a['with_individuals_S1'])}, "
        f"S2 {_ci_text(a['with_individuals_S2'])}",
        f"human-human (ceiling): S1 {_ci_text(a['human_human_S1'])}, "
        f"S2 {_ci_text(a['human_human_S2'])}",
        f"vs the other voters' majority (paired): {_diff_text(a['vs_others_majority'])}",
        f"position: {p['both_orders']} of {p['n_comparisons']} comparisons judged in both "
        f"orders; consistent {_ci_text(p['consistency'])}; {p['counts']}",
        f"  extra calls for full coverage: {p['extra_calls_for_full_coverage']} "
        "(one swapped-order call per comparison judged in one order only; not scheduled)",
        f"  picks the first-shown answer: {_diff_text(p['first_shown'])}",
    ]
    if ln["available"]:
        lines.append(f"length: picks the longer answer: {_diff_text(ln['prefers_longer'])}")
    else:
        lines.append(f"length: not available ({ln['note']})")
    lines.append(f"ties: tie rate {_diff_text(t['tie_rate'])}")
    lines.append(
        "  judge tie rate where humans split: "
        f"{_ci_text(t['judge_tie_rate_where_humans_split'])}; where they agree: "
        f"{_ci_text(t['judge_tie_rate_where_humans_agree'])}"
    )
    return lines

"""Judge mode (MT-Bench): agreement vs the human ceiling, position, length and ties.

Small hand-built vote tables whose answers can be checked by hand. No network.
"""

import numpy as np
import pytest

from ask_or_act import judge
from ask_or_act.core.types import JSON
from ask_or_act.metrics.data import Scored

BOOT = judge.Boot(n_boot=200)


def votes(rows: list[tuple[str, str, str, str, str]], q: str | None = None) -> Scored:
    """rows: (comparison "x-vs-y", model shown first, human label, judge label, voter)."""
    meta: list[dict[str, JSON]] = []
    for k, (comp, a, _, _, voter) in enumerate(rows):
        x, y = comp.split("-vs-")
        b = y if a == x else x
        meta.append(
            {
                "comparison": comp,
                "model_a": a,
                "model_b": b,
                "judge": voter,
                "question_id": k if q is None else 1,
            }
        )
    gold = tuple(r[2] for r in rows)
    pred = tuple(r[3] for r in rows)
    return Scored(
        ids=tuple(f"human:{i}" for i in range(len(rows))),
        gold=gold,
        pred=pred,
        confidence=np.full(len(rows), 0.8),
        correct=np.array([p == g for p, g in zip(pred, gold, strict=True)]),
        probabilities=tuple({} for _ in rows),
        options=("model_a", "model_b", "tie"),
        groups=tuple(f"q{m['question_id']}" for m in meta),
        meta=tuple(meta),
        dataset="mt_bench",
        n_errors=sum(p == "<error>" for p in pred),
    )


def test_position_consistency_and_bias_kinds() -> None:
    s = votes(
        [
            # a-vs-b: x first says model_a (= a); b first says model_b (= a): consistent.
            ("a-vs-b", "a", "model_a", "model_a", "h1"),
            ("a-vs-b", "b", "model_b", "model_b", "h2"),
            # c-vs-d: "model_a" in both orders: always the first answer.
            ("c-vs-d", "c", "model_a", "model_a", "h1"),
            ("c-vs-d", "d", "model_b", "model_a", "h2"),
            # e-vs-f: "model_b" both times: always the second.
            ("e-vs-f", "e", "model_a", "model_b", "h1"),
            ("e-vs-f", "f", "model_b", "model_b", "h2"),
            # g-vs-h: a tie in one order: other.
            ("g-vs-h", "g", "tie", "tie", "h1"),
            ("g-vs-h", "h", "tie", "model_a", "h2"),
            # i-vs-j: judged in one order only → one extra call.
            ("i-vs-j", "i", "model_a", "model_a", "h1"),
        ]
    )
    p = judge.position(s, BOOT)
    assert p["counts"] == {"consistent": 1, "always_first": 1, "always_second": 1, "other": 1}
    assert (p["n_comparisons"], p["both_orders"], p["one_order_only"]) == (5, 4, 1)
    assert p["extra_calls_for_full_coverage"] == 1
    assert p["consistency"]["estimate"] == pytest.approx(0.25)


def test_length_bias_counts_only_decisive_rows_with_different_lengths() -> None:
    s = votes(
        [
            ("a-vs-b", "a", "model_a", "model_a", "h1"),  # A longer: both pick longer
            ("c-vs-d", "c", "model_b", "model_a", "h1"),  # A longer: judge yes, human no
            ("e-vs-f", "e", "tie", "model_a", "h1"),  # human tie: excluded
            ("g-vs-h", "g", "model_a", "model_b", "h1"),  # equal lengths: excluded
        ]
    )
    lengths = {"human:0": (90, 10), "human:1": (50, 5), "human:2": (9, 1), "human:3": (7, 7)}
    r = judge.length(s, lengths, BOOT)["prefers_longer"]
    assert (r["n"], r["judge"], r["human"]) == (2, 1.0, 0.5)
    assert judge.length(s, None)["available"] is False


def test_ties_table_and_paired_rates() -> None:
    s = votes(
        [
            ("a-vs-b", "a", "tie", "model_a", "h1"),
            ("c-vs-d", "c", "tie", "tie", "h1"),
            ("e-vs-f", "e", "model_b", "model_b", "h1"),
            ("g-vs-h", "g", "model_a", "<error>", "h1"),
        ]
    )
    t = judge.ties(s, BOOT)
    assert (t["n"], t["n_errors"]) == (3, 1)
    assert t["table"]["tie"] == {"model_a": 1, "model_b": 0, "tie": 1}
    assert t["tie_rate"]["judge"] == pytest.approx(1 / 3)
    assert t["tie_rate"]["human"] == pytest.approx(2 / 3)


def test_leave_one_out_pairs_judge_and_human_on_the_same_reference() -> None:
    # One comparison, three voters: x, x, y (y shown first to voter 3). The judge says x.
    s = votes(
        [
            ("x-vs-y", "x", "model_a", "model_a", "h1"),
            ("x-vs-y", "x", "model_a", "model_a", "h2"),
            ("x-vs-y", "y", "model_a", "model_b", "h3"),  # voter 3: y; judge: x
        ],
        q="same",
    )
    loo = judge.agreement(s, BOOT)["vs_others_majority"]
    # Voter 1: others = {x, y}: no strict majority → skipped. Same for voter 2.
    # Voter 3: others = {x, x} → x. Human said y (0), judge said x (1).
    assert (loo["n"], loo["judge"], loo["human"]) == (1, 1.0, 0.0)
    a = judge.agreement(s, BOOT)
    assert a["with_majority"]["estimate"] == 1.0  # majority x, judge x
    assert a["human_human_S1"]["estimate"] == pytest.approx(1 / 3)


def test_answer_lengths_use_the_judged_turn() -> None:
    conv = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "aaaa"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "bb"},
    ]
    other = [*conv[:3], {"role": "assistant", "content": "cccccc"}]
    data = {"turn": 2, "conversation_a": conv, "conversation_b": other}
    assert judge.answer_lengths({"x": data}) == {"x": (2, 6)}


def test_non_mt_bench_runs_are_refused() -> None:
    s = votes([("a-vs-b", "a", "tie", "tie", "h1")])
    s = Scored(**{**s.__dict__, "dataset": "banking77"})
    with pytest.raises(ValueError, match="MT-Bench"):
        judge.judge_report(s)


def test_describe_runs_on_a_full_report() -> None:
    s = votes(
        [
            ("a-vs-b", "a", "model_a", "model_a", "h1"),
            ("a-vs-b", "b", "model_b", "tie", "h2"),
            ("c-vs-d", "c", "tie", "model_b", "h1"),
        ]
    )
    lines = judge.describe(judge.judge_report(s, {"human:0": (5, 1)}, BOOT))
    assert lines[0] == "MT-Bench judge: 3 votes, 0 failed"
    assert any(line.startswith("position:") for line in lines)

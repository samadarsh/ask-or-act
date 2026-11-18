"""MT-Bench: display-order normalization, tie setups, human–human ceiling."""

import pytest

from ask_or_act.core.types import JSON, Decision, Example
from ask_or_act.metrics import mt_bench
from tests.metrics_helpers import scored

OPTIONS = ("model_a", "model_b", "tie")


def vote(
    i: int, a: str, b: str, winner: str, judge: str, comparison: str, question: int = 81
) -> Example:
    meta: dict[str, JSON] = {
        "model_a": a,
        "model_b": b,
        "judge": judge,
        "comparison": comparison,
        "question_id": question,
    }
    return Example(f"human:{i}", "x", OPTIONS, winner, meta)


def test_to_model_name() -> None:
    assert mt_bench.to_model_name("model_a", "gpt-4", "vicuna") == "gpt-4"
    assert mt_bench.to_model_name("model_b", "gpt-4", "vicuna") == "vicuna"
    assert mt_bench.to_model_name("tie", "gpt-4", "vicuna") == "tie"
    assert mt_bench.to_model_name("???", "gpt-4", "vicuna") is None


def test_swapped_display_order_still_agrees() -> None:
    # Both experts preferred gpt-4, but saw it in opposite positions: the raw labels
    # differ ("model_a" vs "model_b"), the normalized ones agree.
    ex = [
        vote(0, "gpt-4", "vicuna", "model_a", "j1", "c1"),
        vote(1, "vicuna", "gpt-4", "model_b", "j2", "c1"),
    ]
    s = scored([Decision(e.id, e.gold, 1.0) for e in ex], ex, "mt_bench")
    hh = mt_bench.human_human(s)
    assert hh.rate("S1") == 1.0
    assert hh.n_pairs("S1") == 1


def test_tie_setups_and_same_judge_exclusion() -> None:
    ex = [
        vote(0, "a", "b", "model_a", "j1", "c1"),
        vote(1, "a", "b", "tie", "j2", "c1"),
        vote(2, "a", "b", "model_a", "j3", "c1"),
        vote(3, "a", "b", "model_b", "j1", "c1"),  # same judge as row 0: pair excluded
        vote(4, "a", "c", "model_a", "j1", "c2", question=82),  # alone: no pairs
    ]
    s = scored([Decision(e.id, e.gold, 1.0) for e in ex], ex, "mt_bench")
    hh = mt_bench.human_human(s)
    # Pairs (different judges) in c1: (0,1) (0,2) (1,2) (1,3) (2,3) → 5 pairs
    # S1 agree: (0,2) only → 1/5.  S2 (no ties): (0,2) agree, (2,3) disagree → 1/2
    assert hh.n_pairs("S1") == 5
    assert hh.rate("S1") == pytest.approx(1 / 5)
    assert hh.n_pairs("S2") == 2
    assert hh.rate("S2") == pytest.approx(1 / 2)


def test_decider_human_agreement_and_errors() -> None:
    ex = [
        vote(0, "a", "b", "model_a", "j1", "c1"),
        vote(1, "a", "b", "tie", "j2", "c1"),
        vote(2, "b", "a", "model_b", "j3", "c1"),
    ]
    decisions = [
        Decision("human:0", "model_a", 0.9),  # agrees
        Decision("human:1", "model_a", 0.9),  # vs tie: counts in S1 only
        Decision.failed("human:2", "timeout"),
    ]  # error: no pair at all
    da = mt_bench.decider_human(scored(decisions, ex, "mt_bench"))
    assert (da.n_pairs("S1"), da.rate("S1")) == (2, 0.5)
    assert (da.n_pairs("S2"), da.rate("S2")) == (1, 1.0)


def test_rate_undefined_without_pairs() -> None:
    ex = [vote(0, "a", "b", "tie", "j1", "c1")]
    hh = mt_bench.human_human(scored([Decision("human:0", "tie", 1.0)], ex, "mt_bench"))
    assert hh.rate("S2") is None


def test_report_includes_agreement_section() -> None:
    from ask_or_act.metrics.report import Settings, render_single, single_report

    ex = [
        vote(
            i,
            "gpt-4",
            "vicuna",
            ("model_a", "model_b", "tie")[i % 3],
            f"j{i % 4}",
            f"c{i // 4}",
            question=81 + i // 8,
        )
        for i in range(48)
    ]
    s = scored([Decision(e.id, "model_a", 0.7) for e in ex], ex, "mt_bench")
    r = single_report(s, Settings(n_boot=50))
    section = r["mt_bench"]
    assert set(section) >= {
        "decider_human_S1",
        "decider_human_S2",
        "human_human_S1",
        "human_human_S2",
        "published_human_human",
    }
    assert section["decider_human_S1"]["estimate"] == pytest.approx(1 / 3)
    assert "MT-Bench agreement" in render_single(r, "run")

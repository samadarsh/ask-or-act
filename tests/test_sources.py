"""Row -> Candidate conversion, using small fake rows shaped like the real datasets."""

from typing import Any

from ask_or_act.data.sources import (
    aegis_candidates,
    banking77_candidates,
    comparison_key,
    mt_bench_candidates,
)


def test_banking77_maps_label_ids_to_names() -> None:
    rows = [{"text": "where is my card", "label": 1}, {"text": "top up", "label": 0}]
    cands = banking77_candidates(rows, ["top_up", "card_arrival"])
    assert [(c.id, c.label) for c in cands] == [("test:0", "card_arrival"), ("test:1", "top_up")]
    assert cands[0].fingerprint != cands[1].fingerprint


def mt_row(qid: int, a: str, b: str, winner: str, judge: str, turn: int = 1) -> dict[str, Any]:
    return {
        "question_id": qid,
        "model_a": a,
        "model_b": b,
        "winner": winner,
        "judge": judge,
        "turn": turn,
        "conversation_a": [],
        "conversation_b": [],
    }


def test_comparison_key_ignores_presentation_order() -> None:
    assert comparison_key(mt_row(81, "gpt-4", "vicuna", "tie", "j1")) == comparison_key(
        mt_row(81, "vicuna", "gpt-4", "tie", "j2")
    )
    assert comparison_key(mt_row(81, "a", "b", "tie", "j", turn=1)) != comparison_key(
        mt_row(81, "a", "b", "tie", "j", turn=2)
    )


def test_mt_bench_records_vote_counts_and_groups_by_question() -> None:
    rows = [
        mt_row(81, "gpt-4", "vicuna", "model_a", "expert_1"),
        mt_row(81, "vicuna", "gpt-4", "model_b", "expert_2"),  # same comparison, swapped
        mt_row(82, "gpt-4", "llama", "tie", "expert_1"),
    ]
    cands = mt_bench_candidates(rows)
    assert [c.meta["n_votes"] for c in cands] == [2, 2, 1]
    assert cands[0].unit == cands[1].unit != cands[2].unit
    assert [c.group for c in cands] == ["q81", "q81", "q82"]
    assert [c.label for c in cands] == ["model_a", "model_b", "tie"]


def aegis_row(
    id_: str, prompt: str, label: str = "unsafe", source: str = "human"
) -> dict[str, Any]:
    return {
        "id": id_,
        "prompt": prompt,
        "prompt_label": label,
        "prompt_label_source": source,
        "response": None,
        "response_label": None,
    }


def test_aegis_filters_and_deduplicates() -> None:
    cands = aegis_candidates(
        {
            "test.json": [
                aegis_row("1", "p1"),
                aegis_row("2", "p2", source="llm_jury"),  # not human-labelled
                aegis_row("3", "REDACTED"),  # text not distributed
                aegis_row("4", "p1", label="unsafe"),  # duplicate prompt
                aegis_row("5", "p5", label="needs caution"),  # not a binary label
            ],
            "validation.json": [aegis_row("6", "p1"), aegis_row("7", "p7", label="safe")],
        }
    )
    assert [(c.id, c.label, c.meta["source_file"]) for c in cands] == [
        ("1", "unsafe", "test.json"),
        ("7", "safe", "validation.json"),
    ]


def test_aegis_candidates_never_store_text() -> None:
    (c,) = aegis_candidates({"t.json": [aegis_row("1", "SECRET-UNSAFE-TEXT")]})
    assert "SECRET" not in repr(c)

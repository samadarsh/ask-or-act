"""The prompts we pay to send: right content, no gold, MT-Bench turns rendered correctly."""

import json

import pytest

from ask_or_act.core.types import JSON, PublicExample
from ask_or_act.deciders.prompts import (
    OTHER,
    answer_schema,
    llm_prompt,
    options_with_other,
    render_input,
)


def mt_example(turn: int) -> PublicExample:
    conv_a: list[JSON] = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A-first"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A-second"},
    ]
    conv_b: list[JSON] = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "B-first"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "B-second"},
    ]
    return PublicExample(
        "human:1",
        {"turn": turn, "conversation_a": conv_a, "conversation_b": conv_b},
        ("model_a", "model_b", "tie"),
    )


def test_mt_bench_turn_1_shows_only_the_first_turn() -> None:
    text = render_input("mt_bench", mt_example(1))
    assert "A-first" in text
    assert "B-first" in text
    assert "Q2" not in text
    assert "A-second" not in text


def test_mt_bench_turn_2_shows_both_turns_in_order() -> None:
    text = render_input("mt_bench", mt_example(2))
    order = [text.index(s) for s in ("Q1", "A-first", "B-first", "Q2", "A-second", "B-second")]
    assert order == sorted(order)
    assert "turn 2" in llm_prompt("mt_bench", mt_example(2), mt_example(2).options)


@pytest.mark.parametrize(
    ("task", "options"),
    [
        ("banking77", ("card_arrival", "top_up")),
        ("aegis", ("safe", "unsafe")),
    ],
)
def test_text_tasks_list_options_and_the_input(task: str, options: tuple[str, ...]) -> None:
    ex = PublicExample("x", "the input text", options)
    prompt = llm_prompt(task, ex, options)
    assert "the input text" in prompt
    assert all(f"- {o}" in prompt for o in options)
    assert '"confidence"' in prompt


def test_other_option_is_added_once_and_explained() -> None:
    ex = PublicExample("x", "hi", ("a", "b"))
    opts = options_with_other(ex, True)
    assert opts == ("a", "b", OTHER)
    assert options_with_other(PublicExample("y", "hi", ("a", OTHER)), True) == ("a", OTHER)
    assert "only if none of the other options fits" in llm_prompt("banking77", ex, opts)
    assert options_with_other(ex, False) == ("a", "b")


def test_schema_is_strict_and_enumerates_options() -> None:
    schema = answer_schema(("a", "b"))
    assert schema["required"] == ["answer", "confidence"]
    assert schema["additionalProperties"] is False
    assert json.dumps(schema).count('"enum"') == 1


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="no prompt for task"):
        llm_prompt("imagenet", PublicExample("x", "hi", ("a", "b")), ("a", "b"))

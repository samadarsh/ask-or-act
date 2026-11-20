"""Confidence-signal extraction from token logprobs (arXiv 2608.04899 §4)."""

import math

import pytest

from ask_or_act.deciders.signals import (
    Token,
    answer_probability,
    parse_answer,
    verbalized_logprob,
)


def ln(p: float) -> float:
    return math.log(p)


def stream(*parts: tuple[str, float] | Token) -> list[Token]:
    return [p if isinstance(p, Token) else Token(p[0], p[1]) for p in parts]


def test_parse_answer_strict_and_lenient() -> None:
    assert parse_answer('{"answer": "safe", "confidence": 90}').answer == "safe"
    assert parse_answer('{"answer": "safe", "confidence": 90}').confidence == 90
    lenient = parse_answer('Sure! {"answer": "unsafe", "confidence": 7} hope this helps')
    assert (lenient.answer, lenient.confidence) == ("unsafe", 7)
    assert parse_answer("no json here").answer is None
    assert parse_answer('{"answer": "x", "confidence": "high"}').confidence is None


def test_answer_probability_sums_logprobs_over_the_answer_tokens() -> None:
    tokens = stream(
        ('{"answer":"', ln(1.0)),
        ("card", ln(0.9)),
        ("_arr", ln(0.8)),
        ("ival", ln(0.95)),
        ('","confidence":', ln(1.0)),
        ("87", ln(0.5)),
        ("}", ln(1.0)),
    )
    assert answer_probability(tokens) == pytest.approx(0.9 * 0.8 * 0.95)


def test_hybrid_whole_number_tokens() -> None:
    """o200k-style tokenizers emit "87" as one token: expectation over numeric
    alternatives at that position, renormalized (non-numeric alternatives dropped)."""
    conf = Token(
        "87", ln(0.5), top=(("87", ln(0.5)), ("90", ln(0.3)), ("8", ln(0.1)), ("high", ln(0.05)))
    )
    tokens = stream(('{"answer":"safe","confidence":', 0.0), conf, ("}", 0.0))
    h = verbalized_logprob(tokens)
    assert h is not None
    assert h.mode == "number"
    assert h.value == pytest.approx((87 * 0.5 + 90 * 0.3 + 8 * 0.1) / 0.9 / 100)


def test_hybrid_per_digit_matches_the_papers_formula() -> None:
    """Paper §4: Σ 10·d·P(tens=d) + Σ d·P(units=d)."""
    tens = Token("8", ln(0.6), top=(("8", ln(0.6)), ("9", ln(0.4))))
    units = Token("7", ln(0.5), top=(("7", ln(0.5)), ("0", ln(0.5))))
    tokens = stream(('{"answer":"safe","confidence":', 0.0), tens, units, ("}", 0.0))
    h = verbalized_logprob(tokens)
    assert h is not None
    assert h.mode == "digits"
    assert h.value == pytest.approx((10 * (8 * 0.6 + 9 * 0.4) + (7 * 0.5 + 0 * 0.5)) / 100)


def test_hybrid_handles_a_leading_space_inside_the_token() -> None:
    conf = Token(" 70", ln(0.7), top=((" 70", ln(0.7)), (" 80", ln(0.2)), ("70", ln(0.1))))
    tokens = stream(('{"answer":"safe","confidence":', 0.0), conf, ("}", 0.0))
    h = verbalized_logprob(tokens)
    assert h is not None
    assert h.value == pytest.approx((70 * 0.7 + 80 * 0.2) / 0.9 / 100)


def test_hybrid_unavailable_cases() -> None:
    assert verbalized_logprob(stream(('{"answer":"safe"}', 0.0))) is None
    # The number shares a token with the closing brace: can't isolate it.
    glued = stream(('{"answer":"safe","confidence":', 0.0), ("87}", ln(0.9)))
    assert verbalized_logprob(glued) is None


def test_without_alternatives_the_hybrid_equals_the_stated_number() -> None:
    tokens = stream(('{"answer":"safe","confidence":', 0.0), ("64", ln(0.9)), ("}", 0.0))
    h = verbalized_logprob(tokens)
    assert h is not None
    assert h.value == pytest.approx(0.64)

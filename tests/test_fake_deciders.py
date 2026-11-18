"""The fakes must behave exactly as documented, or later metric tests prove nothing."""

import asyncio
import random
from collections.abc import Sequence

import pytest

from ask_or_act.core.types import Decision, Example
from ask_or_act.deciders.fake import FakeDecider
from tests.helpers import fake, make_examples

N = 20_000
EXAMPLES = make_examples(N, k=4)


def decide(d: FakeDecider, examples: Sequence[Example] = EXAMPLES) -> list[Decision]:
    return asyncio.run(d.decide_batch(examples))


def accuracy(decisions: Sequence[Decision], examples: Sequence[Example] = EXAMPLES) -> float:
    return sum(d.choice == ex.gold for d, ex in zip(decisions, examples, strict=True)) / len(
        examples
    )


def mean_confidence(decisions: Sequence[Decision]) -> float:
    return sum(d.confidence or 0.0 for d in decisions) / len(decisions)


def test_perfect() -> None:
    out = decide(fake("perfect"))
    assert accuracy(out) == 1.0
    assert {d.confidence for d in out} == {1.0}


def test_random_is_chance_level() -> None:
    out = decide(fake("random", seed=1))
    assert accuracy(out) == pytest.approx(0.25, abs=0.015)
    assert {d.confidence for d in out} == {0.25}


def test_constant() -> None:
    out = decide(fake("constant", choice="o2", confidence=0.6))
    assert {d.choice for d in out} == {"o2"}
    assert {d.confidence for d in out} == {0.6}
    assert accuracy(out) == pytest.approx(0.25)


def test_constant_with_invalid_choice_raises() -> None:
    with pytest.raises(ValueError, match="not an option"):
        decide(fake("constant", choice="nope"), EXAMPLES[:1])


def test_calibrated_matches_confidence_in_every_bin() -> None:
    out = decide(fake("calibrated", seed=3))
    bins: dict[int, list[tuple[float, bool]]] = {}
    for d, ex in zip(out, EXAMPLES, strict=True):
        assert d.confidence is not None
        bins.setdefault(min(int((d.confidence - 0.5) / 0.1), 4), []).append(
            (d.confidence, d.choice == ex.gold)
        )
    assert len(bins) == 5
    for members in bins.values():
        conf = sum(c for c, _ in members) / len(members)
        acc = sum(ok for _, ok in members) / len(members)
        assert acc == pytest.approx(conf, abs=0.03)
    assert accuracy(out) == pytest.approx(0.75, abs=0.015)


def test_overconfident_has_the_documented_gap() -> None:
    out = decide(fake("overconfident", seed=3))
    gap = mean_confidence(out) - accuracy(out)
    assert gap == pytest.approx(0.125, abs=0.015)  # inflation 0.5 * E[1 - c] = 0.5 * 0.25
    # Same underlying correctness as the calibrated fake with the same seed.
    assert accuracy(out) == accuracy(decide(fake("calibrated", seed=3)))


def test_deterministic_and_order_independent() -> None:
    sample = EXAMPLES[:500]
    shuffled = list(sample)
    random.Random(0).shuffle(shuffled)
    a = {d.example_id: d for d in decide(fake("calibrated", seed=9), sample)}
    b = {d.example_id: d for d in decide(fake("calibrated", seed=9), shuffled)}
    assert a == b
    c = {d.example_id: d for d in decide(fake("calibrated", seed=10), sample)}
    assert a != c


def test_probabilities_sum_to_one() -> None:
    for d in decide(fake("overconfident", seed=1), EXAMPLES[:200]):
        assert sum(d.probabilities.values()) == pytest.approx(1.0)
        assert d.choice is not None
        assert d.probabilities[d.choice] == d.confidence


def test_pricing_flows_into_estimate_and_decisions() -> None:
    d = fake("perfect", input_price_per_mtok=1_000_000.0)  # $1 per input token
    ex = make_examples(2)
    est = d.estimate(ex)
    out = decide(d, ex)
    assert est.usd > 0
    assert sum(x.cost_usd for x in out) == pytest.approx(est.usd + 0.0)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"kind": "psychic"}, "psychic"),
        ({"low": 0.9, "high": 0.1}, "low <= high"),
        ({"inflation": 2.0}, "inflation"),
        ({"seed": "x"}, "must be a number"),
        ({"typo": 1}, "unknown fake decider params"),
        ({"kind": "constant", "confidence": 3.0}, "confidence"),
    ],
)
def test_invalid_params(params: dict[str, float | str], message: str) -> None:
    from ask_or_act.core.types import DeciderConfig

    with pytest.raises(ValueError, match=message):
        FakeDecider(DeciderConfig("fake", "m", params=params))

import dataclasses

import pytest

from ask_or_act.core.decider import Decider
from ask_or_act.core.types import DeciderConfig, Decision, Example, sha256_hex
from ask_or_act.deciders import build_decider
from ask_or_act.deciders.fake import FakeDecider
from tests.helpers import ScriptedDecider


def test_example_validation() -> None:
    Example("e", "text", ("a", "b"), "a")
    with pytest.raises(ValueError, match="at least 2"):
        Example("e", "text", ("a",), "a")
    with pytest.raises(ValueError, match="duplicate"):
        Example("e", "text", ("a", "a"), "a")
    with pytest.raises(ValueError, match="not in options"):
        Example("e", "text", ("a", "b"), "c")


def test_decision_validation() -> None:
    with pytest.raises(ValueError, match="outside"):
        Decision("e", "a", 1.5)
    with pytest.raises(ValueError, match="must have a choice"):
        Decision("e", None, None)
    failed = Decision.failed("e", "boom", attempts=3)
    assert not failed.ok
    assert failed.attempts == 3


def test_decision_is_immutable() -> None:
    d = Decision("e", "a", 0.9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.choice = "b"  # type: ignore[misc]


def test_decision_json_round_trip() -> None:
    d = Decision(
        "e",
        "a",
        0.75,
        probabilities={"a": 0.75, "b": 0.25},
        signals={"verbalized": 0.8},
        model="m",
        input_tokens=10,
        output_tokens=2,
        cost_usd=0.001,
        latency_s=0.2,
        attempts=2,
        raw={"note": "x"},
    )
    assert Decision.from_json(d.to_json()) == d


def test_sha256_is_key_order_independent() -> None:
    assert sha256_hex({"a": 1, "b": [1, 2]}) == sha256_hex({"b": [1, 2], "a": 1})


def test_deciders_satisfy_the_protocol() -> None:
    # The annotations are the real test: mypy checks each class against Decider,
    # like Go's `var _ Decider = (*FakeDecider)(nil)`.
    fake: Decider = FakeDecider(DeciderConfig("fake", "m"))
    scripted: Decider = ScriptedDecider()
    assert fake.max_batch_size == scripted.max_batch_size == 1


def test_registry() -> None:
    assert isinstance(build_decider(DeciderConfig("fake", "m")), FakeDecider)
    with pytest.raises(ValueError, match="unknown decider type"):
        build_decider(DeciderConfig("gpt-17", "m"))

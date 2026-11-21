"""Study plumbing: stratified / by-question subsets, sharing identical inputs within a
run, and self-consistency configs reusing main calls already in the cache."""

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from ask_or_act.core.cache import DecisionCache
from ask_or_act.core.config import ConfigError, parse_config
from ask_or_act.core.types import Example
from ask_or_act.data.examples import _largest_remainder, question_of, select
from ask_or_act.data.manifest import Manifest
from ask_or_act.data.sampling import SampledExample, Split, seeded_order
from ask_or_act.data.sources import BANKING77
from tests.provider_fixtures import Recorder, groq_ok
from tests.test_groq import BANK, make, run


def manifest(entries: list[SampledExample]) -> Manifest:
    return Manifest("toy", "toy", BANKING77, 7, tuple(entries))


def labelled(counts: dict[str, int], split: Split = Split.TEST) -> Manifest:
    return manifest(
        [
            SampledExample(f"{split.value}:{lab}:{i}", split, lab, "f")
            for lab, n in counts.items()
            for i in range(n)
        ]
    )


def votes(questions: dict[str, int]) -> Manifest:
    return manifest(
        [
            SampledExample(f"v:{q}:{i}", Split.TEST, "tie", "f", {"comparison": f"{q}-t1-a-vs-b"})
            for q, n in questions.items()
            for i in range(n)
        ]
    )


# ------------------------------------------------------------------ subsets


def test_per_label_takes_exactly_n_per_label_and_is_nested() -> None:
    m = labelled({"a": 10, "b": 10, "c": 3})
    four = select(m, "test", per_label=4)
    assert Counter(e.label for e in four) == {"a": 4, "b": 4, "c": 3}  # c has only 3
    two = select(m, "test", per_label=2)
    assert {e.id for e in two} <= {e.id for e in four}
    assert select(m, "test", per_label=4) == four  # deterministic


def test_per_label_respects_the_split() -> None:
    m = manifest([*labelled({"a": 5}).examples, *labelled({"a": 5}, Split.CALIBRATION).examples])
    assert {e.split for e in select(m, "calibration", per_label=2)} == {Split.CALIBRATION}


def test_label_stratified_limit_keeps_proportions() -> None:
    m = labelled({"safe": 231, "unsafe": 269})
    chosen = select(m, "test", 250, stratify="label")
    assert len(chosen) == 250
    assert Counter(e.label for e in chosen) == {
        "safe": 116,
        "unsafe": 134,
    }  # 115.5/134.5, tie by name
    small = select(m, "test", 100, stratify="label")
    assert {e.id for e in small} <= {e.id for e in chosen}


def test_largest_remainder_sums_to_total() -> None:
    assert _largest_remainder({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    assert sum(_largest_remainder({"x": 7, "y": 5, "z": 1}, 9).values()) == 9


def test_question_subset_takes_whole_questions() -> None:
    m = votes({"q1": 5, "q2": 7, "q3": 4, "q4": 6})
    chosen = select(m, "test", 10, stratify="question")
    qs = Counter(question_of(e) for e in chosen)
    sizes = {"q1": 5, "q2": 7, "q3": 4, "q4": 6}
    assert all(qs[q] == sizes[q] for q in qs)  # never a partial question
    assert len(chosen) >= 10
    assert len(chosen) - max(sizes[q] for q in qs) < 10  # stops right after reaching 10
    bigger = select(m, "test", 15, stratify="question")
    assert {e.id for e in chosen} <= {e.id for e in bigger}


def test_plain_limit_is_unchanged_so_old_caches_stay_valid() -> None:
    m = labelled({"a": 12, "b": 8})
    expected = sorted(m.examples, key=lambda e: seeded_order(e.id, m.seed))[:5]
    assert select(m, "test", 5) == sorted(expected, key=lambda e: (e.split, e.id))


def _cfg(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "x",
        "dataset": {"manifest": "m.json", **dataset},
        "decider": {"type": "fake", "model": "m"},
        "budget": {"max_usd": 0.0},
    }


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        ({"per_label": 0}, "per_label must be >= 1"),
        ({"per_label": 2, "limit": 5}, "either limit or per_label"),
        ({"stratify": "label"}, "stratify needs limit"),
        ({"limit": 5, "stratify": "intent"}, "random, label or question"),
    ],
)
def test_subset_config_errors(dataset: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(_cfg(dataset))


def test_subset_config_parses() -> None:
    assert parse_config(_cfg({"per_label": 4})).dataset.per_label == 4
    ds = parse_config(_cfg({"limit": 200, "stratify": "question"})).dataset
    assert (ds.limit, ds.stratify) == (200, "question")


# ------------------------------------------------- identical inputs within a run


def test_identical_inputs_share_one_call() -> None:
    same = [Example(f"test:{i}", "the same comparison", BANK, "card_arrival") for i in range(3)]
    other = Example("test:9", "another input", BANK, "top_up")
    rec = Recorder([groq_ok("card_arrival", 80), groq_ok("top_up", 60)])
    outcome = run(make("openai/gpt-oss-120b", rec), [*same, other])
    assert len(rec.requests) == 2
    assert outcome.plan.duplicates == {"test:1": "test:0", "test:2": "test:0"}
    assert outcome.plan.estimate.calls == 2
    d0, d1, d2, d9 = outcome.decisions
    assert [d.example_id for d in outcome.decisions] == ["test:0", "test:1", "test:2", "test:9"]
    assert d1.choice == d2.choice == d0.choice == "card_arrival"
    assert (d1.cached, d1.charged_usd) == (True, 0.0)
    assert d1.cost_usd == d0.cost_usd  # true cost kept, as for a cache hit
    assert d9.choice == "top_up"


# ------------------------------------------- self-consistency reuses main calls


def test_self_consistency_config_reuses_the_main_only_run(tmp_path: Path) -> None:
    """Plan M: main calls on the whole half first (samples = 0), then k = 3 on a subset.
    The second config pays only for the samples."""
    examples = [Example("test:0", "where is my card", BANK, "card_arrival")]
    with DecisionCache(tmp_path / "c.sqlite") as cache:
        rec = Recorder([groq_ok("card_arrival", 80)])
        (main_only,) = run(make("openai/gpt-oss-120b", rec, cache=cache), examples).decisions
        assert main_only.confidence == 0.8

        params = {"samples": 3, "confidence_signal": "self_consistency"}
        sc = make("openai/gpt-oss-120b", Recorder([]), params, cache=cache)
        assert sc.estimate([e.public() for e in examples]).calls == 3  # main is cached

        rec2 = Recorder([groq_ok("card_arrival", 70)] * 2 + [groq_ok("top_up", 70)])
        (d,) = run(make("openai/gpt-oss-120b", rec2, params, cache=cache), examples).decisions
    assert len(rec2.requests) == 3
    assert all(b["temperature"] == 1.0 for b in rec2.bodies())  # samples only
    assert d.raw["main_cached"] is True
    assert d.confidence == pytest.approx(2 / 3)
    assert d.signals["verbalized"] == 0.8  # from the reused main call
    assert d.cost_usd > main_only.cost_usd  # true cost includes the main call
    assert d.charged_usd == 0.0  # free tier


def test_main_identity_ignores_only_sample_settings() -> None:
    base = make("openai/gpt-oss-120b", Recorder([]))._main_identity()
    sc = {"samples": 3, "sample_variation": "shuffle", "confidence_signal": "self_consistency"}
    assert make("openai/gpt-oss-120b", Recorder([]), sc)._main_identity() == base
    other = make("openai/gpt-oss-120b", Recorder([]), {"reasoning_effort": "low"})
    assert other._main_identity() != base

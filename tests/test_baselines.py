"""Free baselines: majority, random at equal cost, classic TF-IDF/length models."""

import asyncio
from collections import Counter

import numpy as np
import pytest

from ask_or_act.core.types import JSON, DeciderConfig, Decision, PublicExample
from ask_or_act.deciders import DeciderContext, build_decider
from ask_or_act.deciders.baselines import equal_cost_fraction, random_at_equal_cost
from ask_or_act.deciders.classic import ClassicDecider, group_folds

OPTS = ("billing", "shipping", "technical")


def test_majority_uses_training_distribution() -> None:
    labels: list[JSON] = [*(["billing"] * 6), *(["shipping"] * 3), "technical"]
    d = build_decider(
        DeciderConfig("majority", "majority", params={"labels": labels}),
        DeciderContext(dataset="toy"),
    )
    out = asyncio.run(d.decide_batch([PublicExample("x", "hi", OPTS)]))
    assert out[0].choice == "billing"
    assert out[0].confidence == pytest.approx(0.6)
    assert out[0].probabilities == {"billing": 0.6, "shipping": 0.3, "technical": 0.1}
    assert d.estimate([]).usd == 0


def test_equal_cost_fraction() -> None:
    assert equal_cost_fraction(1.0, 5.0, 2.0) == pytest.approx(0.25)
    assert equal_cost_fraction(1.0, 5.0, 10.0) == 1.0
    with pytest.raises(ValueError, match="more than"):
        equal_cost_fraction(2.0, 1.0, 1.5)


def test_random_at_equal_cost_hits_the_fraction_and_is_deterministic() -> None:
    cheap = [Decision(f"e{i}", "a", 0.5, cost_usd=1.0) for i in range(4000)]
    strong = [Decision(f"e{i}", "b", 0.9, cost_usd=5.0) for i in range(4000)]
    mix = random_at_equal_cost(cheap, strong, 0.25, seed=3)
    share = Counter(d.raw["routed_to"] for d in mix)["strong"] / 4000
    assert share == pytest.approx(0.25, abs=0.02)
    assert np.mean([d.cost_usd for d in mix]) == pytest.approx(2.0, abs=0.1)
    assert mix == random_at_equal_cost(cheap, strong, 0.25, seed=3)
    with pytest.raises(ValueError, match="same examples"):
        random_at_equal_cost(cheap[:2], strong[1:3], 0.5)


def classic(dataset: str) -> ClassicDecider:
    return ClassicDecider(
        DeciderConfig("classic", "tfidf-logreg", params={"cv": 3}), DeciderContext(dataset=dataset)
    )


def test_classic_text_model_learns_and_gives_probabilities() -> None:
    rng = np.random.default_rng(0)
    vocab = {
        "billing": ["refund", "charged", "invoice"],
        "shipping": ["parcel", "delivery", "tracking"],
        "technical": ["crash", "login", "error"],
    }
    texts, labels = [], []
    for i in range(300):
        label = OPTS[i % 3]
        words = [*rng.choice(vocab[label], 3), "please", "help"]
        texts.append(" ".join(words))
        labels.append(label)
    d = classic("toy")
    d.fit([f"train:{i}" for i in range(300)], texts, labels)
    test = [
        PublicExample("t1", "my parcel delivery is late", OPTS),
        PublicExample("t2", "app crash on login", OPTS),
    ]
    out = asyncio.run(d.decide_batch(test))
    assert [o.choice for o in out] == ["shipping", "technical"]
    for o in out:
        assert sum(o.probabilities.values()) == pytest.approx(1.0)
        assert o.confidence == max(o.probabilities.values())


def test_classic_refuses_to_predict_on_training_ids() -> None:
    d = classic("toy")
    d.fit(
        [f"train:{i}" for i in range(12)],
        ["refund money", "parcel late", "app crash"] * 4,
        ["billing", "shipping", "technical"] * 4,
    )
    with pytest.raises(ValueError, match="training examples"):
        asyncio.run(d.decide_batch([PublicExample("train:1", "parcel late", OPTS)]))


def conv(a_len: int, b_len: int) -> JSON:
    q: dict[str, JSON] = {"role": "user", "content": "q"}
    return {
        "turn": 1,
        "conversation_a": [q, {"role": "assistant", "content": "x" * a_len}],
        "conversation_b": [q, {"role": "assistant", "content": "x" * b_len}],
    }


def test_classic_mt_bench_length_model_learns_verbosity() -> None:
    opts = ("model_a", "model_b", "tie")
    inputs: list[JSON] = []
    labels: list[str] = []
    rng = np.random.default_rng(1)
    for _ in range(200):
        a, b = int(rng.integers(10, 2000)), int(rng.integers(10, 2000))
        inputs.append(conv(a, b))
        labels.append("model_a" if a > 1.5 * b else "model_b" if b > 1.5 * a else "tie")
    d = classic("mt_bench")
    d.fit([f"cal:{i}" for i in range(200)], inputs, labels)
    out = asyncio.run(d.decide_batch([PublicExample("t", conv(1800, 50), opts)]))
    assert out[0].choice == "model_a"


def test_group_folds_keep_groups_together_and_balance_them() -> None:
    groups = [f"q{i // 7}" for i in range(7 * 40)]  # 40 questions, 7 rows each
    folds = group_folds(groups, 5, seed=0)
    by_group: dict[str, set[int]] = {}
    for g, f in zip(groups, folds, strict=True):
        by_group.setdefault(g, set()).add(f)
    assert all(len(s) == 1 for s in by_group.values())  # a group never straddles folds
    counts = [sum(f == k for f in folds) for k in range(5)]
    assert counts == [56] * 5  # 8 questions per fold
    assert folds == group_folds(groups, 5, seed=0)  # deterministic
    assert folds != group_folds(groups, 5, seed=1)
    with pytest.raises(ValueError, match="at least 5 groups"):
        group_folds(["a", "b"], 5, seed=0)


def test_mt_bench_cross_fit_predicts_training_ids_out_of_fold() -> None:
    """Calibration-half examples are predicted by a model that never saw their
    question; test examples by the full model."""
    opts = ("model_a", "model_b", "tie")
    rng = np.random.default_rng(2)
    ids: list[str] = []
    inputs: list[JSON] = []
    labels: list[str] = []
    groups: list[str] = []
    for i in range(300):
        a, b = int(rng.integers(10, 2000)), int(rng.integers(10, 2000))
        ids.append(f"cal:{i}")
        inputs.append(conv(a, b))
        labels.append("model_a" if a > 1.5 * b else "model_b" if b > 1.5 * a else "tie")
        groups.append(f"q{i // 6}")  # 50 questions
    d = classic("mt_bench")
    d.fit(ids, inputs, labels, groups)
    cal = [PublicExample(ids[i], inputs[i], opts) for i in range(0, 300, 37)]
    out = asyncio.run(d.decide_batch([*cal, PublicExample("test:1", conv(1800, 50), opts)]))
    folds = [o.raw.get("cross_fit_fold") for o in out]
    assert all(isinstance(f, int) for f in folds[:-1])
    assert folds[-1] is None
    assert out[-1].choice == "model_a"
    # Every row's fold model was trained without that row's whole question.
    outer = group_folds(groups, 5, seed=1)
    for ex, o in zip(cal, out[:-1], strict=True):
        i = ids.index(ex.id)
        assert o.raw["cross_fit_fold"] == outer[i]
        same_q = {ids[j] for j in range(300) if groups[j] == groups[i]}
        assert all(outer[ids.index(s)] == outer[i] for s in same_q)


def test_resolved_defaults_are_part_of_the_cache_identity() -> None:
    """Changing a default in code must change the cache key, or stale answers would be
    served. Deciders report their config with defaults filled in."""
    implicit = classic("toy").config
    explicit = ClassicDecider(
        DeciderConfig("classic", "tfidf-logreg", params={"cv": 3, "calibration": "temperature"}),
        DeciderContext(dataset="toy"),
    ).config
    assert implicit == explicit
    assert implicit.params["calibration"] == "temperature"
    sigmoid = ClassicDecider(
        DeciderConfig("classic", "tfidf-logreg", params={"cv": 3, "calibration": "sigmoid"}),
        DeciderContext(dataset="toy"),
    ).config
    assert sigmoid != implicit

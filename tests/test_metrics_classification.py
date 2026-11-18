"""Classification metrics against scikit-learn."""

import numpy as np
import pytest
from sklearn import metrics as skm

from ask_or_act.metrics.classification import (
    accuracy,
    cohen_kappa,
    confusion_matrix,
    label_set,
    macro_f1,
)

rng = np.random.default_rng(42)
LABELS = ["a", "b", "c", "d", "e"]


def random_labels(n: int, agree: float) -> tuple[list[str], list[str]]:
    y = [str(x) for x in rng.choice(LABELS, size=n)]
    pred = [t if rng.random() < agree else str(rng.choice(LABELS)) for t in y]
    return y, pred


@pytest.mark.parametrize(("n", "agree"), [(50, 0.3), (400, 0.7), (1000, 0.95)])
def test_matches_sklearn(n: int, agree: float) -> None:
    y, p = random_labels(n, agree)
    labels = label_set(y, p)
    assert np.array_equal(confusion_matrix(y, p), skm.confusion_matrix(y, p, labels=labels))
    assert macro_f1(y, p) == pytest.approx(skm.f1_score(y, p, average="macro", zero_division=0))
    assert cohen_kappa(y, p) == pytest.approx(skm.cohen_kappa_score(y, p))
    correct = np.array([a == b for a, b in zip(y, p, strict=True)])
    assert accuracy(correct) == pytest.approx(skm.accuracy_score(y, p))


def test_labels_only_in_predictions_count_like_sklearn() -> None:
    # "<error>" never appears in gold: it lowers macro-F1 as a class with F1 = 0.
    y = ["a", "a", "b", "b"]
    p = ["a", "<error>", "b", "a"]
    assert macro_f1(y, p) == pytest.approx(skm.f1_score(y, p, average="macro", zero_division=0))
    assert cohen_kappa(y, p) == pytest.approx(skm.cohen_kappa_score(y, p))


def test_numpy_string_arrays_work() -> None:
    y, p = random_labels(100, 0.5)
    assert macro_f1(np.array(y), np.array(p)) == pytest.approx(macro_f1(y, p))


def test_kappa_hand_computed() -> None:
    # 2×2 table [[20, 5], [10, 15]]: p_o = 0.7, p_e = (25·30 + 25·20)/50² = 0.5 → κ = 0.4
    y = ["x"] * 25 + ["y"] * 25
    p = ["x"] * 20 + ["y"] * 5 + ["x"] * 10 + ["y"] * 15
    assert cohen_kappa(y, p) == pytest.approx(0.4)


def test_kappa_undefined_when_one_label_only() -> None:
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None
    assert cohen_kappa([], []) is None

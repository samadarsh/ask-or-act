"""Accuracy, confusion matrix, macro-F1, Cohen's kappa.

Plain functions over label sequences (lists or numpy string arrays). Conventions
match scikit-learn, and the tests check against it:

* The label set is the sorted union of true and predicted labels.
* The F1 of a class with no predictions and no true instances is 0.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ask_or_act.metrics.data import BoolArray

type Labels = Sequence[str] | npt.NDArray[np.str_]


def accuracy(correct: BoolArray) -> float:
    return float(np.mean(correct)) if len(correct) else float("nan")


def label_set(y_true: Labels, y_pred: Labels) -> list[str]:
    return sorted(set(y_true) | set(y_pred))


def confusion_matrix(
    y_true: Labels, y_pred: Labels, labels: Sequence[str] | None = None
) -> npt.NDArray[np.int64]:
    """Rows = true label, columns = predicted label."""
    labels = list(labels) if labels is not None else label_set(y_true, y_pred)
    index = {lb: i for i, lb in enumerate(labels)}
    k = len(labels)
    t = np.fromiter((index[x] for x in y_true), dtype=np.int64, count=len(y_true))
    p = np.fromiter((index[x] for x in y_pred), dtype=np.int64, count=len(y_pred))
    return np.bincount(t * k + p, minlength=k * k).reshape(k, k)


def macro_f1(y_true: Labels, y_pred: Labels) -> float:
    m = confusion_matrix(y_true, y_pred).astype(np.float64)
    tp = np.diag(m)
    denom = m.sum(axis=0) + m.sum(axis=1)  # F1 = 2tp / (2tp + fp + fn) = 2tp / (pred + actual)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def cohen_kappa(a: Labels, b: Labels) -> float | None:
    """Chance-corrected agreement: (p_o − p_e) / (1 − p_e). None when p_e = 1 (undefined)."""
    if len(a) == 0:
        return None
    m = confusion_matrix(a, b).astype(np.float64)
    n = m.sum()
    p_o = np.trace(m) / n
    p_e = float((m.sum(axis=0) * m.sum(axis=1)).sum() / n**2)
    if p_e >= 1.0:
        return None
    return float((p_o - p_e) / (1.0 - p_e))

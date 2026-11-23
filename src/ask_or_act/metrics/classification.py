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


class LabelCodes:
    """Labels encoded once as integers, so the bootstrap's inner loop is one bincount
    per resample instead of a Python loop over every example.

    ``macro_f1`` and ``cohen_kappa`` here equal the functions above on the resampled
    labels (tests/test_metrics_speed.py): classes absent from a resample are left out
    of the macro average exactly as ``label_set`` leaves them out, and contribute
    nothing to κ's chance agreement.
    """

    def __init__(self, y_true: Labels, y_pred: Labels) -> None:
        labels = label_set(y_true, y_pred)
        index = {lb: i for i, lb in enumerate(labels)}
        self.k = len(labels)
        self.t = np.fromiter((index[x] for x in y_true), dtype=np.int64, count=len(y_true))
        self.p = np.fromiter((index[x] for x in y_pred), dtype=np.int64, count=len(y_pred))

    def confusion(self, idx: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
        k = self.k
        cells = np.bincount(self.t[idx] * k + self.p[idx], minlength=k * k)
        return cells.reshape(k, k).astype(np.float64)

    def macro_f1(self, idx: npt.NDArray[np.intp]) -> float:
        m = self.confusion(idx)
        tp = np.diag(m)
        denom = m.sum(axis=0) + m.sum(axis=1)
        present = denom > 0
        if not present.any():
            return float("nan")
        return float((2 * tp[present] / denom[present]).sum() / present.sum())

    def cohen_kappa(self, idx: npt.NDArray[np.intp]) -> float | None:
        if len(idx) == 0:
            return None
        m = self.confusion(idx)
        n = m.sum()
        p_o = np.trace(m) / n
        p_e = float((m.sum(axis=0) * m.sum(axis=1)).sum() / n**2)
        if p_e >= 1.0:
            return None
        return float((p_o - p_e) / (1.0 - p_e))

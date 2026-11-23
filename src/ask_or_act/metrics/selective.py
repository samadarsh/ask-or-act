"""Selective prediction: does confidence tell right answers from wrong ones?

Sort by confidence, highest first, and answer only the top fraction (the *coverage*).
The *risk* is the error rate among answered examples. A useful confidence signal keeps
risk low at low coverage.

Two pitfalls from arXiv 2608.04899 (§3) are handled here:

* **Ties are blocks.** Examples with equal confidence can't be separated by any
  threshold, so the curve only has a point after each whole tie block. Sparse
  verbalized scores (a few distinct values) therefore produce few points, which is the
  truth about them.
* **Stepwise area.** AURC = Σ over blocks of (block size / N) × (risk after that block).
  This is the area under the step function. Trapezoidal interpolation would invent
  operating points that no threshold can reach, and it reorders methods with sparse
  scores. With no ties it equals the standard (1/N) Σ_i risk(top-i).
"""

from dataclasses import dataclass

import numpy as np

from ask_or_act.metrics.data import BoolArray, FloatArray, IntArray


@dataclass(frozen=True)
class RiskCoverage:
    """One point per tie block, in descending threshold order."""

    threshold: FloatArray  # answer when confidence >= threshold
    coverage: FloatArray
    risk: FloatArray


def risk_coverage(confidence: FloatArray, correct: BoolArray) -> RiskCoverage:
    n = len(confidence)
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return RiskCoverage(empty, empty, empty)
    levels, inverse = np.unique(-confidence, return_inverse=True)  # ascending -c = descending c
    block_n = np.bincount(inverse).astype(np.float64)
    block_err = np.bincount(inverse, weights=(~correct).astype(np.float64))
    cum_n, cum_err = np.cumsum(block_n), np.cumsum(block_err)
    return RiskCoverage(threshold=-levels, coverage=cum_n / n, risk=cum_err / cum_n)


def aurc(confidence: FloatArray, correct: BoolArray) -> float:
    """Area under the risk–coverage step curve. Lower is better."""
    rc = risk_coverage(confidence, correct)
    if len(rc.coverage) == 0:
        return float("nan")
    widths = np.diff(np.concatenate([[0.0], rc.coverage]))
    return float((widths * rc.risk).sum())


def optimal_aurc(correct: BoolArray) -> float:
    """AURC of an oracle that ranks every right answer above every wrong one."""
    return aurc(correct.astype(np.float64), correct)


def excess_aurc(confidence: FloatArray, correct: BoolArray) -> float:
    """AURC minus the oracle's. This removes the part that only reflects accuracy."""
    return aurc(confidence, correct) - optimal_aurc(correct)


def auroc(confidence: FloatArray, correct: BoolArray) -> float | None:
    """P(conf of a random right answer > conf of a random wrong one), ties counting ½.

    Computed with the Mann–Whitney U statistic and average ranks for ties. None if all
    answers are right, or all are wrong.
    """
    n_pos = int(correct.sum())
    n_neg = len(correct) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(confidence, kind="mergesort")
    ranks = np.empty(len(confidence), dtype=np.float64)
    sorted_c = confidence[order]
    # Average rank (1-based) for each tie group.
    _, start, counts = np.unique(sorted_c, return_index=True, return_counts=True)
    avg = start + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    u = ranks[correct].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def n_unique(confidence: FloatArray) -> int:
    return len(np.unique(confidence))


class ConfidenceLevels:
    """The distinct confidence values, sorted once, so a resample's AURC and AUROC are
    bincounts instead of a sort (for the bootstrap's inner loop).

    A resample only changes how many examples sit at each level, so the stepwise
    risk–coverage curve is the same sums in the same order; levels absent from a
    resample add zero-width steps, which are skipped. Equal to ``aurc``,
    ``excess_aurc`` and ``auroc`` (tests/test_metrics_speed.py).
    """

    def __init__(self, confidence: FloatArray, correct: BoolArray) -> None:
        levels, inverse = np.unique(-confidence, return_inverse=True)  # descending c
        self.down = inverse.astype(np.intp)  # level index, most confident first
        self.up = (len(levels) - 1 - inverse).astype(np.intp)  # least confident first
        self.size = len(levels)
        self.wrong = (~correct).astype(np.float64)
        self.right = correct.astype(np.float64)

    def aurc(self, idx: IntArray) -> float:
        n = len(idx)
        if n == 0:
            return float("nan")
        block_n = np.bincount(self.down[idx], minlength=self.size).astype(np.float64)
        block_err = np.bincount(self.down[idx], weights=self.wrong[idx], minlength=self.size)
        cum_n, cum_err = np.cumsum(block_n), np.cumsum(block_err)
        present = block_n > 0
        return float((block_n[present] / n * (cum_err[present] / cum_n[present])).sum())

    def excess_aurc(self, idx: IntArray) -> float:
        n = len(idx)
        if n == 0:
            return float("nan")
        wrong = float(self.wrong[idx].sum())
        oracle = (wrong / n) ** 2  # right answers first (risk 0), then all the wrong ones
        return self.aurc(idx) - oracle

    def auroc(self, idx: IntArray) -> float | None:
        n_pos = int(self.right[idx].sum())
        n_neg = len(idx) - n_pos
        if n_pos == 0 or n_neg == 0:
            return None
        count = np.bincount(self.up[idx], minlength=self.size).astype(np.float64)
        pos = np.bincount(self.up[idx], weights=self.right[idx], minlength=self.size)
        start = np.cumsum(count) - count  # 0-based index of each level's first member
        avg_rank = start + (count + 1) / 2.0
        u = float((pos * avg_rank).sum()) - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))

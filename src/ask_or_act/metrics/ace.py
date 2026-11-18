"""Accuracy-controlled calibration comparison (ACE).

Source: Yang, Zhang et al., *When Calibration Rankings Reverse: Accuracy-Controlled
Evaluation for Fair Comparison of LLMs*, arXiv:2606.30814 **v2** (2026-08-31; EMNLP 2026).

Why it's needed: ECE and Brier depend on accuracy, not only on how well confidence tracks
correctness. A more accurate model can look "better calibrated" for that reason alone.
ACE compares two models on the same examples under three alignments (§4):

* **DA, distribution-aligned (§4.2).** The paper's **default** (§6). Reweight the
  higher-accuracy model so its weighted accuracy equals the other's:
  w = a_low/a_high for its right answers and (1 − a_low)/(1 − a_high) for its wrong ones.
  Then compute weighted ECE and Brier. All examples are kept; we report Kish's
  effective sample size, (Σw)²/Σw² (Appendix E.5, where ESS/N < 0.5 is flagged).
* **IA, instance-aligned (§4.1).** The **corroborating check**. Keep only examples
  where both are right or both are wrong, so accuracies are equal by construction.
  We report the share of examples kept.
* **CA, candidate-aligned (§4.3).** A **diagnostic**. Pool both models' answers for
  each input, and have each model score every candidate with its own probability for
  it. Only possible when both runs give per-option probabilities. Duplicate answers
  (both chose the same option) count once; the paper doesn't say, so this is our choice.

A *reversal* means the raw gap and an aligned gap have opposite signs: the
"better-calibrated" model changes once accuracy is controlled. The bin count isn't
stated in the paper. We use the same bins as the rest of the report and keep the
binning fixed across views.
"""

from dataclasses import dataclass, field

import numpy as np

from ask_or_act.metrics.calibration import Binning, brier_top_label, ece
from ask_or_act.metrics.data import ERROR_LABEL, BoolArray, FloatArray, IntArray, Scored

ESS_WARNING = 0.5


@dataclass(frozen=True)
class View:
    name: str
    available: bool
    ece_a: float | None = None
    ece_b: float | None = None
    brier_a: float | None = None
    brier_b: float | None = None
    info: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def ece_gap(self) -> float | None:
        return None if self.ece_a is None or self.ece_b is None else self.ece_a - self.ece_b

    @property
    def brier_gap(self) -> float | None:
        if self.brier_a is None or self.brier_b is None:
            return None
        return self.brier_a - self.brier_b


def _view(
    name: str,
    ca: FloatArray,
    za: BoolArray,
    cb: FloatArray,
    zb: BoolArray,
    wa: FloatArray | None,
    wb: FloatArray | None,
    n_bins: int,
    strategy: Binning,
    info: dict[str, float] | None = None,
    note: str = "",
) -> View:
    return View(
        name,
        True,
        ece_a=ece(ca, za, n_bins=n_bins, strategy=strategy, weights=wa),
        ece_b=ece(cb, zb, n_bins=n_bins, strategy=strategy, weights=wb),
        brier_a=brier_top_label(ca, za, weights=wa),
        brier_b=brier_top_label(cb, zb, weights=wb),
        info=info or {},
        note=note,
    )


def raw(
    ca: FloatArray, za: BoolArray, cb: FloatArray, zb: BoolArray, n_bins: int, strategy: Binning
) -> View:
    return _view("raw", ca, za, cb, zb, None, None, n_bins, strategy)


def instance_aligned(
    ca: FloatArray, za: BoolArray, cb: FloatArray, zb: BoolArray, n_bins: int, strategy: Binning
) -> View:
    keep = za == zb
    if not keep.any():
        return View("IA", False, note="no example where both are right or both wrong")
    return _view(
        "IA",
        ca[keep],
        za[keep],
        cb[keep],
        zb[keep],
        None,
        None,
        n_bins,
        strategy,
        info={"retention": float(keep.mean())},
    )


def da_weights(correct: BoolArray, target_accuracy: float) -> FloatArray | None:
    """Importance weights moving this model's accuracy to ``target_accuracy`` (§4.2)."""
    acc = float(correct.mean())
    if acc in (0.0, 1.0):
        return None  # one correctness class is empty; positivity fails
    return np.where(correct, target_accuracy / acc, (1 - target_accuracy) / (1 - acc))


def kish_ess(w: FloatArray) -> float:
    return float(w.sum() ** 2 / (w**2).sum())


def distribution_aligned(
    ca: FloatArray, za: BoolArray, cb: FloatArray, zb: BoolArray, n_bins: int, strategy: Binning
) -> View:
    acc_a, acc_b = float(za.mean()), float(zb.mean())
    wa = wb = None
    if acc_a > acc_b:
        wa = da_weights(za, acc_b)
        weighted, w = "a", wa
    elif acc_b > acc_a:
        wb = da_weights(zb, acc_a)
        weighted, w = "b", wb
    else:
        return _view(
            "DA",
            ca,
            za,
            cb,
            zb,
            None,
            None,
            n_bins,
            strategy,
            info={"ess_ratio": 1.0},
            note="equal accuracy; no reweighting needed",
        )
    if w is None:
        return View("DA", False, note="the higher-accuracy model is never wrong; can't reweight")
    ess_ratio = kish_ess(w) / len(w)
    note = f"reweighted {weighted.upper()}"
    if ess_ratio < ESS_WARNING:
        note += f"; ESS/N = {ess_ratio:.2f} < {ESS_WARNING} (unstable)"
    return _view(
        "DA", ca, za, cb, zb, wa, wb, n_bins, strategy, info={"ess_ratio": ess_ratio}, note=note
    )


@dataclass(frozen=True)
class CandidateInstances:
    """Precomputed (example, candidate) pairs for the CA view."""

    members: tuple[IntArray, ...]  # per example: positions of its candidate instances
    conf_a: FloatArray
    conf_b: FloatArray
    correct: BoolArray


def candidate_instances(a: Scored, b: Scored) -> CandidateInstances | None:
    members: list[IntArray] = []
    pa: list[float] = []
    pb: list[float] = []
    y: list[bool] = []
    for i in range(len(a)):
        cands = sorted({a.pred[i], b.pred[i]} - {ERROR_LABEL})
        if any(s not in a.probabilities[i] or s not in b.probabilities[i] for s in cands):
            return None
        members.append(np.arange(len(pa), len(pa) + len(cands), dtype=np.intp))
        pa += [a.probabilities[i][s] for s in cands]
        pb += [b.probabilities[i][s] for s in cands]
        y += [s == a.gold[i] for s in cands]
    return CandidateInstances(
        tuple(members), np.array(pa), np.array(pb), np.array(y, dtype=np.bool_)
    )


def candidate_aligned(
    inst: CandidateInstances | None, idx: IntArray | None, n_bins: int, strategy: Binning
) -> View:
    if inst is None:
        return View("CA", False, note="needs per-option probabilities from both runs")
    if idx is None:
        sel = np.arange(len(inst.correct), dtype=np.intp)
    elif len(idx):
        sel = np.concatenate([inst.members[int(i)] for i in idx])
    else:
        sel = np.zeros(0, dtype=np.intp)
    if len(sel) == 0:
        return View("CA", False, note="no candidates")
    z = inst.correct[sel]
    return _view(
        "CA",
        inst.conf_a[sel],
        z,
        inst.conf_b[sel],
        z,
        None,
        None,
        n_bins,
        strategy,
        info={"instances": float(len(sel))},
    )


def all_views(
    a: Scored,
    b: Scored,
    *,
    n_bins: int,
    strategy: Binning,
    idx: IntArray | None = None,
    inst: CandidateInstances | None = None,
) -> dict[str, View]:
    """Raw, DA, IA and CA views, optionally on a (bootstrap) subset ``idx``."""
    sel = np.arange(len(a)) if idx is None else idx
    ca, za, cb, zb = a.confidence[sel], a.correct[sel], b.confidence[sel], b.correct[sel]
    return {
        "raw": raw(ca, za, cb, zb, n_bins, strategy),
        "DA": distribution_aligned(ca, za, cb, zb, n_bins, strategy),
        "IA": instance_aligned(ca, za, cb, zb, n_bins, strategy),
        "CA": candidate_aligned(inst, idx, n_bins, strategy),
    }

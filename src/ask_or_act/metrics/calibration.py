"""Calibration: top-label ECE (equal-width and equal-mass bins), Brier, reliability data.

"Top-label" means we score the decider's confidence in its own chosen answer
(confidence c, correctness z ∈ {0, 1}). This works for every decider, including those
that give no per-option probabilities.

Binning conventions:

* **Equal-width:** B bins over [0, 1]. A value on an edge goes to the lower bin, so
  the bins are [0, 1/B], (1/B, 2/B], … This matches scikit-learn's
  ``calibration_curve``, which the tests compare against.
* **Equal-mass:** each bin holds about N/B examples. Tied confidences always share a
  bin: a cut never separates equal values, so the result doesn't depend on example
  order (the tie pitfall in arXiv 2608.04899). With heavy ties some bins are larger
  and there are fewer than B of them.

Every function takes optional ``weights``, which the accuracy-controlled DA view
(arXiv 2606.30814 §4.2) needs.

ECE is biased upward on finite samples. With ~100 examples per bin a perfectly
calibrated decider still scores ~0.03, so compare deciders on the same sample and bins
and look at the CIs, never at a single ECE value in isolation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ask_or_act.metrics.data import BoolArray, FloatArray, IntArray

type Binning = Literal["width", "mass"]
DEFAULT_BINS = 15


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float
    count: int
    weight: float
    mean_confidence: float
    accuracy: float


def bin_ids(confidence: FloatArray, n_bins: int, strategy: Binning) -> IntArray:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if strategy == "width":
        inner = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        return np.searchsorted(inner, confidence, side="left").astype(np.intp)
    if strategy == "mass":
        n = len(confidence)
        if n == 0:
            return np.zeros(0, dtype=np.intp)
        ordered = np.sort(confidence)
        cuts = {round(k * n / n_bins) for k in range(1, n_bins)}
        edges = np.unique([ordered[c] for c in sorted(cuts) if 0 < c < n])
        # side="right": a value equal to an edge joins the upper bin with all its ties.
        return np.searchsorted(edges, confidence, side="right").astype(np.intp)
    raise ValueError(f"unknown binning {strategy!r}")


def reliability(
    confidence: FloatArray,
    correct: BoolArray,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Binning = "width",
    weights: FloatArray | None = None,
) -> list[Bin]:
    """Per non-empty bin: range, size, mean confidence, accuracy (reliability diagram data)."""
    w = np.ones_like(confidence) if weights is None else weights
    ids = bin_ids(confidence, n_bins, strategy)
    z = correct.astype(np.float64)
    out = []
    for b in np.unique(ids):
        mask = ids == b
        wb = float(w[mask].sum())
        if wb <= 0:
            continue
        if strategy == "width":
            lower, upper = float(b) / n_bins, float(b + 1) / n_bins
        else:
            lower, upper = float(confidence[mask].min()), float(confidence[mask].max())
        out.append(
            Bin(
                lower=lower,
                upper=upper,
                count=int(mask.sum()),
                weight=wb,
                mean_confidence=float((w[mask] * confidence[mask]).sum() / wb),
                accuracy=float((w[mask] * z[mask]).sum() / wb),
            )
        )
    return out


def ece(
    confidence: FloatArray,
    correct: BoolArray,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Binning = "width",
    weights: FloatArray | None = None,
) -> float:
    """Σ_b (W_b / W) · |accuracy_b − confidence_b|. With weights it is DA-ECE."""
    bins = reliability(confidence, correct, n_bins=n_bins, strategy=strategy, weights=weights)
    total = sum(b.weight for b in bins)
    if total == 0:
        return float("nan")
    return float(sum(b.weight / total * abs(b.accuracy - b.mean_confidence) for b in bins))


@dataclass(frozen=True)
class NoiseFloor:
    """ECE that a *perfectly calibrated* decider would get by chance alone."""

    mean: float
    p95: float
    n_sim: int


def ece_noise_floor(
    confidence: FloatArray,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Binning = "width",
    n_sim: int = 500,
    seed: int = 0,
) -> NoiseFloor:
    """Simulate perfectly calibrated outcomes for these exact confidences.

    Each example is right with probability equal to its confidence (z ~ Bernoulli(c)),
    using the same sample size, confidence distribution and bins. Only small-sample
    noise remains, so:

    * an observed ECE inside this range is consistent with perfect calibration;
    * an ECE above ``p95`` is miscalibration that noise doesn't explain (one-sided, 5%).

    The bins depend only on the confidences, so they are fixed across simulations. Each
    simulated ECE is Σ_b |Σ z − Σ c| / N over bins, computed for all simulations at once.
    """
    n = len(confidence)
    if n == 0:
        return NoiseFloor(float("nan"), float("nan"), 0)
    ids = bin_ids(confidence, n_bins, strategy)
    onehot = np.zeros((n, int(ids.max()) + 1))
    onehot[np.arange(n), ids] = 1.0
    conf_per_bin = confidence @ onehot
    z = np.random.default_rng(seed).random((n_sim, n)) < confidence
    sims = np.abs(z.astype(np.float64) @ onehot - conf_per_bin).sum(axis=1) / n
    return NoiseFloor(float(sims.mean()), float(np.quantile(sims, 0.95)), n_sim)


def brier_top_label(
    confidence: FloatArray, correct: BoolArray, *, weights: FloatArray | None = None
) -> float:
    """Weighted mean of (c − z)². With weights it is DA-BS (arXiv 2606.30814 §4.2)."""
    w = np.ones_like(confidence) if weights is None else weights
    if w.sum() == 0:
        return float("nan")
    return float((w * (confidence - correct.astype(np.float64)) ** 2).sum() / w.sum())


def brier_multiclass(
    probabilities: Sequence[Mapping[str, float]], gold: Sequence[str], options: Sequence[str]
) -> float | None:
    """Mean over examples of Σ_k (p_k − 1[k = gold])².

    Needs a probability vector for every example. Returns None otherwise, e.g. when a
    decider only reports a top-label confidence or a call failed.
    """
    if not probabilities:
        return None
    total = 0.0
    for probs, g in zip(probabilities, gold, strict=True):
        if not probs or set(probs) - set(options):
            return None
        total += sum((probs.get(o, 0.0) - (1.0 if o == g else 0.0)) ** 2 for o in options)
    return total / len(gold)

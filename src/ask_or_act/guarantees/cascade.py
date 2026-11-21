"""Cascaded selective decisions: cheap model → strong model → human (act / ask / escalate).

Source: Jung, Brahman, Choi, *Trust or Escalate*, arXiv:2407.18370 **v1**, §2.3,
**Algorithm 1** (routing: the first stage whose confidence clears its threshold
answers, otherwise escalate to a human), **Algorithm 2** and **Appendix A.2**
(calibration).

* Stage i is calibrated with the single-model fixed-sequence test (fixed_sequence.py)
  at level **δ/|M|**, on the calibration examples that *all earlier stages abstained
  on*. With S pre-registered scan starts per stage, each scan runs at δ/(|M|·S).
* A.2's argument: each stage's conditional risk
  Rᵢ = P(stage i wrong | stage i acts, stages < i abstained) is ≤ α with probability
  ≥ 1 − δ/|M|. The cascade's overall risk is a weighted average of the Rᵢ, so it is at
  most maxᵢ Rᵢ. A union bound over the |M| stages gives P(R_cascade > α) ≤ δ.

**Caveat we flag (not in the paper).** A.2 treats each stage's calibration data as an
i.i.d. sample from "abstained by earlier stages". But λ₁ was chosen by a scan whose
stopping test looked at labels of some examples just below λ₁, and those examples
are exactly the ones stage 2 then calibrates on. ``disjoint=True`` removes that
dependence: each stage calibrates on its own seeded share of the calibration set (the
examples assigned to stage i, restricted to those earlier stages abstain on). That is
fully rigorous, at the cost of fewer examples per stage. The default follows the
paper.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ask_or_act.guarantees.fixed_sequence import MultiStartResult, multi_start
from ask_or_act.metrics.data import BoolArray, FloatArray

_EPS = 1e-9


@dataclass(frozen=True)
class Stage:
    name: str
    confidence: FloatArray  # aligned on the same examples for every stage
    correct: BoolArray
    grid: tuple[float, ...]
    starts: tuple[float, ...]  # pre-registered scan starts (fixed_sequence.DEFAULT_STARTS)


@dataclass(frozen=True)
class CascadeResult:
    alpha: float
    delta: float
    thresholds: tuple[float | None, ...]  # per stage; None: the stage never acts
    fits: tuple[MultiStartResult, ...]
    disjoint: bool


def _share(example_id: str, k: int, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest(), 16) % k


def calibrate(
    stages: Sequence[Stage],
    *,
    alpha: float,
    delta: float,
    ids: Sequence[str] | None = None,
    disjoint: bool = False,
    seed: int = 0,
) -> CascadeResult:
    """ToE Algorithm 2. ``ids`` are needed only for ``disjoint`` (the seeded split)."""
    if not stages:
        raise ValueError("need at least one stage")
    n = len(stages[0].confidence)
    if any(len(s.confidence) != n or len(s.correct) != n for s in stages):
        raise ValueError("every stage must be aligned on the same calibration examples")
    if disjoint and (ids is None or len(ids) != n):
        raise ValueError("disjoint=True needs the example ids")
    remaining = np.ones(n, dtype=np.bool_)
    level = delta / len(stages)
    thresholds: list[float | None] = []
    fits = []
    for i, st in enumerate(stages):
        use = remaining.copy()
        if disjoint:
            assert ids is not None
            use &= np.array([_share(x, len(stages), seed) == i for x in ids], dtype=np.bool_)
        fit = multi_start(
            st.confidence[use],
            st.correct[use],
            alpha=alpha,
            delta=level,
            grid=st.grid,
            starts=st.starts,
        )
        fits.append(fit)
        thresholds.append(fit.threshold)
        if fit.threshold is not None:
            remaining &= st.confidence < fit.threshold - _EPS
    return CascadeResult(alpha, delta, tuple(thresholds), tuple(fits), disjoint)


@dataclass(frozen=True)
class Routed:
    """Where each example goes: index of the acting stage, or -1 for the human."""

    stage: np.ndarray
    correct: BoolArray  # correctness of the acting stage (False for escalated)

    def coverage(self) -> float:
        return float((self.stage >= 0).mean()) if len(self.stage) else 0.0

    def risk(self) -> float | None:
        acted = self.stage >= 0
        return float((~self.correct[acted]).mean()) if acted.any() else None

    def share(self, i: int) -> float:
        return float((self.stage == i).mean()) if len(self.stage) else 0.0

    def stage_risk(self, i: int) -> float | None:
        mine = self.stage == i
        return float((~self.correct[mine]).mean()) if mine.any() else None


def route(stages: Sequence[Stage], thresholds: Sequence[float | None]) -> Routed:
    """ToE Algorithm 1 on new examples (e.g. the test half)."""
    n = len(stages[0].confidence)
    stage = np.full(n, -1, dtype=np.intp)
    correct = np.zeros(n, dtype=np.bool_)
    for i, (st, lam) in enumerate(zip(stages, thresholds, strict=True)):
        if lam is None:
            continue
        acts = (stage < 0) & (st.confidence >= lam - _EPS)
        stage[acts] = i
        correct[acts] = st.correct[acts]
    return Routed(stage, correct)

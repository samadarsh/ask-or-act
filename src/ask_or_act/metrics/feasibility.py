"""Minimum-abstention feasibility: can a risk target be met at all, and at what cost?

Source: Kotte, *When Can Conformal Risk Control Certify LLM Outputs?*, arXiv:2606.29054
**v2** (revised 2026-09-07), §2.2, **Proposition 3** ("Sharpened two-sided abstention
floor"). Let μ = E[R] be the base risk and M = ess sup R ≤ 1, with α < M. Any selective
predictor whose emitted risk satisfies E[R | emit] ≤ α must abstain with probability

    P(abstain) ≥ (μ − α) / (M − α) ≥ (μ − α) / (1 − α).

The population statement is exact and holds for *any* selective predictor and any
confidence score. No threshold or method can do better.

Finite samples: v2 (§2.2) gives the plug-in correction ε / (M − α), with
ε = √(log(2/δ) / (2n)), for μ̂ estimated from n held-out points at level δ/2. We
subtract it, so the conservative floor holds with high probability.

TODO(M3b): BLOCKER before milestone 3b uses ``min_abstention_conservative``. Re-check
the finite-sample correction against the full v2 PDF (Appendix E.4, the proof). We only
saw the proof in summary form. The direction (subtract ε from μ̂, i.e. a lower
confidence bound on μ) and the δ/2 split are our reading of the §2.2 remark. The
population floor ``min_abstention`` is verified against the paper's worked numbers; the
conservative one is not. Reports mark it with "*".

Version note: v1 (2026-06-27) states its Proposition 3 as (μ − α)/(1 − α) − O(√(log(1/δ)/n))
for losses in [0, 1]. That is the M = 1 case of v2. We use 0/1 loss (R = 1 if wrong), so
M = 1 whenever any error occurs, and the two versions agree. We cite v2 because it states
the constant explicitly. The proof (v2 Appendix E.4) was checked from a summary only, not
verbatim; see docs/research.md.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Feasibility:
    alpha: float  # target risk on answered (non-abstained) examples
    base_risk: float  # μ̂: error rate when answering everything
    n: int
    delta: float
    min_abstention: float  # plug-in floor max(0, (μ̂ − α)/(M − α))
    min_abstention_conservative: float  # with the ε/(M − α) finite-sample correction
    feasible_without_abstention: bool  # μ̂ ≤ α

    @property
    def max_coverage(self) -> float:
        """At most this fraction of traffic can be answered while meeting α."""
        return 1.0 - self.min_abstention


def min_abstention(
    base_risk: float, alpha: float, *, n: int, delta: float = 0.05, max_loss: float = 1.0
) -> Feasibility:
    if not 0.0 <= base_risk <= max_loss:
        raise ValueError("base_risk must be in [0, max_loss]")
    if not 0.0 <= alpha < max_loss:
        raise ValueError("Proposition 3 requires 0 <= alpha < M (max_loss)")
    if n < 1 or not 0.0 < delta < 1.0:
        raise ValueError("need n >= 1 and 0 < delta < 1")
    eps = math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    floor = max(0.0, (base_risk - alpha) / (max_loss - alpha))
    conservative = max(0.0, (base_risk - eps - alpha) / (max_loss - alpha))
    return Feasibility(alpha, base_risk, n, delta, floor, conservative, base_risk <= alpha)

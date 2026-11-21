"""Minimum-abstention feasibility: can a risk target be met at all, and at what cost?

Source: Kotte, *When Can Conformal Risk Control Certify LLM Outputs?*, arXiv:2606.29054
**v2** (revised 2026-09-07), §2.2, **Proposition 3** ("Sharpened two-sided abstention
floor"). Let μ = E[R] be the base risk and M = ess sup R ≤ 1, with α < M. Any selective
predictor whose emitted risk satisfies E[R | emit] ≤ α must abstain with probability

    P(abstain) ≥ (μ − α) / (M − α) ≥ (μ − α) / (1 − α).

The population statement is exact and holds for *any* selective predictor and any
confidence score. No threshold or method can do better.

Finite samples: v2 **Appendix E.4** ("Plug-in version (δ/2 + δ/2 split)"), read in the
full v2 PDF on 2026-11-20. With μ̂ from n held-out points, a one-sided Hoeffding bound
at level δ/2 gives P(μ < μ̂ − ε) ≤ δ/2 with ε = √(log(2/δ) / (2n)). Substituting
μ ≥ μ̂ − ε into the floor gives

    abstention ≥ (μ̂ − α)/(M − α) − ε/(M − α),

which is ``min_abstention_conservative`` (clipped at 0). The correction is ε/(M − α), not
ε. E.4's worked numbers (n = 1000, δ = 0.1, M = 1: ε ≈ 0.0387, correction ≈ 0.043 at
α = 0.1) are a regression test (tests/test_guarantees.py).

**Joint statement.** E.4 states the floor on the event where *both* the Hoeffding bound
(δ/2) and the risk guarantee of the method it accompanies (δ/2) hold. So when a report
pairs this floor with a guaranteed threshold, the joint probability is 1 − δ only if
the threshold was certified at δ/2. On its own, the floor holds with probability
≥ 1 − δ/2.

Version note: v1 (2026-06-27) states its Proposition 3 as (μ − α)/(1 − α) − O(√(log(1/δ)/n))
for losses in [0, 1]. That is the M = 1 case of v2. We use 0/1 loss (R = 1 if wrong), so
M = 1 whenever any error occurs, and the two versions agree. We cite v2 because it states
the constant explicitly.
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

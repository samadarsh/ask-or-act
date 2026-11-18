"""Minimum abstention (Kotte, arXiv:2606.29054 v2, Proposition 3), against the paper's
own worked numbers (v2 Remark 5, Table 1, §3.3; all with M = 1, α = 0.10)."""

import math

import pytest

from ask_or_act.metrics.feasibility import min_abstention


@pytest.mark.parametrize(
    ("mu", "expected"),
    [
        (0.24, 0.156),  # CoNLL 72B: "≥15.6%" (Remark 5)
        (0.42, 0.356),  # CoNLL 3B: "≥35.6%" (Remark 5)
        (0.356, 0.284),  # CoNLL 5-model average: 28.4% (Table 1)
        (0.46, 0.400),  # NER (§3.3)
        (0.62, 0.578),  # QA (§3.3)
    ],
)
def test_matches_papers_worked_examples(mu: float, expected: float) -> None:
    f = min_abstention(mu, 0.10, n=10_000)
    assert f.min_abstention == pytest.approx(expected, abs=5e-4)
    assert f.max_coverage == pytest.approx(1 - f.min_abstention)


def test_feasible_without_abstention_when_base_risk_is_low() -> None:
    f = min_abstention(0.05, 0.10, n=500)
    assert f.feasible_without_abstention
    assert f.min_abstention == 0.0


def test_conservative_floor_subtracts_the_plug_in_correction() -> None:
    f = min_abstention(0.24, 0.10, n=770, delta=0.05)
    eps = math.sqrt(math.log(2 / 0.05) / (2 * 770))
    assert f.min_abstention_conservative == pytest.approx((0.24 - eps - 0.10) / 0.90)
    assert f.min_abstention_conservative < f.min_abstention


def test_smaller_max_loss_gives_a_stronger_floor() -> None:
    # v2: (μ−α)/(M−α) ≥ (μ−α)/(1−α) when M < 1.
    assert min_abstention(0.3, 0.1, n=100, max_loss=0.5).min_abstention > (
        min_abstention(0.3, 0.1, n=100).min_abstention
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_risk": 0.3, "alpha": 1.0, "n": 10},  # needs α < M
        {"base_risk": 1.5, "alpha": 0.1, "n": 10},
        {"base_risk": 0.3, "alpha": 0.1, "n": 0},
        {"base_risk": 0.3, "alpha": 0.1, "n": 10, "delta": 0.0},
    ],
)
def test_invalid_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - the message differs per case
        min_abstention(**kwargs)  # type: ignore[arg-type]

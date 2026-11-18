"""Accuracy-controlled calibration (arXiv:2606.30814 v2, §4), against hand values."""

import numpy as np
import pytest

from ask_or_act.core.types import Decision, Example
from ask_or_act.metrics import ace
from ask_or_act.metrics.report import Settings, ace_section
from tests.metrics_helpers import scored


def constant_model(n: int, n_correct: int, conf: float) -> tuple[np.ndarray, np.ndarray]:
    return np.full(n, conf), np.array([True] * n_correct + [False] * (n - n_correct))


def test_da_weights_move_accuracy_to_target() -> None:
    z = np.array([True] * 8 + [False] * 2)  # acc 0.8
    w = ace.da_weights(z, 0.5)
    assert w is not None
    assert float((w * z).sum() / w.sum()) == pytest.approx(0.5)
    assert w[0] == pytest.approx(0.5 / 0.8)
    assert w[-1] == pytest.approx(0.5 / 0.2)
    assert ace.da_weights(np.ones(5, dtype=bool), 0.5) is None  # positivity fails


def test_kish_ess() -> None:
    assert ace.kish_ess(np.ones(10)) == pytest.approx(10)
    assert ace.kish_ess(np.array([1.0, 0.0, 0.0])) == pytest.approx(1)


def test_constructed_reversal() -> None:
    """A: 80% right, always says 0.95 (raw ECE 0.15). B: 50% right, always 0.7
    (raw ECE 0.2). Raw says A is better calibrated. At B's accuracy, A's weighted
    accuracy is 0.5 against confidence 0.95, so DA-ECE(A) = 0.45: the ranking reverses."""
    ca, za = constant_model(100, 80, 0.95)
    cb, zb = constant_model(100, 50, 0.70)
    raw = ace.raw(ca, za, cb, zb, 10, "width")
    da = ace.distribution_aligned(ca, za, cb, zb, 10, "width")
    assert raw.ece_gap == pytest.approx(0.15 - 0.20)
    assert da.ece_a == pytest.approx(0.45)
    assert da.ece_gap == pytest.approx(0.45 - 0.20)
    assert raw.ece_gap is not None
    assert da.ece_gap is not None
    assert raw.ece_gap * da.ece_gap < 0
    # DA-BS (§4.2): weighted mean of (c − z)² = 0.5·0.05² + 0.5·0.95²
    assert da.brier_a == pytest.approx(0.5 * 0.05**2 + 0.5 * 0.95**2)


def test_instance_aligned_keeps_shared_outcomes() -> None:
    ca, za = constant_model(10, 8, 0.9)
    cb, zb = constant_model(10, 6, 0.6)
    ia = ace.instance_aligned(ca, za, cb, zb, 10, "width")
    assert ia.info["retention"] == pytest.approx(0.8)  # 6 both right + 2 both wrong
    assert ia.ece_a == pytest.approx(abs(6 / 8 - 0.9))


def test_equal_accuracy_means_no_reweighting() -> None:
    ca, za = constant_model(10, 5, 0.9)
    cb, zb = constant_model(10, 5, 0.4)
    da = ace.distribution_aligned(ca, za, cb, zb, 10, "width")
    assert da.ece_gap == pytest.approx(ace.raw(ca, za, cb, zb, 10, "width").ece_gap)


def _ex(i: int, gold: str) -> Example:
    return Example(f"e{i}", "x", ("p", "q", "r"), gold)


def test_candidate_aligned_pools_both_answers_once() -> None:
    examples = [_ex(0, "p"), _ex(1, "q")]
    a = scored(
        [
            Decision("e0", "p", 0.6, {"p": 0.6, "q": 0.3, "r": 0.1}),
            Decision("e1", "p", 0.5, {"p": 0.5, "q": 0.4, "r": 0.1}),
        ],
        examples,
    )
    b = scored(
        [
            Decision("e0", "p", 0.9, {"p": 0.9, "q": 0.05, "r": 0.05}),
            Decision("e1", "q", 0.8, {"p": 0.1, "q": 0.8, "r": 0.1}),
        ],
        examples,
    )
    inst = ace.candidate_instances(a, b)
    assert inst is not None
    # e0: both chose p → one candidate; e1: {p, q} → two candidates
    assert [len(m) for m in inst.members] == [1, 2]
    view = ace.candidate_aligned(inst, None, 10, "width")
    # CA-BS for A: ((0.6−1)² + (0.5−0)² + (0.4−1)²) / 3
    assert view.brier_a == pytest.approx((0.4**2 + 0.5**2 + 0.6**2) / 3)


def test_candidate_aligned_needs_probabilities() -> None:
    examples = [_ex(0, "p")]
    a = scored([Decision("e0", "p", 0.6)], examples)
    assert ace.candidate_instances(a, a) is None
    assert not ace.candidate_aligned(None, None, 10, "width").available


def test_report_flags_the_reversal() -> None:
    examples = [Example(f"e{i}", "x", ("y", "n"), "y") for i in range(100)]
    a = scored(
        [Decision(e.id, "y" if i < 80 else "n", 0.95) for i, e in enumerate(examples)], examples
    )
    b = scored(
        [Decision(e.id, "y" if i < 50 else "n", 0.70) for i, e in enumerate(examples)], examples
    )
    section = ace_section(a, b, Settings(n_boot=200))
    assert section["recommended"] == "DA"
    da = section["views"]["DA"]["ece_gap"]
    assert da["reversal"] is True
    assert da["reversal_stability"] > 0.95
    assert section["views"]["raw"]["ece_gap"]["reversal"] is False

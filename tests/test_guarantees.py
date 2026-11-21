"""Milestone 3b: exact binomial bound, fixed-sequence thresholds, cascades.

The Monte Carlo tests check the guarantee itself: over many calibration samples from
a known distribution, the *true* risk at the chosen threshold exceeds α in at most a
δ share of them (plus Monte Carlo slack), including when risk is not monotone.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from ask_or_act.cli import main
from ask_or_act.core.types import JSON
from ask_or_act.guarantees.binomial import binom_cdf, p_value, two_sided_interval, upper_bound
from ask_or_act.guarantees.cascade import Stage, calibrate, route
from ask_or_act.guarantees.fixed_sequence import (
    DEFAULT_STARTS,
    fixed_sequence,
    grid_for,
    multi_start,
)
from ask_or_act.guarantees.mt_bench import by_comparison, human_agreement
from ask_or_act.metrics.data import Scored
from ask_or_act.metrics.feasibility import min_abstention
from ask_or_act.run import run_from_config
from tests.test_run_end_to_end import write_config, write_world

PCT = grid_for("verbalized")

# ----------------------------------------------------------------- binomial


def test_binom_cdf_matches_exact_sum() -> None:
    for n, p in [(10, 0.3), (25, 0.07), (1, 0.5)]:
        for k in range(n + 1):
            exact = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
            assert binom_cdf(k, n, p) == pytest.approx(exact, rel=1e-10, abs=1e-15)
    assert binom_cdf(-1, 5, 0.2) == 0.0
    assert binom_cdf(3, 1000, 0.0) == 1.0


def test_upper_bound_is_clopper_pearson() -> None:
    # Zero errors in n: sup{R : (1 - R)^n >= δ} = 1 - δ^(1/n).
    assert upper_bound(0, 10, 0.05) == pytest.approx(1 - 0.05 ** (1 / 10), abs=1e-8)
    assert upper_bound(0, 0, 0.1) == 1.0
    assert upper_bound(7, 7, 0.1) == 1.0
    lo, hi = two_sided_interval(5, 50)
    assert lo < 0.1 < hi


@pytest.mark.parametrize("n", [5, 20, 80])
def test_p_value_form_equals_toe_bound_form(n: int) -> None:
    """ToE Eq. (4) bound <= α  ⇔  P(Bin(n, α) <= k) <= δ."""
    for k in range(n + 1):
        for alpha in (0.05, 0.1, 0.2):
            assert (upper_bound(k, n, 0.1) <= alpha + 1e-9) == (p_value(k, n, alpha) <= 0.1)


# ----------------------------------------------------------- fixed sequence


def test_scan_stops_at_the_first_failure_and_takes_the_last_certified() -> None:
    # 40 answers at 0.99 (0 wrong), 40 at 0.9 (0 wrong), 40 at 0.8 (20 wrong), 40 at 0.7.
    conf = np.repeat([0.99, 0.9, 0.8, 0.7], 40)
    correct = np.concatenate([np.ones(80, bool), np.arange(40) % 2 == 0, np.ones(40, bool)])
    r = fixed_sequence(conf, correct, alpha=0.1, delta=0.1, grid=PCT, start=0.99)
    # 0.81..0.90 share 0.9's emit set (no calibration answers in between), so each is
    # a certified hypothesis with the same test; ToE Eq. (5) takes the infimum.
    assert r.threshold == 0.81
    assert [s.threshold for s in r.steps] == [0.99, 0.9, 0.8]
    assert [s.certified for s in r.steps] == [True, True, False]
    assert r.coverage == 0.5
    assert r.risk == 0.0
    # 0.7 would pass again (risk 20/160), but the scan stopped: fixed-sequence rule.


def test_tied_grid_points_share_one_test_and_report_the_lowest() -> None:
    conf = np.full(50, 0.95)
    r = fixed_sequence(conf, np.ones(50, bool), alpha=0.1, delta=0.1, grid=PCT, start=0.95)
    assert r.threshold == 0.0  # every grid point below 0.95 has the same emit set
    assert len(r.steps) == 1


def test_an_empty_top_stops_a_single_scan() -> None:
    conf = np.full(100, 0.9)
    r = fixed_sequence(conf, np.ones(100, bool), alpha=0.1, delta=0.1, grid=PCT)
    assert r.threshold is None
    assert (r.steps[0].threshold, r.steps[0].n) == (1.0, 0)


def test_multi_start_unions_scans_at_split_level() -> None:
    conf = np.full(100, 0.9)
    r = multi_start(
        conf, np.ones(100, bool), alpha=0.1, delta=0.1, grid=PCT, starts=(1.0, 0.95, 0.9)
    )
    assert [s.delta for s in r.scans] == pytest.approx([0.1 / 3] * 3)
    assert r.threshold == 0.0
    assert r.coverage_and_risk(conf, np.ones(100, bool)) == (1.0, 0.0)


def test_grid_validation() -> None:
    with pytest.raises(ValueError, match="strictly decreasing"):
        fixed_sequence(np.zeros(1), np.ones(1, bool), alpha=0.1, delta=0.1, grid=[0.1, 0.2])
    with pytest.raises(ValueError, match="k"):
        grid_for("self_consistency")
    assert grid_for("self_consistency", samples=3) == (1.0, 2 / 3, 1 / 3, 0.0)


# ------------------------------------------------------------ Monte Carlo


def _true_risk(values: np.ndarray, weights: np.ndarray, risks: np.ndarray, lam: float) -> float:
    emit = values >= lam - 1e-9
    return float((weights[emit] * risks[emit]).sum() / weights[emit].sum())


def _failure_rate(
    values: np.ndarray, weights: np.ndarray, risks: np.ndarray, *, n: int, reps: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    alpha, delta = 0.1, 0.1
    failures = certified = 0
    for _ in range(reps):
        idx = rng.choice(len(values), size=n, p=weights)
        conf = values[idx]
        correct = rng.random(n) >= risks[idx]
        r = multi_start(
            conf, correct, alpha=alpha, delta=delta, grid=PCT, starts=DEFAULT_STARTS["verbalized"]
        )
        if r.threshold is not None:
            certified += 1
            failures += _true_risk(values, weights, risks, r.threshold) > alpha
    return failures / reps, certified / reps


def test_guarantee_holds_for_a_calibrated_decider() -> None:
    values = np.array([1.0, 0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5])
    weights = np.array([0.15, 0.2, 0.2, 0.15, 0.1, 0.08, 0.05, 0.04, 0.03])
    risks = 1.0 - values  # perfectly calibrated
    fail, cert = _failure_rate(values, weights, risks, n=300, reps=400, seed=1)
    assert cert > 0.9  # the procedure is not vacuous here
    assert fail <= 0.1 + 3 * math.sqrt(0.1 * 0.9 / 400)


def test_guarantee_holds_when_risk_is_not_monotone() -> None:
    """Overconfident top: answers at "100" are wrong 30% of the time, 0.95–0.99 are
    nearly always right. No monotonicity is assumed (Kotte v2 Lemma E.1)."""
    values = np.array([1.0, 0.99, 0.95, 0.9, 0.8])
    weights = np.array([0.2, 0.3, 0.3, 0.1, 0.1])
    risks = np.array([0.3, 0.01, 0.02, 0.15, 0.4])
    fail, _ = _failure_rate(values, weights, risks, n=300, reps=400, seed=2)
    assert fail <= 0.1 + 3 * math.sqrt(0.1 * 0.9 / 400)


def test_guarantee_holds_near_the_boundary() -> None:
    """True risk just above α at every threshold below the top: the hardest case."""
    values = np.array([1.0, 0.95, 0.9])
    weights = np.array([0.4, 0.3, 0.3])
    risks = np.array([0.08, 0.13, 0.13])  # R(1.0) = 0.08 < α; R(0.95) ≈ 0.101 > α
    fail, _ = _failure_rate(values, weights, risks, n=400, reps=400, seed=3)
    assert fail <= 0.1 + 3 * math.sqrt(0.1 * 0.9 / 400)


# --------------------------------------------------------------- cascade


def _stage(name: str, conf: np.ndarray, correct: np.ndarray) -> Stage:
    return Stage(name, conf, correct, PCT, DEFAULT_STARTS["verbalized"])


def test_cascade_routes_cheap_then_strong_then_human() -> None:
    n = 1000  # at 400 the strong stage's test at δ/(2·4) is underpowered
    rng = np.random.default_rng(0)
    cheap_conf = rng.choice([1.0, 0.7], size=n, p=[0.5, 0.5])
    cheap_ok = np.where(cheap_conf == 1.0, rng.random(n) > 0.02, rng.random(n) > 0.4)
    strong_conf = rng.choice([0.99, 0.6], size=n, p=[0.6, 0.4])
    strong_ok = np.where(strong_conf == 0.99, rng.random(n) > 0.03, rng.random(n) > 0.5)
    stages = [_stage("cheap", cheap_conf, cheap_ok), _stage("strong", strong_conf, strong_ok)]
    fit = calibrate(stages, alpha=0.1, delta=0.1)
    assert fit.thresholds[0] == pytest.approx(0.71)  # cheap acts only on its "100"s
    assert fit.thresholds[1] == pytest.approx(0.61)  # strong acts on its "99"s
    assert [f.delta for f in fit.fits] == [0.05, 0.05]  # δ/|M| (ToE Algorithm 2)
    routed = route(stages, fit.thresholds)
    assert routed.share(0) == pytest.approx((cheap_conf == 1.0).mean())
    assert routed.share(1) > 0.2
    assert 0 < 1 - routed.coverage() < 0.3  # the rest goes to a human
    risk = routed.risk()
    assert risk is not None
    assert risk < 0.1


def test_disjoint_cascade_uses_each_stage_share_only() -> None:
    conf = np.full(200, 1.0)
    ids = [f"x{i}" for i in range(200)]
    stages = [_stage("a", conf, np.ones(200, bool)), _stage("b", conf, np.ones(200, bool))]
    fit = calibrate(stages, alpha=0.1, delta=0.1, ids=ids, disjoint=True)
    n_a = fit.fits[0].scans[0].steps[0].n
    assert 60 < n_a < 140  # about half the calibration set
    with pytest.raises(ValueError, match="ids"):
        calibrate(stages, alpha=0.1, delta=0.1, disjoint=True)


# ------------------------------------------------- Kotte finite-sample floor


def test_kotte_v2_e4_worked_numbers() -> None:
    """Kotte v2 Appendix E.4: n = 1000, δ = 0.1, M = 1 gives ε ≈ 0.0387, and at
    α = 0.1 the correction ε/(1 − α) ≈ 0.043."""
    f = min_abstention(0.3, 0.1, n=1000, delta=0.1)
    eps = math.sqrt(math.log(2 / 0.1) / (2 * 1000))
    assert eps == pytest.approx(0.0387, abs=5e-5)
    assert f.min_abstention - f.min_abstention_conservative == pytest.approx(0.043, abs=5e-4)


# --------------------------------------------------------------------- CLI


def test_thresholds_cli_fits_on_calibration_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = write_world(tmp_path, monkeypatch)
    runs = {}
    for split in ("calibration", "test"):
        cfg = write_config(tmp_path, manifest_path)
        cfg.write_text(cfg.read_text().replace('split = "test"', f'split = "{split}"'))
        result = run_from_config(cfg, raw_dir=tmp_path, out=lambda _: None)
        assert result.run_dir is not None
        runs[split] = str(result.run_dir)
    assert main(["thresholds", "--cal", runs["test"]]) == 2  # never fit on the test half
    assert "test-half run" in capsys.readouterr().err
    assert main(["thresholds", "--cal", runs["calibration"], "--test", runs["test"]]) == 0
    out = capsys.readouterr().out
    assert "feasibility (Kotte v2 Prop. 3)" in out
    assert "escalate: everything else" in out
    assert "test (held out, n = 6)" in out


# ------------------------------------------------------ pre-registration (frozen)


def test_preregistered_starts_and_delta_split_are_frozen() -> None:
    """Frozen 2026-11-21 before any LLM data (docs/guarantees.md). If this fails, the
    pre-registration was changed: don't, unless starting a new, re-registered study."""
    assert DEFAULT_STARTS == {
        "verbalized": (1.0, 0.99, 0.95, 0.9),
        "self_consistency": (1.0,),
        "continuous": (0.999, 0.99, 0.95, 0.9),
    }
    conf = np.full(10, 1.0)
    r = multi_start(conf, np.ones(10, bool), alpha=0.1, delta=0.1, grid=PCT, starts=(1.0, 0.9))
    assert [s.delta for s in r.scans] == [0.05, 0.05]  # equal split over starts
    stages = [_stage("a", conf, np.ones(10, bool)), _stage("b", conf, np.ones(10, bool))]
    fit = calibrate(stages, alpha=0.1, delta=0.1)
    assert [f.delta for f in fit.fits] == [0.05, 0.05]  # then over cascade stages
    assert [s.delta for s in fit.fits[0].scans] == [0.0125] * 4


# ------------------------------------------------------------------ MT-Bench


def _votes(rows: list[tuple[str, str, str, str]]) -> Scored:
    """rows: (comparison, displayed model_a, human label, decider choice)."""
    meta: list[dict[str, JSON]] = []
    for comp, a, _, _ in rows:
        names = comp.split("-vs-")
        b = names[1] if a == names[0] else names[0]
        meta.append({"comparison": comp, "model_a": a, "model_b": b, "question_id": 1})
    pred = tuple(r[3] for r in rows)
    gold = tuple(r[2] for r in rows)
    return Scored(
        ids=tuple(f"human:{i}" for i in range(len(rows))),
        gold=gold,
        pred=pred,
        confidence=np.linspace(0.5, 0.9, len(rows)),
        correct=np.array([p == g for p, g in zip(pred, gold, strict=True)]),
        probabilities=tuple({} for _ in rows),
        options=("model_a", "model_b", "tie"),
        groups=tuple("q1" for _ in rows),
        meta=tuple(meta),
        dataset="mt_bench",
        n_errors=0,
    )


def test_by_comparison_maps_display_order_to_model_names() -> None:
    s = _votes(
        [
            # x-vs-y shown both ways: 2 votes for x as model_a, 1 for x as model_b.
            ("x-vs-y", "x", "model_a", "model_a"),
            ("x-vs-y", "x", "model_a", "model_a"),
            ("x-vs-y", "y", "model_b", "model_a"),  # = x; this order's decision is y
            # p-vs-q: one vote each way, for different models: no majority.
            ("p-vs-q", "p", "model_a", "model_a"),
            ("p-vs-q", "p", "model_b", "model_a"),
            # u-vs-w: a single tie vote.
            ("u-vs-w", "u", "tie", "model_b"),
        ]
    )
    u = by_comparison(s)
    assert (u.n_votes, u.n_comparisons, u.n_no_majority) == (6, 3, 1)
    assert u.scored.ids == ("u-vs-w", "x-vs-y")
    assert u.scored.gold == ("tie", "x")
    assert u.scored.pred == ("w", "x")  # x-vs-y uses the order with more votes (x first)
    assert list(u.scored.correct) == [False, True]


def test_human_ceiling_is_leave_one_out() -> None:
    s = _votes(
        [
            ("x-vs-y", "x", "model_a", "model_a"),
            ("x-vs-y", "x", "model_a", "model_a"),
            ("x-vs-y", "x", "model_b", "model_a"),
            ("solo-vs-z", "solo", "tie", "tie"),  # alone: skipped
        ]
    )
    h = human_agreement(s)
    # Votes 1 and 2: others are {x, y}: no majority, skipped. Vote 3: others {x, x}: x ≠ y.
    assert (h.n_votes_used, h.rate, h.n_votes) == (1, 0.0, 4)
    assert h.alpha == 1.0


# ------------------------------------------- pre-registered target ([guarantee])


def test_guarantee_table_parses_and_validates() -> None:
    from ask_or_act.core.config import ConfigError, parse_config

    base = {
        "name": "x",
        "dataset": {"manifest": "m.json"},
        "decider": {"type": "fake", "model": "m"},
        "budget": {"max_usd": 0.0},
    }
    cfg = parse_config({**base, "guarantee": {"alpha": 0.25, "delta": 0.1, "registered": "d"}})
    assert (cfg.guarantee.alpha, cfg.guarantee.delta, cfg.guarantee.registered) == (
        0.25,
        0.1,
        "d",
    )
    assert parse_config(base).guarantee.alpha is None
    with pytest.raises(ConfigError, match="alpha must be in"):
        parse_config({**base, "guarantee": {"alpha": 1.5}})
    with pytest.raises(ConfigError, match="unknown keys"):
        parse_config({**base, "guarantee": {"target": 0.2}})


def test_thresholds_cli_uses_the_preregistered_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = write_world(tmp_path, monkeypatch)
    cfg = write_config(tmp_path, manifest_path)
    text = cfg.read_text().replace('split = "test"', 'split = "calibration"')
    cfg.write_text(text + '\n[guarantee]\nalpha = 0.2\ndelta = 0.05\nregistered = "2026-11-22"\n')
    result = run_from_config(cfg, raw_dir=tmp_path, out=lambda _: None)
    assert result.run_dir is not None
    assert main(["thresholds", "--cal", str(result.run_dir)]) == 0
    out = capsys.readouterr().out
    assert "target α = 0.2: pre-registered in the run config, registered 2026-11-22" in out
    assert "risk ≤ 0.2 on acted-on examples with probability ≥ 0.950 (δ = 0.05)" in out
    assert main(["thresholds", "--cal", str(result.run_dir), "--alpha", "0.3"]) == 0
    assert "α = 0.3 is NOT the pre-registered target (α = 0.2" in capsys.readouterr().out

"""scripts/analyze_study.py / ask_or_act.study_analysis: the M7 analysis as pre-registered
(docs/analysis-plan.md). Fakes only: no network and no study data."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ask_or_act import study_analysis as sa
from ask_or_act.core.runs import GoldRecord, save_run
from ask_or_act.core.types import Decision
from ask_or_act.metrics.report import Settings
from ask_or_act.study_fakes import make_fake_study

FAST = Settings(n_boot=60)  # the plan's 2,000 in the script; fewer here for speed


@pytest.fixture(scope="module")
def fake_spec(tmp_path_factory: pytest.TempPathFactory) -> sa.Spec:
    root = tmp_path_factory.mktemp("fakes")
    return sa.load_spec(make_fake_study(root))


@pytest.fixture(scope="module")
def result(fake_spec: sa.Spec) -> dict[str, Any]:
    return sa.analyze(fake_spec, settings=FAST)


def test_the_script_uses_the_plans_fixed_settings() -> None:
    assert (sa.SETTINGS.n_boot, sa.SETTINGS.seed, sa.SETTINGS.level) == (2000, 0, 0.95)
    assert sa.FWER == 0.05
    assert sa.DRIFT_SHARE == 0.05
    assert sa.EQUIVALENCE == {"ece_gap": 0.02, "accuracy": 0.03, "coverage": 0.10}


def test_violation_p_value_is_the_upper_binomial_tail() -> None:
    assert sa.p_violation(0, 50, 0.1) == 1.0
    assert sa.p_violation(3, 0, 0.1) == 1.0
    assert sa.p_violation(10, 10, 0.1) == pytest.approx(1e-10)
    # P(Bin(20, 0.1) >= 2) = 1 - 0.9^20 - 20 * 0.1 * 0.9^19
    assert sa.p_violation(2, 20, 0.1) == pytest.approx(1 - 0.9**20 - 2 * 0.9**19)


def test_families_have_the_planned_sizes(result: dict[str, Any]) -> None:
    p = result["primary"]
    # fakes: Banking77 3 LLMs + classic, Aegis 4 + classic, MT-Bench 2 + classic
    assert p["Q1"]["family"]["size"] == 6 + 10 + 3
    assert p["Q1b"]["family"]["size"] == 3 + 4 + 2
    f4 = p["Q4"]["family"]
    assert f4["size"] + len(f4["dropped"]) == 6 + 12 + 2
    assert all("certifies nothing" in d["reason"] for d in p["Q4"]["dropped"])
    f2 = p["Q2"]["family"]
    certified = [r for r in p["Q2"]["configs"] if r["certified"]]
    assert f2["size"] + len(f2["dropped"]) == len(certified)


def test_holm_and_rejections_are_consistent(result: dict[str, Any]) -> None:
    for fam_key, rows, key in (
        ("Q1", result["primary"]["Q1"]["pairs"], "excess"),
        ("Q1b", result["primary"]["Q1b"]["pairs"], "excess"),
        ("Q4", result["primary"]["Q4"]["pairs"], "violation"),
    ):
        tested = [r[key] for r in rows if r[key]["p"] is not None]
        assert all(t["p_holm"] >= t["p"] for t in tested), fam_key
        assert all(t["rejected"] == (t["p_holm"] <= 0.05) for t in tested), fam_key


def test_the_overconfident_fake_is_found_worse_calibrated(result: dict[str, Any]) -> None:
    pair = next(
        p
        for p in result["primary"]["Q1"]["pairs"]
        if p["dataset"] == "banking77"
        and {p["a"], p["b"]} == {"fake-calibrated", "fake-overconfident"}
    )
    assert pair["excess"]["estimate"] < -0.05  # a − b with a = calibrated
    assert pair["excess"]["p"] == pytest.approx(1 / (FAST.n_boot + 1))  # the smallest p possible


def test_q1_tests_the_excess_ece_gap_and_keeps_the_rest_descriptive(
    fake_spec: sa.Spec, result: dict[str, Any]
) -> None:
    """Amendment 2: the tested number is the excess-ECE gap; raw ECE, Brier, DA and IA
    are reported without entering a family."""
    from ask_or_act.metrics.calibration import ece_fast

    pair = result["primary"]["Q1"]["pairs"][0]
    data = sa.load(fake_spec, sa.select(fake_spec))["banking77"]
    a, b = (next(d for d in data if d.name == n) for n in (pair["a"], pair["b"]))
    ctx = sa.Ctx(sa.PRIMARY, {}, FAST)
    sa_, sb = ctx.common([ctx.units(a, "test"), ctx.units(b, "test")])
    raw = ece_fast(sa_.confidence, sa_.correct, n_bins=15) - ece_fast(
        sb.confidence, sb.correct, n_bins=15
    )
    floors = [nf["mean"] for nf in pair["noise_floor"]]
    assert pair["raw_ece_gap"]["estimate"] == pytest.approx(raw)
    assert pair["excess"]["estimate"] == pytest.approx(raw - (floors[0] - floors[1]))
    assert [pair["excess"]["floor_a"], pair["excess"]["floor_b"]] == pytest.approx(floors)
    d = pair["diagnostics"]
    for g in (d["da_ece_gap"], d["ia_ece_gap"], pair["raw_ece_gap"], pair["brier_gap"]):
        assert "p" not in g  # kept only as p_not_tested, outside every family
        assert "p_holm" not in g
    for g in (pair["excess"], pair["raw_ece_gap"], pair["brier_gap"], d["da_ece_gap"]):
        assert g["interval"] == "inverts the centered p"


def _toy(conf: list[float], correct: list[bool]) -> Any:
    from ask_or_act.metrics.data import Scored

    n = len(conf)
    ids = tuple(f"x{i}" for i in range(n))
    return Scored(
        ids,
        ("g",) * n,
        ("g",) * n,
        np.array(conf),
        np.array(correct),
        ({},) * n,
        ("g",),
        ids,
        ({},) * n,
        "toy",
        0,
    )


def test_centered_matches_paired_bootstrap_and_gives_the_basic_interval() -> None:
    from ask_or_act.metrics.stats import paired_bootstrap

    rng = np.random.default_rng(3)
    a = _toy(list(rng.uniform(0.5, 1, 80)), list(rng.random(80) < 0.7))
    b = _toy(list(rng.uniform(0.5, 1, 80)), list(rng.random(80) < 0.8))
    fa, fb = sa._ece_stat(a, FAST), sa._ece_stat(b, FAST)
    ref = paired_bootstrap(fa, fb, 80, n_boot=FAST.n_boot, seed=FAST.seed)
    got = sa.centered(fa, fb, 80, FAST)
    assert got["p"] == pytest.approx(ref.p_value)
    assert got["estimate"] == pytest.approx(ref.diff)
    lo, hi = got["low"], got["high"]
    assert hi - got["estimate"] == pytest.approx(got["estimate"] - lo)  # symmetric
    assert (lo > 0 or hi < 0) == (got["p"] <= 0.05)
    # a fixed shift moves the estimate and the interval, not the resampled spread
    sh = sa.centered(fa, fb, 80, FAST, shift=0.01)
    assert sh["estimate"] == pytest.approx(got["estimate"] - 0.01)
    assert sh["high"] - sh["low"] == pytest.approx(got["high"] - got["low"])


def _gaps(x: Any) -> list[dict[str, Any]]:
    """Every reported gap in a result: dicts with a p-value and an interval."""
    if isinstance(x, dict):
        here = [x] if {"low", "high", "interval"} <= set(x) else []
        return here + [g for v in x.values() for g in _gaps(v)]
    if isinstance(x, list):
        return [g for v in x for g in _gaps(v)]
    return []


def test_every_gap_interval_agrees_with_its_p_value(result: dict[str, Any]) -> None:
    """Amendment 2: each interval inverts its centered p, so it excludes 0 exactly when
    p <= 0.05. Covers Q1, Q1b (tested, raw, diagnostics, per decider), the coverage
    differences and Q4's comparisons with B's own threshold, in every variant."""
    gaps = [g for g in _gaps(result) if g.get("p", g.get("p_not_tested")) is not None]
    assert len(gaps) > 500
    assert {g["interval"] for g in gaps} == {"inverts the centered p"}
    for g in gaps:
        p = g.get("p", g.get("p_not_tested"))
        assert p is not None
        assert (g["low"] > 0 or g["high"] < 0) == (p <= 0.05), g


def test_the_interval_inverts_the_p_value_at_the_edges() -> None:
    boot = [float(x) for x in np.linspace(-1, 1, 2000)]
    for d in (0.0, 0.3, 0.9, 0.95, 0.951, 0.96, 1.2):
        r = sa.invert(d, [b + d for b in boot], 0.95)
        assert (r["low"] > 0 or r["high"] < 0) == (r["p"] <= 0.05), d
    assert sa.invert(None, boot, 0.95)["p"] is None


def test_every_variant_and_section_is_present(result: dict[str, Any]) -> None:
    assert [v["key"] for v in result["sensitivity"]] == ["S1", "S2", "S3", "S4", "S5"]
    s1 = result["sensitivity"][0]
    n_primary = {p["dataset"]: p["n"] for p in result["primary"]["Q1"]["pairs"]}
    n_s1 = {p["dataset"]: p["n"] for p in s1["Q1"]["pairs"]}
    assert all(n_s1[d] < n_primary[d] for d in n_primary)  # the seen ids were dropped
    s2 = result["sensitivity"][1]
    assert {p["dataset"] for p in s2["Q1"]["pairs"]} == {"mt_bench"}
    assert s2["Q1"]["pairs"][0]["n"] == 500  # votes, not comparisons
    assert set(result["exploratory"]) >= {"alpha_grid", "coverage_differences", "judge"}


def test_drift_flags_only_the_planted_new_fingerprint(result: dict[str, Any]) -> None:
    models = result["S6"]["models"]
    changes = {m: list(v["changes"]) for m, v in models.items() if v["changes"]}
    assert changes == {"fake-overconfident": ["mt_bench"]}
    ch = models["fake-overconfident"]["changes"]["mt_bench"]
    assert ch["fingerprints"] == ["fp_fakenew1"]
    assert ch["day"] == "2026-12-02"
    mix = models["fake-overconfident"]["mix"]["mt_bench"]
    assert mix["only_test"] == ["fp_fakenew1"]
    assert mix["tv_distance"] > 0.3
    calm = models["fake-calibrated"]["mix"]["banking77"]
    assert calm["only_test"] == []
    assert calm["tv_distance"] < 0.1
    (split,) = result["S6"]["splits"]
    assert split["model"] == "fake-overconfident"
    assert split["q2_restricted"]  # rerun on calls with fingerprints seen in both halves


def test_the_analysis_is_deterministic(fake_spec: sa.Spec, result: dict[str, Any]) -> None:
    again = sa.analyze(fake_spec, settings=FAST)
    assert json.dumps(again, sort_keys=True) == json.dumps(result, sort_keys=True)


def test_markdown_has_the_plan_order_and_no_text(result: dict[str, Any]) -> None:
    md = sa.to_markdown(result)
    order = [
        "## Primary analysis",
        "### Q1.",
        "### Q1b.",
        "### Q2.",
        "### Q3.",
        "### Q4.",
        "## Sensitivity analyses",
        "### S6",
        "## Exploratory",
    ]
    positions = [md.index(h) for h in order]
    assert positions == sorted(positions)
    assert "Family F1: 19 test(s)" in md
    assert "dc2ea31" in md
    assert "Amendments 1, 2" in md
    assert "Diagnostics, not tests (Amendment 2)" in md
    assert "excess ECE gap" in md
    assert "Fingerprint mix, calibration half vs test half" in md


# ----------------------------------------------------------------- the read guard


def _config(path: Path, name: str, runs: Path) -> Path:
    samples = Path(__file__).parents[1] / "data" / "samples" / "banking77.json"
    path.write_text(
        f'name = "{name}"\n[dataset]\nmanifest = "{samples}"\nsplit = "test"\n'
        '[decider]\ntype = "fake"\nmodel = "m"\n[budget]\nmax_usd = 0.0\n'
        f'[cache]\nenabled = false\n[output]\nruns_dir = "{runs}"\n'
    )
    return path


def _save(cfg_path: Path, runs: Path, status: str) -> None:
    from ask_or_act.core.config import load_config

    cfg = load_config(cfg_path)
    d = Decision("x1", "a", 0.9)
    meta: dict[str, Any] = {
        "name": cfg.name,
        "status": status,
        "config": {"path": str(cfg_path), "sha256": cfg.sha256, "contents": cfg.raw},
        "dataset": {"name": "banking77", "split": "test"},
    }
    save_run(
        runs,
        f"20261201T000000Z-{cfg.name}-{status[:6]}",
        meta,
        [d],
        [GoldRecord("x1", "a", ("a", "b"))],
    )


def _spec(tmp: Path, test_cfg: Path, schedule: Path | None) -> Path:
    lines = [] if schedule is None else [f'schedule = "{schedule}"']
    lines += ["[[decider]]", 'name = "m"', 'dataset = "banking77"', f'test = "{test_cfg}"']
    p = tmp / "spec.toml"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_study_runs_are_not_read_until_the_whole_schedule_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    done = _config(tmp_path / "a.toml", "study-a", runs)
    pending = _config(tmp_path / "b.toml", "study-b", runs)
    _save(done, runs, "completed")
    _save(pending, runs, "quota_stopped")  # a partial run never counts
    schedule = tmp_path / "schedule.toml"
    schedule.write_text(f'configs = ["{done}", "{pending}"]\n')

    def boom(path: Path) -> object:
        raise AssertionError(f"read {path} before the study was complete")

    monkeypatch.setattr(sa, "load_run", boom)
    spec = sa.load_spec(_spec(tmp_path, done, schedule))
    with pytest.raises(sa.NotReadyError) as exc:
        sa.analyze(spec, settings=FAST)
    assert exc.value.missing == [str(pending)]


def test_a_spec_with_study_configs_must_name_the_schedule(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    cfg = _config(tmp_path / "a.toml", "study-a", runs)
    _save(cfg, runs, "completed")
    with pytest.raises(ValueError, match="schedule"):
        sa.select(sa.load_spec(_spec(tmp_path, cfg, None)))


def test_partial_runs_are_never_selected(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    cfg = _config(tmp_path / "a.toml", "fake-a", runs)
    _save(cfg, runs, "quota_stopped")
    with pytest.raises(sa.NotReadyError):
        sa.select(sa.load_spec(_spec(tmp_path, cfg, None)))
    _save(cfg, runs, "completed")
    (chosen,) = sa.select(sa.load_spec(_spec(tmp_path, cfg, None))).values()
    assert chosen.name.endswith("-comple")


def test_the_study_spec_covers_the_schedule_and_the_planned_cascades() -> None:
    root = Path(__file__).parents[1]
    spec = sa.load_spec(root / "configs" / "analysis" / "study.toml", root)
    from ask_or_act.study import load_schedule

    assert spec.schedule is not None
    in_spec = {p for d in spec.deciders for p in d.slots.values()}
    assert {root / p for p in load_schedule(spec.schedule).configs} == {
        p for p in in_spec if "configs/study/" in p.as_posix()
    }
    assert {(c.dataset, c.stages) for c in spec.cascades} == {
        ("banking77", ("gpt-oss-20b", "gpt-oss-120b")),
        ("banking77", ("classic", "gpt-oss-120b")),
        ("aegis", ("gpt-oss-safeguard-20b", "gpt-oss-120b")),
        ("aegis", ("gpt-oss-20b", "gpt-oss-120b")),
        ("aegis", ("classic", "gpt-oss-120b")),
    }
    counts = {
        ds: sum(d.dataset == ds and d.kind == "llm" for d in spec.deciders)
        for ds in ("banking77", "aegis", "mt_bench")
    }
    assert counts == {"banking77": 3, "aegis": 4, "mt_bench": 2}

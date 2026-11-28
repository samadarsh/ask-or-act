"""The milestone 7 analysis, exactly as pre-registered in docs/analysis-plan.md.

The plan was FROZEN by commit dc2ea31 (2026-11-27 23:12:15 UTC), with Amendment 1 (a
date correction only) and Amendment 2 (Q1/Q1b test the excess-ECE gap; every gap gets
the interval that inverts its centered p; DA and IA become diagnostics; S6's UTC-day
reading and its fingerprint-mix check). Section names here follow the plan: Q1, Q1b,
Q2, Q3, Q4, the families F1, F1b, F2 and F4, and the sensitivity checks S1–S6.

**What it reads, and when.** A *spec* (TOML) names the run configs: for each decider and
dataset, its calibration and test halves, and optionally its self-consistency halves.
For each config, the newest completed run with the same config hash is used
(``select``). That step reads only ``run.json``: name, status and config hash.

- **The study guard.** If a spec names any study config, either under configs/study/
  or with a name starting "study-", **every** config in the study schedule must have a
  completed run. Otherwise nothing is read beyond that ``run.json`` metadata, and
  NotReadyError is raised, listing what's missing. So no study decision is read until the
  whole study is done.
- **Completed runs only.** Partial runs are never selected, and ``from_run`` refuses
  them again (``require_completed``).

**Fixed seeds.** Every bootstrap uses Settings() (2,000 resamples, seed 0, 95%). The
random-routing baseline uses seed 0. Re-running gives identical output.

**Paper-backed pieces** (cited where they are implemented):
- ACE DA and IA views (diagnostics only, Amendment 2): metrics/ace.py, arXiv 2606.30814
  v2 §4.
- The fixed-sequence test: guarantees/fixed_sequence.py, ToE arXiv 2407.18370 v1 §2.1
  Thm 1; Kotte arXiv 2606.29054 v2 Lemma E.1.
- The cascade: guarantees/cascade.py, ToE Alg. 2 and A.2.
- The abstention floor: metrics/feasibility.py, Kotte v2 Prop. 3, App. E.4.

The Q2/Q4 violation test is an ordinary one-sided exact binomial test, not a guarantee.

**Output:** ids and aggregates only, never input text. Aegis rows are never printed.
"""

import itertools
import json
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ask_or_act import judge
from ask_or_act.core.config import RunConfig, load_config
from ask_or_act.core.runs import RunRecord, load_run
from ask_or_act.core.types import Decision
from ask_or_act.guarantees.binomial import binom_cdf, two_sided_interval
from ask_or_act.guarantees.cascade import Stage, calibrate, route
from ask_or_act.guarantees.fixed_sequence import FastFit
from ask_or_act.guarantees.mt_bench import by_comparison, human_agreement
from ask_or_act.guarantees.recommend import GridChoice, grid_for_run, run_guarantee
from ask_or_act.metrics import ace
from ask_or_act.metrics.calibration import ece_fast
from ask_or_act.metrics.data import ERROR_LABEL, IntArray, Scored, align, from_run
from ask_or_act.metrics.feasibility import Feasibility, min_abstention
from ask_or_act.metrics.report import (
    Settings,
    noise_floors,
    single_report,
    to_jsonable,
)
from ask_or_act.metrics.stats import holm, iter_resamples
from ask_or_act.report.build import load_seen, without_seen
from ask_or_act.study import load_schedule

PLAN = "docs/analysis-plan.md (FROZEN dc2ea31, 2026-11-27 23:12:15 UTC; Amendments 1, 2)"
SETTINGS = Settings()  # 15 bins, 2,000 resamples, seed 0, 95%: the M2 metric settings
FWER = 0.05  # each family, Holm
ALPHA_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)  # exploratory (Q3)
DRIFT_SHARE = 0.05  # S6: a new fingerprint on > 5% of a run's calls is a change
EQUIVALENCE = {"ece_gap": 0.02, "accuracy": 0.03, "coverage": 0.10}  # "practically equivalent"
_EPS = 1e-9
SLOTS = ("cal", "test", "sc_cal", "sc_test")


class NotReadyError(RuntimeError):
    """Some config has no completed run yet; nothing beyond run.json was read."""

    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = list(missing)
        super().__init__(
            f"{len(self.missing)} config(s) without a completed run: "
            + ", ".join(self.missing[:8])
            + (" …" if len(self.missing) > 8 else "")
        )


# ----------------------------------------------------------------------------- spec


@dataclass(frozen=True)
class DeciderSpec:
    name: str
    dataset: str
    kind: str  # "llm" | "classic" | "majority"
    slots: Mapping[str, Path]  # some of SLOTS -> config path ("test" is required)


@dataclass(frozen=True)
class CascadeSpec:
    dataset: str
    stages: tuple[str, ...]  # decider names, cheapest first


@dataclass(frozen=True)
class Spec:
    path: Path
    label: str
    deciders: tuple[DeciderSpec, ...]
    cascades: tuple[CascadeSpec, ...]
    schedule: Path | None  # the study schedule: all of it must be complete
    seen_ids: Path | None  # S1
    notes: Mapping[str, str] = field(default_factory=dict)  # per dataset, with headlines
    targets: Mapping[str, tuple[float, float]] = field(default_factory=dict)  # fakes only
    deviations: tuple[str, ...] = ()


def load_spec(path: Path, root: Path | None = None) -> Spec:
    """Relative paths in the spec resolve against ``root`` (default: the current folder)."""
    base = Path.cwd() if root is None else root
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    def p(v: object) -> Path:
        q = Path(str(v))
        return q if q.is_absolute() else base / q

    deciders = []
    for d in data.get("decider", []):
        kind = str(d.get("kind", "llm"))
        if kind not in ("llm", "classic", "majority"):
            raise ValueError(f"{path}: decider kind must be llm, classic or majority")
        slots = {k: p(d[k]) for k in SLOTS if k in d}
        if "test" not in slots:
            raise ValueError(f"{path}: decider {d.get('name')!r} needs a test config")
        deciders.append(DeciderSpec(str(d["name"]), str(d["dataset"]), kind, slots))
    names = Counter((d.dataset, d.name) for d in deciders)
    if any(v > 1 for v in names.values()):
        raise ValueError(f"{path}: a decider name appears twice on one dataset")
    cascades = tuple(
        CascadeSpec(str(c["dataset"]), tuple(str(s) for s in c["stages"]))
        for c in data.get("cascade", [])
    )
    for c in cascades:
        if len(c.stages) != 2 or any((c.dataset, s) not in names for s in c.stages):
            raise ValueError(f"{path}: a cascade needs two deciders of its dataset")
    targets = {
        str(k): (float(v["alpha"]), float(v["delta"])) for k, v in data.get("targets", {}).items()
    }
    return Spec(
        path,
        str(data.get("label", path.stem)),
        tuple(deciders),
        cascades,
        p(data["schedule"]) if "schedule" in data else None,
        p(data["seen_ids"]) if "seen_ids" in data else None,
        {str(k): str(v) for k, v in data.get("notes", {}).items()},
        targets,
        tuple(str(x) for x in data.get("deviations", [])),
    )


# ------------------------------------------------------------------ run selection


def _completed_dirs(cfg: RunConfig) -> list[Path]:
    """Completed runs of exactly this config (same name and hash), oldest first. Reads
    run.json only."""
    if not cfg.runs_dir.exists():
        return []
    out = []
    for meta_path in sorted(cfg.runs_dir.glob(f"*-{cfg.name}-*/run.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("name") == cfg.name
            and meta.get("status") == "completed"
            and meta.get("config", {}).get("sha256") == cfg.sha256
        ):
            out.append(meta_path.parent)
    return out


def _is_study(path: Path, cfg: RunConfig) -> bool:
    return cfg.name.startswith("study-") or "configs/study/" in path.as_posix()


def select(spec: Spec) -> dict[Path, Path]:
    """Config path → newest completed run dir. Raises NotReadyError (having read only run.json
    files) if any is missing, or, for a spec with study configs, if any config of the
    study schedule has no completed run."""
    paths = sorted({p for d in spec.deciders for p in d.slots.values()})
    cfgs = {p: load_config(p) for p in paths}
    if any(_is_study(p, c) for p, c in cfgs.items()):
        if spec.schedule is None:
            raise ValueError("a spec with study configs must name the study schedule")
        sched = load_schedule(spec.schedule).configs
        missing = [str(p) for p in sched if not _completed_dirs(load_config(p))]
        if missing:
            raise NotReadyError(missing)
    chosen: dict[Path, Path] = {}
    missing = []
    for p, c in cfgs.items():
        dirs = _completed_dirs(c)
        if dirs:
            chosen[p] = dirs[-1]  # run ids start with the UTC start time
        else:
            missing.append(str(p))
    if missing:
        raise NotReadyError(missing)
    return chosen


# --------------------------------------------------------------------- loading


@dataclass(frozen=True)
class Slot:
    record: RunRecord
    scored: Scored  # vote level for MT-Bench
    grid: GridChoice
    target: tuple[float, float] | None


@dataclass(frozen=True)
class Decider:
    spec: DeciderSpec
    slots: Mapping[str, Slot]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def kind(self) -> str:
        return self.spec.kind


def load(spec: Spec, chosen: Mapping[Path, Path]) -> dict[str, list[Decider]]:
    """Deciders by dataset, in spec order. Call only after ``select``."""
    out: dict[str, list[Decider]] = defaultdict(list)
    for d in spec.deciders:
        slots = {}
        for k, cfg_path in d.slots.items():
            rec = load_run(chosen[cfg_path])
            s = from_run(rec)  # refuses a run that isn't completed
            if s.dataset != d.dataset:
                raise ValueError(f"{cfg_path}: dataset {s.dataset!r}, spec says {d.dataset!r}")
            g = run_guarantee(rec)
            target = (
                (g.alpha, g.delta)
                if g is not None and g.alpha is not None and g.delta is not None
                else spec.targets.get(d.dataset)
            )
            slots[k] = Slot(rec, s, grid_for_run(rec, None), target)
        out[d.dataset].append(Decider(d, slots))
    return dict(out)


# ------------------------------------------------------------------- variants


@dataclass(frozen=True)
class Variant:
    key: str
    title: str
    unit: str = "comparison"  # MT-Bench unit
    drop_seen: bool = False  # S1: test halves without the pilot-seen ids
    drop_errors: bool = False  # S5
    disjoint: bool = False  # S4
    vs_human: bool = False  # S3: α = 1 − h on the calibration votes (MT-Bench)
    questions: tuple[str, ...] = ("Q1", "Q1b", "Q2", "Q3", "Q4")
    datasets: tuple[str, ...] | None = None  # None: all


PRIMARY = Variant("primary", "Primary analysis")
SENSITIVITY = (
    Variant("S1", "S1: without the pilot-seen examples", drop_seen=True),
    Variant("S2", "S2: MT-Bench at vote level", unit="vote", datasets=("mt_bench",)),
    Variant(
        "S3",
        "S3: MT-Bench α = 1 − h (human ceiling, calibration votes)",
        vs_human=True,
        questions=("Q2", "Q3"),
        datasets=("mt_bench",),
    ),
    Variant("S4", "S4: disjoint cascade calibration", disjoint=True, questions=("Q3",)),
    Variant(
        "S5",
        "S5: failed calls excluded",
        drop_errors=True,
        questions=("Q1", "Q1b", "Q2", "Q3"),
    ),
)


@dataclass
class Ctx:
    variant: Variant
    seen: Mapping[str, set[str]]
    settings: Settings

    def scored(self, d: Decider, slot: str, *, keep: set[str] | None = None) -> Scored:
        """Vote level, after S1 (test halves) and an optional id filter (S6)."""
        s = d.slots[slot].scored
        if keep is not None:
            s = s.take(np.array([k for k, i in enumerate(s.ids) if i in keep], dtype=np.intp))
        if self.variant.drop_seen and slot in ("test", "sc_test"):
            s = without_seen(s, self.seen.get(s.dataset, set()))
        return s

    def unit(self, s: Scored) -> Scored:
        if s.dataset == "mt_bench" and self.variant.unit == "comparison":
            return by_comparison(s).scored
        return s

    def units(self, d: Decider, slot: str, *, keep: set[str] | None = None) -> Scored:
        return self.unit(self.scored(d, slot, keep=keep))

    def common(self, ss: Sequence[Scored]) -> list[Scored]:
        """The same examples, in the same order, for every run (paired analyses). S5
        also drops every example on which any of them failed."""
        ids = set(ss[0].ids).intersection(*(set(s.ids) for s in ss[1:]))
        if self.variant.drop_errors:
            ids -= {i for s in ss for i, p in zip(s.ids, s.pred, strict=True) if p == ERROR_LABEL}
        cut = [
            s.take(np.array([k for k, i in enumerate(s.ids) if i in ids], dtype=np.intp))
            for s in ss
        ]
        return [cut[0], *(align(cut[0], c)[1] for c in cut[1:])]


def _groups(s: Scored) -> tuple[str, ...] | None:
    return s.groups if s.dataset == "mt_bench" else None


# ------------------------------------------------------------------ fits and tests


@dataclass(frozen=True)
class Fit:
    names: tuple[str, ...]
    alpha: float
    delta: float
    thresholds: tuple[float | None, ...]
    grids: tuple[GridChoice, ...]
    n_cal: int
    feasibility: Feasibility | None
    cal_coverage: float
    cal_risk: float | None

    @property
    def certified(self) -> bool:
        return any(t is not None for t in self.thresholds)


@dataclass(frozen=True)
class Applied:
    n: int
    n_act: int
    errors: int
    shares: tuple[float, ...]  # per stage, of all test examples
    coverage_ci: tuple[float, float] | None
    risk_ci: tuple[float, float] | None
    p_violation: float  # one-sided exact binomial, H0: risk ≤ α

    @property
    def coverage(self) -> float:
        return self.n_act / self.n if self.n else 0.0

    @property
    def risk(self) -> float | None:
        return self.errors / self.n_act if self.n_act else None


def p_violation(errors: int, n_act: int, alpha: float) -> float:
    """P(Bin(n_act, α) ≥ errors): small when the observed errors are too many for a risk
    of at most α. One-sided exact binomial test of H0: risk ≤ α."""
    if n_act == 0 or errors == 0:
        return 1.0
    return max(0.0, 1.0 - binom_cdf(errors - 1, n_act, alpha))


def _stages(names: Sequence[str], ss: Sequence[Scored], grids: Sequence[GridChoice]) -> list[Stage]:
    return [
        Stage(n, s.confidence, s.correct, g.grid, g.starts)
        for n, s, g in zip(names, ss, grids, strict=True)
    ]


def fit(
    names: Sequence[str],
    cal: Sequence[Scored],
    grids: Sequence[GridChoice],
    alpha: float,
    delta: float,
    *,
    disjoint: bool = False,
) -> Fit:
    """The pre-registered procedure on calibration examples (already common across stages):
    Kotte v2 Prop. 3 floor (δ_floor = δ), then ToE Alg. 2 (one stage: ToE Thm 1)."""
    n = len(cal[0])
    if n == 0:
        return Fit(tuple(names), alpha, delta, (None,) * len(cal), tuple(grids), 0, None, 0.0, None)
    stages = _stages(names, cal, grids)
    feas = min_abstention(1.0 - float(cal[0].correct.mean()), alpha, n=n, delta=delta)
    res = calibrate(stages, alpha=alpha, delta=delta, ids=list(cal[0].ids), disjoint=disjoint)
    routed = route(stages, res.thresholds)
    return Fit(
        tuple(names),
        alpha,
        delta,
        res.thresholds,
        tuple(grids),
        n,
        feas,
        routed.coverage(),
        routed.risk(),
    )


def apply(f: Fit, test: Sequence[Scored]) -> Applied:
    """Route the (common) test examples with the fitted thresholds."""
    n = len(test[0])
    if n == 0:
        return Applied(0, 0, 0, (0.0,) * len(test), None, None, 1.0)
    routed = route(_stages(f.names, test, f.grids), f.thresholds)
    acted = routed.stage >= 0
    n_act = int(acted.sum())
    errors = int((acted & ~routed.correct).sum())
    return Applied(
        n,
        n_act,
        errors,
        tuple(routed.share(i) for i in range(len(test))),
        two_sided_interval(n_act, n),
        two_sided_interval(errors, n_act) if n_act else None,
        p_violation(errors, n_act, f.alpha),
    )


_TOL = 1e-12  # the same tie tolerance as metrics.stats._paired


def invert(d: float | None, boot: Iterable[float | None], level: float) -> dict[str, Any]:
    """The centered bootstrap p-value of "gap = 0" and the interval that inverts it
    (Amendment 2).

    With e_b = |d*_b − d| over the B resamples where the gap is defined:

    - p = (1 + #{e_b ≥ |d|}) / (B + 1), as metrics.stats.paired_bootstrap;
    - the interval is d ± h: every θ whose test of "gap = θ" has p > 1 − level. That
      holds when #{e_b ≥ |d − θ|} > m, with m = ⌊(1 − level)(B + 1)⌋ − 1, so h is the
      (m + 1)-th largest e_b (plus the tie tolerance).

    So the interval excludes 0 exactly when p ≤ 1 − level. It is symmetric because the
    p-value is. The basic interval was tried first and disagreed with this p-value (the
    resampled ECE carries its own upward bias); see the plan, Amendment 2."""
    vals = np.array([x for x in boot if x is not None], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if d is None or not np.isfinite(d) or len(vals) == 0:
        return {"estimate": d, "low": None, "high": None, "p": None, "n_boot": len(vals)}
    e = np.sort(np.abs(vals - d))
    big = len(e)
    m = int(np.floor((1 - level) * (big + 1) + 1e-9)) - 1
    h = float("inf") if m < 0 or m >= big else float(e[big - m - 1]) + _TOL
    extreme = int(np.sum(e >= abs(d) - _TOL))
    return {
        "estimate": float(d),
        "low": float(d - h),
        "high": float(d + h),
        "p": min(1.0, (1 + extreme) / (big + 1)),
        "n_boot": big,
        "interval": "inverts the centered p",
    }


def untested(r: Mapping[str, Any]) -> dict[str, Any]:
    """A descriptive or diagnostic gap (Amendment 2): its p is kept only as
    ``p_not_tested``, so the interval can be checked against it. It never enters a
    family or a table."""
    out = dict(r)
    out["p_not_tested"] = out.pop("p")
    return out


def _diff(x: float | None, y: float | None, shift: float = 0.0) -> float | None:
    return None if x is None or y is None else x - y - shift


def centered(
    stat_a: Callable[[IntArray], float | None],
    stat_b: Callable[[IntArray], float | None],
    n: int,
    settings: Settings,
    *,
    groups: Sequence[str] | None = None,
    shift: float = 0.0,
) -> dict[str, Any]:
    """Centered paired bootstrap of the gap d = stat_a − stat_b − shift, where ``shift``
    is a fixed adjustment (it moves the estimate, not the resampled spread). p and the
    interval are from ``invert``. The same resamples as paired_bootstrap with the same
    seed, so with shift = 0 the p is identical (tests/test_study_analysis.py)."""
    full = np.arange(n, dtype=np.intp)
    d = _diff(stat_a(full), stat_b(full), shift)
    boot = [
        _diff(stat_a(idx), stat_b(idx), shift)
        for idx in iter_resamples(n, n_boot=settings.n_boot, seed=settings.seed, groups=groups)
    ]
    return invert(d, boot, settings.level)


def single(
    stat: Callable[[IntArray], float | None],
    n: int,
    settings: Settings,
    *,
    groups: Sequence[str] | None = None,
    shift: float = 0.0,
) -> dict[str, Any]:
    """``centered`` for one statistic minus a fixed shift (per-decider excess ECE)."""
    return centered(stat, lambda idx: 0.0, n, settings, groups=groups, shift=shift)


def _ece_stat(s: Scored, settings: Settings) -> Callable[[IntArray], float]:
    return lambda idx: ece_fast(
        s.confidence[idx], s.correct[idx], n_bins=settings.n_bins, strategy="width"
    )


def floor_mean(s: Scored, settings: Settings) -> float:
    """The ECE noise-floor mean: the expected ECE of a perfectly calibrated decider with
    these n and confidence values (metrics.calibration.ece_noise_floor: 500 seeded
    simulations, the frozen bins). The same number as in the per-decider table."""
    return float(noise_floors(s, settings)["ece_width"]["mean"])


def excess_test(a: Scored, b: Scored, settings: Settings) -> dict[str, Any]:
    """Q1/Q1b's test (Amendment 2): the paired gap in excess ECE, (ECE_A − floor_A) −
    (ECE_B − floor_B), raw top-label ECE with the frozen 15 equal-width bins, by the
    centered paired bootstrap. The floors are fixed, so this asks whether the ECE gap is
    beyond what the two noise floors explain. Negative: A better calibrated. Not a
    guarantee, so no theorem to cite; its null behaviour is checked by simulation
    (scripts/excess_ece_null.py)."""
    fa, fb = floor_mean(a, settings), floor_mean(b, settings)
    r = centered(
        _ece_stat(a, settings),
        _ece_stat(b, settings),
        len(a),
        settings,
        groups=_groups(a),
        shift=fa - fb,
    )
    return {**r, "floor_a": fa, "floor_b": fb}


def raw_ece_gap(a: Scored, b: Scored, settings: Settings) -> dict[str, Any]:
    """The raw ECE gap A − B, descriptive since Amendment 2."""
    ea, eb = _ece_stat(a, settings), _ece_stat(b, settings)
    return untested(centered(ea, eb, len(a), settings, groups=_groups(a)))


def ace_gaps(a: Scored, b: Scored, settings: Settings) -> dict[str, Any]:
    """Q1's diagnostics (Amendment 2): the ACE DA and IA ECE gaps and the raw Brier gap
    (arXiv 2606.30814 v2 §4; the views of metrics/ace.py with width bins, as
    ace_section), each with the interval of ``invert``, plus the reversal flags and DA's
    ESS/N. Reported, never tested."""
    inst = ace.candidate_instances(a, b)
    point = ace.all_views(a, b, n_bins=settings.n_bins, strategy="width", inst=inst)
    keys = (("DA", "ece"), ("IA", "ece"), ("raw", "brier"))

    def gap(v: ace.View, metric: str) -> float | None:
        return v.ece_gap if metric == "ece" else v.brier_gap

    boot: dict[tuple[str, str], list[float | None]] = {k: [] for k in keys}
    resamples = iter_resamples(
        len(a), n_boot=settings.n_boot, seed=settings.seed, groups=_groups(a)
    )
    for idx in resamples:
        views = ace.all_views(a, b, n_bins=settings.n_bins, strategy="width", idx=idx, inst=inst)
        for k in keys:
            boot[k].append(gap(views[k[0]], k[1]))
    raw_ece = point["raw"].ece_gap
    out: dict[str, Any] = {
        f"{v.lower()}_{m}_gap": untested(invert(gap(point[v], m), boot[(v, m)], settings.level))
        for v, m in keys
    }
    for v in ("DA", "IA"):
        g = point[v].ece_gap
        out[f"{v.lower()}_reversal"] = g is not None and raw_ece is not None and g * raw_ece < 0
    ess = point["DA"].info.get("ess_ratio")
    out |= {
        "ia_kept": point["IA"].info.get("instances"),
        "ess_ratio": ess,
        "ess_flag": (ess or 1.0) < ace.ESS_WARNING,
    }
    return out


def _equivalent(low: float | None, high: float | None, margin: float) -> bool:
    return low is not None and high is not None and -margin < low and high < margin


# --------------------------------------------------------------------- Q1, Q1b


def _describe(s: Scored, settings: Settings) -> dict[str, Any]:
    """Per-decider numbers: descriptive only, never a test (plan, Q1)."""
    r = single_report(s, settings)
    keep = ("accuracy", "ece_width", "ece_mass", "brier", "aurc", "mean_confidence")
    return {
        "n": r["n"],
        "n_errors": r["n_errors"],
        "n_unique_confidences": r["n_unique_confidences"],
        "metrics": {k: r["metrics"][k] for k in keep if k in r["metrics"]},
        "noise_floor": r["noise_floor"],
        "excess_ece": untested(
            single(
                _ece_stat(s, settings),
                len(s),
                settings,
                groups=_groups(s),
                shift=r["noise_floor"]["ece_width"]["mean"],
            )
        ),
    }


def _pair(a: Scored, b: Scored, names: tuple[str, str], ctx: Ctx) -> dict[str, Any]:
    """One Q1/Q1b pair (Amendment 2). ``excess`` is the test. The raw ECE gap and the
    ``diagnostics`` (ACE DA and IA, the raw Brier gap) are reported, never tested. Every
    interval inverts its centered p-value."""
    t = excess_test(a, b, ctx.settings)
    d = ace_gaps(a, b, ctx.settings)
    return {
        "a": names[0],
        "b": names[1],
        "n": len(a),
        "accuracy": [float(a.correct.mean()), float(b.correct.mean())] if len(a) else None,
        "excess": t,
        "raw_ece_gap": raw_ece_gap(a, b, ctx.settings),  # descriptive
        "noise_floor": [
            noise_floors(s, ctx.settings)["ece_width"] for s in (a, b)
        ],  # mean and p95 under perfect calibration, for A and B
        "brier_gap": d.pop("raw_brier_gap"),  # descriptive: depends on accuracy
        "diagnostics": d,
        "equivalent": _equivalent(t["low"], t["high"], EQUIVALENCE["ece_gap"]),
    }


def q1(data: Mapping[str, list[Decider]], ctx: Ctx) -> dict[str, Any]:
    pairs, per = [], {}
    for ds, deciders in data.items():
        eligible = [d for d in deciders if d.kind in ("llm", "classic")]
        for d in eligible:
            per[f"{ds}/{d.name}"] = _describe(ctx.units(d, "test"), ctx.settings)
        for a, b in itertools.combinations(eligible, 2):
            sa, sb = ctx.common([ctx.units(a, "test"), ctx.units(b, "test")])
            pairs.append({"dataset": ds, **_pair(sa, sb, (a.name, b.name), ctx)})
    return {"pairs": pairs, "per_decider": per, "family": _family("F1", pairs, "excess")}


def q1b(data: Mapping[str, list[Decider]], ctx: Ctx) -> dict[str, Any]:
    pairs, per = [], {}
    for ds, deciders in data.items():
        for d in deciders:
            if d.kind != "llm" or "sc_test" not in d.slots:
                continue
            sc = ctx.units(d, "sc_test")
            main, sc = ctx.common([ctx.units(d, "test"), sc])
            per[f"{ds}/{d.name}/self_consistency"] = _describe(sc, ctx.settings)
            per[f"{ds}/{d.name}/verbalized (same examples)"] = _describe(main, ctx.settings)
            names = (f"{d.name} verbalized", f"{d.name} self-consistency")
            pairs.append({"dataset": ds, "decider": d.name, **_pair(main, sc, names, ctx)})
    return {"pairs": pairs, "per_decider": per, "family": _family("F1b", pairs, "excess")}


def _family(name: str, rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Holm across a family, in place: adds ``p_holm`` and ``rejected``. A test without a
    p-value (e.g. DA undefined) is dropped and listed."""
    usable = [r for r in rows if r[key]["p"] is not None]
    dropped = [
        {"test": _label(r), "reason": r[key].get("why") or "no p-value (statistic undefined)"}
        for r in rows
        if r[key]["p"] is None
    ]
    adjusted = holm([r[key]["p"] for r in usable])
    for r, q in zip(usable, adjusted, strict=True):
        r[key]["p_holm"] = q
        r[key]["rejected"] = q <= FWER
    return {
        "name": name,
        "size": len(usable),
        "fwer": FWER,
        "rejected": sum(r[key]["rejected"] for r in usable),
        "dropped": dropped,
    }


def _label(r: Mapping[str, Any]) -> str:
    if "a" in r and "b" in r:
        return f"{r['dataset']}: {r['a']} vs {r['b']}"
    return f"{r['dataset']}: {r.get('name', '?')}"


# --------------------------------------------------------------------- Q2, Q3


def _target(d: Decider, slot: str, ctx: Ctx) -> tuple[float, float]:
    t = d.slots[slot].target
    if t is None:
        raise ValueError(f"{d.spec.dataset}/{d.name}: no [guarantee] target in its config")
    if ctx.variant.vs_human:
        if d.spec.dataset != "mt_bench":
            raise ValueError("S3 applies to MT-Bench only")
        return human_agreement(ctx.scored(d, slot)).alpha, t[1]
    return t


def _config_row(
    ds: str, name: str, signal: str, f: Fit, a: Applied, note: str = ""
) -> dict[str, Any]:
    feas = f.feasibility
    return {
        "dataset": ds,
        "name": name,
        "signal": signal,
        "alpha": f.alpha,
        "delta": f.delta,
        "thresholds": list(f.thresholds),
        "certified": f.certified,
        "n_cal": f.n_cal,
        "cal_coverage": f.cal_coverage,
        "cal_risk": f.cal_risk,
        "floor": None
        if feas is None
        else {
            "base_risk": feas.base_risk,
            "plug_in": feas.min_abstention,
            "conservative": feas.min_abstention_conservative,
            "delta_floor": feas.delta,
        },
        "n_test": a.n,
        "n_act": a.n_act,
        "errors": a.errors,
        "test_coverage": a.coverage,
        "test_coverage_ci": a.coverage_ci,
        "test_abstention": 1.0 - a.coverage,
        "test_risk": a.risk,
        "test_risk_ci": a.risk_ci,
        "shares": list(a.shares),
        "above_alpha": a.risk is not None and a.risk > f.alpha,
        "violation": {"p": a.p_violation if f.certified and a.n_act else None}
        | (
            {"why": "certified, but nothing acted on in the test half"}
            if f.certified and not a.n_act
            else {}
        ),
        "note": note,
    }


def single_configs(
    data: Mapping[str, list[Decider]], ctx: Ctx, *, keep: Mapping[str, set[str]] | None = None
) -> list[dict[str, Any]]:
    """One row per (dataset, decider, signal) with a calibration half (Q2, Q3)."""
    rows = []
    for ds, deciders in data.items():
        for d in deciders:
            for signal, cal_slot, test_slot in (
                ("verbalized" if d.kind == "llm" else "native", "cal", "test"),
                ("self_consistency", "sc_cal", "sc_test"),
            ):
                if d.kind == "majority" or cal_slot not in d.slots or test_slot not in d.slots:
                    continue
                k = None if keep is None else keep.get(f"{ds}/{d.name}")
                alpha, delta = _target(d, cal_slot, ctx)
                (cal,) = ctx.common([ctx.units(d, cal_slot, keep=k)])
                (test,) = ctx.common([ctx.units(d, test_slot, keep=k)])
                f = fit([d.name], [cal], [d.slots[cal_slot].grid], alpha, delta)
                rows.append(_config_row(ds, d.name, signal, f, apply(f, [test])))
    return rows


def _cost_per_decision(d: Decider) -> float:
    ds = d.slots["test"].record.decisions
    return float(np.mean([x.cost_usd for x in ds])) if ds else 0.0


def cascades(data: Mapping[str, list[Decider]], spec: Spec, ctx: Ctx) -> list[dict[str, Any]]:
    rows = []
    for c in spec.cascades:
        if c.dataset not in data or (
            ctx.variant.datasets and c.dataset not in ctx.variant.datasets
        ):
            continue
        by = {d.name: d for d in data[c.dataset]}
        ds = [by[s] for s in c.stages]
        alpha, delta = _target(ds[0], "cal", ctx)
        cal = ctx.common([ctx.units(d, "cal") for d in ds])
        test = ctx.common([ctx.units(d, "test") for d in ds])
        f = fit(
            list(c.stages),
            cal,
            [d.slots["cal"].grid for d in ds],
            alpha,
            delta,
            disjoint=ctx.variant.disjoint,
        )
        a = apply(f, test)
        row = _config_row(c.dataset, " → ".join(c.stages), "cascade", f, a)
        cost = [_cost_per_decision(d) for d in ds]
        reach_strong = 1.0 - a.shares[0]
        row["cost_per_decision_usd"] = cost[0] + reach_strong * cost[1]
        row["baselines"] = _cascade_baselines(test, cost, reach_strong, ctx.settings.seed)
        row["shares_named"] = {"act": a.shares[0], "ask": a.shares[1], "escalate": 1 - a.coverage}
        rows.append(row)
    return rows


def _cascade_baselines(
    test: Sequence[Scored], cost: Sequence[float], reach_strong: float, seed: int
) -> dict[str, Any]:
    """Always cheap, always strong, and random routing at equal cost (same share sent to
    the strong model, chosen at random; 2,000 seeded draws). Descriptive, not tests."""
    cheap, strong = test
    n = len(cheap)
    out: dict[str, Any] = {
        "always_cheap": {"risk": 1 - float(cheap.correct.mean()) if n else None, "cost": cost[0]},
        "always_strong": {"risk": 1 - float(strong.correct.mean()) if n else None, "cost": cost[1]},
    }
    m = round(reach_strong * n)
    if n:
        rng = np.random.default_rng(seed)
        risks = []
        for _ in range(2000):
            pick = np.zeros(n, dtype=np.bool_)
            pick[rng.choice(n, size=m, replace=False)] = True
            right = np.where(pick, strong.correct, cheap.correct)
            risks.append(1.0 - float(right.mean()))
        lo, hi = np.quantile(risks, [0.025, 0.975])
        out["random_equal_cost"] = {
            "share_to_strong": m / n,
            "risk_mean": float(np.mean(risks)),
            "risk_95": [float(lo), float(hi)],
            "cost": cost[0] + (m / n) * cost[1],
            "draws": 2000,
        }
    return out


def alpha_grid(data: Mapping[str, list[Decider]], ctx: Ctx) -> list[dict[str, Any]]:
    """Exploratory: coverage and risk on the test half over ALPHA_GRID (not pre-registered)."""
    rows = []
    for ds, deciders in data.items():
        for d in deciders:
            if d.kind == "majority" or "cal" not in d.slots:
                continue
            (cal,) = ctx.common([ctx.units(d, "cal")])
            (test,) = ctx.common([ctx.units(d, "test")])
            _, delta = _target(d, "cal", ctx)
            for alpha in ALPHA_GRID:
                f = fit([d.name], [cal], [d.slots["cal"].grid], alpha, delta)
                a = apply(f, [test])
                rows.append(
                    {
                        "dataset": ds,
                        "name": d.name,
                        "alpha": alpha,
                        "threshold": f.thresholds[0],
                        "coverage": a.coverage,
                        "risk": a.risk,
                    }
                )
    return rows


def coverage_differences(data: Mapping[str, list[Decider]], ctx: Ctx) -> list[dict[str, Any]]:
    """Exploratory: paired coverage differences at the registered target, with bootstrap
    CIs that refit both thresholds on each calibration resample (approximate)."""
    s = ctx.settings
    rows = []
    for ds, deciders in data.items():
        eligible = [d for d in deciders if d.kind in ("llm", "classic") and "cal" in d.slots]
        for a, b in itertools.combinations(eligible, 2):
            alpha, delta = _target(a, "cal", ctx)
            ca, cb = ctx.common([ctx.units(a, "cal"), ctx.units(b, "cal")])
            ta, tb = ctx.common([ctx.units(a, "test"), ctx.units(b, "test")])
            if not len(ca) or not len(ta):
                continue
            ga, gb = a.slots["cal"].grid, b.slots["cal"].grid
            fa = FastFit(alpha=alpha, delta=delta, grid=ga.grid, starts=ga.starts, n=len(ca))
            fb = FastFit(alpha=alpha, delta=delta, grid=gb.grid, starts=gb.starts, n=len(cb))

            def cov(lam: float | None, t: Scored, idx: IntArray) -> float:
                return 0.0 if lam is None else float((t.confidence[idx] >= lam - _EPS).mean())

            full_t = np.arange(len(ta), dtype=np.intp)
            point = cov(fa.threshold(ca.confidence, ca.correct), ta, full_t) - cov(
                fb.threshold(cb.confidence, cb.correct), tb, full_t
            )
            diffs = []
            for ic, it in zip(
                iter_resamples(len(ca), n_boot=s.n_boot, seed=s.seed, groups=_groups(ca)),
                iter_resamples(len(ta), n_boot=s.n_boot, seed=s.seed + 1, groups=_groups(ta)),
                strict=True,
            ):
                la = fa.threshold(ca.confidence[ic], ca.correct[ic])
                lb = fb.threshold(cb.confidence[ic], cb.correct[ic])
                diffs.append(cov(la, ta, it) - cov(lb, tb, it))
            iv = invert(point, diffs, s.level)  # Amendment 2
            rows.append(
                {
                    "dataset": ds,
                    "a": a.name,
                    "b": b.name,
                    "coverage_diff": point,
                    "low": iv["low"],
                    "high": iv["high"],
                    "p": iv["p"],
                    "interval": iv.get("interval"),
                }
            )
    return rows


# --------------------------------------------------------------------------- Q4


def q4(
    data: Mapping[str, list[Decider]],
    ctx: Ctx,
    *,
    signal: str = "verbalized",
    keep: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    """λ̂ fitted on A's calibration half, applied unchanged to B's test half. F4: the
    verbalized LLM pairs. Pairs whose A certifies nothing drop out and are listed."""
    cal_slot, test_slot = ("sc_cal", "sc_test") if signal == "self_consistency" else ("cal", "test")
    rows, dropped = [], []
    for ds, deciders in data.items():
        if signal == "classic":
            sources = [d for d in deciders if d.kind == "classic" and "cal" in d.slots]
            targets = [d for d in deciders if d.kind == "llm"]
            pairs = [(a, b) for a in sources for b in targets]
        else:
            llms = [
                d
                for d in deciders
                if d.kind == "llm" and cal_slot in d.slots and test_slot in d.slots
            ]
            pairs = list(itertools.permutations(llms, 2))
        for a, b in pairs:
            ka = None if keep is None else keep.get(f"{ds}/{a.name}")
            kb = None if keep is None else keep.get(f"{ds}/{b.name}")
            alpha, delta = _target(b, cal_slot, ctx)
            (ca,) = ctx.common([ctx.units(a, cal_slot, keep=ka)])
            (cb,) = ctx.common([ctx.units(b, cal_slot, keep=kb)])
            (tb,) = ctx.common([ctx.units(b, test_slot, keep=kb)])
            fa = fit([a.name], [ca], [a.slots[cal_slot].grid], alpha, delta)
            fb = fit([b.name], [cb], [b.slots[cal_slot].grid], alpha, delta)
            label = f"{ds}: {a.name} → {b.name}"
            if not fa.certified:
                dropped.append({"test": label, "reason": f"{a.name} certifies nothing"})
                continue
            lam_a, lam_b = fa.thresholds[0], fb.thresholds[0]
            on_b = replace(fa, names=(b.name,))
            ap = apply(on_b, [tb])
            native = apply(fb, [tb])
            rows.append(
                {
                    "dataset": ds,
                    "a": a.name,
                    "b": b.name,
                    "threshold_a": lam_a,
                    "threshold_b": lam_b,
                    "alpha": alpha,
                    "transfer": _config_row(ds, label, signal, on_b, ap),
                    "native": _config_row(ds, b.name, signal, fb, native),
                    "vs_native": _vs_native(tb, lam_a, lam_b, ctx.settings),
                    "violation": {
                        "p": ap.p_violation if ap.n_act else None,
                        **({} if ap.n_act else {"why": "nothing acted on at λ̂_A"}),
                    },
                }
            )
    out: dict[str, Any] = {"pairs": rows, "dropped": dropped}
    if signal == "verbalized":
        out["family"] = _family("F4", rows, "violation")
        out["family"]["dropped"] = dropped + out["family"]["dropped"]
    return out


def _vs_native(t: Scored, lam_a: float | None, lam_b: float | None, s: Settings) -> dict[str, Any]:
    """Coverage and risk at λ̂_A minus at λ̂_B on B's test examples, both thresholds held
    fixed: the CI leaves out calibration-half variability."""

    def cov(lam: float | None) -> Callable[[IntArray], float | None]:
        return lambda idx: 0.0 if lam is None else float((t.confidence[idx] >= lam - _EPS).mean())

    def risk(lam: float | None) -> Callable[[IntArray], float | None]:
        def f(idx: IntArray) -> float | None:
            if lam is None:
                return None
            emit = t.confidence[idx] >= lam - _EPS
            return float((~t.correct[idx][emit]).mean()) if emit.any() else None

        return f

    if not len(t):
        return {}
    out = {}
    stats = {"coverage": (cov(lam_a), cov(lam_b)), "risk": (risk(lam_a), risk(lam_b))}
    for k, (fa, fb) in stats.items():
        r = centered(fa, fb, len(t), s, groups=_groups(t))  # Amendment 2
        out[k] = {
            "diff": r["estimate"],
            "low": r["low"],
            "high": r["high"],
            "p": r["p"],
            "interval": r.get("interval"),
            "note": "thresholds held fixed",
        }
    return out


# --------------------------------------------------------------------------- S6


def _calls(raw: Mapping[str, Any] | None) -> list[tuple[str | None, str | None, int | None]]:
    """(response id, system_fingerprint, created) of every provider response in a
    decision's raw extras: the call itself, its main call, its samples."""
    if not isinstance(raw, Mapping):
        return []
    out = []
    if "system_fingerprint" in raw or "created" in raw:
        c = raw.get("created")
        out.append(
            (
                raw.get("id"),
                raw.get("system_fingerprint"),
                int(c) if isinstance(c, int | float) else None,
            )
        )
    out += _calls(raw.get("main"))
    for smp in raw.get("samples") or []:
        if isinstance(smp, Mapping):
            out += _calls(smp.get("raw"))
    return out


def _main_call(d: Decision) -> tuple[str | None, int | None]:
    calls = _calls(d.raw)
    return (calls[0][1], calls[0][2]) if calls else (None, None)


def _day(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).date().isoformat()


def drift(data: Mapping[str, list[Decider]]) -> dict[str, Any]:
    """S6. For each LLM: the UTC dates of every run's calls; each fingerprint's first
    appearance across all of that model's runs in the spec; and, per dataset, whether a
    fingerprint that first appeared after the model's earliest calibration call there
    serves more than DRIFT_SHARE of the calls of one of its runs there."""
    runs_by_model: dict[str, list[tuple[str, str, Decider, Slot]]] = defaultdict(list)
    for ds, deciders in data.items():
        for d in deciders:
            if d.kind == "llm":
                for k, sl in d.slots.items():
                    runs_by_model[d.name].append((ds, k, d, sl))
    models: dict[str, Any] = {}
    for model, runs in runs_by_model.items():
        first: dict[str, int] = {}
        per_run: list[dict[str, Any]] = []
        any_fp = False
        for ds, k, _, sl in runs:
            seen_ids: set[str] = set()
            fps: Counter[str] = Counter()
            days: Counter[str] = Counter()
            for dec in sl.record.decisions:
                for rid, fp, ts in _calls(dec.raw):
                    if rid is not None:
                        if rid in seen_ids:
                            continue
                        seen_ids.add(rid)
                    if ts is not None:
                        days[_day(ts)] += 1
                    if fp is not None:
                        any_fp = True
                        fps[fp] += 1
                        if ts is not None:
                            first[fp] = min(first.get(fp, ts), ts)
            per_run.append(
                {
                    "dataset": ds,
                    "slot": k,
                    "run": sl.record.path.name,
                    "days": dict(sorted(days.items())),
                    "fingerprints": dict(fps),
                    "n_calls": sum(fps.values()),
                }
            )
        changes = {}
        for ds in sorted({r["dataset"] for r in per_run}):
            cal_ts = [
                ts
                for (dds, k, _, sl) in runs
                if dds == ds and k in ("cal", "sc_cal")
                for dec in sl.record.decisions
                for (_, _, ts) in _calls(dec.raw)
                if ts is not None
            ]
            if not cal_ts:
                continue
            t0 = min(cal_ts)
            # Compared by UTC day: on the first day the whole pool appears within
            # seconds, so "after the earliest call" means a later day than its day.
            new = {fp for fp, ts in first.items() if _day(ts) > _day(t0)}
            flagged = set()
            for r in per_run:
                if r["dataset"] != ds or not r["n_calls"]:
                    continue
                for fp in new:
                    if r["fingerprints"].get(fp, 0) / r["n_calls"] > DRIFT_SHARE:
                        flagged.add(fp)
            if flagged:
                changes[ds] = {
                    "fingerprints": sorted(flagged),
                    "day": min(_day(first[fp]) for fp in flagged),
                    "earliest_calibration_call": datetime.fromtimestamp(t0, UTC).isoformat(),
                }
        mix = {ds: _mix(per_run, ds) for ds in sorted({r["dataset"] for r in per_run})}
        for r in per_run:
            r["fingerprints"] = dict(sorted(r["fingerprints"].items()))
        models[model] = {
            "has_fingerprints": any_fp,
            "runs": per_run,
            "first_seen": {
                fp: datetime.fromtimestamp(ts, UTC).isoformat() for fp, ts in sorted(first.items())
            },
            "changes": changes,
            "mix": {ds: m for ds, m in mix.items() if m is not None},
        }
    return {
        "rule": "(UTC-day reading, Amendment 2) a fingerprint first seen (across the model's "
        "runs in this spec) on a later UTC day than the model's earliest calibration call "
        "on a dataset, used by more than "
        f"{DRIFT_SHARE:.0%} of the calls of one of its runs on that dataset",
        "models": models,
        "limitation": "an unchanged fingerprint pool doesn't prove an unchanged model; Groq "
        "doesn't document what system_fingerprint means",
    }


def _mix(per_run: Sequence[Mapping[str, Any]], ds: str) -> dict[str, Any] | None:
    """S6's descriptive check (Amendment 2): the fingerprint mix of a model's calibration
    calls against its test calls on one dataset. Shares, one-half-only fingerprints and
    the total-variation distance; not a test."""
    halves: dict[str, Counter[str]] = {"calibration": Counter(), "test": Counter()}
    for r in per_run:
        if r["dataset"] == ds:
            half = "calibration" if r["slot"] in ("cal", "sc_cal") else "test"
            halves[half].update(r["fingerprints"])
    n = {h: sum(c.values()) for h, c in halves.items()}
    if not n["calibration"] or not n["test"]:
        return None
    fps = sorted(set(halves["calibration"]) | set(halves["test"]))
    shares = {fp: [halves[h][fp] / n[h] for h in ("calibration", "test")] for fp in fps}
    return {
        "n_calls": n,
        "shares": shares,
        "only_calibration": [fp for fp in fps if not halves["test"][fp]],
        "only_test": [fp for fp in fps if not halves["calibration"][fp]],
        "tv_distance": 0.5 * sum(abs(c - t) for c, t in shares.values()),
    }


def drift_splits(
    data: Mapping[str, list[Decider]], d6: Mapping[str, Any], ctx: Ctx
) -> list[dict[str, Any]]:
    """When S6 finds a change: that model's test results split by period (calls before /
    from the change day), and Q2/Q4 on calls whose fingerprints appear in both halves."""
    out = []
    for model, m in d6["models"].items():
        for ds, ch in m["changes"].items():
            d = next(x for x in data[ds] if x.name == model)
            entry: dict[str, Any] = {
                "model": model,
                "dataset": ds,
                "change_day": ch["day"],
                "periods": {},
            }
            for test_slot, cal_slot in (("test", "cal"), ("sc_test", "sc_cal")):
                if test_slot not in d.slots or cal_slot not in d.slots:
                    continue
                sl = d.slots[test_slot]
                when = {dec.example_id: _main_call(dec)[1] for dec in sl.record.decisions}
                before = {i for i, ts in when.items() if ts is not None and _day(ts) < ch["day"]}
                after = {i for i, ts in when.items() if ts is not None and _day(ts) >= ch["day"]}
                alpha, delta = _target(d, cal_slot, ctx)
                (cal,) = ctx.common([ctx.units(d, cal_slot)])
                f = fit([d.name], [cal], [d.slots[cal_slot].grid], alpha, delta)
                for label, ids in (("before", before), ("from", after)):
                    (t,) = ctx.common([ctx.units(d, test_slot, keep=ids)])
                    a = apply(f, [t])
                    entry["periods"][f"{test_slot} {label} {ch['day']}"] = {
                        "n": len(t),
                        "accuracy": float(t.correct.mean()) if len(t) else None,
                        "coverage": a.coverage,
                        "risk": a.risk,
                        "risk_ci": a.risk_ci,
                    }
            fp_cal = _fps_of(d, ("cal", "sc_cal"))
            fp_test = _fps_of(d, ("test", "sc_test"))
            both = fp_cal & fp_test
            keep_ids = {
                dec.example_id
                for k in d.slots
                for dec in d.slots[k].record.decisions
                if _main_call(dec)[0] in both
            }
            keep = {f"{ds}/{model}": keep_ids}
            entry["shared_fingerprints"] = sorted(both)
            sub = {ds: data[ds]}
            entry["q2_restricted"] = [
                r for r in single_configs(sub, ctx, keep=keep) if r["name"] == model
            ]
            entry["q4_restricted"] = [
                r for r in q4(sub, ctx, keep=keep)["pairs"] if model in (r["a"], r["b"])
            ]
            out.append(entry)
    return out


def _fps_of(d: Decider, slots: Iterable[str]) -> set[str]:
    return {
        fp
        for k in slots
        if k in d.slots
        for dec in d.slots[k].record.decisions
        for (_, fp, _) in _calls(dec.raw)
        if fp is not None
    }


# ------------------------------------------------------------------------ the run


def _within(data: Mapping[str, list[Decider]], v: Variant) -> dict[str, list[Decider]]:
    return {k: x for k, x in data.items() if v.datasets is None or k in v.datasets}


def run_variant(
    data: Mapping[str, list[Decider]],
    spec: Spec,
    v: Variant,
    seen: Mapping[str, set[str]],
    settings: Settings,
) -> dict[str, Any]:
    ctx = Ctx(v, seen, settings)
    sub = _within(data, v)
    out: dict[str, Any] = {"key": v.key, "title": v.title}
    if "Q1" in v.questions:
        out["Q1"] = q1(sub, ctx)
    if "Q1b" in v.questions:
        out["Q1b"] = q1b(sub, ctx)
    if "Q2" in v.questions or "Q3" in v.questions:
        configs = single_configs(sub, ctx)
        casc = cascades(sub, spec, ctx)
        rows = configs + casc
        certified = [r for r in rows if r["certified"]]
        for r in rows:
            if not r["certified"]:
                r["violation"]["why"] = "not certified (coverage 0)"
        if "Q2" in v.questions:
            out["Q2"] = {
                "configs": rows,
                "family": _family("F2", certified, "violation"),
                "n_above_alpha": sum(r["above_alpha"] for r in rows),
                "not_certified": [
                    _label(r) + f" ({r['signal']})" for r in rows if not r["certified"]
                ],
            }
        if "Q3" in v.questions:
            out["Q3"] = {"cascades": casc}
    if "Q4" in v.questions:
        out["Q4"] = q4(sub, ctx)
    return out


def analyze(spec: Spec, *, settings: Settings = SETTINGS) -> dict[str, Any]:
    """Everything in the plan, in its order. Raises NotReadyError before reading any decision
    if a run is missing."""
    chosen = select(spec)
    data = load(spec, chosen)
    seen = load_seen(spec.seen_ids)
    primary = run_variant(data, spec, PRIMARY, seen, settings)
    ctx = Ctx(PRIMARY, seen, settings)
    d6 = drift(data)
    result: dict[str, Any] = {
        "plan": PLAN,
        "spec": spec.label,
        "settings": settings,
        "notes": dict(spec.notes),
        "deviations": list(spec.deviations),
        "runs": {str(k): v.name for k, v in sorted(chosen.items())},
        "primary": primary,
        "sensitivity": [run_variant(data, spec, v, seen, settings) for v in SENSITIVITY],
        "S6": {**d6, "splits": drift_splits(data, d6, ctx)},
        "exploratory": {
            "alpha_grid": alpha_grid(data, ctx),
            "coverage_differences": coverage_differences(data, ctx),
            "transfer_self_consistency": q4(data, ctx, signal="self_consistency"),
            "transfer_from_classic": q4(data, ctx, signal="classic"),
            "judge": {
                d.name: judge.describe(judge.judge_report(d.slots["test"].scored))
                for d in data.get("mt_bench", [])
                if d.kind == "llm"
            },
        },
    }
    return to_jsonable(result)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------- Markdown


def _f(x: Any, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _pct(x: Any) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _ci(lo: Any, hi: Any, fmt: Callable[[Any], str] = _f) -> str:
    return "—" if lo is None or hi is None else f"[{fmt(lo)}, {fmt(hi)}]"


def _est(m: Mapping[str, Any], fmt: Callable[[Any], str] = _f, key: str = "estimate") -> str:
    return f"{fmt(m[key])} {_ci(m['low'], m['high'], fmt)}"


def _cip(pair: Any, fmt: Callable[[Any], str] = _f) -> str:
    return "" if not pair else _ci(pair[0], pair[1], fmt)


def _table(head: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _fam(f: Mapping[str, Any]) -> str:
    text = (
        f"Family {f['name']}: {f['size']} test(s), Holm at FWER {f['fwer']:g}; "
        f"{f['rejected']} rejected."
    )
    if f["dropped"]:
        text += " Dropped: " + "; ".join(f"{d['test']} ({d['reason']})" for d in f["dropped"]) + "."
    return text


def _pairs_md(q: Mapping[str, Any]) -> list[str]:
    """The tested excess-ECE table, then the untested diagnostics (Amendment 2)."""
    rows, diag = [], []
    for p in q["pairs"]:
        t = p["excess"]
        verdict = (
            ("separated: " + (p["a"] if (t["estimate"] or 0) < 0 else p["b"]) + " better")
            if t.get("rejected")
            else ("practically equivalent (±0.02)" if p["equivalent"] else "not separated")
        )
        nf_a, nf_b = p["noise_floor"]
        acc = p["accuracy"] or [None, None]
        rows.append(
            [
                p["dataset"],
                f"{p['a']} vs {p['b']}",
                str(p["n"]),
                f"{_f(acc[0])} / {_f(acc[1])}",
                _f(t["estimate"]),
                _ci(t["low"], t["high"]),
                _f(t["p"], 4),
                _f(t.get("p_holm"), 4),
                _est(p["raw_ece_gap"]),
                f"{_f(nf_a['mean'])} ({_f(nf_a['p95'])}) / {_f(nf_b['mean'])} ({_f(nf_b['p95'])})",
                verdict,
            ]
        )
        d = p["diagnostics"]
        flags = ", ".join(
            x
            for x, on in (
                ("DA reversal", d["da_reversal"]),
                ("IA reversal", d["ia_reversal"]),
                ("ESS/N < 0.5", d["ess_flag"]),
            )
            if on
        )
        diag.append(
            [
                p["dataset"],
                f"{p['a']} vs {p['b']}",
                _est(d["da_ece_gap"]),
                _est(d["ia_ece_gap"]),
                _f(d["ess_ratio"], 2),
                _est(p["brier_gap"]),
                flags or "—",
            ]
        )
    return [
        *_table(
            [
                "dataset",
                "A vs B",
                "n",
                "accuracy A / B",
                "excess ECE gap",
                "95% CI",
                "p",
                "p (Holm)",
                "raw ECE gap [95% CI]",
                "noise floor A / B: mean (p95)",
                "result",
            ],
            rows,
        ),
        "",
        "Diagnostics, not tests (Amendment 2). Under DA a perfectly calibrated decider "
        "looks miscalibrated whenever the two accuracies differ: in a simulation of two "
        "perfectly calibrated deciders at accuracy 0.800 vs 0.626, raw +0.000, DA +0.159, "
        "IA −0.144 (A − B). DA penalises the more accurate decider and IA favours it. The "
        'paper (arXiv 2606.30814 v2 §4.2) says DA "aligns only the overall distribution '
        'of correct and incorrect outcomes" and that the views are read jointly. The raw '
        "Brier gap is descriptive: it depends on accuracy.",
        "",
        *_table(
            [
                "dataset",
                "A vs B",
                "DA ECE gap [95% CI]",
                "IA ECE gap [95% CI]",
                "DA ESS/N",
                "raw Brier gap",
                "flags",
            ],
            diag,
        ),
    ]


def _per_md(per: Mapping[str, Any]) -> list[str]:
    rows = []
    for k, r in per.items():
        m = r["metrics"]
        nf = r["noise_floor"]["ece_width"]
        rows.append(
            [
                k,
                str(r["n"]),
                str(r["n_errors"]),
                str(r["n_unique_confidences"]),
                _est(m["accuracy"]),
                _f(m["ece_width"]["estimate"]),
                f"{_f(nf['mean'])} / {_f(nf['p95'])}",
                _est(r["excess_ece"]),
                _f(m["ece_mass"]["estimate"]),
                _f(m["brier"]["estimate"]),
                _f(m["aurc"]["estimate"]),
            ]
        )
    return _table(
        [
            "decider",
            "n",
            "errors",
            "unique conf.",
            "accuracy [95% CI]",
            "raw ECE (width)",
            "ECE noise floor mean / p95",
            "excess ECE [95% interval]",
            "ECE mass",
            "Brier",
            "AURC",
        ],
        rows,
    )


def _configs_md(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    out = []
    for r in rows:
        lam = ", ".join("none" if t is None else f"{t:g}" for t in r["thresholds"])
        floor = r["floor"]
        v = r["violation"]
        out.append(
            [
                r["dataset"],
                r["name"],
                r["signal"],
                f"{r['alpha']:g} / {r['delta']:g}",
                lam,
                "—"
                if floor is None
                else f"{_pct(floor['plug_in'])} ({_pct(floor['conservative'])})",
                f"{r['n_act']}/{r['n_test']}",
                f"{_pct(r['test_coverage'])} {_cip(r['test_coverage_ci'], _pct)}",
                f"{_f(r['test_risk'])} {_cip(r['test_risk_ci'])}",
                "yes" if r["above_alpha"] else "no",
                _f(v.get("p"), 4) if v.get("p") is not None else (v.get("why") or "—"),
                _f(v.get("p_holm"), 4),
            ]
        )
    return _table(
        [
            "dataset",
            "decider",
            "signal",
            "α / δ",
            "λ̂",
            "floor (cons.)",
            "acted/n",
            "test coverage [95% CI]",
            "test risk [95% CI]",
            "risk > α",
            "p (violation)",
            "p (Holm)",
        ],
        out,
    )


def _q4_md(q: Mapping[str, Any]) -> list[str]:
    rows = []
    for p in q["pairs"]:
        t, nat, vs = p["transfer"], p["native"], p["vs_native"]
        rows.append(
            [
                p["dataset"],
                f"{p['a']} → {p['b']}",
                _f(p["threshold_a"], 2),
                _f(p["threshold_b"], 2),
                f"{t['n_act']}/{t['n_test']}",
                _f(t["test_risk"]),
                _cip(t["test_risk_ci"]),
                _pct(t["test_coverage"]),
                _pct(nat["test_coverage"]),
                "—" if not vs else _est(vs["coverage"], _pct, "diff"),
                _f(p["violation"].get("p"), 4),
                _f(p["violation"].get("p_holm"), 4),
            ]
        )
    return _table(
        [
            "dataset",
            "A → B",
            "λ̂_A",
            "λ̂_B",
            "acted/n",
            "risk on B",
            "95% CI",
            "coverage (λ̂_A)",
            "coverage (λ̂_B)",
            "Δ coverage [95% CI, thresholds fixed]",
            "p (violation)",
            "p (Holm)",
        ],
        rows,
    )


def _variant_md(v: Mapping[str, Any], notes: Mapping[str, str], level: str = "##") -> list[str]:
    lines: list[str] = [f"{level} {v['title']}", ""]
    if "Q1" in v:
        lines += [
            f"{level}# Q1. Which decider is best calibrated?",
            "",
            "Paired gap in excess ECE (A − B; negative: A better calibrated), tested by a "
            "centered paired bootstrap (Amendment 2). Excess ECE is raw top-label ECE (15 "
            "equal-width bins) minus the decider's own noise-floor mean, so the test asks "
            "whether the ECE gap is beyond what the two floors explain. Every interval is "
            "the one that inverts the centered p-value: it excludes 0 exactly when p ≤ 0.05. "
            "The raw ECE gap is descriptive.",
            "",
        ]
        lines += [*_pairs_md(v["Q1"]), "", _fam(v["Q1"]["family"]), ""]
        lines += [
            "Per decider (descriptive). Raw ECE is biased upward on finite samples, more "
            "so for small n and many distinct confidence values: a perfectly calibrated "
            "decider's expected ECE is its noise-floor mean, not 0. Excess ECE is raw ECE "
            "minus that mean; near 0 is consistent with perfect calibration. Accuracy CIs "
            "are percentile intervals; excess-ECE intervals invert the centered p-value.",
            "",
            *_per_md(v["Q1"]["per_decider"]),
            "",
        ]
    if "Q1b" in v:
        lines += [f"{level}# Q1b. Self-consistency vs verbalized (same model, same examples)", ""]
        lines += [*_pairs_md(v["Q1b"]), "", _fam(v["Q1b"]["family"]), ""]
    if "Q2" in v:
        q = v["Q2"]
        lines += [
            f"{level}# Q2. Do the pre-registered guarantees hold on the test halves?",
            "",
            "A rejection is evidence that a guarantee failed on this data; a "
            "non-rejection is not proof that it holds.",
            "",
        ]
        lines += [
            *_configs_md(q["configs"]),
            "",
            _fam(q["family"]),
            f"Point estimates above α: {q['n_above_alpha']} of {len(q['configs'])} "
            "(expected now and then even when every guarantee holds).",
            "Not certified (coverage 0): " + (", ".join(q["not_certified"]) or "none") + ".",
            "",
        ]
    if "Q3" in v:
        lines += [
            f"{level}# Q3. How much can be automated at the target?",
            "",
            "Coverage is in the Q2 table. Cascades:",
            "",
        ]
        rows = []
        for c in v["Q3"]["cascades"]:
            b = c["baselines"]
            rnd = b.get("random_equal_cost", {})
            rows.append(
                [
                    c["dataset"],
                    c["name"],
                    _pct(c["shares_named"]["act"]),
                    _pct(c["shares_named"]["ask"]),
                    _pct(c["shares_named"]["escalate"]),
                    _f(c["test_risk"]),
                    f"${c['cost_per_decision_usd']:.6f}",
                    f"{_f(b['always_cheap']['risk'])} / {_f(b['always_strong']['risk'])}",
                    f"{_f(rnd.get('risk_mean'))} {_cip(rnd.get('risk_95'))}",
                ]
            )
        lines += [
            *_table(
                [
                    "dataset",
                    "cascade",
                    "act",
                    "ask",
                    "escalate",
                    "risk (acted)",
                    "cost / decision (list)",
                    "always cheap / strong risk",
                    "random at equal cost: risk [95%]",
                ],
                rows,
            ),
            "",
        ]
    if "Q4" in v:
        lines += [f"{level}# Q4. Do thresholds transfer across models?", ""]
        lines += [*_q4_md(v["Q4"]), "", _fam(v["Q4"]["family"]), ""]
    return lines


def to_markdown(r: Mapping[str, Any]) -> str:
    lines = [
        f"# Study results ({r['spec']})",
        "",
        f"Generated by scripts/analyze_study.py from the plan {r['plan']}. Seeds fixed "
        f"({r['settings']['n_boot']:,} bootstrap resamples, seed {r['settings']['seed']}). "
        "Output of the script, unedited; interpretation is in a separate commit. Ids and "
        "aggregates only.",
        "",
    ]
    if r["notes"]:
        lines += ["**Notes that go with every headline:**", ""]
        lines += [f"- {k}: {v}" for k, v in r["notes"].items()] + [""]
    lines += ["**Deviations from the plan:** " + ("; ".join(r["deviations"]) or "none") + ".", ""]
    lines += ["<details><summary>Runs used</summary>", ""]
    lines += [f"- `{Path(k).name}` → `{v}`" for k, v in r["runs"].items()] + ["", "</details>", ""]
    lines += _variant_md(r["primary"], r["notes"])
    lines += ["## Sensitivity analyses", ""]
    for v in r["sensitivity"]:
        lines += _variant_md(v, r["notes"], level="###")
    s6 = r["S6"]
    lines += ["### S6: model drift", "", f"Rule: {s6['rule']}.", ""]
    for model, m in s6["models"].items():
        if not m["has_fingerprints"]:
            lines.append(
                f"- {model}: no system_fingerprint in the saved responses "
                "(limitation: drift can't be checked)."
            )
            continue
        ch = (
            ", ".join(
                f"{ds}: {', '.join(c['fingerprints'])} from {c['day']}"
                for ds, c in m["changes"].items()
            )
            or "no change"
        )
        lines.append(f"- {model}: {len(m['first_seen'])} fingerprints; {ch}.")
        for run in m["runs"]:
            days = ", ".join(f"{d} ({n})" for d, n in run["days"].items())
            lines.append(f"  - {run['dataset']} {run['slot']}: calls on {days or '—'}")
    lines += [
        "",
        "Fingerprint mix, calibration half vs test half (descriptive, Amendment 2; "
        "TV = total-variation distance between the two mixes, 0 = identical, 1 = disjoint):",
        "",
    ]
    mix_rows = []
    for model, m in s6["models"].items():
        for ds, x in m.get("mix", {}).items():
            moved = sorted(x["shares"].items(), key=lambda kv: -abs(kv[1][1] - kv[1][0]))[:3]
            mix_rows.append(
                [
                    model,
                    ds,
                    f"{x['n_calls']['calibration']} / {x['n_calls']['test']}",
                    _f(x["tv_distance"]),
                    ", ".join(x["only_calibration"]) or "—",
                    ", ".join(x["only_test"]) or "—",
                    "; ".join(f"{fp} {_pct(c)} → {_pct(t)}" for fp, (c, t) in moved),
                ]
            )
    lines += _table(
        [
            "model",
            "dataset",
            "calls cal / test",
            "TV",
            "only in calibration",
            "only in test",
            "largest share changes",
        ],
        mix_rows,
    )
    for sp in s6["splits"]:
        lines += ["", f"{sp['model']} on {sp['dataset']}, change on {sp['change_day']}:", ""]
        lines += _table(
            ["period", "n", "accuracy", "coverage", "risk [95% CI]"],
            [
                [
                    k,
                    str(p["n"]),
                    _f(p["accuracy"]),
                    _pct(p["coverage"]),
                    f"{_f(p['risk'])} {_cip(p['risk_ci'])}",
                ]
                for k, p in sp["periods"].items()
            ],
        )
        lines += ["", "Restricted to fingerprints seen in both halves (Q2):", ""]
        lines += _configs_md(sp["q2_restricted"])
    lines += ["", f"Limitation: {s6['limitation']}.", ""]
    e = r["exploratory"]
    lines += [
        "## Exploratory (not pre-registered as tests; unadjusted)",
        "",
        "### Coverage over α (not the pre-registered target)",
        "",
    ]
    lines += _table(
        ["dataset", "decider", "α", "λ̂", "test coverage", "test risk"],
        [
            [
                x["dataset"],
                x["name"],
                f"{x['alpha']:g}",
                _f(x["threshold"], 3),
                _pct(x["coverage"]),
                _f(x["risk"]),
            ]
            for x in e["alpha_grid"]
        ],
    )
    lines += ["", "### Coverage differences (thresholds refitted per resample; approximate)", ""]
    lines += _table(
        ["dataset", "A − B", "Δ coverage", "95% CI"],
        [
            [
                x["dataset"],
                f"{x['a']} − {x['b']}",
                _pct(x["coverage_diff"]),
                _ci(x["low"], x["high"], _pct),
            ]
            for x in e["coverage_differences"]
        ],
    )
    for key, title in (
        ("transfer_self_consistency", "Transfer, self-consistency signal"),
        ("transfer_from_classic", "Transfer from the classic baseline"),
    ):
        lines += ["", f"### {title}", "", *_q4_md(e[key])]
        if e[key]["dropped"]:
            lines += [
                "",
                "Dropped: " + "; ".join(f"{d['test']} ({d['reason']})" for d in e[key]["dropped"]),
            ]
    if e["judge"]:
        lines += ["", "### MT-Bench judge statistics (vote level)", ""]
        for name, desc in e["judge"].items():
            lines += [f"**{name}**", "", *[f"    {x}" for x in desc], ""]
    return "\n".join(lines).rstrip() + "\n"

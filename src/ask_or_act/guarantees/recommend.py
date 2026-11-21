"""``ask-or-act thresholds``: recommend act / ask / escalate thresholds from a calibration
run, and check them on a held-out test run.

Order of operations (each step uses only what it is allowed to see):

1. **Feasibility first** (Kotte v2 Proposition 3, metrics/feasibility.py), on the
   calibration half: if the base risk μ̂ exceeds α, *any* valid rule must escalate at
   least (μ̂ − α)/(1 − α) of traffic. Reported before searching.
2. **Threshold** by the fixed-sequence test (fixed_sequence.py, ToE Theorem 1) on the
   calibration half only. One stage: act / escalate. Several stages (cheap → strong):
   the cascade (cascade.py, ToE Algorithm 2): act / ask the stronger model / escalate.
3. **Check on the test half**, which never influenced the threshold: coverage and
   observed risk with a 95% Clopper–Pearson interval. Observed risk above α on one
   test half doesn't disprove a 1 − δ guarantee, but it is worth a look.

The grid comes from the run's **config** (its confidence signal and number of
samples), not from its data, so it is fixed before the calibration labels are seen.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from ask_or_act.core.runs import RunRecord
from ask_or_act.guarantees.binomial import two_sided_interval
from ask_or_act.guarantees.cascade import Stage, calibrate, route
from ask_or_act.guarantees.fixed_sequence import DEFAULT_STARTS, grid_for
from ask_or_act.guarantees.mt_bench import HumanAgreement, by_comparison, human_agreement
from ask_or_act.metrics.data import Scored, align
from ask_or_act.metrics.feasibility import Feasibility, min_abstention
from ask_or_act.metrics.selective import n_unique


def _table(value: object, key: str) -> dict[str, object]:
    inner = value.get(key) if isinstance(value, dict) else None
    return inner if isinstance(inner, dict) else {}


def run_params(record: RunRecord) -> dict[str, object]:
    """The decider params from the config saved in run.json."""
    contents = _table(record.metadata, "config")
    return _table(_table(_table(contents, "contents"), "decider"), "params")


@dataclass(frozen=True)
class Registered:
    alpha: float | None
    delta: float | None
    registered: str | None


def run_guarantee(record: RunRecord) -> Registered | None:
    """The ``[guarantee]`` table of the config saved in run.json, if any."""
    g = _table(_table(_table(record.metadata, "config"), "contents"), "guarantee")
    if not g:
        return None

    def num(key: str) -> float | None:
        v = g.get(key)
        return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None

    reg = g.get("registered")
    return Registered(num("alpha"), num("delta"), reg if isinstance(reg, str) else None)


def run_split(record: RunRecord) -> str | None:
    split = _table(record.metadata, "dataset").get("split")
    return split if isinstance(split, str) else None


@dataclass(frozen=True)
class GridChoice:
    grid: tuple[float, ...]
    starts: tuple[float, ...]
    text: str


def grid_for_run(record: RunRecord, signal: str | None) -> GridChoice:
    """Grid and scan starts from the run's **config**, never its data: verbalized →
    percent steps; self_consistency → k + 1 steps; otherwise (probabilities, baselines)
    0.001 steps."""
    params = run_params(record)
    used = signal or str(params.get("confidence_signal", "")) or None
    if used == "self_consistency":
        k = int(str(params.get("samples", 0)))
        return GridChoice(
            grid_for(used, samples=k),
            DEFAULT_STARTS["self_consistency"],
            f"self-consistency, k = {k}: {k + 1} steps",
        )
    if used == "verbalized":
        return GridChoice(grid_for(used), DEFAULT_STARTS["verbalized"], "verbalized %")
    return GridChoice(grid_for(None), DEFAULT_STARTS["continuous"], "continuous, 0.001 steps")


@dataclass(frozen=True)
class StageReport:
    name: str
    threshold: float | None
    grid: str
    n_unique_cal: int
    cal_coverage: float
    cal_risk: float | None
    test_share: float | None = None


@dataclass(frozen=True)
class Recommendation:
    alpha: float
    delta: float
    dataset: str
    n_cal: int
    feasibility: Feasibility  # of the first (cheapest) stage on the calibration half
    stages: tuple[StageReport, ...]
    cal_coverage: float
    cal_risk: float | None
    n_test: int | None = None
    test_coverage: float | None = None
    test_risk: float | None = None
    test_risk_ci: tuple[float, float] | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    delta_floor: float = 0.1
    unit: str = "example"  # MT-Bench: "comparison" (primary) or "vote" (sensitivity)
    unit_note: str = ""
    human: HumanAgreement | None = None  # set when α is relative to the human ceiling
    human_margin: float = 0.0


def recommend(
    cal: Sequence[tuple[str, Scored, GridChoice]],
    test: Sequence[Scored] | None = None,
    *,
    alpha: float | None,
    delta: float,
    delta_floor: float | None = None,
    disjoint: bool = False,
    mt_bench_unit: str = "comparison",
    alpha_vs_human: float | None = None,
) -> Recommendation:
    """``cal``: one (name, calibration run, grid choice) per stage, cheapest first.
    ``test``: the same stages' test-half runs, in the same order.

    * ``delta_floor``: the feasibility floor's own δ (default: ``delta``). Floor and
      threshold are separate statements; a joint one holds with probability
      ≥ 1 − (δ_floor/2 + δ), the failure probabilities summed.
    * ``mt_bench_unit``: "comparison" (pre-registered, guarantees/mt_bench.py) or
      "vote" (the sensitivity check).
    * ``alpha_vs_human``: set α = 1 − h + margin, where h is the human ceiling on the
      calibration half's votes (MT-Bench only); ``alpha`` must then be None.
    """
    if (alpha is None) == (alpha_vs_human is None):
        raise ValueError("give exactly one of alpha and alpha_vs_human")
    if mt_bench_unit not in ("comparison", "vote"):
        raise ValueError("mt_bench_unit must be comparison or vote")
    names = [c[0] for c in cal]
    scored = [c[1] for c in cal]
    human: HumanAgreement | None = None
    if alpha_vs_human is not None:
        if scored[0].dataset != "mt_bench":
            raise ValueError("a target relative to the human ceiling needs MT-Bench votes")
        human = human_agreement(scored[0])
        alpha = human.alpha + alpha_vs_human
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"human-relative α = {alpha:.3f} is outside (0, 1)")
    assert alpha is not None
    unit, unit_note = "example", ""
    if scored[0].dataset == "mt_bench":
        unit = mt_bench_unit
        if unit == "comparison":
            units = [by_comparison(x) for x in scored]
            u = units[0]
            unit_note = (
                f"one example per comparison with the majority human label: {u.n_votes} "
                f"votes → {u.n_comparisons} comparisons, {u.n_no_majority} excluded (no "
                "strict majority)"
            )
            scored = [x.scored for x in units]
            test = None if test is None else [by_comparison(t).scored for t in test]
        else:
            unit_note = "every vote an example (sensitivity check; votes are not independent)"
    for other in scored[1:]:  # same calibration examples for every stage
        align(scored[0], other)
    base = scored[0]
    aligned = [base, *(align(base, s)[1] for s in scored[1:])]
    stages = [
        Stage(n, s.confidence, s.correct, g.grid, g.starts)
        for n, s, (_, _, g) in zip(names, aligned, cal, strict=True)
    ]
    warnings: list[str] = []
    d_floor = delta if delta_floor is None else delta_floor
    feas = min_abstention(1.0 - float(base.correct.mean()), alpha, n=len(base), delta=d_floor)
    if not feas.feasible_without_abstention:
        warnings.append(
            f"{names[0]}: base risk {feas.base_risk:.3f} > α = {alpha:.3g}: any valid rule "
            f"must escalate ≥ {feas.min_abstention:.1%} (Kotte v2 Prop. 3)"
        )
    if base.dataset == "mt_bench":
        n_groups = len(set(base.groups))
        what = "votes" if unit == "vote" else "comparisons"
        warnings.append(
            f"MT-Bench {what} on one question are not independent ({len(base)} {what}, "
            f"{n_groups} questions): the binomial test assumes i.i.d. examples, so this "
            "guarantee is optimistic"
            + (" (less so per comparison than per vote)" if unit == "comparison" else "")
        )
    fit = calibrate(stages, alpha=alpha, delta=delta, ids=list(base.ids), disjoint=disjoint)
    cal_routed = route(stages, fit.thresholds)
    reports = []
    for i, (st, f, c, s) in enumerate(zip(stages, fit.fits, cal, aligned, strict=True)):
        reports.append(
            StageReport(
                st.name,
                f.threshold,
                f"{c[2].text}; starts {', '.join(f'{x:g}' for x in st.starts)}",
                n_unique(s.confidence),
                cal_routed.share(i),
                cal_routed.stage_risk(i),
            )
        )
        if f.threshold is None:
            firsts = [
                f"{x.steps[0].threshold:g}: {x.steps[0].errors}/{x.steps[0].n} wrong"
                for x in f.scans
                if x.steps
            ]
            warnings.append(
                f"{st.name}: nothing certified at α = {alpha:.3g}; every scan failed its first "
                f"test ({'; '.join(firsts) or 'no examples reached this stage'})"
            )
    rec = Recommendation(
        alpha,
        delta,
        base.dataset,
        len(base),
        feas,
        tuple(reports),
        cal_routed.coverage(),
        cal_routed.risk(),
        warnings=tuple(warnings),
        delta_floor=d_floor,
        unit=unit,
        unit_note=unit_note,
        human=human,
        human_margin=alpha_vs_human or 0.0,
    )
    if test is None:
        return rec
    t0 = test[0]
    t_aligned = [t0, *(align(t0, t)[1] for t in test[1:])]
    t_stages = [
        Stage(st.name, t.confidence, t.correct, st.grid, st.starts)
        for st, t in zip(stages, t_aligned, strict=True)
    ]
    routed = route(t_stages, fit.thresholds)
    acted = routed.stage >= 0
    errors = int((acted & ~routed.correct).sum())
    n_acted = int(acted.sum())
    return Recommendation(
        **{
            **rec.__dict__,
            "stages": tuple(
                StageReport(**{**r.__dict__, "test_share": routed.share(i)})
                for i, r in enumerate(reports)
            ),
            "n_test": len(t0),
            "test_coverage": routed.coverage(),
            "test_risk": routed.risk(),
            "test_risk_ci": two_sided_interval(errors, n_acted) if n_acted else None,
        }
    )


def describe(rec: Recommendation) -> list[str]:
    a, d = rec.alpha, rec.delta
    f = rec.feasibility
    lines = []
    if rec.human is not None:
        h = rec.human
        lines.append(
            f"human ceiling (calibration votes): a typical human agrees with the others' "
            f"majority {h.rate:.1%} of the time ({h.n_votes_used} of {h.n_votes} votes "
            f"usable) → α = 1 − h{f' + {rec.human_margin:g}' if rec.human_margin else ''} "
            f"= {a:.3f}"
        )
    lines += [
        f"target: risk ≤ {a:.3g} on acted-on examples (calibration n = {rec.n_cal}, {rec.dataset})",
        *([f"unit: {rec.unit_note}"] if rec.unit_note else []),
        f"[statement 1, own δ] feasibility (Kotte v2 Prop. 3): base risk "
        f"{f.base_risk:.3f}; must escalate ≥ {f.min_abstention:.1%} (plug-in), "
        f"≥ {f.min_abstention_conservative:.1%} with probability ≥ {1 - f.delta / 2:.3f} "
        f"(finite-sample correction, App. E.4, δ_floor = {f.delta})",
        f"[statement 2, own δ] threshold (ToE Thm 1): risk ≤ {a:.3g} on acted-on "
        f"examples with probability ≥ {1 - d:.3f} (δ = {d})",
        f"  (both at once: probability ≥ {1 - d - f.delta / 2:.3f}, the failure "
        "probabilities summed)",
    ]
    tiers = ["act"] + ["ask"] * (len(rec.stages) - 1)
    for tier, s in zip(tiers, rec.stages, strict=True):
        lam = "never (no threshold certified)" if s.threshold is None else f"≥ {s.threshold:g}"
        test = "" if s.test_share is None else f", test {s.test_share:.1%}"
        lines.append(
            f"  {tier}: {s.name} when confidence {lam} [{s.grid}; {s.n_unique_cal} distinct "
            f"values] → calibration {s.cal_coverage:.1%}{test}"
        )
    lines.append(
        f"  escalate: everything else → calibration {1 - rec.cal_coverage:.1%}"
        + ("" if rec.test_coverage is None else f", test {1 - rec.test_coverage:.1%}")
    )
    if rec.cal_risk is not None:
        lines.append(f"calibration: coverage {rec.cal_coverage:.1%}, risk {rec.cal_risk:.3f}")
    if rec.test_coverage is not None:
        risk = "n/a" if rec.test_risk is None else f"{rec.test_risk:.3f}"
        ci = rec.test_risk_ci
        ci_text = "" if ci is None else f" [95% CI {ci[0]:.3f}, {ci[1]:.3f}]"
        held = "" if rec.test_risk is None else ("  ≤ α" if rec.test_risk <= a else "  > α")
        lines.append(
            f"test (held out, n = {rec.n_test}): coverage {rec.test_coverage:.1%}, "
            f"risk {risk}{ci_text}{held}"
        )
    lines += [f"warning: {w}" for w in rec.warnings]
    return lines

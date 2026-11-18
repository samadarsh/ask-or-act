"""Assemble metric reports for one run, or a paired comparison of two runs.

Every scalar metric is a function of example indices (a ``Stat``). That one shape
serves the point estimate (all indices), bootstrap CIs (resampled indices) and paired
tests (the same resampled indices for both runs).
"""

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import numpy as np

from ask_or_act.metrics import ace, mt_bench
from ask_or_act.metrics.calibration import (
    DEFAULT_BINS,
    brier_top_label,
    ece,
    ece_noise_floor,
    reliability,
)
from ask_or_act.metrics.classification import cohen_kappa, confusion_matrix, label_set, macro_f1
from ask_or_act.metrics.data import ERROR_LABEL, IntArray, Scored, align
from ask_or_act.metrics.feasibility import min_abstention
from ask_or_act.metrics.selective import aurc, auroc, excess_aurc, n_unique, risk_coverage
from ask_or_act.metrics.stats import (
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    Stat,
    bootstrap_ci,
    holm,
    mcnemar,
    paired_bootstrap,
    resamples,
)

LOWER_IS_BETTER = frozenset(
    {"ece_width", "ece_mass", "brier", "brier_multiclass", "aurc", "excess_aurc"}
)
# Metrics in the multiple-comparison family; mean_confidence describes, it doesn't score.
FAMILY = (
    "accuracy",
    "macro_f1",
    "cohen_kappa",
    "ece_width",
    "ece_mass",
    "brier",
    "brier_multiclass",
    "aurc",
    "excess_aurc",
    "auroc",
)


@dataclass(frozen=True)
class Settings:
    n_bins: int = DEFAULT_BINS
    n_boot: int = DEFAULT_N_BOOT
    seed: int = DEFAULT_SEED
    level: float = 0.95
    alphas: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
    delta: float = 0.05


def _groups(s: Scored) -> tuple[str, ...] | None:
    """Cluster bootstrap only where rows are correlated (MT-Bench: per question)."""
    return s.groups if s.dataset == "mt_bench" else None


def stats_for(s: Scored, n_bins: int) -> dict[str, Stat]:
    gold, pred = np.array(s.gold), np.array(s.pred)
    c, z = s.confidence, s.correct
    fns: dict[str, Stat] = {
        "accuracy": lambda i: float(z[i].mean()),
        "macro_f1": lambda i: macro_f1(gold[i], pred[i]),
        "cohen_kappa": lambda i: cohen_kappa(gold[i], pred[i]),
        "ece_width": lambda i: ece(c[i], z[i], n_bins=n_bins, strategy="width"),
        "ece_mass": lambda i: ece(c[i], z[i], n_bins=n_bins, strategy="mass"),
        "brier": lambda i: brier_top_label(c[i], z[i]),
        "aurc": lambda i: aurc(c[i], z[i]),
        "excess_aurc": lambda i: excess_aurc(c[i], z[i]),
        "auroc": lambda i: auroc(c[i], z[i]),
        "mean_confidence": lambda i: float(c[i].mean()),
    }
    # Multiclass Brier, when every decision has a probability vector: precompute
    # per-example terms so resampling is a cheap mean.
    if all(s.probabilities) and all(set(p) <= set(s.options) for p in s.probabilities):
        terms = np.array(
            [
                sum((p.get(o, 0.0) - (o == g)) ** 2 for o in s.options)
                for p, g in zip(s.probabilities, s.gold, strict=True)
            ]
        )
        fns["brier_multiclass"] = lambda i: float(terms[i].mean())
    return fns


def _ci_dict(ci: Any) -> dict[str, Any]:
    return {"estimate": ci.estimate, "low": ci.low, "high": ci.high}


# Metrics an errored decision distorts: "wrong with confidence 0" is perfectly calibrated
# and perfectly ranked, so it flatters ECE, Brier and the selective metrics.
CONFIDENCE_METRICS = (
    "ece_width",
    "ece_mass",
    "brier",
    "brier_multiclass",
    "aurc",
    "excess_aurc",
    "auroc",
    "mean_confidence",
)


def _metrics_with_ci(
    s: Scored, settings: Settings, only: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    kw = {"n_boot": settings.n_boot, "seed": settings.seed, "level": settings.level}
    return {
        name: _ci_dict(bootstrap_ci(fn, len(s), groups=_groups(s), **kw))  # type: ignore[arg-type]
        for name, fn in stats_for(s, settings.n_bins).items()
        if only is None or name in only
    }


def noise_floors(s: Scored, settings: Settings) -> dict[str, dict[str, float]]:
    out = {}
    for strategy in ("width", "mass"):
        nf = ece_noise_floor(
            s.confidence, n_bins=settings.n_bins, strategy=strategy, seed=settings.seed
        )
        out[f"ece_{strategy}"] = {"mean": nf.mean, "p95": nf.p95, "n_sim": nf.n_sim}
    return out


def single_report(s: Scored, settings: Settings = Settings()) -> dict[str, Any]:
    n = len(s)
    metrics = _metrics_with_ci(s, settings)
    rc = risk_coverage(s.confidence, s.correct)
    labels = label_set(s.gold, s.pred)
    base_risk = 1.0 - float(s.correct.mean())
    report: dict[str, Any] = {
        "dataset": s.dataset,
        "n": n,
        "n_errors": s.n_errors,
        "n_unique_confidences": n_unique(s.confidence),
        "settings": asdict(settings),
        "metrics": metrics,
        "noise_floor": noise_floors(s, settings),
        "reliability": {
            strategy: [
                asdict(b)
                for b in reliability(
                    s.confidence, s.correct, n_bins=settings.n_bins, strategy=strategy
                )
            ]
            for strategy in ("width", "mass")
        },
        "risk_coverage": {"threshold": rc.threshold, "coverage": rc.coverage, "risk": rc.risk},
        "confusion": {"labels": labels, "matrix": confusion_matrix(s.gold, s.pred, labels)},
        "feasibility": [
            {**asdict(f), "max_coverage": f.max_coverage}
            for f in (
                min_abstention(base_risk, a, n=n, delta=settings.delta) for a in settings.alphas
            )
        ],
    }
    if s.n_errors:
        ok = s.take(np.flatnonzero(np.array(s.pred) != ERROR_LABEL).astype(np.intp))
        report["excluding_errors"] = {
            "n": len(ok),
            "note": "calibration/selective metrics on answered examples only; errors "
            "counted as wrong with confidence 0 look perfectly calibrated",
            "metrics": _metrics_with_ci(ok, settings, only=CONFIDENCE_METRICS),
            "noise_floor": noise_floors(ok, settings) if len(ok) else {},
        }
    if s.dataset == "mt_bench":
        report["mt_bench"] = mt_bench_section(s, settings)
    return report


def mt_bench_section(s: Scored, settings: Settings) -> dict[str, Any]:
    out: dict[str, Any] = {"published_human_human": mt_bench.PUBLISHED_HUMAN_HUMAN}
    for name, counts in (
        ("decider_human", mt_bench.decider_human(s)),
        ("human_human", mt_bench.human_human(s)),
    ):
        for setup in ("S1", "S2"):
            ci = bootstrap_ci(
                functools.partial(_rate, counts, setup),
                len(counts.groups),
                n_boot=settings.n_boot,
                seed=settings.seed,
                level=settings.level,
            )
            out[f"{name}_{setup}"] = {**_ci_dict(ci), "pairs": counts.n_pairs(setup)}
    return out


def _rate(counts: mt_bench.AgreementCounts, setup: str, idx: IntArray) -> float | None:
    return counts.rate(setup, idx)


def compare_report(
    a: Scored, b: Scored, settings: Settings = Settings(), names: Sequence[str] = ("A", "B")
) -> dict[str, Any]:
    a, b = align(a, b)
    n, groups = len(a), _groups(a)
    kw = {"n_boot": settings.n_boot, "seed": settings.seed, "level": settings.level}
    fa, fb = stats_for(a, settings.n_bins), stats_for(b, settings.n_bins)
    paired = {
        name: asdict(paired_bootstrap(fa[name], fb[name], n, groups=groups, **kw))  # type: ignore[arg-type]
        for name in fa
        if name in fb
    }
    mc = mcnemar(a.correct, b.correct)
    family = [m for m in FAMILY if m in paired]
    raw_p = [mc.p_value if m == "accuracy" else paired[m]["p_value"] for m in family]
    usable = [i for i, p in enumerate(raw_p) if p is not None]
    adjusted = holm([raw_p[i] for i in usable])
    tests = {}
    for k, i in enumerate(usable):
        m = family[i]
        tests[m] = {
            "test": "mcnemar_exact" if m == "accuracy" else "paired_bootstrap",
            "p_value": raw_p[i],
            "p_holm": adjusted[k],
            "better": _better(m, paired[m]["diff"], names),
        }
    return {
        "dataset": a.dataset,
        "n": n,
        "names": list(names),
        "settings": asdict(settings),
        "paired": paired,
        "mcnemar": asdict(mc),
        "tests": tests,
        "ace": ace_section(a, b, settings),
    }


def _better(metric: str, diff: float | None, names: Sequence[str]) -> str | None:
    if diff is None or diff == 0:
        return None
    a_better = diff < 0 if metric in LOWER_IS_BETTER else diff > 0
    return names[0] if a_better else names[1]


def ace_section(a: Scored, b: Scored, settings: Settings) -> dict[str, Any]:
    """ACE views with paired-bootstrap CIs on each gap, plus reversal flags (§5.2, E.6)."""
    inst = ace.candidate_instances(a, b)
    point = ace.all_views(a, b, n_bins=settings.n_bins, strategy="width", inst=inst)
    boot: dict[str, dict[str, list[float]]] = {v: {"ece": [], "brier": []} for v in point}
    idxs: list[IntArray] = resamples(
        len(a), n_boot=settings.n_boot, seed=settings.seed, groups=_groups(a)
    )
    for idx in idxs:
        for name, view in ace.all_views(
            a, b, n_bins=settings.n_bins, strategy="width", idx=idx, inst=inst
        ).items():
            if view.ece_gap is not None:
                boot[name]["ece"].append(view.ece_gap)
            if view.brier_gap is not None:
                boot[name]["brier"].append(view.brier_gap)
    lo_q, hi_q = (1 - settings.level) / 2, 1 - (1 - settings.level) / 2
    out: dict[str, Any] = {"recommended": "DA", "views": {}}
    raw_gaps = {"ece": point["raw"].ece_gap, "brier": point["raw"].brier_gap}
    for name, view in point.items():
        entry: dict[str, Any] = {
            "available": view.available,
            "note": view.note,
            "info": view.info,
            "ece_a": view.ece_a,
            "ece_b": view.ece_b,
            "brier_a": view.brier_a,
            "brier_b": view.brier_b,
        }
        for metric, gap in (("ece", view.ece_gap), ("brier", view.brier_gap)):
            samples = np.array(boot[name][metric])
            ci = (
                (None, None)
                if len(samples) == 0
                else tuple(float(x) for x in np.quantile(samples, [lo_q, hi_q]))
            )
            raw_gap = raw_gaps[metric]
            reversed_ = (
                name != "raw" and gap is not None and raw_gap is not None and gap * raw_gap < 0
            )
            stability = None
            if reversed_ and len(samples) and gap is not None:
                stability = float(np.mean(np.sign(samples) == np.sign(gap)))
            entry[f"{metric}_gap"] = {
                "estimate": gap,
                "low": ci[0],
                "high": ci[1],
                "reversal": reversed_,
                "reversal_stability": stability,
            }
        out["views"][name] = entry
    return out


# ---------------------------------------------------------------------------- output


def to_jsonable(x: Any) -> Any:
    """numpy → Python, dataclasses → dicts, NaN/inf → None (JSON has no NaN)."""
    if is_dataclass(x) and not isinstance(x, type):
        return to_jsonable(asdict(x))
    if isinstance(x, Mapping):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, np.bool_ | bool):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        f = float(x)
        return f if math.isfinite(f) else None
    return x


def _fmt(x: Any, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _metric_rows(metrics: Mapping[str, Any], floors: Mapping[str, Any], level: float) -> list[str]:
    rows = [
        f"  {'metric':<18}{'estimate':>9}  {int(level * 100)}% CI{'':<9}"
        "noise floor (perfectly calibrated: mean / 95th pct)"
    ]
    for name, m in metrics.items():
        arrow = "↓" if name in LOWER_IS_BETTER else " "
        row = f"  {name:<17}{arrow}{_fmt(m['estimate']):>9}  [{_fmt(m['low'])}, {_fmt(m['high'])}]"
        if name in floors:
            f = floors[name]
            above = m["estimate"] is not None and m["estimate"] > f["p95"]
            row += f"   {_fmt(f['mean'])} / {_fmt(f['p95'])}"
            row += "  ← above noise" if above else "  (within noise)"
        rows.append(row)
    return rows


def render_single(r: Mapping[str, Any], title: str) -> str:
    lines = [
        f"{title}  ({r['dataset']}, n={r['n']}, errors={r['n_errors']}, "
        f"unique confidences={r['n_unique_confidences']})",
        "",
    ]
    lines += _metric_rows(r["metrics"], r["noise_floor"], r["settings"]["level"])
    if "excluding_errors" in r:
        ex = r["excluding_errors"]
        lines += [
            "",
            f"  excluding {r['n_errors']} errored decision(s) (n={ex['n']}): "
            "errors as 'wrong, confidence 0' flatter these metrics",
        ]
        lines += _metric_rows(ex["metrics"], ex["noise_floor"], r["settings"]["level"])
    lines += ["", "  feasibility (Kotte v2 Prop. 3): min. abstention to reach risk ≤ α"]
    for f in r["feasibility"]:
        lines.append(
            f"    α={f['alpha']:<5} base risk {f['base_risk']:.3f} → abstain ≥ "
            f"{f['min_abstention']:.1%} (conservative* {f['min_abstention_conservative']:.1%})"
        )
    lines.append(
        "    * TODO(M3b): finite-sample correction not yet verified against the full "
        "PDF; don't rely on it"
    )
    if "mt_bench" in r:
        mb = r["mt_bench"]
        lines += ["", "  MT-Bench agreement   S1 (with ties)        S2 (without ties)"]
        for who in ("decider_human", "human_human"):
            s1, s2 = mb[f"{who}_S1"], mb[f"{who}_S2"]
            lines.append(
                f"    {who:<17}{_fmt(s1['estimate'])} [{_fmt(s1['low'])}, "
                f"{_fmt(s1['high'])}]  {_fmt(s2['estimate'])} [{_fmt(s2['low'])}, "
                f"{_fmt(s2['high'])}]"
            )
        lines.append(
            "    published human-human (Zheng et al. Table 5): "
            "turn 1 0.63 / 0.81, turn 2 0.67 / 0.82"
        )
    return "\n".join(lines)


def render_compare(r: Mapping[str, Any]) -> str:
    lines = [
        f"paired comparison, A − B  ({r['dataset']}, n={r['n']}, same examples)",
        "",
        f"  {'metric':<18}{'A':>8}{'B':>8}{'A−B':>9}  {'95% CI':<18}{'p':>8}{'p_holm':>8}  better",
    ]
    for name, p in r["paired"].items():
        t = r["tests"].get(name, {})
        arrow = "↓" if name in LOWER_IS_BETTER else " "
        lines.append(
            f"  {name:<17}{arrow}{_fmt(p['a']):>8}{_fmt(p['b']):>8}{_fmt(p['diff']):>9}  "
            f"{'[' + _fmt(p['low']) + ', ' + _fmt(p['high']) + ']':<18}"
            f"{_fmt(t.get('p_value'), 4):>8}{_fmt(t.get('p_holm'), 4):>8}  {t.get('better') or ''}"
        )
    mc = r["mcnemar"]
    lines += [
        "",
        f"  accuracy test: exact McNemar, only A right={mc['only_a_right']}, "
        f"only B right={mc['only_b_right']}",
    ]
    lines += [
        "",
        "  accuracy-controlled calibration (ACE, arXiv 2606.30814; DA is the default)",
        f"    {'view':<5}{'ECE gap A−B':>14}  {'95% CI':<18}{'Brier gap':>11}  note",
    ]
    for name, v in r["ace"]["views"].items():
        if not v["available"]:
            lines.append(f"    {name:<5}{'n/a':>14}  {'':<18}{'':>11}  {v['note']}")
            continue
        e, bgap = v["ece_gap"], v["brier_gap"]
        flag = " REVERSAL" if e["reversal"] or bgap["reversal"] else ""
        lines.append(
            f"    {name:<5}{_fmt(e['estimate'], 4):>14}  "
            f"{'[' + _fmt(e['low'], 4) + ', ' + _fmt(e['high'], 4) + ']':<18}"
            f"{_fmt(bgap['estimate'], 4):>11}  {v['note']}{flag}"
        )
    return "\n".join(lines)

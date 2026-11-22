"""``ask-or-act report RUN...``: one static HTML file with every result for a set of runs.

What it contains, per dataset:

1. **Accuracy and calibration** with 95% bootstrap CIs, and the ECE **noise floor**:
   the ECE a perfectly calibrated decider with the same confidences would show
   (metrics/calibration.py). An ECE below the floor's 95th percentile is
   indistinguishable from perfect calibration at this n.
2. **Reliability diagrams** and **risk–coverage curves** (stepwise).
3. **Guaranteed thresholds**, for every decider that has both a calibration-half and a
   test-half run. The target registered in the run's config is labelled
   PRE-REGISTERED. Everything else is labelled EXPLORATORY or SENSITIVITY: other α
   values, MT-Bench per vote, α relative to the human ceiling.
4. **Cascades** (cheap → strong, ordered by true cost per decision), for deciders that
   share calibration and test examples.
5. **Judge mode** for MT-Bench runs (judge.py): agreement against the human ceiling,
   position, length and tie behaviour.
6. **Sensitivity without pilot-seen examples** (pre-registered, docs/study.md): test-half
   results again without the ids in ``configs/pilot/seen_test_ids.json``. For MT-Bench,
   every comparison that contains a seen vote is dropped.

The input is a list of decision tables (report/table.py): one row per decision with
its id, prediction, confidence and human label. Saved runs are converted into tables;
any other source that fills one gets the same report. Nothing is called. The page holds ids and
numbers only, never input text (rule: no raw unsafe text in reports).
"""

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from html import escape
from importlib import resources
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ask_or_act import judge
from ask_or_act.guarantees.recommend import Recommendation, recommend
from ask_or_act.metrics.data import Scored, align
from ask_or_act.metrics.report import Settings, single_report
from ask_or_act.report import svg
from ask_or_act.report.table import DecisionTable, Target, load

DEFAULT_SEEN = Path("configs/pilot/seen_test_ids.json")
EXPLORATORY_ALPHAS = (0.05, 0.1, 0.2, 0.25)
DATASET_TITLES = {
    "banking77": "Banking77 routing",
    "aegis": "Aegis 2.0 safety",
    "mt_bench": "MT-Bench judging",
}


# --------------------------------------------------------------------------- loading


def load_seen(path: Path | None) -> dict[str, set[str]]:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {k: set(v["ids"]) for k, v in data.get("datasets", {}).items()}


def without_seen(s: Scored, seen: set[str]) -> Scored:
    """Drop seen examples. MT-Bench: drop every vote of a comparison with a seen vote."""
    if s.dataset == "mt_bench":
        bad = {str(m["comparison"]) for i, m in zip(s.ids, s.meta, strict=True) if i in seen}
        keep = [k for k, m in enumerate(s.meta) if str(m["comparison"]) not in bad]
    else:
        keep = [k for k, i in enumerate(s.ids) if i not in seen]
    return s.take(np.array(keep, dtype=np.intp))


# --------------------------------------------------------------------------- thresholds


@dataclass(frozen=True)
class ThresholdRow:
    names: tuple[str, ...]
    target: Target
    rec: Recommendation | None
    rec_unseen: Recommendation | None  # test half without pilot-seen examples
    error: str = ""


def _pairs(runs: Sequence[DecisionTable]) -> list[tuple[DecisionTable, DecisionTable]]:
    """(calibration run, test run) per decider: same dataset and resolved decider
    config. The latest run of each wins."""
    latest: dict[tuple[str, str, str | None], DecisionTable] = {}
    for r in sorted(runs, key=lambda r: r.order):
        latest[(r.dataset, r.decider_key, r.split)] = r
    out = []
    for (ds, key, split), cal in latest.items():
        if split == "calibration" and (ds, key, "test") in latest:
            out.append((cal, latest[(ds, key, "test")]))
    return sorted(out, key=lambda p: (p[0].dataset, p[0].cost_per_decision, p[0].name))


def _recommend(
    cal: Sequence[DecisionTable],
    test: Sequence[Scored],
    target: Target,
    **kw: Any,
) -> Recommendation:
    stages = [(r.name, r.scored, r.grid) for r in cal]
    return recommend(
        stages, list(test), alpha=kw.pop("alpha", target.alpha), delta=target.delta, **kw
    )


def threshold_rows(
    cal: Sequence[DecisionTable], test: Sequence[DecisionTable], seen: Mapping[str, set[str]]
) -> list[ThresholdRow]:
    names = tuple(r.name for r in cal)
    reg = cal[0].target
    targets: list[tuple[Target, dict[str, Any]]] = []
    if reg is not None:
        targets.append((reg, {}))
    for a in EXPLORATORY_ALPHAS:
        if reg is None or abs(a - reg.alpha) > 1e-12:
            targets.append(
                (Target(a, reg.delta if reg else 0.1, "EXPLORATORY", "not a registered target"), {})
            )
    if cal[0].dataset == "mt_bench" and reg is not None:
        targets.append(
            (
                Target(reg.alpha, reg.delta, "SENSITIVITY", "per vote, not per comparison"),
                {"mt_bench_unit": "vote"},
            )
        )
        targets.append(
            (
                Target(
                    reg.alpha,
                    reg.delta,
                    "SENSITIVITY",
                    "α = 1 − h (human ceiling, calibration votes)",
                ),
                {"alpha": None, "alpha_vs_human": 0.0},
            )
        )
    test_s = [t.scored for t in test]
    seen_ds = seen.get(cal[0].dataset, set())
    unseen_s = [without_seen(t, seen_ds) for t in test_s] if seen_ds else None
    rows = []
    for target, kw in targets:
        try:
            rec = _recommend(cal, test_s, target, **dict(kw))
            rec_u = _recommend(cal, unseen_s, target, **dict(kw)) if unseen_s else None
            if kw.get("alpha_vs_human") is not None:
                target = replace(target, alpha=rec.alpha)
            rows.append(ThresholdRow(names, target, rec, rec_u))
        except ValueError as exc:
            rows.append(ThresholdRow(names, target, None, None, str(exc)))
    return rows


def cascades(
    pairs: Sequence[tuple[DecisionTable, DecisionTable]], seen: Mapping[str, set[str]]
) -> tuple[list[ThresholdRow], list[str]]:
    """Two-stage cascades (cheap → strong) at the registered target, per dataset."""
    rows, notes = [], []
    by_ds: dict[str, list[tuple[DecisionTable, DecisionTable]]] = defaultdict(list)
    for p in pairs:
        by_ds[p[0].dataset].append(p)
    for ds, ps in sorted(by_ds.items()):
        if len(ps) < 2:
            notes.append(
                f"{ds}: a cascade needs two or more deciders with calibration and test runs "
                f"(found {len(ps)})"
            )
            continue
        for cheap, strong in combinations(ps, 2):  # already sorted cheapest first
            cal, test = [cheap[0], strong[0]], [cheap[1], strong[1]]
            try:
                align(cal[0].scored, cal[1].scored)
                align(test[0].scored, test[1].scored)
            except ValueError:
                notes.append(f"{cheap[0].name} → {strong[0].name}: different examples; skipped")
                continue
            target = cheap[0].target or Target(0.1, 0.1, "EXPLORATORY", "no registered target")
            seen_ds = seen.get(ds, set())
            t_s = [t.scored for t in test]
            try:
                rec = _recommend(cal, t_s, target)
                rec_u = (
                    _recommend(cal, [without_seen(t, seen_ds) for t in t_s], target)
                    if seen_ds
                    else None
                )
                rows.append(ThresholdRow((cal[0].name, cal[1].name), target, rec, rec_u))
            except ValueError as exc:
                rows.append(ThresholdRow((cal[0].name, cal[1].name), target, None, None, str(exc)))
    return rows, notes


# --------------------------------------------------------------------------- formatting


def _pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}%}"


def _num(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _ci(m: Mapping[str, Any] | None, fmt: Callable[[float | None], str] = _num) -> str:
    if not m or m.get("estimate") is None:
        return "—"
    if m.get("low") is None:
        return fmt(m["estimate"])
    return f'{fmt(m["estimate"])} <span class="ci">[{fmt(m["low"])}, {fmt(m["high"])}]</span>'


def _badge(kind: str) -> str:
    cls = {"PRE-REGISTERED": "pre", "SENSITIVITY": "sens"}.get(kind, "expl")
    return f'<span class="badge {cls}">{escape(kind)}</span>'


def _table(head: Sequence[str], rows: Sequence[Sequence[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (
        '<div class="scroll"><table '
        f'class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def _short(name: str, n: int = 24) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


# --------------------------------------------------------------------------- sections


def metrics_section(runs: Sequence[DecisionTable], reports: Mapping[str, dict[str, Any]]) -> str:
    rows = []
    for r in runs:
        rep = reports[r.key]
        m, nf = rep["metrics"], rep["noise_floor"]["ece_width"]
        ece = m["ece_width"]["estimate"]
        within = ece is not None and ece <= nf["p95"]
        floor = f"{_num(nf['mean'])} / {_num(nf['p95'])}" + (
            ' <span class="ok">within noise</span>' if within else ""
        )
        rows.append(
            [
                f"<code>{escape(r.name)}</code>",
                escape(r.split or "?"),
                str(rep["n"]),
                str(rep["n_errors"]),
                _ci(m["accuracy"], _pct),
                _ci(m["cohen_kappa"]),
                _ci(m["ece_width"]),
                floor,
                _ci(m["brier"]),
                _ci(m["aurc"]),
                _ci(m["auroc"]),
                str(rep["n_unique_confidences"]),
            ]
        )
    head = [
        "run",
        "split",
        "n",
        "errors",
        "accuracy",
        "Cohen's κ",
        "ECE (equal-width bins)",
        "ECE noise floor (mean / p95)",
        "Brier",
        "AURC",
        "AUROC",
        "distinct confidences",
    ]
    return _table(head, rows, "num")


def charts_section(runs: Sequence[DecisionTable], reports: Mapping[str, dict[str, Any]]) -> str:
    rel = []
    for r in runs:
        bins = [
            svg.Bin(b["lower"], b["upper"], b["count"], b["mean_confidence"], b["accuracy"])
            for b in reports[r.key]["reliability"]["width"]
        ]
        rel.append(
            f"<figure><figcaption>{escape(_short(r.name, 44))}</figcaption>"
            f"{svg.reliability(bins, title=f'Reliability: {r.name}')}</figure>"
        )
    rc = []
    for k in range(0, len(runs), svg.MAX_SERIES):
        chunk = runs[k : k + svg.MAX_SERIES]
        series = [
            svg.Series(
                r.name,
                reports[r.key]["risk_coverage"]["coverage"],
                reports[r.key]["risk_coverage"]["risk"],
            )
            for r in chunk
        ]
        rc.append(
            "<figure>"
            + svg.step_lines(series, title="Risk vs coverage", x_label="coverage", y_label="risk")
            + svg.legend([r.name for r in chunk])
            + "</figure>"
        )
    return (
        "<h3>Reliability diagrams</h3>"
        '<p class="note">Bars: accuracy per confidence bin (darker = more examples). '
        "Tick: the bin's mean confidence. Diagonal: perfect "
        "calibration. Hover a bar for counts.</p>"
        f'<div class="grid">{"".join(rel)}</div>'
        "<h3>Risk–coverage</h3>"
        '<p class="note">Error rate among the most confident '
        "examples, as the share answered grows. "
        "Stepwise (tied confidences enter as one block). Lower is better.</p>"
        f'<div class="grid">{"".join(rc)}</div>'
    )


def _rec_cells(rec: Recommendation | None) -> list[str]:
    if rec is None:
        return ["—"] * 4
    lam = " → ".join("none" if s.threshold is None else f"≥ {s.threshold:g}" for s in rec.stages)
    risk = _num(rec.test_risk)
    if rec.test_risk_ci is not None:
        risk += (
            f' <span class="ci">[{_num(rec.test_risk_ci[0])}, {_num(rec.test_risk_ci[1])}]</span>'
        )
    held = ""
    if rec.test_risk is not None:
        held = (
            ' <span class="ok">≤ α</span>'
            if rec.test_risk <= rec.alpha
            else ' <span class="warn">&gt; α</span>'
        )
    return [lam, _pct(rec.cal_coverage), _pct(rec.test_coverage), risk + held]


def thresholds_table(rows: Sequence[ThresholdRow], *, show_names: bool = False) -> str:
    out = []
    for row in rows:
        t, rec = row.target, row.rec
        feas = "—"
        if rec is not None:
            f = rec.feasibility
            feas = (
                "none needed"
                if f.feasible_without_abstention
                else f"≥ {_pct(f.min_abstention_conservative)}"
            )
        cells = [
            _badge(t.kind) + f'<div class="sub">{escape(t.note)}</div>',
            f"{t.alpha:.3g}",
            f"{t.delta:g}",
            feas,
            *_rec_cells(rec),
        ]
        if row.rec_unseen is not None:
            u = row.rec_unseen
            cells.append(f"{_pct(u.test_coverage)} / {_num(u.test_risk)} (n = {u.n_test})")
        else:
            cells.append("—")
        if show_names:
            cells.insert(0, " → ".join(f"<code>{escape(n)}</code>" for n in row.names))
        if row.error:
            cells[-1] = f'<span class="warn">{escape(row.error)}</span>'
        out.append(cells)
    head = [
        "target",
        "α",
        "δ",
        "escalation floor (Kotte)",
        "threshold λ̂",
        "cal coverage",
        "test coverage",
        "test risk [95% CI]",
        "test without seen ids: coverage / risk",
    ]
    if show_names:
        head.insert(0, "cascade (act → ask)")
    return _table(head, out, "num")


def guarantees_section(
    pairs: Sequence[tuple[DecisionTable, DecisionTable]], seen: Mapping[str, set[str]], dataset: str
) -> str:
    mine = [p for p in pairs if p[0].dataset == dataset]
    if not mine:
        return (
            '<p class="empty">No decider has both a calibration-half and a test-half run here, so '
            "no threshold can be fitted. (Pilot runs are test-half only; the study adds the "
            "calibration halves.)</p>"
        )
    parts = []
    for cal, test in mine:
        rows = threshold_rows([cal], [test], seen)
        parts.append(
            f"<h4><code>{escape(cal.name)}</code> → <code>{escape(test.name)}</code></h4>"
            + thresholds_table(rows)
        )
        first = rows[0].rec
        if first is not None and first.warnings:
            parts.append(
                '<ul class="warnings">'
                + "".join(f"<li>{escape(w)}</li>" for w in first.warnings)
                + "</ul>"
            )
    return "".join(parts)


def judge_section(runs: Sequence[DecisionTable], raw_dir: Path) -> str:
    mt = [r for r in runs if r.dataset == "mt_bench"]
    if not mt:
        return ""
    reps = {
        r.key: judge.judge_report(r.scored, judge.load_lengths(r.scored.ids, raw_dir)) for r in mt
    }
    rows: list[list[str]] = []
    bias_rows: list[list[str]] = []
    agree_iv, ceiling_iv, first_iv, long_iv, tie_iv, stacks = [], [], [], [], [], []
    for r in mt:
        j = reps[r.key]
        a, p, ln, t = j["agreement"], j["position"], j["length"], j["ties"]
        loo = a["vs_others_majority"]
        lab = _short(r.name, 21)
        agree_iv += [
            svg.Interval(
                lab,
                a["with_individuals_S1"]["estimate"],
                a["with_individuals_S1"]["low"],
                a["with_individuals_S1"]["high"],
                1,
                f"{r.name}: judge vs each human (S1)",
            ),
            svg.Interval(
                lab,
                a["human_human_S1"]["estimate"],
                a["human_human_S1"]["low"],
                a["human_human_S1"]["high"],
                2,
                f"{r.name}: human vs human (S1, same votes)",
            ),
        ]
        ceiling_iv += [
            svg.Interval(
                lab, loo["judge"], None, None, 1, f"{r.name}: judge vs the other voters' majority"
            ),
            svg.Interval(
                lab, loo["human"], None, None, 2, f"{r.name}: human vs the other voters' majority"
            ),
        ]
        fs = p["first_shown"]
        first_iv += [
            svg.Interval(lab, fs["judge"], None, None, 1, f"{r.name}: judge"),
            svg.Interval(lab, fs["human"], None, None, 2, f"{r.name}: humans, same rows"),
        ]
        if ln["available"]:
            pl = ln["prefers_longer"]
            long_iv += [
                svg.Interval(lab, pl["judge"], None, None, 1, f"{r.name}: judge"),
                svg.Interval(lab, pl["human"], None, None, 2, f"{r.name}: humans, same rows"),
            ]
        tr = t["tie_rate"]
        tie_iv += [
            svg.Interval(lab, tr["judge"], None, None, 1, f"{r.name}: judge"),
            svg.Interval(lab, tr["human"], None, None, 2, f"{r.name}: humans, same rows"),
        ]
        c = p["counts"]
        stacks.append(
            (
                lab,
                [
                    ("consistent", c["consistent"]),
                    ("always the first answer", c["always_first"]),
                    ("always the second answer", c["always_second"]),
                    ("other (a tie in one order)", c["other"]),
                ],
            )
        )

        def diff(x: Mapping[str, Any]) -> str:
            if x.get("diff") is None:
                return "—"
            ci = (
                ""
                if x.get("low") is None
                else f' <span class="ci">[{x["low"]:+.1%}, {x["high"]:+.1%}]</span>'
            )
            return f"{_pct(x['judge'])} vs {_pct(x['human'])}: {x['diff']:+.1%}{ci} (n = {x['n']})"

        m = a["with_majority"]
        bias_rows.append(
            [
                f"<code>{escape(r.name)}</code>",
                f'{_ci(p["consistency"], _pct)} <div class="sub">{p["both_orders"]} of '
                f"{p['n_comparisons']} comparisons in both orders; "
                f"{p['extra_calls_for_full_coverage']} extra calls to cover all</div>",
                diff(fs),
                diff(ln["prefers_longer"]) if ln["available"] else escape(ln["note"]),
                diff(tr),
            ]
        )
        rows.append(
            [
                f"<code>{escape(r.name)}</code>",
                f'{_ci(m, _pct)} <div class="sub">{m["n"]} comparisons, '
                f"{m['n_no_majority']} excluded</div>",
                _ci(a["with_individuals_S1"], _pct),
                _ci(a["human_human_S1"], _pct),
                _ci(a["with_individuals_S2"], _pct),
                _ci(a["human_human_S2"], _pct),
                diff(loo),
            ]
        )
    head = [
        "judge run",
        "agrees with human majority",
        "with each human (S1, ties)",
        "human–human (S1)",
        "with each human (S2, no ties)",
        "human–human (S2)",
        "vs other voters' majority: judge vs human (paired)",
    ]
    bias_head = [
        "judge run",
        "position consistency (both orders)",
        "picks first-shown answer: judge vs humans",
        "picks longer answer: judge vs humans",
        "says tie: judge vs humans",
    ]
    pair_legend = svg.legend(["judge", "humans (same rows)"])
    charts = [
        (
            "Agreement with individual humans (S1)",
            svg.intervals(
                agree_iv,
                title="Agreement with individual humans",
                x_label="agreement, with ties as a label",
            )
            + svg.legend(["judge vs each human", "human vs human (ceiling)"]),
        ),
        (
            "Against the other voters' majority (paired)",
            svg.intervals(
                ceiling_iv, title="Against the other voters' majority", x_label="agreement"
            )
            + svg.legend(["judge", "human (the ceiling)"]),
        ),
        (
            "Position: verdict when the answers swap",
            svg.stacked_share(stacks, title="Position consistency")
            + '<div class="legend">'
            + "".join(
                f'<span class="key"><i style="background:var(--series-{k + 1})"></i>{n}</span>'
                for k, n in enumerate(
                    ["consistent", "always the first answer", "always the second answer", "other"]
                )
            )
            + "</div>",
        ),
        (
            "Picks the first-shown answer",
            svg.intervals(
                first_iv,
                title="Picks the first-shown answer",
                x_label="share of non-tie verdicts (dashed: 50%)",
                band=(0.5, 0.5, "no preference"),
            )
            + pair_legend,
        ),
    ]
    if long_iv:
        charts.append(
            (
                "Picks the longer answer",
                svg.intervals(
                    long_iv, title="Picks the longer answer", x_label="share of non-tie verdicts"
                )
                + pair_legend,
            )
        )
    charts.append(
        (
            "Says tie",
            svg.intervals(tie_iv, title="Tie rate", x_label="share of verdicts that are tie")
            + pair_legend,
        )
    )
    figs = "".join(f"<figure><figcaption>{escape(t)}</figcaption>{c}</figure>" for t, c in charts)
    return (
        '<section id="judge"><h2>Judge mode (MT-Bench)</h2>'
        '<p class="note">Agreement follows Zheng et al. 2023 '
        "(arXiv:2306.05685 v4, Table 5): S1 counts "
        '"tie" as a label, S2 drops pairs with a tie. The ceiling is '
        "the experts' agreement with each "
        "other on the same votes. The paired column compares judge and human against the same "
        "reference (the other voters' majority) on the same rows. Position bias needs both display "
        "orders: comparisons judged in one order only would need one extra call each "
        "(listed, never scheduled automatically). 95% CIs resample whole questions.</p>"
        + "<h3>Agreement</h3>"
        + _table(head, rows, "num")
        + "<h3>Position, length and tie behaviour</h3>"
        + _table(bias_head, bias_rows, "num")
        + f'<div class="grid">{figs}</div></section>'
    )


def sensitivity_section(
    runs: Sequence[DecisionTable], seen: Mapping[str, set[str]], settings: Settings
) -> str:
    rows = []
    for r in runs:
        if r.split != "test" or r.dataset not in seen:
            continue
        u = without_seen(r.scored, seen[r.dataset])
        if len(u) == 0:
            rows.append(
                [
                    f"<code>{escape(r.name)}</code>",
                    str(len(r.scored)),
                    "0",
                    '<span class="sub">every example was seen in the smoke/pilot runs</span>',
                    "",
                    "",
                    "",
                ]
            )
            continue
        rep = single_report(u, settings)
        m = rep["metrics"]
        extra = ""
        if r.dataset == "mt_bench":
            maj = judge.agreement(u)["with_majority"]
            extra = f'{_ci(maj, _pct)} <div class="sub">{maj["n"]} comparisons</div>'
        rows.append(
            [
                f"<code>{escape(r.name)}</code>",
                str(len(r.scored)),
                str(len(u)),
                _ci(m["accuracy"], _pct),
                _ci(m["ece_width"]),
                _ci(m["brier"]),
                extra or "—",
            ]
        )
    if not rows:
        return '<p class="empty">No test-half runs in this report.</p>'
    head = [
        "test run",
        "n (all)",
        "n without seen",
        "accuracy",
        "ECE",
        "Brier",
        "MT-Bench: agrees with majority",
    ]
    return _table(head, rows, "num")


# --------------------------------------------------------------------------- page


def build(
    sources: Sequence[Path | DecisionTable],
    *,
    raw_dir: Path = Path("data/raw"),
    seen_path: Path | None = DEFAULT_SEEN,
    title: str = "ask-or-act report",
    settings: Settings | None = None,
    now: datetime | None = None,
) -> str:
    settings = settings or Settings()
    runs = load(sources)
    if not runs:
        raise ValueError("no runs or decision tables given")
    seen = load_seen(seen_path)
    reports = {r.key: single_report(r.scored, settings) for r in runs}
    pairs = _pairs(runs)
    datasets = sorted(
        {r.dataset for r in runs},
        key=lambda d: list(DATASET_TITLES).index(d) if d in DATASET_TITLES else 99,
    )
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")

    overview = _table(
        ["run", "dataset", "split", "model", "n", "errors", "true cost / decision", "source"],
        [
            [
                f"<code>{escape(r.name)}</code>",
                escape(r.dataset),
                escape(r.split or "?"),
                escape(r.model),
                str(len(r.scored)),
                str(r.scored.n_errors),
                f"${r.cost_per_decision:.6f}",
                f"<code>{escape(r.source)}</code>",
            ]
            for r in runs
        ],
        "num",
    )
    body = [
        f"<header><h1>{escape(title)}</h1>"
        f'<p class="meta">{len(runs)} runs · generated {stamp} · '
        "computed from saved decisions, no API calls · "
        "ids and numbers only, never input text</p>"
        '<button id="theme" type="button" aria-label="Toggle light or dark">◐</button></header>',
        '<section><h2>How to read this</h2><ul class="howto">'
        f"<li>{_badge('PRE-REGISTERED')} the target fixed in the run's config before the results "
        "(docs/guarantees.md). This is the headline.</li>"
        f"<li>{_badge('EXPLORATORY')} any other target. Shown for context; never a claim.</li>"
        f"<li>{_badge('SENSITIVITY')} the pre-registered target under a different analysis choice "
        "(MT-Bench per vote; α from the human ceiling).</li>"
        "<li>Brackets are 95% bootstrap CIs (2,000 resamples; MT-Bench resampled by question). "
        "Errors count as wrong with confidence 0.</li>"
        "<li>The ECE noise floor is what a perfectly calibrated decider with these confidences "
        "would score at this n. ECE is biased upward at small n; compare deciders paired on the "
        "same examples.</li></ul></section>",
        f"<section><h2>Runs</h2>{overview}</section>",
    ]
    for ds in datasets:
        mine = [r for r in runs if r.dataset == ds]
        body.append(
            f'<section id="{escape(ds)}"><h2>{escape(DATASET_TITLES.get(ds, ds))}</h2>'
            "<h3>Accuracy and calibration</h3>"
            + metrics_section(mine, reports)
            + charts_section(mine, reports)
            + "<h3>Guaranteed thresholds (act / escalate)</h3>"
            '<p class="note">Fixed-sequence test on the calibration half (Trust or Escalate, '
            "arXiv:2407.18370 v1, Theorem 1), checked on the held-out test half. The guarantee: "
            "with probability ≥ 1 − δ, risk ≤ α on acted-on examples. The escalation floor "
            "(Kotte, arXiv:2606.29054 v2, Prop. 3) says how much any valid rule must escalate.</p>"
            + guarantees_section(pairs, seen, ds)
            + "</section>"
        )
    casc, notes = cascades(pairs, seen)
    body.append(
        '<section id="cascades"><h2>Cascades (act → ask → escalate)</h2>'
        '<p class="note">The cheaper decider acts when confident, '
        "otherwise it asks the stronger one, "
        "which acts or escalates to a human (ToE Algorithm 2, δ split over the stages). Ordered by "
        "true cost per decision.</p>"
        + (thresholds_table(casc, show_names=True) if casc else "")
        + ("".join(f'<p class="empty">{escape(n)}</p>' for n in notes) if notes else "")
        + "</section>"
    )
    body.append(judge_section(runs, raw_dir))
    body.append(
        '<section id="sensitivity"><h2>Sensitivity: without pilot-seen examples</h2>'
        '<p class="note">Pre-registered in docs/study.md: the smoke and pilot runs looked at 20 '
        "test-half examples per dataset before the study settings were final. Every test-half "
        "result is also reported without them. For MT-Bench, every comparison that contains a seen "
        "vote is dropped. The threshold tables above carry the same check in their last column.</p>"
        + sensitivity_section(runs, seen, settings)
        + "</section>"
    )
    return _page().replace("{title}", escape(title)).replace("{body}", "\n".join(body))


def write_report(sources: Sequence[Path | DecisionTable], out: Path, **kw: Any) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(sources, **kw), encoding="utf-8")
    return out


def _page() -> str:
    """The page shell (styles, light/dark tokens, theme toggle), shipped with the package."""
    return resources.files("ask_or_act.report").joinpath("template.html").read_text("utf-8")

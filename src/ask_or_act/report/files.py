"""``ask-or-act report FILE...``: one static HTML page for users' own predictions files.

One file gets one model's page; two or three files get them side by side, with the
paired comparison of ``ask-or-act compare`` at the top. Everything is computed with
the public API (``find_thresholds``, ``evaluate``, ``compare``), and the summary at
the top is exactly the text those commands print, so the page and the terminal can't
disagree. The page then shows:

* the thresholds at the user's TARGET α (the ``--alpha`` given, or the default). It
  is never labelled PRE-REGISTERED: that label means the target was fixed in a
  benchmark config before any result, which a user file can't show. Other α values
  are EXPLORATORY: context, never a claim;
* the reliability diagram and the risk–coverage curve (stepwise, test half), with the
  chosen threshold marked on the curve and α as a dashed line;
* the coverage estimate on unlabelled rows: how many of them the threshold would act on.

Nothing is called, and the page loads nothing: no scripts, fonts or images from
elsewhere. It holds ids, file names and numbers only, never input text.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from importlib import resources
from itertools import combinations
from typing import Any

from ask_or_act import api
from ask_or_act.predictions import InputError, Problem
from ask_or_act.report import svg
from ask_or_act.report.build import _ci, _num, _pct, _table
from ask_or_act.report.table import DecisionTable

EXPLORATORY_ALPHAS = (0.05, 0.1, 0.2, 0.25)
MAX_FILES = svg.MAX_SERIES  # one risk–coverage chart, every curve distinguishable
CURVE_POINTS = 720  # display only: about two points per pixel of the chart's width


@dataclass(frozen=True)
class Options:
    alpha: float = api.DEFAULT_ALPHA
    delta: float = api.DEFAULT_DELTA
    alpha_given: bool = False  # --alpha on the command line (else the default)
    rows: api.Rows = "test"
    n_boot: int = 2000
    seed: int = 0
    bins: int = 15
    split_seed: int = 0


@dataclass
class Model:
    """One file's results."""

    table: DecisionTable
    label: str
    input_warnings: tuple[Problem, ...]
    ev: api.Evaluation
    how: str  # how the halves were chosen, as the thresholds header says it
    th: api.Thresholds | None = None
    th_problem: list[str] = field(default_factory=list)  # why there is no threshold


@dataclass(frozen=True)
class FilesReport:
    html: str
    n_models: int
    n_labelled: int
    n_paired: int | None  # several files: the ids compared
    notes: tuple[str, ...]  # for stderr: what the page explains but the user should know


# --------------------------------------------------------------------------- computing


def _thresholds(t: DecisionTable, alpha: float, delta: float) -> api.Thresholds | list[str]:
    try:
        return api.find_thresholds(t, alpha=alpha, delta=delta)
    except api.NotEnoughData as exc:
        return exc.explain(alpha, delta)
    except InputError as exc:
        return [p.message for p in exc.problems if p.severity == "error"]


def _models(loaded: Sequence[api.Loaded], o: Options) -> list[Model]:
    labels = [_src(x.table) for x in loaded]
    if len(set(labels)) < len(labels):  # same file name in two folders
        labels = [f"{n} ({k + 1})" for k, n in enumerate(labels)]
    out = []
    for x, label in zip(loaded, labels, strict=True):
        ev = api.evaluate(x.table, rows=o.rows, n_boot=o.n_boot, seed=o.seed, bins=o.bins)
        m = Model(x.table, label, x.warnings, ev, api.split_how(x.table, o.split_seed))
        th = _thresholds(x.table, o.alpha, o.delta)
        if isinstance(th, api.Thresholds):
            m.th = th
        else:
            m.th_problem = th
        out.append(m)
    return out


# --------------------------------------------------------------------------- text


def _src(t: DecisionTable) -> str:
    """The file as the user named it on the command line."""
    return t.source or t.name


def _say(header: str, body: str, command: str) -> str:
    """A command's text output as HTML: the header line, then the summary, keeping its
    indented detail lines. ``data-command`` lets a test compare it with the terminal."""
    lines = [f'<p class="hd">{escape(header)}</p>', '<p class="gap"></p>'] if header else []
    first = True
    for line in body.splitlines():
        if not line.strip():
            lines.append('<p class="gap"></p>')
        elif line.startswith(" "):
            lines.append(f'<p class="in">{escape(line.strip())}</p>')
        else:
            lines.append(f'<p class="{"lead" if first else ""}">{escape(line)}</p>')
            first = False
    return (
        f'<div class="say" data-command="{escape(command)}">{"".join(lines)}</div>'
        f'<p class="from">Same text as <code>{escape(command)}</code></p>'
    )


def _flags(o: Options) -> str:
    out = ""
    if o.alpha != api.DEFAULT_ALPHA:
        out += f" --alpha {o.alpha:g}"
    if o.delta != api.DEFAULT_DELTA:
        out += f" --delta {o.delta:g}"
    return out


def _threshold_text(m: Model, o: Options) -> str:
    cmd = f"ask-or-act thresholds {_src(m.table)}{_flags(o)}"
    if m.th is None:
        return _say("", "\n".join(m.th_problem), cmd)
    return _say(api.thresholds_header([m.table], m.th, m.how), m.th.summary(), cmd)


def _metrics_text(m: Model, o: Options) -> str:
    extra = " --all" if o.rows == "all" else ""
    cmd = f"ask-or-act metrics {_src(m.table)}{extra}"
    return _say(api.metrics_header(m.table, m.ev), m.ev.summary(), cmd)


# --------------------------------------------------------------------------- sections


def _badge(kind: str) -> str:
    cls = {"TARGET": "target", "EXPLORATORY": "expl"}[kind]
    return f'<span class="badge {cls}">{kind}</span>'


def _target_note(o: Options) -> str:
    return "your --alpha" if o.alpha_given else "the default; set it with --alpha"


def thresholds_table(m: Model, o: Options) -> str:
    alphas = [(o.alpha, "TARGET")] + [
        (a, "EXPLORATORY") for a in EXPLORATORY_ALPHAS if abs(a - o.alpha) > 1e-12
    ]
    rows = []
    for a, kind in alphas:
        res = m.th if kind == "TARGET" and m.th is not None else None
        problem: list[str] = []
        if res is None:
            got = _thresholds(m.table, a, o.delta)
            res = got if isinstance(got, api.Thresholds) else None
            problem = got if isinstance(got, list) else []
        note = _target_note(o) if kind == "TARGET" else "context only, not a claim"
        cells = [f'{_badge(kind)}<div class="sub">{note}</div>', f"{a:g}", f"{o.delta:g}"]
        if res is None:
            why = problem[0].rstrip(".") if problem else "not computable"
            cells.append(f'<span class="sub">{escape(why)}</span>')
            cells += [""] * 4
        elif not res.certified:
            cells += ["act on nothing", "0%", "—", _floor(res)]
            cells.append(_unlabelled_cell(res))
        else:
            cal = f'<div class="sub">calibration {_pct(res.cal_coverage)}</div>'
            cov = f"{_pct(res.test_coverage)} {cal}"
            cells += [f"≥ {res.threshold:.3g}", cov, _risk(res), _floor(res), _unlabelled_cell(res)]
        rows.append(cells)
    head = [
        "target",
        "α",
        "δ",
        "act when confidence",
        "covers (test half)",
        "wrong when acting (test) [95% CI]",
        "escalation floor (Kotte)",
        "unlabelled rows acted on",
    ]
    return _table(head, rows, "num")


def _risk(t: api.Thresholds) -> str:
    if t.test_risk is None:
        return "—"
    ci = (
        f' <span class="ci">[{t.test_risk_ci[0]:.1%}, {t.test_risk_ci[1]:.1%}]</span>'
        if t.test_risk_ci
        else ""
    )
    held = (
        ' <span class="ok">≤ α</span>'
        if t.test_risk <= t.alpha
        else ' <span class="warn">&gt; α</span>'
    )
    return f"{t.test_risk:.1%}{ci}{held}"


def _floor(t: api.Thresholds) -> str:
    return "none needed" if t.min_escalation <= 0 else f"≥ {api.share_text(t.min_escalation)}"


def _unlabelled_cell(t: api.Thresholds) -> str:
    if not t.unlabelled:
        return "—"
    return f"{t.unlabelled_acted:,} of {t.unlabelled:,} ({_pct(t.unlabelled_acted / t.unlabelled)})"


def metrics_table(models: Sequence[Model]) -> str:
    rows = []
    for m in models:
        rep = m.ev.report
        mt, nf = rep["metrics"], rep["noise_floor"]["ece_width"]
        ece = mt["ece_width"]["estimate"]
        within = ece is not None and ece <= nf["p95"]
        rows.append(
            [
                f"<code>{escape(m.label)}</code>",
                f"{m.ev.n:,}",
                f"{m.ev.n_errors:,}",
                _ci(mt["accuracy"], _pct),
                _ci(mt["cohen_kappa"]),
                _ci(mt["ece_width"])
                + f'<div class="sub">noise floor {_num(nf["mean"])}, p95 {_num(nf["p95"])}</div>'
                + ('<span class="ok">within noise</span>' if within else ""),
                _ci(mt["brier"]),
                _ci(mt["aurc"]),
                _ci(mt["auroc"]),
                f"{rep['n_unique_confidences']:,}",
            ]
        )
    head = [
        "file",
        "rows",
        "failed",
        "accuracy",
        "Cohen's κ",
        "ECE (and its noise floor)",
        "Brier",
        "AURC",
        "AUROC",
        "distinct conf.",
    ]
    return _table(head, rows, "num")


def _thin(xs: Sequence[float], ys: Sequence[float]) -> tuple[list[float], list[float]]:
    """At most about CURVE_POINTS points, for drawing only: the last point in each
    coverage bucket, so every step's end is kept and the curve's shape with it."""
    xs, ys = list(map(float, xs)), list(map(float, ys))
    if len(xs) <= CURVE_POINTS:
        return xs, ys
    keep: dict[int, int] = {}
    for i, x in enumerate(xs):
        keep[int(x * CURVE_POINTS)] = i
    idx = sorted({0, *keep.values(), len(xs) - 1})
    return [xs[i] for i in idx], [ys[i] for i in idx]


def charts(models: Sequence[Model], o: Options) -> str:
    series: list[svg.Series] = []
    marks: list[svg.Mark] = []
    for k, m in enumerate(models):
        rc = m.ev.report["risk_coverage"]
        xs, ys = _thin(rc["coverage"], rc["risk"])
        series.append(svg.Series(m.label, xs, ys))
        th = m.th
        if (
            o.rows == "test"
            and th is not None
            and th.certified
            and th.test_coverage is not None
            and th.test_risk is not None
        ):
            marks.append(
                svg.Mark(
                    th.test_coverage,
                    th.test_risk,
                    k + 1,
                    f"{m.label}: act when confidence ≥ {th.threshold:.3g}: covers "
                    f"{th.test_coverage:.0%} of the test half, {th.test_risk:.1%} wrong",
                )
            )
    marked = ", ".join(
        f"{m.label} at confidence ≥ {m.th.threshold:.3g}"
        for m in models
        if m.th is not None and m.th.certified
    )
    rc_note = (
        f"Ring: the chosen threshold ({marked}), with a dashed drop to its coverage. "
        if marks
        else "No threshold is certified at the TARGET α, so nothing is marked. "
    )
    rc_fig = (
        "<figure><figcaption>Risk–coverage (test half)</figcaption>"
        + svg.step_lines(
            series,
            title="Risk vs coverage",
            x_label="coverage (share acted on, most confident first)",
            y_label="wrong among those acted on",
            marks=marks,
            ref_y=(o.alpha, f"α = {o.alpha:g}"),
        )
        + svg.legend([m.label for m in models])
        + "</figure>"
    )
    rel = []
    for k, m in enumerate(models):
        bins = [
            svg.Bin(b["lower"], b["upper"], b["count"], b["mean_confidence"], b["accuracy"])
            for b in m.ev.report["reliability"]["width"]
        ]
        rel.append(
            f"<figure><figcaption>Reliability: {escape(m.label)}</figcaption>"
            f"{svg.reliability(bins, title=f'Reliability: {m.label}', series=k + 1)}</figure>"
        )
    rows_word = "test half" if o.rows == "test" else "all labelled rows"
    return (
        '<div class="grid">' + rc_fig + "".join(rel) + "</div>"
        '<p class="note">'
        + escape(
            f"Risk–coverage: the error rate among the most confident decisions as the share "
            f"acted on grows ({rows_word}). Stepwise: tied confidences enter as one block. "
            f"Lower is better. {rc_note}Dashed line: the TARGET α. Reliability: accuracy "
            "per confidence bin (darker = more rows), the bin's mean confidence as a tick, "
            "the diagonal is perfect calibration. Hover any mark for its numbers."
        )
        + "</p>"
    )


def unlabelled_section(models: Sequence[Model], o: Options) -> str:
    tiles = []
    for m in models:
        th = m.th
        if th is None:
            body = '<span class="big">—</span><span class="sub">no threshold: see above</span>'
        elif not th.unlabelled:
            body = (
                '<span class="big">none</span><span class="sub">no unlabelled rows in this '
                "file (rows with an empty human_label)</span>"
            )
        elif not th.certified:
            body = (
                f'<span class="big">0 of {th.unlabelled:,}</span><span class="sub">nothing is '
                f"certified at α = {o.alpha:g}, so every one goes to a human</span>"
            )
        else:
            share = th.unlabelled_acted / th.unlabelled
            test = (
                f"; on the labelled test half it covers {_pct(th.test_coverage)}"
                if th.test_coverage is not None
                else ""
            )
            body = (
                f'<span class="big">{_pct(share)}</span><span class="sub">'
                f"{th.unlabelled_acted:,} of {th.unlabelled:,} unlabelled rows have confidence "
                f"≥ {th.threshold:.3g}{test}</span>"
            )
        tiles.append(
            f'<div class="tile"><div class="sub"><code>{escape(m.label)}</code></div>{body}</div>'
        )
    return (
        f'<div class="tiles">{"".join(tiles)}</div><p class="note">'
        "Unlabelled rows have no human answer, so their error rate is unknown; this is how "
        "much of that workload the threshold would take. The α guarantee carries over to "
        "them only if they come from the same distribution as the calibration rows. If the "
        "unlabelled share differs a lot from the test half's, the new rows may be different "
        "(other topics, another time period): label a sample of them before relying on it.</p>"
    )


def warnings_section(models: Sequence[Model]) -> str:
    items = []
    for m in models:
        for p in m.input_warnings:
            hint = f' <span class="sub">hint: {escape(p.hint)}</span>' if p.hint else ""
            items.append(f"<li><code>{escape(m.label)}</code>: {escape(p.message)}{hint}</li>")
        for w in m.th.warnings if m.th is not None else ():
            w = w.removeprefix(f"{m.table.name}: ")  # the row already names the file
            items.append(f"<li><code>{escape(m.label)}</code>: {escape(w)}</li>")
    if not items:
        return ""
    return (
        f'<section><h2>Notes on the input</h2><ul class="warnings">{"".join(items)}</ul></section>'
    )


def _howto(o: Options, several: bool) -> str:
    items = [
        f"{_badge('TARGET')} the α you asked for ({_target_note(o)}). Its threshold is the "
        "answer. A user's file is never labelled PRE-REGISTERED: that label is kept for "
        "benchmark configs that fixed α before any result.",
        f"{_badge('EXPLORATORY')} other α values, for context. Choosing α after looking at "
        "them voids the guarantee.",
        "Thresholds are fitted on the calibration half and checked on the test half; "
        "metrics and charts use the test half"
        + (" (here: every labelled row, --all)" if o.rows == "all" else "")
        + ". Re-rolling --split-seed until a result looks good also voids the guarantee.",
        "Brackets are 95% bootstrap CIs. Failed decisions (empty prediction) count as wrong.",
        "The ECE noise floor is the ECE a perfectly calibrated model with these confidences "
        "would show at this n; ECE is biased upward on small samples.",
    ]
    if several:
        items.append(
            "Comparisons are paired on the same ids: exact McNemar for accuracy, a paired "
            "bootstrap for the rest, Holm-adjusted within each pair. The ACE line compares "
            "calibration with accuracy controlled (arXiv:2606.30814 v2)."
        )
    return (
        '<section><h2>How to read this</h2><ul class="howto">'
        + "".join(f"<li>{i}</li>" for i in items)
        + "</ul></section>"
    )


# --------------------------------------------------------------------------- page


def build(
    loaded: Sequence[api.Loaded],
    options: Options | None = None,
    *,
    title: str | None = None,
    now: datetime | None = None,
) -> FilesReport:
    """The page for 1 to ``MAX_FILES`` predictions files. Raises ``InputError`` when the
    files can't be scored or compared (e.g. different ids), ``ValueError`` for too many."""
    o = options or Options()
    if not 1 <= len(loaded) <= MAX_FILES:
        raise ValueError(f"give 1 to {MAX_FILES} predictions files (got {len(loaded)})")
    models = _models(loaded, o)
    several = len(models) > 1
    notes: list[str] = []
    for m in models:
        if m.th is None and m.th_problem:
            notes.append(f"{m.label}: no threshold at α = {o.alpha:g}: {m.th_problem[0]}")

    summary: list[str] = []
    n_paired = None
    if several:
        for a, b in combinations(models, 2):
            cmp = api.compare(
                a.table,
                b.table,
                rows=o.rows,
                alpha=o.alpha,
                delta=o.delta,
                n_boot=o.n_boot,
                seed=o.seed,
                bins=o.bins,
            )
            n_paired = cmp.n if n_paired is None else min(n_paired, cmp.n)
            cmd = f"ask-or-act compare {_src(a.table)} {_src(b.table)}"
            if o.rows == "all":
                cmd += " --all"
            summary.append(_say(api.compare_header(cmp), cmp.summary(), cmd + _flags(o)))
    else:
        summary += [_threshold_text(models[0], o), _metrics_text(models[0], o)]

    n_lab = sum(m.table.n_labelled for m in models)
    n_unl = sum(len(m.table) - m.table.n_labelled for m in models)
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    names = ", ".join(m.label for m in models)
    page_title = title or f"ask-or-act report: {names}"
    target = (
        f"{_badge('TARGET')} α = {o.alpha:g}, δ = {o.delta:g}: at most {o.alpha:.0%} of "
        f"acted-on decisions wrong, with probability ≥ {1 - o.delta:.0%}"
    )
    body = [
        f"<header><h1>{escape(page_title)}</h1>"
        f'<p class="meta">{len(models)} file{"s" if several else ""} · {n_lab:,} labelled, '
        f"{n_unl:,} unlabelled rows · generated {stamp} · computed locally, nothing sent "
        "anywhere · ids and numbers only, never input text</p>"
        '<button id="theme" type="button" aria-label="Toggle light or dark">◐</button></header>',
        f'<section id="summary"><h2>Summary</h2><p class="meta">{target}</p>'
        + "".join(summary)
        + "</section>",
    ]
    if several:
        body.append(
            '<section id="thresholds"><h2>Act or ask: thresholds</h2>'
            + "".join(
                f"<h3><code>{escape(m.label)}</code></h3>{_threshold_text(m, o)}"
                + thresholds_table(m, o)
                for m in models
            )
            + '<p class="note">Each model on its own. To chain them (the cheaper one acts '
            "when confident, otherwise asks the next, then a human): "
            f"<code>ask-or-act thresholds {escape(' '.join(_src(m.table) for m in models))}"
            "</code>.</p></section>"
        )
    else:
        body.append(
            '<section id="thresholds"><h2>Act or ask: thresholds</h2>'
            '<p class="note">Fixed-sequence test on the calibration half (Trust or Escalate, '
            "arXiv:2407.18370 v1, Theorem 1; the chosen threshold's validity per Kotte, "
            "arXiv:2606.29054 v2, Lemma E.1), checked on the held-out test half. The "
            "escalation floor (Kotte v2, Prop. 3) is how much any valid rule must escalate."
            "</p>" + thresholds_table(models[0], o) + "</section>"
        )
    body += [
        '<section id="charts"><h2>Calibration and ranking</h2>'
        + metrics_table(models)
        + charts(models, o)
        + "</section>",
        '<section id="unlabelled"><h2>Unlabelled rows: estimated coverage</h2>'
        + unlabelled_section(models, o)
        + "</section>",
        warnings_section(models),
        _howto(o, several),
    ]
    html = _page().replace("{title}", escape(page_title)).replace("{body}", "\n".join(body))
    return FilesReport(html, len(models), n_lab, n_paired, tuple(notes))


def _page() -> str:
    return resources.files("ask_or_act.report").joinpath("template.html").read_text("utf-8")


def saved_line(out: Any, r: FilesReport) -> str:
    what = (
        f"{r.n_models} models, paired on {r.n_paired:,} ids"
        if r.n_models > 1 and r.n_paired is not None
        else f"1 model, {r.n_labelled:,} labelled rows"
    )
    return f"saved: {out} ({what}; opens offline, no external files)"

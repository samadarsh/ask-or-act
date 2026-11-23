"""Inline SVG charts for the HTML report: no JavaScript, no external services.

Colours are CSS custom properties defined by the page (``--series-1`` …, ``--ink``,
``--grid``), so one SVG reads correctly in light and dark mode. Every mark carries a
``<title>``, which the browser shows as a tooltip; every chart also has a table of
the same numbers next to it, so nothing is conveyed by colour alone.

Risk–coverage curves are drawn as **steps** (arXiv 2608.04899: stepwise, never
trapezoidal), matching how AURC is computed.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

W, H = 360, 240
PAD_L, PAD_R, PAD_T, PAD_B = 46, 12, 12, 38
MAX_SERIES = 3  # the palette's all-pairs-safe slots; more series → small multiples


@dataclass(frozen=True)
class Series:
    name: str
    xs: Sequence[float]
    ys: Sequence[float]


@dataclass(frozen=True)
class Mark:
    """A point on a curve, e.g. the chosen threshold: a ring, plus a dashed drop to the
    x axis so the coverage can be read off."""

    x: float
    y: float
    series: int  # palette slot of the curve it sits on
    tip: str


def _fmt(v: float) -> str:
    return f"{v:.1f}"


class _Frame:
    def __init__(self, x0: float, x1: float, y0: float, y1: float) -> None:
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def x(self, v: float) -> float:
        return PAD_L + (v - self.x0) / (self.x1 - self.x0 or 1) * (W - PAD_L - PAD_R)

    def y(self, v: float) -> float:
        return H - PAD_B - (v - self.y0) / (self.y1 - self.y0 or 1) * (H - PAD_T - PAD_B)


def _axes(f: _Frame, x_label: str, y_label: str, ticks: int = 5, pct: bool = True) -> list[str]:
    out = []
    for k in range(ticks + 1):
        v = f.y0 + (f.y1 - f.y0) * k / ticks
        y = f.y(v)
        out.append(
            f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{_fmt(y)}" y2="{_fmt(y)}"/>'
        )
        label = f"{v:.0%}" if pct else f"{v:.2g}"
        out.append(
            f'<text class="tick" x="{PAD_L - 6}" y="{_fmt(y + 3)}" text-anchor="end">{label}</text>'
        )
        xv = f.x0 + (f.x1 - f.x0) * k / ticks
        x = f.x(xv)
        xl = f"{xv:.0%}" if pct else f"{xv:.2g}"
        out.append(
            f'<text class="tick" x="{_fmt(x)}" y="{H - PAD_B + 14}" '
            f'text-anchor="middle">{xl}</text>'
        )
    out.append(
        f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{H - PAD_B}" y2="{H - PAD_B}"/>'
    )
    out.append(
        f'<text class="label" x="{(PAD_L + W - PAD_R) / 2}" y="{H - 6}" text-anchor="middle">'
        f"{escape(x_label)}</text>"
    )
    out.append(
        f'<text class="label" transform="translate(12 {(PAD_T + H - PAD_B) / 2}) rotate(-90)" '
        f'text-anchor="middle">{escape(y_label)}</text>'
    )
    return out


def _svg(body: list[str], title: str, height: int = H) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {W} {height}" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title>{''.join(body)}</svg>"
    )


def legend(names: Sequence[str]) -> str:
    """HTML legend (≥ 2 series); swatch + name, text in ink colour."""
    if len(names) < 2:
        return ""
    items = "".join(
        f'<span class="key"><i style="background:var(--series-{k + 1})"></i>{escape(n)}</span>'
        for k, n in enumerate(names)
    )
    return f'<div class="legend">{items}</div>'


def step_lines(
    series: Sequence[Series],
    *,
    title: str,
    x_label: str,
    y_label: str,
    y_max: float | None = None,
    marks: Sequence[Mark] = (),
    ref_y: tuple[float, str] | None = None,
) -> str:
    """Stepwise curves (e.g. risk vs coverage), at most ``MAX_SERIES``. ``marks``: points
    to ring (the chosen threshold); ``ref_y``: a dashed horizontal line (e.g. α)."""
    if len(series) > MAX_SERIES:
        raise ValueError(f"at most {MAX_SERIES} series per chart; use small multiples")
    top = y_max if y_max is not None else max((max(s.ys, default=0.0) for s in series), default=1.0)
    if ref_y is not None:
        top = max(top, ref_y[0])
    top = nice_top(top)
    f = _Frame(0.0, 1.0, 0.0, top)
    body = _axes(f, x_label, y_label)
    for k, s in enumerate(series):
        if len(s.xs) == 0:
            continue
        pts = sorted(zip(s.xs, s.ys, strict=True))
        d = [f"M{_fmt(f.x(pts[0][0]))},{_fmt(f.y(pts[0][1]))}"]
        for _, (x1, y1) in itertools.pairwise(pts):
            d.append(f"H{_fmt(f.x(x1))}V{_fmt(f.y(y1))}")
        body.append(
            f'<path class="line" style="stroke:var(--series-{k + 1})" d="{"".join(d)}">'
            f"<title>{escape(s.name)}</title></path>"
        )
        x_end, y_end = pts[-1]
        body.append(
            f'<circle class="end" cx="{_fmt(f.x(x_end))}" cy="{_fmt(f.y(y_end))}" r="4" '
            f'style="fill:var(--series-{k + 1})"><title>{escape(s.name)}: risk '
            f"{y_end:.1%} at full coverage</title></circle>"
        )
    if ref_y is not None:
        v, name = ref_y
        y = f.y(v)
        body.append(
            f'<line class="ref" x1="{PAD_L}" x2="{W - PAD_R}" y1="{_fmt(y)}" y2="{_fmt(y)}">'
            f"<title>{escape(name)}</title></line>"
            f'<text class="tick" x="{PAD_L + 4}" y="{_fmt(y - 4)}">{escape(name)}</text>'
        )
    for m in marks:
        cx, cy = f.x(m.x), f.y(m.y)
        color = f"var(--series-{m.series})"
        body.append(
            f'<line class="drop" x1="{_fmt(cx)}" x2="{_fmt(cx)}" y1="{_fmt(cy)}" '
            f'y2="{H - PAD_B}" style="stroke:{color}"/>'
            f'<circle class="mark" cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="6" '
            f'style="stroke:{color}"><title>{escape(m.tip)}</title></circle>'
        )
    return _svg(body, title)


def nice_top(value: float) -> float:
    """Smallest axis top ≥ value whose 5 ticks are round percentages."""
    for step in (0.01, 0.02, 0.05, 0.1, 0.2):
        if step * 5 >= value * 1.02:
            return step * 5
    return 1.0


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    confidence: float | None
    accuracy: float | None


def reliability(bins: Sequence[Bin], *, title: str, series: int = 1) -> str:
    """Accuracy per confidence bin as bars from the baseline, with the diagonal
    (perfect calibration) and each bin's mean confidence as a tick."""
    f = _Frame(0.0, 1.0, 0.0, 1.0)
    body = _axes(f, "confidence", "accuracy")
    body.append(
        f'<line class="diag" x1="{_fmt(f.x(0))}" y1="{_fmt(f.y(0))}" '
        f'x2="{_fmt(f.x(1))}" y2="{_fmt(f.y(1))}"/>'
    )
    total = sum(b.count for b in bins) or 1
    for b in bins:
        if not b.count or b.accuracy is None:
            continue
        x0, x1 = f.x(b.lo) + 1, f.x(b.hi) - 1
        y = f.y(b.accuracy)
        h = f.y(0) - y
        opacity = 0.35 + 0.65 * min(1.0, b.count / total * 4)
        tip = f"confidence {b.lo:.2f}–{b.hi:.2f}: {b.count} examples, accuracy {b.accuracy:.1%}" + (
            "" if b.confidence is None else f", mean confidence {b.confidence:.1%}"
        )
        body.append(
            f'<rect class="bar" x="{_fmt(x0)}" y="{_fmt(y)}" width="{_fmt(max(1.0, x1 - x0))}" '
            f'height="{_fmt(max(0.0, h))}" rx="2" style="fill:var(--series-{series});'
            f'opacity:{opacity:.2f}"><title>{escape(tip)}</title></rect>'
        )
        if b.confidence is not None:
            cy = f.y(b.confidence)
            body.append(
                f'<line class="conf" x1="{_fmt(x0)}" x2="{_fmt(x1)}" y1="{_fmt(cy)}" '
                f'y2="{_fmt(cy)}"><title>{escape(tip)}</title></line>'
            )
    return _svg(body, title)


@dataclass(frozen=True)
class Interval:
    label: str  # the row
    estimate: float | None
    low: float | None
    high: float | None
    series: int = 1  # palette slot
    note: str = ""


def intervals(
    rows: Sequence[Interval],
    *,
    title: str,
    x_label: str,
    band: tuple[float, float, str] | None = None,
    x_range: tuple[float, float] = (0.0, 1.0),
    signed: bool = False,
) -> str:
    """Dot + CI whiskers, one row per ``Interval`` (rows with the same label share a
    line, offset). ``band``: a shaded reference range, e.g. the human ceiling's CI.
    ``signed``: values are differences, drawn around a zero line."""
    labels: list[str] = []
    for r in rows:
        if r.label not in labels:
            labels.append(r.label)
    row_h = 26
    left = 150
    height = PAD_T + row_h * len(labels) + 34
    x0, x1 = x_range

    def x(v: float) -> float:
        return left + (min(max(v, x0), x1) - x0) / (x1 - x0) * (W - left - PAD_R)

    body = []
    for k in range(5):
        v = x0 + (x1 - x0) * k / 4
        body.append(
            f'<line class="grid" x1="{_fmt(x(v))}" x2="{_fmt(x(v))}" y1="{PAD_T}" '
            f'y2="{height - 30}"/>'
        )
        tick = f"{v:+.0%}" if signed else f"{v:.0%}"
        body.append(
            f'<text class="tick" x="{_fmt(x(v))}" y="{height - 18}" '
            f'text-anchor="middle">{tick}</text>'
        )
    body.append(
        f'<text class="label" x="{(left + W - PAD_R) / 2}" y="{height - 4}" '
        f'text-anchor="middle">{escape(x_label)}</text>'
    )
    if signed:
        body.append(
            f'<line class="axis" x1="{_fmt(x(0))}" x2="{_fmt(x(0))}" y1="{PAD_T}" '
            f'y2="{height - 30}"/>'
        )
    if band is not None:
        lo, hi, name = band
        if hi - lo < 1e-9:  # a reference line, e.g. 50% = no preference
            body.append(
                f'<line class="ref" x1="{_fmt(x(lo))}" x2="{_fmt(x(lo))}" y1="{PAD_T}" '
                f'y2="{height - 30}"><title>{escape(name)}</title></line>'
            )
        else:
            body.append(
                f'<rect class="band" x="{_fmt(x(lo))}" y="{PAD_T}" '
                f'width="{_fmt(x(hi) - x(lo))}" height="{height - 30 - PAD_T}">'
                f"<title>{escape(name)}: {lo:.1%}–{hi:.1%}</title></rect>"
            )
    for i, label in enumerate(labels):
        yc = PAD_T + row_h * i + row_h / 2
        body.append(
            f'<text class="rowlabel" x="{left - 8}" y="{_fmt(yc + 4)}" text-anchor="end">'
            f"{escape(label)}</text>"
        )
        mine = [r for r in rows if r.label == label]
        for j, r in enumerate(mine):
            y = yc + (j - (len(mine) - 1) / 2) * 8
            if r.estimate is None:
                continue
            color = f"var(--series-{r.series})"
            tip = (
                f"{r.note or label}: {r.estimate:+.1%}"
                if signed
                else f"{r.note or label}: {r.estimate:.1%}"
            )
            if r.low is not None and r.high is not None:
                ci = (
                    f" [{r.low:+.1%}, {r.high:+.1%}]" if signed else f" [{r.low:.1%}, {r.high:.1%}]"
                )
                tip += ci
                body.append(
                    f'<line class="whisker" x1="{_fmt(x(r.low))}" x2="{_fmt(x(r.high))}" '
                    f'y1="{_fmt(y)}" y2="{_fmt(y)}" style="stroke:{color}"/>'
                )
            body.append(
                f'<circle class="dot" cx="{_fmt(x(r.estimate))}" cy="{_fmt(y)}" r="4.5" '
                f'style="fill:{color}"><title>{escape(tip)}</title></circle>'
            )
    return _svg(body, title, height)


def stacked_share(rows: Sequence[tuple[str, Sequence[tuple[str, int]]]], *, title: str) -> str:
    """One 100% bar per row; segments in fixed palette order with a 2px surface gap."""
    row_h = 30
    left = 150
    height = PAD_T + row_h * len(rows) + 10
    body = []
    for i, (label, parts) in enumerate(rows):
        y = PAD_T + row_h * i + 6
        body.append(
            f'<text class="rowlabel" x="{left - 8}" y="{y + 13}" '
            f'text-anchor="end">{escape(label)}</text>'
        )
        total = sum(n for _, n in parts)
        if not total:
            body.append(
                f'<text class="tick" x="{left}" y="{y + 13}">no '
                "comparison judged in both orders</text>"
            )
            continue
        x = float(left)
        width = W - left - PAD_R
        for k, (name, n) in enumerate(parts):
            if not n:
                continue
            w = width * n / total
            body.append(
                f'<rect x="{_fmt(x)}" y="{y}" width="{_fmt(max(0.5, w - 2))}" height="18" rx="2" '
                f'style="fill:var(--series-{k + 1})"><title>{escape(name)}: {n} of {total} '
                f"({n / total:.0%})</title></rect>"
            )
            x += w
    return _svg(body, title, height)

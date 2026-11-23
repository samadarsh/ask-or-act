"""The public Python API the predictions commands are built on (docs/cli-design.md §5.3).

    import ask_or_act as aoa
    th = aoa.find_thresholds("preds.csv", alpha=0.1)
    print(th.summary())

Every function takes a predictions file, a mapping of columns, a ``DecisionTable``, or
a saved run folder. ``result.to_json()`` is exactly what the CLI prints with ``--json``,
so the CLI and the API can't drift apart. Nothing here makes a network call.

The statistics are the existing, tested pieces, unchanged:

* thresholds: ``guarantees.recommend`` (fixed-sequence test of Trust or Escalate,
  arXiv:2407.18370 v1 Theorem 1; validity of the selected threshold per Kotte,
  arXiv:2606.29054 v2 Lemma E.1; feasibility floor Kotte v2 Prop. 3);
* metrics and paired comparison: ``metrics.report`` (seeded bootstrap CIs, exact
  McNemar, centered paired bootstrap, Holm).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ask_or_act.core.types import JSON
from ask_or_act.guarantees.recommend import Recommendation, recommend
from ask_or_act.guarantees.sample_size import min_rows
from ask_or_act.metrics.data import Scored, align
from ask_or_act.metrics.report import Settings, compare_report, single_report
from ask_or_act.predictions import InputError, Problem, check
from ask_or_act.report.table import DecisionTable, from_run

type Data = DecisionTable | str | Path | Mapping[str, Sequence[object]]
type Rows = Literal["test", "all"]

DEFAULT_ALPHA = 0.1
DEFAULT_DELTA = 0.1


class NotEnoughData(ValueError):  # noqa: N818 - a domain error, named as the docs call it
    """Too few labelled calibration rows for the requested guarantee (CLI exit 7)."""

    def __init__(self, have: int, need: int, typical: int | None, hint: str) -> None:
        self.have, self.need, self.typical, self.hint = have, need, typical, hint
        super().__init__(f"{have} labelled calibration rows; a guarantee needs at least {need}")


def _plain(x: dict[str, Any]) -> dict[str, JSON]:
    """numpy arrays and scalars → lists and floats; NaN → None (JSON has no NaN)."""
    from ask_or_act.metrics.report import to_jsonable

    out: dict[str, JSON] = to_jsonable(x)
    return out


# --------------------------------------------------------------------------- loading


@dataclass(frozen=True)
class Loaded:
    table: DecisionTable
    warnings: tuple[Problem, ...] = ()


def load(data: Data, **read_options: Any) -> Loaded:
    """A table from any supported source, with the input warnings ``check`` found."""
    if isinstance(data, DecisionTable):
        return Loaded(data)
    if isinstance(data, str | Path) and Path(data).is_dir():
        if not (Path(data) / "run.json").is_file():
            raise InputError(
                [Problem("unreadable", "error", f"{data} is a folder without a run.json")]
            )
        return Loaded(from_run(Path(data)))
    result = check(data, **read_options)
    if result.table is None:
        raise InputError(result.problems)
    return Loaded(result.table, result.warnings)


def halves(t: DecisionTable) -> dict[str, DecisionTable]:
    """{"calibration": ..., "test": ...}, whichever exist."""
    return {h.split: h for h in t.by_split() if h.split is not None}


def _pct(x: float | None, digits: int = 0) -> str:
    return "—" if x is None or math.isnan(x) else f"{x:.{digits}%}"


def _range(lo: float, hi: float) -> str:
    return f"[{lo:.1%}–{hi:.1%}]"


# --------------------------------------------------------------------------- thresholds


@dataclass(frozen=True)
class Stage:
    name: str
    threshold: float | None  # act when confidence ≥ threshold; None: never acts
    cal_share: float
    test_share: float | None


@dataclass(frozen=True)
class Thresholds:
    alpha: float
    delta: float
    stages: tuple[Stage, ...]
    n_cal: int
    n_test: int | None
    cal_coverage: float
    cal_risk: float | None
    test_coverage: float | None
    test_risk: float | None
    test_risk_ci: tuple[float, float] | None
    min_escalation: float  # Kotte floor; 0 when α is reachable without escalating
    unlabelled: int
    unlabelled_acted: int
    closest: "Attempt | None"  # when nothing is certified: the scan's nearest miss
    grid: str
    warnings: tuple[str, ...] = ()
    input_warnings: tuple[Problem, ...] = field(default_factory=tuple)

    @property
    def certified(self) -> bool:
        return any(s.threshold is not None for s in self.stages)

    @property
    def threshold(self) -> float | None:
        """The first stage's threshold (the single-model answer)."""
        return self.stages[0].threshold

    @property
    def coverage(self) -> float:
        """Share acted on: the test half's if there is one, else the calibration half's."""
        return self.test_coverage if self.test_coverage is not None else self.cal_coverage

    def summary(self) -> str:
        a, d = f"{self.alpha:g}", f"{self.delta:g}"
        out: list[str] = []
        if not self.certified:
            out.append(
                f"Act on nothing: no confidence threshold keeps errors at or below "
                f"{self.alpha:.0%} with δ = {d}."
            )
            if self.closest is not None:
                out.append("  " + self.closest.text(self.alpha, self.delta))
            out.append(
                "  Escalate everything, label more rows, or try a better-calibrated confidence."
            )
            return "\n".join(out)
        if len(self.stages) == 1:
            out.append(
                f"Act when confidence ≥ {_thr(self.stages[0].threshold)}; ask a human below. "
                f"Covers {_pct(self.coverage)} of decisions."
            )
        else:
            parts = []
            acted_before = False
            for i, s in enumerate(self.stages):
                share = s.test_share if s.test_share is not None else s.cal_share
                lead = "" if i == 0 else ("otherwise ask " if acted_before else "so ask ")
                if s.threshold is None:
                    parts.append(
                        f"{s.name} never acts" if i == 0 else f"{lead}{s.name}: never acts"
                    )
                else:
                    which = "acts" if i == 0 else "which acts"
                    sep = " " if i == 0 else ", "
                    parts.append(
                        f"{lead}{s.name}{sep}{which} when confidence ≥ {_thr(s.threshold)} "
                        f"({_pct(share)} of decisions)"
                    )
                    acted_before = True
            out.append(";\n".join(parts) + ";")
            out.append(f"escalate the remaining {_pct(1 - self.coverage)} to a human.")
        out.append("")
        out.append(
            f"  Guarantee: at most {self.alpha:.0%} of acted-on decisions are wrong "
            f"(α = {a}), unless the calibration sample was unlucky, which has probability "
            f"≤ {self.delta:.0%} (δ = {d})."
            + (" δ is split over the stages." if len(self.stages) > 1 else "")
        )
        if self.test_risk is not None and self.test_coverage is not None:
            ci = f" {_range(*self.test_risk_ci)}" if self.test_risk_ci else ""
            verdict = (
                "within α"
                if self.test_risk <= self.alpha
                else "above α: either the ≤ δ bad luck, or the test data differs"
            )
            out.append(
                f"  Test check: {self.test_risk:.1%} wrong when acting{ci}, covering "
                f"{_pct(self.test_coverage)}: {verdict}."
            )
        if self.unlabelled:
            out.append(
                f"  Unlabelled rows: this would act on {self.unlabelled_acted:,} of "
                f"{self.unlabelled:,} ({_pct(self.unlabelled_acted / self.unlabelled)})."
            )
        if self.min_escalation > 0:
            out.append(
                f"  No rule can reach α = {a} without escalating at least "
                f"{_pct(self.min_escalation)} of decisions."
            )
        return "\n".join(out)

    def to_json(self) -> dict[str, JSON]:
        return {
            "certified": self.certified,
            "alpha": self.alpha,
            "delta": self.delta,
            "stages": [
                {
                    "name": s.name,
                    "threshold": s.threshold,
                    "cal_share": s.cal_share,
                    "test_share": s.test_share,
                }
                for s in self.stages
            ],
            "threshold": self.threshold,
            "coverage": self.coverage,
            "calibration": {"n": self.n_cal, "coverage": self.cal_coverage, "risk": self.cal_risk},
            "test": {
                "n": self.n_test,
                "coverage": self.test_coverage,
                "risk": self.test_risk,
                "risk_ci": list(self.test_risk_ci) if self.test_risk_ci else None,
            },
            "min_escalation": self.min_escalation,
            "unlabelled": {"n": self.unlabelled, "acted": self.unlabelled_acted},
            "grid": self.grid,
            "notes": list(self.warnings),
        }


def _thr(t: float | None) -> str:
    return "—" if t is None else f"{t:.3g}"


@dataclass(frozen=True)
class Attempt:
    """The pre-registered scan start whose first test came closest to passing."""

    threshold: float
    n: int
    wrong: int
    need: int | None  # rows at this error rate the test would need; None if rate ≥ α

    def text(self, alpha: float, delta: float) -> str:
        rate = self.wrong / self.n if self.n else 0.0
        head = (
            f"Closest attempt: {self.n:,} calibration rows with confidence ≥ "
            f"{self.threshold:g} had {self.wrong:,} wrong ({rate:.1%})"
        )
        if self.need is None:
            return f"{head}, which is not below α = {alpha:g}."
        return (
            f"{head}. At that error rate, proving ≤ {alpha:.0%} with δ = {delta:g} takes "
            f"about {self.need:,} such rows."
        )


def _closest_attempt(
    s: Scored, starts: Sequence[float], alpha: float, delta: float
) -> Attempt | None:
    from ask_or_act.guarantees.binomial import p_value

    best: tuple[float, Attempt] | None = None
    for lam in starts:
        mask = s.confidence >= lam - 1e-9
        n, wrong = int(mask.sum()), int((mask & ~s.correct).sum())
        if n == 0:
            continue
        p = p_value(wrong, n, alpha)
        if best is None or p < best[0]:
            rate = wrong / n
            need = (
                min_rows(alpha, delta, starts=len(starts), error_rate=rate)
                if rate < alpha
                else None
            )
            best = (p, Attempt(lam, n, wrong, need))
    return None if best is None else best[1]


def _unlabelled_acted(
    tables: Sequence[DecisionTable], thresholds: Sequence[float | None]
) -> tuple[int, int]:
    """(n unlabelled, n the cascade would act on), joining the tables by id."""
    first = [r for r in tables[0].rows if r.label is None]
    confs = [
        {r.id: (r.confidence if r.prediction is not None else None) for r in t.rows} for t in tables
    ]
    acted = 0
    for r in first:
        for conf, thr in zip(confs, thresholds, strict=True):
            c = conf.get(r.id)
            if thr is not None and c is not None and c >= thr - 1e-9:
                acted += 1
                break
    return len(first), acted


def _need_rows(alpha: float, delta: float, have: int, starts: int) -> None:
    need = min_rows(alpha, delta, starts=starts)
    if need is None or have >= need:
        return
    typical = min_rows(alpha, delta, starts=starts, error_rate=alpha / 2)
    hint = f"label at least {need - have:,} more rows"
    for a in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        n = min_rows(a, delta, starts=starts) if a > alpha else None
        if n is not None and n <= have:
            hint += f", or loosen the target: --alpha {a:g} needs {n:,}"
            break
    raise NotEnoughData(have, need, typical, hint)


def find_thresholds(
    data: Data | Sequence[Data],
    *,
    alpha: float = DEFAULT_ALPHA,
    delta: float = DEFAULT_DELTA,
    test: Data | Sequence[Data] | None = None,
    disjoint: bool = False,
    **read_options: Any,
) -> Thresholds:
    """Act/ask/escalate thresholds with a guarantee: with probability ≥ 1 − δ over the
    calibration rows, at most α of acted-on decisions are wrong.

    ``data``: one source, or several (a cascade, cheapest first, joined on id). Without
    ``test``, each source's split column (or the hashed split) gives the halves; with
    ``test``, ``data`` is all calibration and ``test`` all test. Raises ``InputError``
    for bad input and ``NotEnoughData`` when no threshold could be certified because
    there are too few labelled calibration rows.
    """
    sources: list[Data] = list(data) if isinstance(data, list | tuple) else [data]  # type: ignore[list-item]
    loaded = [load(d, **read_options) for d in sources]
    tables = [x.table for x in loaded]
    if test is None:
        parts = [halves(t) for t in tables]
        cal = [p.get("calibration") for p in parts]
        tst = [p.get("test") for p in parts]
    else:
        tests: list[Data] = list(test) if isinstance(test, list | tuple) else [test]  # type: ignore[list-item]
        if len(tests) != len(tables):
            raise ValueError("give one test source per calibration source")
        cal = list(tables)
        tst = [load(d, **read_options).table for d in tests]
    if any(c is None for c in cal):
        raise InputError([Problem("no_calibration", "error", "no calibration rows to fit on")])
    cal_t = [c for c in cal if c is not None]
    starts = len(cal_t[0].grid.starts)
    n_cal = cal_t[0].n_labelled
    _need_rows(alpha, delta, n_cal, starts)
    test_t = [t for t in tst if t is not None] if all(t is not None for t in tst) else None
    try:
        rec: Recommendation = recommend(
            [(c.name, c.scored, c.grid) for c in cal_t],
            None if test_t is None else [t.scored for t in test_t],
            alpha=alpha,
            delta=delta,
            disjoint=disjoint,
        )
    except ValueError as exc:  # e.g. the files cover different ids
        raise InputError([Problem("ids_differ", "error", str(exc))]) from exc
    thresholds = [s.threshold for s in rec.stages]
    n_unl, n_acted = _unlabelled_acted(tables, thresholds)
    feas = rec.feasibility
    return Thresholds(
        alpha=rec.alpha,
        delta=rec.delta,
        stages=tuple(Stage(s.name, s.threshold, s.cal_coverage, s.test_share) for s in rec.stages),
        n_cal=rec.n_cal,
        n_test=rec.n_test,
        cal_coverage=rec.cal_coverage,
        cal_risk=rec.cal_risk,
        test_coverage=rec.test_coverage,
        test_risk=rec.test_risk,
        test_risk_ci=rec.test_risk_ci,
        min_escalation=0.0
        if feas.feasible_without_abstention
        else float(feas.min_abstention_conservative),
        unlabelled=n_unl,
        unlabelled_acted=n_acted,
        closest=None
        if any(st.threshold is not None for st in rec.stages)
        else _closest_attempt(cal_t[0].scored, cal_t[0].grid.starts, rec.alpha, rec.delta),
        grid=rec.stages[0].grid,
        warnings=rec.warnings,
        input_warnings=tuple(w for x in loaded for w in x.warnings),
    )


# --------------------------------------------------------------------------- metrics


def _rows_of(t: DecisionTable, rows: Rows) -> DecisionTable:
    if rows == "all" or t.split is not None:
        return t
    part = halves(t).get("test")
    if part is None:
        raise InputError([Problem("no_test", "error", "no test rows to score", "pass --all")])
    return part


@dataclass(frozen=True)
class Evaluation:
    name: str
    rows: Rows
    n: int
    n_errors: int
    report: Mapping[str, Any]  # metrics.report.single_report
    input_warnings: tuple[Problem, ...] = ()

    def metric(self, name: str) -> Mapping[str, Any]:
        return dict(self.report["metrics"][name])

    @property
    def accuracy(self) -> float:
        return float(self.report["metrics"]["accuracy"]["estimate"])

    def summary(self) -> str:
        m, nf = self.report["metrics"], self.report["noise_floor"]["ece_width"]
        acc, conf = m["accuracy"], m["mean_confidence"]["estimate"]
        ece, kappa = m["ece_width"], m["cohen_kappa"]
        out = [
            f"Accuracy {acc['estimate']:.1%} {_range(acc['low'], acc['high'])}. Agreement with "
            f"humans: κ = {kappa['estimate']:.2f} [{kappa['low']:.2f}–{kappa['high']:.2f}]."
        ]
        gap = conf - acc["estimate"]
        if ece["estimate"] > nf["p95"]:
            word = "overstates" if gap > 0 else "understates"
            out.append(
                f"Calibration: confidence {word} accuracy by about {abs(gap) * 100:.0f} points "
                f"(mean confidence {conf:.1%})."
            )
            out.append(
                f"  ECE {ece['estimate']:.3f} [{ece['low']:.3f}–{ece['high']:.3f}]; a perfectly "
                f"calibrated model would show about {nf['mean']:.3f} at this n (95th "
                f"percentile {nf['p95']:.3f}), so this is beyond noise."
            )
        else:
            out.append(
                f"Calibration: no clear miscalibration. ECE {ece['estimate']:.3f} is within what "
                f"a perfectly calibrated model shows at this n (up to {nf['p95']:.3f})."
            )
        rc = self.report["risk_coverage"]
        half = next(
            (1 - r for c, r in zip(rc["coverage"], rc["risk"], strict=True) if c >= 0.5), None
        )
        auroc = m["auroc"]
        if half is not None and auroc["estimate"] is not None:
            out.append(
                f"Ranking: the most confident half is {half:.1%} right; AUROC "
                f"{auroc['estimate']:.2f} [{auroc['low']:.2f}–{auroc['high']:.2f}]."
            )
        out.append(f"Confidence has {self.report['n_unique_confidences']:,} distinct values.")
        return "\n".join(out)

    def to_json(self) -> dict[str, JSON]:
        r = self.report
        return _plain(
            {
                "name": self.name,
                "rows": self.rows,
                "n": self.n,
                "n_errors": self.n_errors,
                "metrics": r["metrics"],
                "noise_floor": r["noise_floor"],
                "n_unique_confidences": r["n_unique_confidences"],
                "reliability": r["reliability"]["width"],
                "risk_coverage": r["risk_coverage"],
                "feasibility": r["feasibility"],
            }
        )


def evaluate(
    data: Data,
    *,
    rows: Rows = "test",
    n_boot: int = 2000,
    seed: int = 0,
    bins: int = 15,
    **read_options: Any,
) -> Evaluation:
    """Accuracy, κ, calibration (ECE with its noise floor, Brier) and selective metrics,
    with 95% bootstrap CIs, on the test half (``rows="all"``: every labelled row)."""
    loaded = load(data, **read_options)
    part = _rows_of(loaded.table, rows)
    s = part.scored
    if len(s) == 0:
        raise InputError([Problem("no_labels", "error", "no labelled rows to score")])
    report = single_report(s, Settings(n_bins=bins, n_boot=n_boot, seed=seed))
    return Evaluation(loaded.table.name, rows, len(s), s.n_errors, report, loaded.warnings)


# --------------------------------------------------------------------------- compare


@dataclass(frozen=True)
class Comparison:
    names: tuple[str, str]
    rows: Rows
    n: int
    report: Mapping[str, Any]  # metrics.report.compare_report
    thresholds: tuple[Thresholds | None, Thresholds | None]
    alpha: float

    def _line(self, metric: str, label: str, fmt: str, scale: float, unit: str = "") -> str:
        p = self.report["paired"][metric]
        t = self.report["tests"].get(metric, {})
        if p["diff"] is None:
            return f"{label}: not computable."
        better, ph = t.get("better"), t.get("p_holm")
        sign = 1.0 if better in (None, self.names[0]) else -1.0
        d, lo, hi = sign * p["diff"], sign * p["low"], sign * p["high"]
        if sign < 0:
            lo, hi = hi, lo
        ci = f"{d * scale:+{fmt}}{unit} [{lo * scale:+{fmt}}, {hi * scale:+{fmt}}]"
        test = "exact McNemar" if metric == "accuracy" else "paired bootstrap"
        pv = f"{test} p = {t['p_value']:.2g}, Holm-adjusted p = {ph:.2g}" if ph is not None else ""
        if ph is not None and ph < 0.05 and better:
            other = self.names[1] if better == self.names[0] else self.names[0]
            what = "more accurate" if metric == "accuracy" else f"better on {label}"
            va, vb = (p["a"], p["b"]) if better == self.names[0] else (p["b"], p["a"])
            vals = f"{va:.1%} vs {vb:.1%}" if metric == "accuracy" else f"{va:.3f} vs {vb:.3f}"
            return f"{better} is {what} than {other}: {vals} ({ci}; {pv})."
        return f"{label}: no clear difference ({ci}; {pv})."

    def _ace_line(self) -> str | None:
        """Calibration with accuracy controlled: the DA view (ACE, arXiv:2606.30814 v2,
        §4.2, the paper's default), so a more accurate model can't look better
        calibrated for that reason alone."""
        view = self.report.get("ace", {}).get("views", {}).get("DA")
        if not view or not view.get("available"):
            return None
        g = view["ece_gap"]
        if g["estimate"] is None or g["low"] is None:
            return None
        a, b = self.names
        head = "With accuracy controlled (ACE, DA view)"
        gap = f"A − B {g['estimate']:+.3f} [{g['low']:+.3f}, {g['high']:+.3f}]"
        if g["low"] > 0 or g["high"] < 0:
            better = b if g["estimate"] > 0 else a
            text = (
                f"{head}, {better} is better calibrated: ECE {view['ece_a']:.3f} vs "
                f"{view['ece_b']:.3f} ({gap})"
            )
        else:
            text = f"{head}: no clear calibration difference ({gap})"
        if g.get("reversal"):
            text += "; this reverses the raw ECE comparison, which partly reflects accuracy"
        return text + "."

    def summary(self) -> str:
        a, b = self.names
        ace_line = self._ace_line()
        out = [
            self._line("accuracy", "Accuracy", ".1f", 100.0, " points"),
            self._line("ece_width", "calibration (ECE)", ".3f", 1.0),
            *([ace_line] if ace_line else []),
            self._line("brier", "Brier score", ".3f", 1.0),
        ]
        ta, tb = self.thresholds
        if ta is not None and tb is not None:

            def acts(t: Thresholds) -> str:
                return f"acts on {_pct(t.coverage)}" if t.certified else "acts on nothing"

            out.append(f"At α = {self.alpha:g}: {a} {acts(ta)}, {b} {acts(tb)}.")
        return "\n".join(out)

    def to_json(self) -> dict[str, JSON]:
        r = self.report
        return _plain(
            {
                "names": list(self.names),
                "rows": self.rows,
                "n": self.n,
                "paired": r["paired"],
                "mcnemar": r["mcnemar"],
                "tests": r["tests"],
                "ace": r.get("ace"),
                "thresholds": [None if t is None else t.to_json() for t in self.thresholds],
                "alpha": self.alpha,
            }
        )


def compare(
    a: Data,
    b: Data,
    *,
    rows: Rows = "test",
    alpha: float = DEFAULT_ALPHA,
    delta: float = DEFAULT_DELTA,
    n_boot: int = 2000,
    seed: int = 0,
    bins: int = 15,
    **read_options: Any,
) -> Comparison:
    """Paired comparison of two models on the same ids: exact McNemar for accuracy,
    centered paired bootstrap for the rest, Holm-adjusted; plus each one's coverage
    at α."""
    la, lb = load(a, **read_options), load(b, **read_options)
    names = (la.table.name, lb.table.name)
    if names[0] == names[1]:
        names = (f"{names[0]} (A)", f"{names[1]} (B)")
    sa, sb = _rows_of(la.table, rows).scored, _rows_of(lb.table, rows).scored
    try:
        sa, sb = align(sa, sb)
    except ValueError as exc:
        raise InputError([Problem("ids_differ", "error", str(exc))]) from exc
    report = compare_report(sa, sb, Settings(n_bins=bins, n_boot=n_boot, seed=seed), names)
    ths: list[Thresholds | None] = []
    for x in (la, lb):
        try:
            ths.append(find_thresholds(x.table, alpha=alpha, delta=delta))
        except (NotEnoughData, InputError):
            ths.append(None)
    return Comparison(names, rows, len(sa), report, (ths[0], ths[1]), alpha)


__all__ = [
    "Comparison",
    "Evaluation",
    "NotEnoughData",
    "Thresholds",
    "compare",
    "evaluate",
    "find_thresholds",
    "load",
]

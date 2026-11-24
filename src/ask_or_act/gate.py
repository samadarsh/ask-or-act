"""``ask-or-act gate``: pass or fail a new predictions file against a saved baseline.

The gate fails (exit 1) only when the new file is **worse than the baseline by more
than a tolerance, beyond statistical noise**. Both files are paired on the same ids:

* **accuracy** on the test half: Tango's score test with the tolerance as its margin
  (metrics/tango.py; Tango 1998, Eq. (24)–(26) and (33));
* **calibration** on the test half: the ACE gap, ECE_new − ECE_baseline under the DA
  view, so a change in accuracy alone can't move it (ACE, arXiv:2606.30814 v2 §4.2,
  the paper's default; metrics/ace.py). Raw ECE is shown but doesn't decide;
* **coverage at α**: each file's threshold fitted on its own calibration half (the
  fixed-sequence test, guarantees/), coverage measured on the test half;
* **certified**: a baseline that certifies a threshold at α, δ while the new file
  doesn't is a failure (``require_certified``).

**"Beyond noise".** Each paired check tests H0: "the new file is at most ``tolerance``
worse" with a one-sided test at ``level``. The three p-values are Holm-adjusted (Holm
1979, as in ``metrics.stats.holm``), so when nothing really got worse a false FAIL
happens at most ``level`` of the time in total, not per check. A check that can show
neither "worse" nor "within the tolerance" is *inconclusive*. It passes unless
``strict`` (``--fail-on-inconclusive``) is set. The intervals shown are 1 − 2·level
(90% at the default 0.05): their ends are exactly the two one-sided tests.

**Bootstrap (calibration and coverage).** The centered paired bootstrap that
``compare`` uses, one-sided and shifted by the tolerance:
p = (1 + #{u*_b − û ≤ û + tol}) / (B + 1) for an improvement û, with the basic
interval [2û − q_hi, 2û − q_lo], so the interval and the test agree. Test-half and
calibration-half rows are resampled separately, with the same indices for both files.

**Coverage refits both thresholds in every resample** (guarantees.fixed_sequence
.FastFit, identical to ``multi_start``). Otherwise noise in choosing a threshold would
look like a significant coverage change. Bootstrapping a threshold search (a
discontinuous function of the data) is only approximately valid, so the coverage
check is approximate; the simulation in docs/gate-simulation.md measures how it
behaves.
"""

import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from statistics import NormalDist
from typing import Any, Literal

import numpy as np

from ask_or_act.api import (
    DEFAULT_ALPHA,
    DEFAULT_DELTA,
    Data,
    Thresholds,
    _plain,
    find_thresholds,
    halves,
    load,
)
from ask_or_act.core.types import JSON
from ask_or_act.guarantees.fixed_sequence import _EPS, FastFit
from ask_or_act.metrics.ace import distribution_aligned
from ask_or_act.metrics.calibration import ece
from ask_or_act.metrics.data import Scored, align
from ask_or_act.metrics.stats import holm, iter_resamples
from ask_or_act.metrics.tango import paired_accuracy
from ask_or_act.predictions import InputError, Problem
from ask_or_act.report.table import DecisionTable

type Status = Literal["pass", "fail", "inconclusive", "warn"]

PAIRED = ("accuracy", "calibration", "coverage")


# --------------------------------------------------------------------------- policy


@dataclass(frozen=True)
class Tolerances:
    """How much worse the new file may be before the gate can fail. Accuracy and
    coverage are fractions (0.01 = 1 point); calibration is an ECE difference."""

    accuracy: float = 0.01
    calibration: float = 0.02
    coverage: float = 0.05

    def __post_init__(self) -> None:
        for name in ("accuracy", "calibration", "coverage"):
            v = getattr(self, name)
            if not 0.0 <= v < 1.0:
                raise ValueError(f"tolerance {name} must be in [0, 1), got {v}")

    def text(self) -> str:
        return (
            f"accuracy {self.accuracy * 100:g} point{'' if self.accuracy == 0.01 else 's'}, "
            f"calibration {self.calibration:g}, coverage {self.coverage * 100:g} points"
        )


@dataclass(frozen=True)
class Policy:
    alpha: float = DEFAULT_ALPHA
    delta: float = DEFAULT_DELTA
    tolerance: Tolerances = field(default_factory=Tolerances)
    level: float = 0.05
    require_certified: bool = True
    strict: bool = False  # inconclusive checks fail
    require_expectation: bool = False
    n_boot: int = 2000
    seed: int = 0
    bins: int = 15
    registered: str | None = None  # when the policy was fixed (UTC date), for the record

    def __post_init__(self) -> None:
        if not 0.0 < self.level < 0.5:
            raise ValueError(f"level must be in (0, 0.5), got {self.level}")
        if self.n_boot < 100:
            raise ValueError("n_boot must be at least 100")

    def to_json(self) -> dict[str, JSON]:
        t = self.tolerance
        return {
            "alpha": self.alpha,
            "delta": self.delta,
            "tolerance": {
                "accuracy": t.accuracy,
                "calibration": t.calibration,
                "coverage": t.coverage,
            },
            "level": self.level,
            "require_certified": self.require_certified,
            "strict": self.strict,
            "require_expectation": self.require_expectation,
            "n_boot": self.n_boot,
            "seed": self.seed,
            "bins": self.bins,
            "registered": self.registered,
        }


# --------------------------------------------------------------------------- expectations

METRICS = {  # metric → (unit, what the number is)
    "accuracy": ("points", "accuracy change, new − baseline, in percentage points"),
    "coverage": ("points", "coverage change at α, in percentage points"),
    "calibration": ("abs", "accuracy-controlled ECE change (ACE gap); lower is better"),
    "ece": ("abs", "raw ECE change; lower is better"),
}
_NUM = r"[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)"
_RANGE = re.compile(
    rf"^\s*(?P<metric>[a-z]+)\s+(?:(?P<lo>{_NUM})\s+to\s+(?P<hi>{_NUM})|(?P<op>>=|≥|<=|≤)\s*(?P<x>{_NUM}))"
    r"\s*(?P<unit>points?|pts|abs)?\s*$"
)
EXPECT_HELP = (
    "'<metric> <low> to <high> [points]' or '<metric> >= <x>' (also <=, ≥, ≤). Metrics: "
    "accuracy and coverage in percentage points, calibration (ACE gap) and ece as "
    "absolute ECE changes. Changes are new − baseline, e.g. 'accuracy +0 to +2 points', "
    "'calibration <= 0'"
)


@dataclass(frozen=True)
class Expectation:
    """A prediction of one metric's change, written before seeing the result."""

    metric: str
    low: float  # in the metric's own scale (fractions for accuracy/coverage)
    high: float
    text: str

    @staticmethod
    def parse(text: str) -> "Expectation":
        m = _RANGE.match(text.strip().lower())
        if not m:
            raise ValueError(f"can't read the expectation {text!r}: write {EXPECT_HELP}")
        metric = m["metric"]
        if metric not in METRICS:
            raise ValueError(
                f"unknown metric {metric!r} in {text!r}; choose from {', '.join(METRICS)}"
            )
        unit = METRICS[metric][0]
        given = m["unit"]
        if given and (given.startswith("p") != (unit == "points")):
            raise ValueError(f"{metric} is measured in {unit}, not {given!r}: {text!r}")

        def num(s: str) -> float:
            v = float(s.replace("−", "-"))
            return v / 100.0 if unit == "points" else v

        if m["op"]:
            x = num(m["x"])
            lo, hi = (x, math.inf) if m["op"] in (">=", "≥") else (-math.inf, x)
        else:
            lo, hi = num(m["lo"]), num(m["hi"])
            if lo > hi:
                raise ValueError(f"the range in {text!r} goes from high to low")
        return Expectation(metric, lo, hi, text.strip())

    def to_json(self) -> dict[str, JSON]:
        return {
            "metric": self.metric,
            "low": None if math.isinf(self.low) else self.low,
            "high": None if math.isinf(self.high) else self.high,
            "unit": METRICS[self.metric][0],
            "text": self.text,
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "Expectation":
        lo, hi = d.get("low"), d.get("high")
        return Expectation(
            str(d["metric"]),
            -math.inf if lo is None else float(lo),
            math.inf if hi is None else float(hi),
            str(d["text"]),
        )


@dataclass(frozen=True)
class Verdict:
    expectation: Expectation
    observed: float | None
    low: float | None  # the observed change's interval (1 − 2·level)
    high: float | None

    @property
    def held(self) -> bool | None:
        """The point estimate is inside the predicted range."""
        if self.observed is None:
            return None
        return self.expectation.low <= self.observed <= self.expectation.high

    @property
    def beyond_noise(self) -> bool:
        """The whole interval is outside the predicted range: a clear miss."""
        if self.low is None or self.high is None:
            return False
        return self.high < self.expectation.low or self.low > self.expectation.high

    def text(self) -> str:
        e = self.expectation
        if self.observed is None:
            return f"{e.text}: not computable"
        obs = _fmt(e.metric, self.observed)
        if self.held:
            return f"{e.text}: held (observed {obs})"
        return f"{e.text}: missed (observed {obs}{', beyond noise' if self.beyond_noise else ''})"

    def to_json(self) -> dict[str, JSON]:
        return {
            **self.expectation.to_json(),
            "observed": self.observed,
            "low": self.low,
            "high": self.high,
            "held": self.held,
            "beyond_noise": self.beyond_noise,
        }


def _fmt(metric: str, x: float) -> str:
    return f"{x * 100:+.1f} points" if METRICS[metric][0] == "points" else f"{x:+.3f}"


# --------------------------------------------------------------------------- checks


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    text: str
    new: float | None = None
    baseline: float | None = None
    diff: float | None = None  # new − baseline, in the metric's own direction
    low: float | None = None
    high: float | None = None
    tolerance: float | None = None
    p_worse: float | None = None  # one-sided p for "worse by more than the tolerance"
    p_holm: float | None = None
    p_within: float | None = None  # one-sided p for "within the tolerance"
    test: str = ""

    def to_json(self) -> dict[str, JSON]:
        return {
            "name": self.name,
            "status": self.status,
            "text": self.text,
            "new": self.new,
            "baseline": self.baseline,
            "diff": self.diff,
            "low": self.low,
            "high": self.high,
            "tolerance": self.tolerance,
            "p_worse": self.p_worse,
            "p_holm": self.p_holm,
            "p_within": self.p_within,
            "test": self.test,
        }


MARKS = {"pass": "✓", "fail": "✗", "inconclusive": "?", "warn": "!"}


@dataclass(frozen=True)
class GateResult:
    names: tuple[str, str]  # (new, baseline)
    n_cal: int
    n_test: int
    policy: Policy
    checks: tuple[Check, ...]
    thresholds: tuple[Thresholds, Thresholds]  # (new, baseline)
    verdicts: tuple[Verdict, ...]
    observed: Mapping[str, Mapping[str, float | None]]
    seconds: float
    warnings: tuple[str, ...] = ()

    def failing(self) -> list[Check]:
        bad: set[Status] = {"fail", "inconclusive"} if self.policy.strict else {"fail"}
        return [c for c in self.checks if c.status in bad]

    @property
    def passed(self) -> bool:
        return not self.failing()

    def rows_needed(self) -> dict[str, int | None]:
        """Approximate test rows each paired check needs to show "within the tolerance",
        for strict mode.

        A normal approximation, not a guarantee: the standard error is read off this
        run's interval (half-width / z_{1−level}) and scaled by √(n_now / n), assuming
        the calibration half grows with the test half. The count is the n at which an
        unchanged model (true difference 0) passes with 80% probability:
        n = n_now · ((z_{1−level} + z_{0.8}) · SE_now / tolerance)².
        ``None`` means not provable at any size (a 0 tolerance) or not computable.
        """
        z_lvl = NormalDist().inv_cdf(1 - self.policy.level)
        z_pow = NormalDist().inv_cdf(0.8)
        out: dict[str, int | None] = {}
        for c in self.checks:
            if c.name not in PAIRED:
                continue
            if None in (c.diff, c.low, c.high, c.tolerance) or not c.tolerance:
                out[c.name] = None
                continue
            assert c.diff is not None
            assert c.low is not None
            assert c.high is not None
            # the worse side: calibration is a gap (higher is worse), the others improvements
            half = (c.high - c.diff) if c.name == "calibration" else (c.diff - c.low)
            se = max(half, 0.0) / z_lvl
            out[c.name] = max(1, math.ceil(self.n_test * ((z_lvl + z_pow) * se / c.tolerance) ** 2))
        return out

    def rows_needed_text(self) -> str:
        parts = [
            f"{k} ~{v:,}" if v is not None else f"{k}: not provable (tolerance 0)"
            for k, v in self.rows_needed().items()
        ]
        return (
            f"strict: test rows needed to show each check within its tolerance (approx., "
            f"80% chance when nothing changed): {', '.join(parts)}; this run has "
            f"{self.n_test:,}."
        )

    def header(self) -> str:
        p = self.policy
        return (
            f"gate: {self.names[0]} vs baseline {self.names[1]}, paired on {self.n_test:,} test "
            f"/ {self.n_cal:,} calibration ids (α = {p.alpha:g}, δ = {p.delta:g}; tolerances: "
            f"{p.tolerance.text()}{'; strict' if p.strict else ''})"
        )

    def summary(self) -> str:
        width = max(len(c.name) for c in self.checks)
        lines = [f"  {MARKS[c.status]} {c.name:<{width}}  {c.text}" for c in self.checks]
        failing = self.failing()
        inconclusive = [c for c in self.checks if c.status == "inconclusive"]
        n = len(self.checks)
        if failing:
            names = ", ".join(c.name for c in failing)
            lines.append(f"FAIL: {len(failing)} of {n} checks failed ({names}).")
        elif inconclusive:
            names = ", ".join(c.name for c in inconclusive)
            lines.append(
                f"PASS, with {len(inconclusive)} inconclusive ({names}): not shown worse, not "
                "shown within the tolerance (--fail-on-inconclusive fails these)."
            )
        else:
            lines.append(f"PASS: {n} of {n} checks passed.")
        if self.policy.strict:
            lines.append(self.rows_needed_text())
        if self.verdicts:
            lines.append("expected:")
            lines.extend(f"  {v.text()}" for v in self.verdicts)
        lvl = self.policy.level
        lines.append(
            f"Intervals are {1 - 2 * lvl:.0%} (two one-sided {lvl:g} tests); the paired "
            f"checks' p-values are Holm-adjusted. Coverage refits both thresholds in each "
            f"of {self.policy.n_boot:,} resamples (approximate). {self.seconds:.1f} s."
        )
        return "\n".join(lines)

    def to_json(self) -> dict[str, JSON]:
        tn, tb = self.thresholds
        return _plain(
            {
                "passed": self.passed,
                "names": {"new": self.names[0], "baseline": self.names[1]},
                "n_calibration": self.n_cal,
                "n_test": self.n_test,
                "policy": self.policy.to_json(),
                "checks": [c.to_json() for c in self.checks],
                "expectations": [v.to_json() for v in self.verdicts],
                "observed": dict(self.observed),
                "thresholds": {"new": tn.to_json(), "baseline": tb.to_json()},
                "seconds": self.seconds,
                "warnings": list(self.warnings),
                "rows_needed": self.rows_needed(),
            }
        )


# --------------------------------------------------------------------------- the run


def _pair(new: DecisionTable, base: DecisionTable) -> tuple[Scored, Scored, Scored, Scored]:
    hn, hb = halves(new), halves(base)
    for name, h in ((new.name, hn), (base.name, hb)):
        for part in ("calibration", "test"):
            if part not in h or len(h[part].scored) == 0:
                raise InputError(
                    [Problem("no_" + part, "error", f"{name}: no labelled {part} rows")]
                )
    try:
        cn, cb = align(hn["calibration"].scored, hb["calibration"].scored)
        tn, tb = align(hn["test"].scored, hb["test"].scored)
    except ValueError as exc:
        raise InputError(
            [
                Problem(
                    "ids_differ",
                    "error",
                    f"the new file and the baseline must have the same labelled ids in each "
                    f"half: {exc}",
                    "both files need the same ids (and the same split column, if any)",
                )
            ]
        ) from exc
    return cn, cb, tn, tb


def _coverage(conf: np.ndarray, lam: float | None) -> float:
    return 0.0 if lam is None or len(conf) == 0 else float(np.mean(conf >= lam - _EPS))


def _boot(
    point: float, draws: np.ndarray, tol: float, level: float
) -> tuple[float, float, float, float]:
    """For an improvement ``point`` (new better > 0) and its bootstrap draws:
    (p_worse, p_within, low, high), centered and shifted by ``tol``."""
    dev = draws - point
    b = len(draws)
    p_worse = (1 + int(np.sum(dev <= point + tol + 1e-12))) / (b + 1)
    p_within = (1 + int(np.sum(dev >= point + tol - 1e-12))) / (b + 1)
    q_lo, q_hi = np.quantile(draws, [level, 1 - level])
    return min(1.0, p_worse), min(1.0, p_within), float(2 * point - q_hi), float(2 * point - q_lo)


def run_gate(
    new: Data,
    baseline: Data,
    *,
    policy: Policy = Policy(),
    expectations: Sequence[Expectation] = (),
    **read_options: Any,
) -> GateResult:
    """Run every check. Raises ``InputError`` (exit 6) and ``NotEnoughData`` (exit 7)."""
    start = time.perf_counter()
    ln, lb = load(new, **read_options), load(baseline, **read_options)
    cn, cb, tn, tb = _pair(ln.table, lb.table)
    p, tol = policy, policy.tolerance
    th_new = find_thresholds(ln.table, alpha=p.alpha, delta=p.delta)
    th_base = find_thresholds(lb.table, alpha=p.alpha, delta=p.delta)
    lam_n, lam_b = th_new.threshold, th_base.threshold
    warnings: list[str] = []

    # accuracy: Tango's score test, exact formulas, no resampling
    acc = paired_accuracy(tn.correct, tb.correct, tolerance=tol.accuracy, level=p.level)
    if len(set(tn.groups)) < len(tn):
        warnings.append(
            "rows are clustered (several per group): the accuracy test treats them as "
            "independent and is optimistic; the bootstrap checks resample whole groups"
        )

    # calibration and coverage: one paired bootstrap, refitting thresholds each time
    def da_gap(ci: np.ndarray | None) -> float | None:
        sel = slice(None) if ci is None else ci
        v = distribution_aligned(
            tn.confidence[sel],
            tn.correct[sel],
            tb.confidence[sel],
            tb.correct[sel],
            p.bins,
            "width",
        )
        return v.ece_gap

    def raw_gap(ci: np.ndarray | None) -> float:
        sel = slice(None) if ci is None else ci
        e_new = ece(tn.confidence[sel], tn.correct[sel], n_bins=p.bins, strategy="width")
        e_base = ece(tb.confidence[sel], tb.correct[sel], n_bins=p.bins, strategy="width")
        return float(e_new - e_base)

    da_view = distribution_aligned(
        tn.confidence, tn.correct, tb.confidence, tb.correct, p.bins, "width"
    )
    gap0 = da_view.ece_gap
    if da_view.available and da_view.info.get("ess_ratio", 1.0) < 0.5:
        warnings.append(f"calibration: {da_view.note}")
    raw0 = raw_gap(None)
    cov_n0, cov_b0 = _coverage(tn.confidence, lam_n), _coverage(tb.confidence, lam_b)
    fit_n = FastFit(
        alpha=p.alpha,
        delta=p.delta,
        grid=ln.table.grid.grid,
        starts=ln.table.grid.starts,
        n=len(cn),
    )
    fit_b = FastFit(
        alpha=p.alpha,
        delta=p.delta,
        grid=lb.table.grid.grid,
        starts=lb.table.grid.starts,
        n=len(cb),
    )
    gaps: list[float] = []
    raws: list[float] = []
    covs: list[float] = []
    cal_draws = iter_resamples(len(cn), n_boot=p.n_boot, seed=p.seed, groups=cn.groups)
    test_draws = iter_resamples(len(tn), n_boot=p.n_boot, seed=p.seed + 1, groups=tn.groups)
    for ci, ti in zip(cal_draws, test_draws, strict=True):
        ln_ = fit_n.threshold(cn.confidence[ci], cn.correct[ci])
        lb_ = fit_b.threshold(cb.confidence[ci], cb.correct[ci])
        covs.append(_coverage(tn.confidence[ti], ln_) - _coverage(tb.confidence[ti], lb_))
        raws.append(raw_gap(ti))
        if gap0 is not None:
            g = da_gap(ti)
            if g is not None:
                gaps.append(g)

    checks: dict[str, Check] = {}
    # accuracy
    checks["accuracy"] = Check(
        "accuracy",
        "pass",
        "",
        acc.new,
        acc.baseline,
        acc.diff,
        acc.low,
        acc.high,
        tol.accuracy,
        acc.p_worse,
        None,
        acc.p_within,
        "Tango score test (1998), Eq. (24); interval Eq. (33)",
    )
    # calibration (improvement = −gap)
    if gap0 is None or len(gaps) < p.n_boot // 2:
        checks["calibration"] = Check(
            "calibration",
            "warn",
            f"not computable ({da_view.note or 'too few resamples with both classes'})",
            test="ACE DA view, paired bootstrap",
        )
    else:
        pw, pi, lo, hi = _boot(-gap0, -np.asarray(gaps), tol.calibration, p.level)
        checks["calibration"] = Check(
            "calibration",
            "pass",
            "",
            float(da_view.ece_a or 0.0),
            float(da_view.ece_b or 0.0),
            gap0,
            -hi,
            -lo,
            tol.calibration,
            pw,
            None,
            pi,
            "ACE DA view (arXiv:2606.30814 v2 §4.2), centered paired bootstrap",
        )
    # coverage
    cov0 = cov_n0 - cov_b0
    pw, pi, lo, hi = _boot(cov0, np.asarray(covs), tol.coverage, p.level)
    checks["coverage"] = Check(
        "coverage",
        "pass",
        "",
        cov_n0,
        cov_b0,
        cov0,
        lo,
        hi,
        tol.coverage,
        pw,
        None,
        pi,
        "centered paired bootstrap, thresholds refitted per resample (approximate)",
    )

    # Holm across the paired checks that could be computed, then each one's status
    live = [k for k in PAIRED if checks[k].p_worse is not None]
    adjusted = holm([float(checks[k].p_worse or 1.0) for k in live])
    for k, ph in zip(live, adjusted, strict=True):
        c = checks[k]
        status: Status = (
            "fail" if ph < p.level else "pass" if (c.p_within or 1.0) < p.level else "inconclusive"
        )
        checks[k] = replace(c, p_holm=ph, status=status, text=_text(c, status, ph, p))

    out = [checks[k] for k in PAIRED]
    out.append(_certified(th_new, th_base, p))
    if p.require_expectation and not expectations:
        out.append(
            Check(
                "expectation",
                "fail",
                "the policy requires an expected effect (--expect, or ask-or-act changelog "
                "expect before the run)",
            )
        )

    observed: dict[str, dict[str, float | None]] = {
        "accuracy": {
            "new": acc.new,
            "baseline": acc.baseline,
            "diff": acc.diff,
            "low": acc.low,
            "high": acc.high,
        },
        "calibration": {
            "new": checks["calibration"].new,
            "baseline": checks["calibration"].baseline,
            "diff": gap0,
            "low": checks["calibration"].low,
            "high": checks["calibration"].high,
        },
        "coverage": {"new": cov_n0, "baseline": cov_b0, "diff": cov0, "low": lo, "high": hi},
    }
    _, _, r_lo, r_hi = _boot(-raw0, -np.asarray(raws), 0.0, p.level)
    observed["ece"] = {"diff": raw0, "low": -r_hi, "high": -r_lo}
    verdicts = tuple(
        Verdict(
            e, observed[e.metric]["diff"], observed[e.metric]["low"], observed[e.metric]["high"]
        )
        for e in expectations
    )
    return GateResult(
        (ln.table.source or ln.table.name, lb.table.source or lb.table.name),
        len(cn),
        len(tn),
        policy,
        tuple(out),
        (th_new, th_base),
        verdicts,
        observed,
        time.perf_counter() - start,
        tuple(warnings),
    )


def _text(c: Check, status: Status, p_holm: float, p: Policy) -> str:
    assert c.diff is not None
    assert c.low is not None
    assert c.high is not None
    assert c.tolerance is not None
    assert c.new is not None
    assert c.baseline is not None
    if c.name == "calibration":
        values = f"ECE {c.new:.3f} vs {c.baseline:.3f} accuracy-controlled"
        change = f"{c.diff:+.3f} [{c.low:+.3f}, {c.high:+.3f}]"
        tol = f"{c.tolerance:g}"
        worst = f"{c.high:+.3f}"
    else:
        digits = 1 if c.name == "accuracy" else 0
        values = f"{c.new * 100:.{digits}f}% vs {c.baseline * 100:.{digits}f}%"
        if c.name == "coverage":
            values += f" at α = {p.alpha:g}"
        change = f"{c.diff * 100:+.1f} points [{c.low * 100:+.1f}, {c.high * 100:+.1f}]"
        tol = f"{c.tolerance * 100:g}-point"
        worst = f"{-c.low * 100:.1f} points"
    head = f"{values}: {change}"
    if status == "fail":
        return f"{head}, worse beyond the {tol} tolerance (Holm p = {p_holm:.2g})"
    if status == "pass":
        return f"{head}, within the {tol} tolerance"
    return f"{head}, inconclusive: could be up to {worst} worse (tolerance {tol})"


def _certified(new: Thresholds, base: Thresholds, p: Policy) -> Check:
    def at(t: Thresholds) -> str:
        return f"act ≥ {t.threshold:g}" if t.threshold is not None else "nothing certified"

    both = f"new: {at(new)}; baseline: {at(base)} (α = {p.alpha:g}, δ = {p.delta:g})"
    if new.certified:
        return Check("certified", "pass", both)
    if base.certified:
        if p.require_certified:
            return Check("certified", "fail", f"lost: {both}")
        return Check("certified", "warn", f"lost (require_certified = false): {both}")
    return Check("certified", "warn", f"neither file certifies a threshold: {both}")

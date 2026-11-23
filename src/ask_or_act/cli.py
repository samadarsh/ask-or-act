"""Command-line entry point (docs/cli-design.md).

The predictions commands (``check``; more to come) read the user's own file, run
locally and make no network connection. The benchmark commands (``run``, ``study``,
``data``, ``judge``, ``audit-redaction``) are unchanged until the study ends.
"""

import argparse
import difflib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from ask_or_act import __version__
from ask_or_act.data import manifest
from ask_or_act.data.sources import DATASETS, DEFAULT_SEED

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_SAMPLES_DIR = Path("data/samples")


LOCAL_ONLY = (
    "The analysis commands (check, thresholds, metrics, compare, report) run locally: "
    "no network connections, no telemetry."
)
DOCS = "https://github.com/samadarsh/ask-or-act/blob/main/docs/cli-design.md"


def _parser_options() -> dict[str, Any]:
    """New-style parser settings: no prefix abbreviations (clig.dev), and on 3.14+
    argparse's own typo suggestions, with our colour decision (term.py)."""
    from ask_or_act import term

    opts: dict[str, Any] = {
        "allow_abbrev": False,
        "formatter_class": argparse.RawDescriptionHelpFormatter,
    }
    if sys.version_info >= (3, 14):
        opts["suggest_on_error"] = True
        opts["color"] = term.use_color(sys.stdout)
    return opts


def _columns(text: str) -> dict[str, str]:
    out = {}
    for part in text.split(","):
        ours, sep, theirs = part.partition("=")
        if not sep or not ours.strip() or not theirs.strip():
            raise argparse.ArgumentTypeError(
                f"expected name=column pairs, e.g. prediction=pred,confidence=score; got {text!r}"
            )
        out[ours.strip()] = theirs.strip()
    return out


def _unit_interval(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0.0 < v < 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, exclusive; got {text}")
    return v


def _output_flags() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    g = p.add_argument_group("output")
    g.add_argument("--json", action="store_true", help="print one JSON document on stdout")
    g.add_argument("-q", "--quiet", action="store_true", help="only the result and errors")
    g.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="colour (default auto: on for a terminal, off with NO_COLOR)",
    )
    return p


def _input_flags() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    g = p.add_argument_group("reading the file")
    g.add_argument("--format", choices=("csv", "tsv", "jsonl"), help="default: from the extension")
    g.add_argument(
        "--columns",
        type=_columns,
        metavar="NAME=COLUMN,...",
        help="map our column names to yours, e.g. prediction=pred,confidence=score",
    )
    g.add_argument(
        "--confidence-scale",
        default="probability",
        metavar="SCALE",
        help="probability (default, 0-1), percent (0-100), votes:K, or logprob",
    )
    g.add_argument(
        "--confidence-of",
        choices=("predicted", "positive"),
        default="predicted",
        help="predicted (default): confidence in the predicted answer; "
        "positive: P(positive class), for two-label tasks",
    )
    g.add_argument(
        "--positive", metavar="LABEL", help="the positive label for --confidence-of positive"
    )
    g.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="seed for the automatic split when there is no split column (default 0). "
        "Choose it before looking at results: re-rolling seeds voids the guarantee",
    )
    g.add_argument(
        "--cal-fraction",
        type=_unit_interval,
        default=0.5,
        help="share of rows in the calibration half for the automatic split (default 0.5)",
    )
    g.add_argument(
        "--split-map",
        type=_columns,
        metavar="VALUE=SPLIT,...",
        help="rename split values, e.g. dev=calibration,holdout=test",
    )
    return p


THRESHOLDS_EPILOG = f"""\
examples:
  # act above what confidence, at most 10% wrong (the default alpha)?
  ask-or-act thresholds preds.csv

  # a stricter target
  ask-or-act thresholds preds.csv --alpha 0.05

  # cheap model first, strong model second, then a human
  ask-or-act thresholds small.csv large.csv

exit codes: 0 done (even "act on nothing"), 6 bad input, 7 too few labelled rows.
runs locally: no network connections, no telemetry.
docs: {DOCS}
"""

METRICS_EPILOG = f"""\
examples:
  # the test half (the calibration half is for fitting thresholds)
  ask-or-act metrics preds.csv

  # every labelled row
  ask-or-act metrics preds.csv --all --json

exit codes: 0 done, 6 bad input.
runs locally: no network connections, no telemetry.
docs: {DOCS}
"""

COMPARE_EPILOG = f"""\
examples:
  ask-or-act compare small.csv large.csv
  ask-or-act compare small.csv large.csv --json | jq .result.tests

exit codes: 0 done, 6 bad input (e.g. the files cover different ids).
runs locally: no network connections, no telemetry.
docs: {DOCS}
"""

REPORT_EPILOG = f"""\
examples:
  # one model: thresholds, reliability, risk-coverage, unlabelled coverage
  ask-or-act report preds.csv -o report.html

  # two models side by side, paired on the same ids
  ask-or-act report small.csv large.csv -o compare.html --alpha 0.05

exit codes: 0 saved, 2 usage, 6 bad input (e.g. the files cover different ids).
runs locally: no network connections, no telemetry; the page loads nothing external.
docs: {DOCS}
"""

CHECK_EPILOG = """\
examples:
  # is my file usable, and what does it support?
  ask-or-act check preds.csv

  # my columns are called pred, conf and label
  ask-or-act check preds.csv --columns prediction=pred,confidence=conf,human_label=label

  # confidence is P(unsafe) from a binary classifier
  ask-or-act check moderation.csv --confidence-of positive --positive unsafe

input columns: id, prediction, confidence, human_label (required; empty prediction =
a failed decision, empty human_label = unlabelled), options, split, group (optional).

exit codes: 0 usable (maybe with warnings), 1 warnings under --strict, 6 errors.
runs locally: no network connections, no telemetry.
docs: https://github.com/samadarsh/ask-or-act/blob/main/docs/cli-design.md
"""


def build_parser() -> argparse.ArgumentParser:
    opts = _parser_options()
    parser = argparse.ArgumentParser(
        prog="ask-or-act",
        description="Evaluate and calibrate AI decisions: when to act, ask, or escalate.",
        epilog=f"{LOCAL_ONLY}\nNew here? ask-or-act check --help",
        **{k: v for k, v in opts.items() if k != "allow_abbrev"},
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    chk = sub.add_parser(
        "check",
        help="validate a predictions file and say what it supports (local, no network)",
        description="Validate a predictions file (CSV, TSV or JSONL) and say what it supports. "
        "Every problem is reported at once, with line numbers and a fix. Use - for stdin.",
        epilog=CHECK_EPILOG,
        parents=[_input_flags(), _output_flags()],
        **opts,
    )
    chk.add_argument("file", help="predictions file, or - for stdin (with --format)")
    chk.add_argument("--alpha", type=_unit_interval, default=0.1, help="target risk (default 0.1)")
    chk.add_argument(
        "--delta",
        type=_unit_interval,
        default=0.1,
        help="allowed failure probability (default 0.1)",
    )
    chk.add_argument("--strict", action="store_true", help="treat warnings as failures (exit 1)")

    data = sub.add_parser("data", help="benchmark dataset samples")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    for cmd, help_text in [
        ("sample", "download pinned sources and write sample manifests"),
        ("verify", "check committed manifests still match the pinned sources"),
    ]:
        p = data_sub.add_parser(cmd, help=help_text)
        p.add_argument(
            "names",
            nargs="*",
            metavar="NAME",
            help=f"datasets (default: all of {', '.join(DATASETS)})",
        )
        p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
        p.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
        if cmd == "sample":
            p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    run = sub.add_parser("run", help="run a decider over a dataset sample (see configs/)")
    run.add_argument("config", type=Path, help="run config file (TOML)")
    run.add_argument("--dry-run", action="store_true", help="show plan and cost; call nothing")
    run.add_argument("--yes", action="store_true", help="approve a run with nonzero cost")
    run.add_argument(
        "--refresh",
        action="store_true",
        help="ignore cached decisions (re-call the API); still stores new ones",
    )
    run.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)

    met = sub.add_parser(
        "metrics",
        help="accuracy, calibration and ranking with 95%% CIs (a predictions file, or runs)",
        description="Accuracy, Cohen's kappa, calibration (ECE with its noise floor, Brier) "
        "and ranking (AURC, AUROC), with 95%% bootstrap CIs. A predictions file is scored on "
        "its test half (--all: every labelled row). Run folders keep the benchmark report.",
        epilog=METRICS_EPILOG,
        parents=[_input_flags(), _output_flags()],
        **opts,
    )
    met.add_argument(
        "runs",
        nargs="+",
        type=Path,
        metavar="FILE_OR_RUN",
        help="a predictions file, or runs/<id> [runs/<id2>]",
    )
    met.add_argument(
        "--all", action="store_true", help="score every labelled row, not the test half"
    )
    met.add_argument("--bins", type=int, default=15, help="calibration bins (default 15)")
    met.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    met.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    met.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.2],
        help="risk targets for the abstention feasibility check",
    )
    met.add_argument("--out", type=Path, help="JSON output path")
    met.add_argument(
        "--signal",
        help="score this recorded signal as the confidence, e.g. "
        "verbalized, token_logprob, verbalized_logprob, self_consistency, native",
    )
    rep = sub.add_parser(
        "report",
        help="one static HTML report (predictions files, or runs); opens offline",
        description="One static HTML page. One predictions file: its thresholds at your "
        "TARGET alpha, the reliability diagram, the risk-coverage curve with the threshold "
        "marked, and coverage on unlabelled rows. Two or three files: side by side, paired "
        "on the same ids (as compare). Run folders keep the benchmark report.",
        epilog=REPORT_EPILOG,
        parents=[_input_flags()],
        **opts,
    )
    rep.add_argument(
        "runs",
        nargs="+",
        type=Path,
        metavar="FILE_OR_RUN",
        help="predictions file(s), or runs/<id> ...",
    )
    rep.add_argument(
        "-o", "--out", type=Path, default=Path("reports/report.html"), help="HTML path"
    )
    rep.add_argument("--title", help="page title")
    rep.add_argument(
        "--alpha",
        type=_unit_interval,
        help="files: the TARGET risk (default 0.1); run folders use their registered target",
    )
    rep.add_argument(
        "--delta", type=_unit_interval, default=0.1, help="files: failure probability (default 0.1)"
    )
    rep.add_argument("--all", action="store_true", help="files: score every labelled row")
    rep.add_argument("--seed", type=int, default=0, help="files: bootstrap seed")
    rep.add_argument("--bins", type=int, default=15, help="files: calibration bins (default 15)")
    rep.add_argument("-q", "--quiet", action="store_true", help="only the saved line and errors")
    rep.add_argument(
        "--color", choices=("auto", "always", "never"), default="auto", help="colour on stderr"
    )
    rep.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="for MT-Bench lengths")
    rep.add_argument(
        "--seen-ids",
        type=Path,
        default=Path("configs/pilot/seen_test_ids.json"),
        help="ids for the pre-registered sensitivity check (docs/study.md)",
    )
    rep.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    jud = sub.add_parser("judge", help="judge-mode analysis of an MT-Bench run (no calls)")
    jud.add_argument("runs", nargs="+", type=Path, help="runs/<id> of MT-Bench runs")
    jud.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="for answer lengths")
    thr = sub.add_parser(
        "thresholds",
        help="act/ask/escalate thresholds with a risk guarantee (a predictions file, or runs)",
        description="Find the confidence above which a model can act, with a guarantee: with "
        "probability at least 1 - delta, at most alpha of acted-on decisions are wrong. "
        "Fitted on the calibration half, checked on the test half. Two or more files make a "
        "cascade (cheapest first, joined on id).",
        epilog=THRESHOLDS_EPILOG,
        parents=[_input_flags(), _output_flags()],
        **opts,
    )
    thr.add_argument(
        "files",
        nargs="*",
        type=Path,
        metavar="FILE",
        help="predictions file(s); 2+ make a cascade, cheapest first",
    )
    thr.add_argument(
        "--cal",
        nargs="+",
        type=Path,
        help="calibration half: run folder(s), or predictions file(s) used whole",
    )
    thr.add_argument(
        "--test", nargs="+", type=Path, help="the same deciders' test-half runs, same order"
    )
    thr.add_argument(
        "--alpha",
        type=float,
        help="target risk (default: the run config's pre-registered [guarantee] alpha, else 0.1)",
    )
    thr.add_argument(
        "--alpha-vs-human",
        type=float,
        default=0.0,
        metavar="MARGIN",
        help="MT-Bench sensitivity analysis (always shown): α = 1 − h + MARGIN, h = how "
        "often a typical human agrees with the others' majority (default MARGIN 0)",
    )
    thr.add_argument(
        "--delta-floor",
        type=float,
        help="δ for the feasibility floor, a separate statement (default: --delta)",
    )
    thr.add_argument(
        "--mt-bench-unit",
        choices=("comparison", "vote"),
        default="comparison",
        help="MT-Bench example unit (default comparison; vote-level is always also "
        "shown as a sensitivity check)",
    )
    thr.add_argument(
        "--delta",
        type=float,
        help="allowed failure probability (default: pre-registered [guarantee] delta, else 0.1)",
    )
    thr.add_argument("--signal", help="score this recorded signal as the confidence")
    thr.add_argument(
        "--disjoint",
        action="store_true",
        help="cascade: calibrate each stage on its own share of the calibration set",
    )

    cmp = sub.add_parser(
        "compare",
        help="is model B better than model A? paired tests on the same ids (local, no network)",
        description="Paired comparison of two models' predictions on the same ids (test half "
        "by default): exact McNemar for accuracy, a centered paired bootstrap for the other "
        "metrics, Holm-adjusted, and what each would act on at alpha.",
        epilog=COMPARE_EPILOG,
        parents=[_input_flags(), _output_flags()],
        **opts,
    )
    cmp.add_argument("a", type=Path, metavar="FILE_A", help="model A's predictions")
    cmp.add_argument("b", type=Path, metavar="FILE_B", help="model B's predictions")
    cmp.add_argument("--all", action="store_true", help="every labelled row, not the test half")
    cmp.add_argument("--alpha", type=_unit_interval, default=0.1, help="target risk (default 0.1)")
    cmp.add_argument("--delta", type=_unit_interval, default=0.1, help="failure probability (0.1)")
    cmp.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples (2000)")
    cmp.add_argument("--seed", type=int, default=0, help="bootstrap seed (0)")
    cmp.add_argument("--bins", type=int, default=15, help="calibration bins (15)")

    study = sub.add_parser("study", help="run a multi-day schedule of configs, one day at a time")
    study_sub = study.add_subparsers(dest="study_command", required=True)
    s_run = study_sub.add_parser(
        "run", help="run every unfinished config, models in parallel, until done or quota"
    )
    s_run.add_argument("schedule", type=Path)
    s_run.add_argument("--yes", action="store_true", help="approve provider calls")
    s_run.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    s_stat = study_sub.add_parser("status", help="progress and quota use; calls nothing")
    s_stat.add_argument("schedule", type=Path)
    s_stat.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)

    aud = sub.add_parser(
        "audit-redaction",
        help="fail if any Aegis prompt text appears in runs, caches or logs",
    )
    aud.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files or directories to scan (default: runs/, .cache/, logs/)",
    )
    aud.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    aud.add_argument("--ngram", type=int, default=8, help="words per matched n-gram")
    aud.add_argument(
        "--min-words", type=int, default=5, help="skip prompts shorter than this many words"
    )
    return parser


def _run_thresholds(args: argparse.Namespace) -> int:
    from ask_or_act.core.runs import load_run
    from ask_or_act.guarantees.recommend import (
        describe,
        grid_for_run,
        recommend,
        run_guarantee,
        run_split,
    )
    from ask_or_act.metrics.data import from_run

    if args.test and len(args.test) != len(args.cal):
        print("error: give one --test run per --cal run, in the same order", file=sys.stderr)
        return 2
    try:
        cal = []
        registered = []
        for path in args.cal:
            record = load_run(path)
            registered.append(run_guarantee(record))
            grid = grid_for_run(record, args.signal)
            if run_split(record) == "test":
                print(f"error: {path} is a test-half run; fit on calibration", file=sys.stderr)
                return 2
            name = str(record.metadata.get("name", path.name))
            cal.append((name, from_run(record, args.signal), grid))
        test = [from_run(load_run(p), args.signal) for p in args.test] if args.test else None
        pre = registered[0]
        if any(r != pre for r in registered[1:]):
            raise ValueError("the --cal runs pre-register different [guarantee] targets")
        notes = []
        alpha, delta = args.alpha, args.delta
        if pre is not None and pre.alpha is not None:
            when = f", registered {pre.registered}" if pre.registered else ""
            if alpha is None:
                alpha = pre.alpha
                notes.append(f"target α = {alpha:g}: pre-registered in the run config{when}")
            elif alpha != pre.alpha:
                notes.append(
                    f"NOTE: α = {alpha:g} is NOT the pre-registered target "
                    f"(α = {pre.alpha:g}{when}); treat this as exploratory"
                )
        if pre is not None and pre.delta is not None and delta is None:
            delta = pre.delta
        alpha = 0.1 if alpha is None else alpha
        delta = 0.1 if delta is None else delta
        kw = {"delta": delta, "delta_floor": args.delta_floor, "disjoint": args.disjoint}
        rec = recommend(cal, test, alpha=alpha, mt_bench_unit=args.mt_bench_unit, **kw)
        extra = []
        if rec.dataset == "mt_bench":
            flip = "vote" if args.mt_bench_unit == "comparison" else "comparison"
            label = "sensitivity" if flip == "vote" else "pre-registered unit"
            extra.append(
                (
                    f"{label}: MT-Bench per {flip}, same α",
                    recommend(cal, test, alpha=alpha, mt_bench_unit=flip, **kw),
                )
            )
            extra.append(
                (
                    "sensitivity: α relative to the human ceiling (not the registered target)",
                    recommend(
                        cal,
                        test,
                        alpha=None,
                        alpha_vs_human=args.alpha_vs_human,
                        mt_bench_unit=args.mt_bench_unit,
                        **kw,
                    ),
                )
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for line in [*notes, *describe(rec)]:
        print(line)
    for title, r in extra:
        print(f"\n--- {title}")
        for line in describe(r):
            print(line)
    return 0


def _run_report(args: argparse.Namespace) -> int:
    from ask_or_act.metrics.report import Settings
    from ask_or_act.report.build import write_report

    if args.alpha is not None or args.all:
        print("error: --alpha and --all apply to predictions files only", file=sys.stderr)
        return 2
    try:
        out = write_report(
            args.runs,
            args.out,
            raw_dir=args.raw_dir,
            seen_path=args.seen_ids,
            title=args.title or "ask-or-act report",
            settings=Settings(n_boot=args.n_boot),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"saved: {out}")
    return 0


def _run_judge(args: argparse.Namespace) -> int:
    from ask_or_act import judge
    from ask_or_act.core.runs import load_run
    from ask_or_act.metrics.data import from_run

    for path in args.runs:
        try:
            s = from_run(load_run(path))
            rep = judge.judge_report(s, judge.load_lengths(s.ids, args.raw_dir))
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            return 2
        print(f"== {path.name}")
        for line in judge.describe(rep):
            print(line)
    return 0


def _run_study(args: argparse.Namespace) -> int:
    from ask_or_act import study
    from ask_or_act.core.config import ConfigError

    try:
        schedule = study.load_schedule(args.schedule)
        if args.study_command == "status":
            study.status(schedule, args.raw_dir)
            return 0
        if not args.yes:
            print(
                "study run calls provider APIs (free tier uses the daily quota): "
                "check `ask-or-act study status` and re-run with --yes",
                file=sys.stderr,
            )
            return 3
        return study.run(schedule, args.raw_dir, approve=True)
    except (ConfigError, manifest.ManifestError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_audit(args: argparse.Namespace) -> int:
    from ask_or_act.audit import DEFAULT_TARGETS, run_audit

    return run_audit(
        args.paths or list(DEFAULT_TARGETS),
        args.raw_dir,
        n=args.ngram,
        min_words=args.min_words,
    )


def _run_metrics(args: argparse.Namespace) -> int:
    from ask_or_act.metrics.command import run_metrics
    from ask_or_act.metrics.report import Settings

    if len(args.runs) > 2:
        print("error: give one run, or two runs to compare", file=sys.stderr)
        return 2
    settings = Settings(
        n_bins=args.bins, n_boot=args.n_boot, seed=args.seed, alphas=tuple(args.alpha)
    )
    try:
        run_metrics(args.runs, settings, out_path=args.out, signal=args.signal)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _run_run(args: argparse.Namespace) -> int:
    from ask_or_act.core.config import ConfigError
    from ask_or_act.run import run_from_config

    try:
        return run_from_config(
            args.config,
            raw_dir=args.raw_dir,
            dry_run=args.dry_run,
            approve_spend=args.yes,
            refresh=args.refresh,
        ).exit_code
    except (ConfigError, manifest.ManifestError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_data(args: argparse.Namespace) -> int:
    try:
        from ask_or_act.data import hub
    except ImportError:  # pragma: no cover - depends on the environment
        print("error: install the data extra first: uv sync --extra data", file=sys.stderr)
        return 1

    names: list[str] = args.names or list(DATASETS)
    status = 0
    for name in names:
        path = args.samples_dir / f"{name}.json"
        if args.data_command == "sample":
            m = hub.build_manifest(name, args.raw_dir, args.seed)
            manifest.write(m, path)
            print(f"{name}: wrote {path} {m.counts()}")
        else:
            m = manifest.read(path)
            try:
                manifest.verify(m, hub.current_fingerprints(name, args.raw_dir))
                print(f"{name}: OK ({len(m.examples)} examples)")
            except manifest.ManifestError as exc:
                print(f"{name}: MISMATCH {exc}", file=sys.stderr)
                status = 1
    return status


def _run_check(args: argparse.Namespace) -> int:
    from ask_or_act import exit_codes, predictions, summaries, term

    source: Any
    if args.file == "-":
        if sys.stdin.isatty():
            build_parser().parse_args(["check", "--help"])
        source = sys.stdin
    else:
        source = Path(args.file)
    try:
        scale = predictions.Scale.parse(args.confidence_scale)
    except ValueError as exc:
        print(term.Painter(term.use_color(sys.stderr, args.color)).error(str(exc)), file=sys.stderr)
        return exit_codes.USAGE
    result = predictions.check(
        source,
        format=args.format,
        columns=args.columns,
        confidence_scale=scale,
        confidence_of=args.confidence_of,
        positive=args.positive,
        split_seed=args.split_seed,
        cal_fraction=args.cal_fraction,
        split_map=args.split_map,
        alpha=args.alpha,
        delta=args.delta,
    )
    if args.json:
        print(json.dumps(summaries.check_json(result), indent=2, ensure_ascii=False))
    else:
        paint = term.Painter(term.use_color(sys.stdout, args.color))
        print(summaries.check_text(result, paint, quiet=args.quiet))
    if not result.ok:
        return exit_codes.BAD_INPUT
    if args.strict and result.warnings:
        return exit_codes.NO
    return exit_codes.OK


def _is_predictions(paths: Sequence[Path]) -> bool:
    """A file (or stdin) rather than a run folder."""
    return any(str(p) == "-" or not p.is_dir() for p in paths)


def _read_options(args: argparse.Namespace) -> dict[str, Any]:
    from ask_or_act import predictions

    return {
        "format": args.format,
        "columns": args.columns,
        "confidence_scale": predictions.Scale.parse(args.confidence_scale),
        "confidence_of": args.confidence_of,
        "positive": args.positive,
        "split_seed": args.split_seed,
        "cal_fraction": args.cal_fraction,
        "split_map": args.split_map,
    }


class _Out:
    """stdout for the result, stderr for everything else (docs/cli-design.md §5.2)."""

    def __init__(self, args: argparse.Namespace) -> None:
        from ask_or_act import term

        self.args = args
        self.err = term.Painter(term.use_color(sys.stderr, args.color))

    def say(self, text: str) -> None:
        from ask_or_act import summaries

        print(summaries.wrap(text), file=sys.stderr)

    def problem(self, p: Any, where: str = "") -> None:
        msg = f"{where}: {p.message}" if where else p.message
        self.say(self.err.error(msg) if p.severity == "error" else self.err.warning(msg))
        if p.hint:
            self.say(f"  hint: {p.hint}")

    def warnings(self, problems: Sequence[Any], where: str = "") -> None:
        if not self.args.quiet:
            for p in problems:
                self.problem(p, where)


# Facts the result already states; repeating them as warnings is noise.
_QUIET_WARNINGS = frozenset({"few_calibration", "small_test"})


def _load_tables(paths: Sequence[Path], args: argparse.Namespace, out: _Out) -> list[Any] | int:
    """Read every file (all problems shown), or return the exit code."""
    from ask_or_act import api, exit_codes, predictions

    try:
        opts = _read_options(args)
    except ValueError as exc:
        print(out.err.error(str(exc)), file=sys.stderr)
        return exit_codes.USAGE
    tables = []
    failed = False
    for path in paths:
        source: Any = sys.stdin if str(path) == "-" else path
        try:
            loaded = api.load(source, **opts)
        except predictions.InputError as exc:
            failed = True
            for p in exc.problems:
                if p.severity == "error":
                    out.problem(p, path.name)
            out.say(f"  (ask-or-act check {path} shows every problem)")
            continue
        out.warnings([w for w in loaded.warnings if w.code not in _QUIET_WARNINGS], path.name)
        tables.append(loaded.table)
    return exit_codes.BAD_INPUT if failed else tables


def _emit(args: argparse.Namespace, header: str, result: Any, command: str, inputs: Any) -> None:
    from ask_or_act import summaries

    if args.json:
        doc = summaries.envelope(command, result.to_json(), [], inputs)
        print(json.dumps(summaries.to_jsonable(doc), indent=2, ensure_ascii=False))
    else:
        print(summaries.wrap(f"{header}\n\n{result.summary()}"))


def _not_enough(out: _Out, exc: Any, alpha: float, delta: float) -> None:
    first, *rest = exc.explain(alpha, delta)
    out.say(out.err.error(first))
    for line in rest:
        out.say(line)


def _run_thresholds_files(args: argparse.Namespace) -> int:
    from ask_or_act import api, exit_codes, predictions

    out = _Out(args)
    if args.files and args.cal:
        print(out.err.error("give files, or --cal/--test, not both"), file=sys.stderr)
        return exit_codes.USAGE
    if args.signal or args.mt_bench_unit != "comparison":
        print(
            out.err.error("--signal and --mt-bench-unit apply to run folders only"),
            file=sys.stderr,
        )
        return exit_codes.USAGE
    cal = _load_tables(args.files or args.cal, args, out)
    if isinstance(cal, int):
        return cal
    test = None
    if args.test:
        loaded = _load_tables(args.test, args, out)
        if isinstance(loaded, int):
            return loaded
        test = loaded
    alpha = 0.1 if args.alpha is None else args.alpha
    delta = 0.1 if args.delta is None else args.delta
    try:
        th = api.find_thresholds(cal, alpha=alpha, delta=delta, test=test, disjoint=args.disjoint)
    except api.NotEnoughData as exc:
        _not_enough(out, exc, alpha, delta)
        return exit_codes.NOT_ENOUGH_DATA
    except predictions.InputError as exc:
        for p in exc.problems:
            out.problem(p)
        return exit_codes.BAD_INPUT
    how = "--cal/--test files" if test else api.split_how(cal[0], args.split_seed)
    header = api.thresholds_header(cal, th, how)
    inputs = {"sources": [t.source or t.name for t in cal], "split": how}
    _emit(args, header, th, "thresholds", inputs)
    return exit_codes.OK


def _run_metrics_file(args: argparse.Namespace) -> int:
    from ask_or_act import api, exit_codes, predictions

    out = _Out(args)
    if len(args.runs) != 1:
        print(
            out.err.error("give one predictions file; to compare two: ask-or-act compare A B"),
            file=sys.stderr,
        )
        return exit_codes.USAGE
    tables = _load_tables(args.runs, args, out)
    if isinstance(tables, int):
        return tables
    rows: Literal["test", "all"] = "all" if args.all else "test"
    try:
        ev = api.evaluate(tables[0], rows=rows, n_boot=args.n_boot, seed=args.seed, bins=args.bins)
    except predictions.InputError as exc:
        for p in exc.problems:
            out.problem(p)
        return exit_codes.BAD_INPUT
    t = tables[0]
    header = api.metrics_header(t, ev)
    _emit(args, header, ev, "metrics", {"source": t.source or t.name, "rows": rows})
    if args.out:
        from ask_or_act import summaries

        args.out.write_text(json.dumps(summaries.to_jsonable(ev.to_json()), indent=2))
    return exit_codes.OK


def _run_compare(args: argparse.Namespace) -> int:
    from ask_or_act import api, exit_codes, predictions

    out = _Out(args)
    tables = _load_tables([args.a, args.b], args, out)
    if isinstance(tables, int):
        return tables
    rows: Literal["test", "all"] = "all" if args.all else "test"
    try:
        cmp = api.compare(
            tables[0],
            tables[1],
            rows=rows,
            alpha=args.alpha,
            delta=args.delta,
            n_boot=args.n_boot,
            seed=args.seed,
            bins=args.bins,
        )
    except predictions.InputError as exc:
        for p in exc.problems:
            out.problem(p)
        return exit_codes.BAD_INPUT
    header = api.compare_header(cmp)
    _emit(args, header, cmp, "compare", {"a": str(args.a), "b": str(args.b), "rows": rows})
    return exit_codes.OK


def _run_report_files(args: argparse.Namespace) -> int:
    from ask_or_act import api, exit_codes, predictions
    from ask_or_act.report import files

    out = _Out(args)
    if not 1 <= len(args.runs) <= files.MAX_FILES:
        print(
            out.err.error(
                f"give 1 to {files.MAX_FILES} predictions files (got {len(args.runs)}); "
                "one chart keeps at most that many curves apart"
            ),
            file=sys.stderr,
        )
        return exit_codes.USAGE
    tables = _load_tables(args.runs, args, out)
    if isinstance(tables, int):
        return tables
    opts = files.Options(
        alpha=api.DEFAULT_ALPHA if args.alpha is None else args.alpha,
        delta=args.delta,
        alpha_given=args.alpha is not None,
        rows="all" if args.all else "test",
        n_boot=args.n_boot,
        seed=args.seed,
        bins=args.bins,
        split_seed=args.split_seed,
    )
    try:
        page = files.build([api.Loaded(t) for t in tables], opts, title=args.title)
    except predictions.InputError as exc:
        for p in exc.problems:
            out.problem(p)
        return exit_codes.BAD_INPUT
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page.html, encoding="utf-8")
    if not args.quiet:
        for note in page.notes:
            out.say(out.err.warning(note))
    print(files.saved_line(args.out, page))
    return exit_codes.OK


def _guarded(run: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> int:
    """New commands: Ctrl-C exits 130 at once; a crash exits 70 with a report link,
    never 1, so CI can't mistake a bug for a failed gate."""
    from ask_or_act import exit_codes

    try:
        return run(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return exit_codes.INTERRUPTED
    except Exception as exc:
        print(
            f"error: internal error (a bug in ask-or-act {__version__}): "
            f"{type(exc).__name__}: {exc}\n  please report it: {exit_codes.ISSUES_URL}",
            file=sys.stderr,
        )
        return exit_codes.INTERNAL


def _suggest_command(parser: argparse.ArgumentParser, argv: Sequence[str]) -> int | None:
    """Python < 3.14: "did you mean" for a mistyped command (3.14's argparse does this)."""
    if sys.version_info >= (3, 14) or not argv or argv[0].startswith("-"):
        return None
    commands = [
        name
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
        for name in action.choices
    ]
    if argv[0] in commands:
        return None
    close = difflib.get_close_matches(argv[0], commands, n=1)
    hint = f", maybe you meant {close[0]!r}?" if close else ""
    parser.print_usage(sys.stderr)
    print(
        f"ask-or-act: error: unknown command {argv[0]!r}{hint} (choose from {', '.join(commands)})",
        file=sys.stderr,
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    bad = _suggest_command(parser, argv)
    if bad is not None:
        return bad
    args = parser.parse_args(argv)
    if args.command == "check":
        return _guarded(_run_check, args)
    if args.command == "data":
        unknown = sorted(set(args.names) - set(DATASETS))
        if unknown:
            parser.error(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)}")
        return _run_data(args)
    if args.command == "run":
        return _run_run(args)
    if args.command == "metrics":
        if _is_predictions(args.runs):
            return _guarded(_run_metrics_file, args)
        return _run_metrics(args)
    if args.command == "thresholds":
        if args.files or (args.cal and _is_predictions(args.cal)):
            return _guarded(_run_thresholds_files, args)
        if not args.cal:
            parser.error("give a predictions file, or --cal run folder(s)")
        return _run_thresholds(args)
    if args.command == "compare":
        return _guarded(_run_compare, args)
    if args.command == "report":
        if _is_predictions(args.runs):
            return _guarded(_run_report_files, args)
        return _run_report(args)
    if args.command == "judge":
        return _run_judge(args)
    if args.command == "study":
        return _run_study(args)
    if args.command == "audit-redaction":
        return _run_audit(args)
    parser.print_help()
    return 0

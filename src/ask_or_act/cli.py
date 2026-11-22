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
from typing import Any

from ask_or_act import __version__
from ask_or_act.data import manifest
from ask_or_act.data.sources import DATASETS, DEFAULT_SEED

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_SAMPLES_DIR = Path("data/samples")


LOCAL_ONLY = "The analysis commands (check) run locally: no network connections, no telemetry."
NEW_COMMANDS = ("check",)


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

    met = sub.add_parser("metrics", help="metrics for a run, or a paired comparison of two")
    met.add_argument("runs", nargs="+", type=Path, help="runs/<id> [runs/<id2>]")
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
    rep = sub.add_parser("report", help="one static HTML report for a set of runs (no calls)")
    rep.add_argument("runs", nargs="+", type=Path, help="runs/<id> ...")
    rep.add_argument("--out", type=Path, default=Path("reports/report.html"), help="HTML path")
    rep.add_argument("--title", default="ask-or-act report")
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
        help="act/ask/escalate thresholds with a risk guarantee (fit on calibration runs)",
    )
    thr.add_argument(
        "--cal",
        nargs="+",
        type=Path,
        required=True,
        help="calibration-half run(s), cheapest first; 2+ runs make a cascade",
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

    try:
        out = write_report(
            args.runs,
            args.out,
            raw_dir=args.raw_dir,
            seen_path=args.seen_ids,
            title=args.title,
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
        return _run_metrics(args)
    if args.command == "thresholds":
        return _run_thresholds(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "judge":
        return _run_judge(args)
    if args.command == "study":
        return _run_study(args)
    if args.command == "audit-redaction":
        return _run_audit(args)
    parser.print_help()
    return 0

"""Command-line entry point. More subcommands arrive in milestone 5."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ask_or_act import __version__
from ask_or_act.data import manifest
from ask_or_act.data.sources import DATASETS, DEFAULT_SEED

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_SAMPLES_DIR = Path("data/samples")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ask-or-act",
        description="Evaluate and calibrate AI decisions: when to act, ask, or escalate.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
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
    if args.command == "judge":
        return _run_judge(args)
    if args.command == "study":
        return _run_study(args)
    if args.command == "audit-redaction":
        return _run_audit(args)
    parser.print_help()
    return 0

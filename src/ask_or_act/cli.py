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
    "The analysis commands (check, thresholds, metrics, compare, report, gate, changelog, demo) "
    "run locally: "
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

GATE_EPILOG = f"""\
examples:
  # fail only if new.csv is worse than the baseline beyond noise
  ask-or-act gate new.csv --baseline baselines/main.csv

  # record what you changed and what you expect, before seeing the result
  ask-or-act gate new.csv --baseline baselines/main.csv --change "prompt v4" \\
      --why "shorter instructions" --expect "accuracy -1 to +1 points"

  # the policy in ask-or-act.toml ([gate] table): baseline, alpha, tolerances, ...
  ask-or-act gate new.csv

checks (paired on the same ids): accuracy (Tango score test), calibration (ACE gap,
accuracy-controlled), coverage at alpha (thresholds refitted per resample), certified.
A check fails only when the new file is worse by more than its tolerance beyond noise
(one-sided tests at --level, Holm-adjusted); "inconclusive" passes unless
--fail-on-inconclusive. Every run appends an entry to the decision changelog.

exit codes: 0 pass, 1 fail, 2 usage or config error, 6 bad input (e.g. different ids),
7 too few labelled rows.
runs locally: no network connections, no telemetry.
docs: {DOCS}
"""

CHANGELOG_EPILOG = f"""\
examples:
  ask-or-act changelog --last 5
  # write the expected effect before generating the new predictions
  ask-or-act changelog expect --change "prompt v4" --why "cut latency" \\
      --expect "accuracy -1 to +1 points" --expect "coverage >= 0 points"
  # append the entry a CI gate run produced (its --json output)
  ask-or-act changelog add gate.json

exit codes: 0 done, 2 usage, 6 unreadable changelog or file.
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
        epilog=f"{LOCAL_ONLY}\nNew here? ask-or-act demo",
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

    gate = sub.add_parser(
        "gate",
        help="pass/fail a new predictions file against a baseline, for CI (local, no network)",
        description="Compare a new predictions file with a saved baseline on the same ids and "
        "exit 1 only if it is worse beyond noise and beyond a tolerance: accuracy, "
        "accuracy-controlled calibration, coverage at alpha, and certification. Appends an "
        "entry to the decision changelog.",
        epilog=GATE_EPILOG,
        parents=[_input_flags(), _output_flags()],
        **opts,
    )
    gate.add_argument("new", type=Path, metavar="NEW", help="the new predictions file")
    gate.add_argument("--baseline", type=Path, help="the baseline file (default: [gate] baseline)")
    pol = gate.add_argument_group("policy (flags > ask-or-act.toml [gate] > defaults)")
    pol.add_argument(
        "--policy", type=Path, help="config file (default: the nearest ask-or-act.toml)"
    )
    pol.add_argument("--no-config", action="store_true", help="ignore any config file")
    pol.add_argument("--alpha", type=_unit_interval, help="target risk (default 0.1)")
    pol.add_argument("--delta", type=_unit_interval, help="failure probability (default 0.1)")
    pol.add_argument(
        "--tolerance",
        type=_columns,
        metavar="NAME=VALUE,...",
        help="how much worse is allowed: accuracy=0.01 (1 point), calibration=0.02, "
        "coverage=0.05 (the defaults)",
    )
    pol.add_argument(
        "--level", type=_unit_interval, help="one-sided test level, Holm-adjusted (default 0.05)"
    )
    pol.add_argument(
        "--fail-on-inconclusive",
        action="store_true",
        default=None,
        help="strict mode: a check that is neither shown worse nor shown within its "
        "tolerance fails",
    )
    pol.add_argument("--n-boot", type=int, help="bootstrap resamples (2000)")
    pol.add_argument("--seed", type=int, help="bootstrap seed (0)")
    pol.add_argument("--bins", type=int, help="calibration bins (15)")
    log = gate.add_argument_group("decision changelog")
    log.add_argument("--change", default="", help="what changed")
    log.add_argument("--why", default="", help="why")
    log.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="TEXT",
        help="the expected effect, before the run (repeatable): e.g. 'accuracy +0 to +2 points'",
    )
    log.add_argument("--model", help="context: model name")
    log.add_argument("--prompt", help="context: prompt version")
    log.add_argument("--dataset", help="context: dataset")
    log.add_argument(
        "--changelog",
        type=Path,
        help="the changelog file (default: [gate] changelog or ask-or-act-changelog.jsonl)",
    )
    log.add_argument(
        "--no-record",
        action="store_true",
        help="don't append (CI: the entry is in the --json output; append it locally with "
        "ask-or-act changelog add)",
    )

    chl = sub.add_parser(
        "changelog",
        help="the decision changelog: predicted vs observed for each gate run",
        description="Show the decision changelog, record an expected effect before a run, or "
        "append an entry from a CI gate run.",
        epilog=CHANGELOG_EPILOG,
        **opts,
    )
    chl.add_argument(
        "--file",
        type=Path,
        help="the changelog (default: [gate] changelog or ask-or-act-changelog.jsonl)",
    )
    chl.add_argument("--no-config", action="store_true", help="ignore any config file")
    chl.add_argument("--last", type=int, metavar="N", help="only the last N entries")
    chl.add_argument("--json", action="store_true", help="print one JSON document on stdout")
    chl_sub = chl.add_subparsers(dest="changelog_command")
    exp = chl_sub.add_parser("expect", help="record the expected effect before the run", **opts)
    exp.add_argument("--change", required=True, help="what will change")
    exp.add_argument("--why", default="", help="why")
    exp.add_argument(
        "--expect",
        action="append",
        required=True,
        metavar="TEXT",
        help="the expected effect (repeatable)",
    )
    exp.add_argument("--file", type=Path, dest="sub_file", help=argparse.SUPPRESS)
    add = chl_sub.add_parser("add", help="append the entry from a gate --json output", **opts)
    add.add_argument("gate_json", type=Path, metavar="GATE_JSON", help="gate --json output file")

    dem = sub.add_parser(
        "demo",
        help="example data and a report in a new folder, in seconds (local, no network)",
        description="Write a synthetic predictions file (a support-ticket router whose model "
        "is overconfident; no text, no download) and its HTML report into a new folder, "
        "and print what thresholds and metrics say about it.",
        epilog=LOCAL_ONLY,
        **opts,
    )
    dem.add_argument(
        "--dir",
        type=Path,
        default=Path("ask-or-act-demo"),
        help="folder to create (default ask-or-act-demo; never overwritten: -2, -3, ... if it "
        "exists)",
    )
    dem.add_argument("-q", "--quiet", action="store_true", help="only the saved lines")
    dem.add_argument(
        "--color", choices=("auto", "always", "never"), default="auto", help="colour on stderr"
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
    from ask_or_act.core.runs import load_run, require_completed
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
            require_completed(record)  # before the split check: say the real reason
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


def _run_demo(args: argparse.Namespace) -> int:
    import shlex

    from ask_or_act import api, demo, exit_codes, summaries
    from ask_or_act.report import files

    out = _Out(args)
    folder = demo.new_folder(args.dir)
    data = demo.write_data(folder)
    n = demo.N_LABELLED + demo.N_NEW
    if args.quiet:
        print(f"saved: {data}")
    else:
        print(
            summaries.wrap(
                f"Wrote {data}: {n:,} synthetic predictions from a support-ticket router "
                f"({demo.N_LABELLED:,} with a human label, {demo.N_NEW:,} new tickets without "
                f"one). Ids, labels and scores only, generated from seed {demo.SEED}; no download."
            )
        )
    table = api.load(data).table  # its one warning (the unlabelled rows) is said above
    if not args.quiet:
        th = api.find_thresholds([table])
        ev = api.evaluate(table)
        how = api.split_how(table)
        for header, result in (
            (api.thresholds_header([table], th, how), th),
            (api.metrics_header(table, ev), ev),
        ):
            print()
            print(summaries.wrap(f"{header}\n\n{result.summary()}"))
    opts = files.Options(
        alpha=api.DEFAULT_ALPHA,
        delta=api.DEFAULT_DELTA,
        alpha_given=False,
        rows="test",
        n_boot=2000,
        seed=0,
        bins=15,
        split_seed=0,
    )
    page = files.build([api.Loaded(table)], opts, title=demo.TITLE)
    report = folder / demo.REPORT_NAME
    report.write_text(page.html, encoding="utf-8")
    if not args.quiet:
        for note in page.notes:
            out.say(out.err.warning(note))
        print()
    print(files.saved_line(report, page))
    if not args.quiet:
        q = shlex.quote(str(data))
        print(
            "\nTry next:\n"
            f"  ask-or-act check {q}\n"
            f"  ask-or-act thresholds {q} --alpha 0.05   # stricter target, less coverage\n"
            f"  ask-or-act metrics {q}\n"
            "Your own file needs: id, prediction, confidence, human_label "
            "(ask-or-act check --help)."
        )
    return exit_codes.OK


def _config(args: argparse.Namespace, out: _Out) -> Any:
    """The project config, or None; raises ConfigError. Using it is never silent."""
    from ask_or_act import project_config

    if args.no_config:
        return None
    policy = getattr(args, "policy", None)
    cfg = project_config.load(policy) if policy else project_config.find()
    if cfg is not None and not getattr(args, "quiet", False):
        try:
            shown = cfg.path.relative_to(Path.cwd().resolve())
        except ValueError:
            shown = cfg.path
        out.say(f"config: {shown}")
    return cfg


GATE_KEYS: dict[str, type | tuple[type, ...]] = {
    "baseline": str,
    "alpha": (float, int),
    "delta": (float, int),
    "tolerance": dict,
    "level": (float, int),
    "require_certified": bool,
    "strict": bool,
    "require_expectation": bool,
    "changelog": str,
    "registered": str,
    "n_boot": int,
    "seed": int,
    "bins": int,
}


def _gate_policy(args: argparse.Namespace, cfg: Any) -> tuple[Any, Path | None, Path]:
    """(Policy, baseline path, changelog path): flags > [gate] > defaults."""
    from ask_or_act import api, changelog, gate, project_config

    table: dict[str, Any] = cfg.table("gate") if cfg is not None else {}
    where = f"{cfg.path} [gate]" if cfg is not None else "[gate]"
    project_config.check_keys(where, table, GATE_KEYS)
    tol_table = dict(table.get("tolerance", {}))
    tol_flags = getattr(args, "tolerance", None) or {}
    tol: dict[str, float] = {}
    for src, values in (("config", tol_table), ("--tolerance", tol_flags)):
        for k, v in values.items():
            if k not in ("accuracy", "calibration", "coverage"):
                raise project_config.ConfigError(
                    f"{src}: unknown tolerance {k!r} (accuracy, calibration, coverage)"
                )
            try:
                tol[k] = float(v)
            except (TypeError, ValueError):
                raise project_config.ConfigError(f"{src}: tolerance {k} must be a number") from None

    def pick(flag: Any, key: str, default: Any) -> Any:
        return flag if flag is not None else table.get(key, default)

    try:
        policy = gate.Policy(
            alpha=float(pick(args.alpha, "alpha", api.DEFAULT_ALPHA)),
            delta=float(pick(args.delta, "delta", api.DEFAULT_DELTA)),
            tolerance=gate.Tolerances(**tol),
            level=float(pick(args.level, "level", 0.05)),
            require_certified=bool(table.get("require_certified", True)),
            strict=bool(pick(args.fail_on_inconclusive, "strict", False)),
            require_expectation=bool(table.get("require_expectation", False)),
            n_boot=int(pick(args.n_boot, "n_boot", 2000)),
            seed=int(pick(args.seed, "seed", 0)),
            bins=int(pick(args.bins, "bins", 15)),
            registered=table.get("registered"),
        )
    except ValueError as exc:
        raise project_config.ConfigError(f"{where}: {exc}") from exc
    for key in ("alpha", "delta"):
        if not 0.0 < getattr(policy, key) < 1.0:
            raise project_config.ConfigError(f"{where}: {key} must be between 0 and 1")
    folder = cfg.folder if cfg is not None else Path()

    def rel(p: str) -> Path:
        """Relative to the config's folder, shown (and logged) relative to the cwd."""
        path = Path(p)
        if path.is_absolute():
            return path
        full = (folder / path).resolve()
        try:
            return full.relative_to(Path.cwd().resolve())
        except ValueError:
            return full

    baseline = args.baseline or (rel(table["baseline"]) if "baseline" in table else None)
    log = args.changelog or (
        rel(table["changelog"]) if "changelog" in table else changelog.DEFAULT_PATH
    )
    return policy, baseline, log


def _run_gate(args: argparse.Namespace) -> int:
    from ask_or_act import api, changelog, exit_codes, gate, predictions, project_config, summaries

    out = _Out(args)
    try:
        cfg = _config(args, out)
        policy, baseline, log_path = _gate_policy(args, cfg)
    except project_config.ConfigError as exc:
        print(out.err.error(str(exc)), file=sys.stderr)
        return exit_codes.USAGE
    if baseline is None:
        print(
            out.err.error("no baseline: give --baseline FILE, or set baseline in [gate]"),
            file=sys.stderr,
        )
        return exit_codes.USAGE
    try:
        expectations = [gate.Expectation.parse(t) for t in args.expect]
    except ValueError as exc:
        print(out.err.error(str(exc)), file=sys.stderr)
        return exit_codes.USAGE
    try:
        entries = changelog.read(log_path)
    except changelog.ChangelogError as exc:
        print(out.err.error(str(exc)), file=sys.stderr)
        return exit_codes.BAD_INPUT
    pending = changelog.open_expectation(entries)
    if pending is not None:
        if expectations:
            print(
                out.err.error(
                    f"{log_path} already holds an expectation for this run ({pending['id']}: "
                    f"{pending.get('change', '')}); drop --expect, so the one written before "
                    "the run counts"
                ),
                file=sys.stderr,
            )
            return exit_codes.USAGE
        expectations = [gate.Expectation.from_json(d) for d in pending.get("expected", [])]
        out.say(
            f"expectation: {pending['id']} ({pending.get('change', '')}), recorded "
            f"{pending['time']}"
        )
    tables = _load_tables([args.new, baseline], args, out)
    if isinstance(tables, int):
        return tables
    try:
        result = gate.run_gate(tables[0], tables[1], policy=policy, expectations=expectations)
    except api.NotEnoughData as exc:
        _not_enough(out, exc, policy.alpha, policy.delta)
        return exit_codes.NOT_ENOUGH_DATA
    except predictions.InputError as exc:
        for p in exc.problems:
            out.problem(p)
        return exit_codes.BAD_INPUT
    t = tables[0]
    context = {
        "model": args.model or (t.model if t.model != "?" else None),
        "prompt": args.prompt,
        "dataset": args.dataset or t.dataset,
    }
    entry = changelog.gate_entry(
        result,
        entries=entries,
        new_path=None if str(args.new) == "-" else args.new,
        baseline_path=baseline,
        change=args.change,
        why=args.why,
        context=context,
        expectation=pending,
    )
    if not args.no_record:
        changelog.append(log_path, entry)
    if not args.quiet:
        for w in result.warnings:
            out.say(out.err.warning(w))
        if entry["rerun_of"]:
            out.say(
                out.err.warning(
                    f"these files were gated before ({entry['rerun_of']}); that run's "
                    "expectation is the one that counts"
                )
            )
    if args.json:
        res = result.to_json()
        res["changelog_entry"] = entry
        inputs: dict[str, Any] = {
            "new": str(args.new),
            "baseline": str(baseline),
            "config": str(cfg.path) if cfg is not None else None,
        }
        doc = summaries.envelope("gate", res, [], inputs)
        print(json.dumps(summaries.to_jsonable(doc), indent=2, ensure_ascii=False))
    else:
        print(summaries.wrap(f"{result.header()}\n{result.summary()}"))
    if not args.no_record and not args.quiet:
        out.say(f"changelog: appended {entry['id']} to {log_path}")
    return exit_codes.OK if result.passed else exit_codes.NO


def _run_changelog(args: argparse.Namespace) -> int:
    from ask_or_act import changelog, exit_codes, gate, project_config, summaries

    out = _Out(argparse.Namespace(quiet=args.json, color="auto"))
    try:
        cfg = _config(argparse.Namespace(no_config=args.no_config, quiet=args.json), out)
        table: dict[str, Any] = cfg.table("gate") if cfg is not None else {}
    except project_config.ConfigError as exc:
        print(out.err.error(str(exc)), file=sys.stderr)
        return exit_codes.USAGE
    chosen = getattr(args, "sub_file", None) or args.file
    if chosen is None and "changelog" in table:
        p = Path(str(table["changelog"]))
        chosen = p if p.is_absolute() else cfg.folder / p
    path = chosen or changelog.DEFAULT_PATH
    try:
        entries = changelog.read(path)
    except changelog.ChangelogError as exc:
        print(out.err.error(str(exc)), file=sys.stderr)
        return exit_codes.BAD_INPUT
    if args.changelog_command == "expect":
        try:
            exps = [gate.Expectation.parse(t) for t in args.expect]
        except ValueError as exc:
            print(out.err.error(str(exc)), file=sys.stderr)
            return exit_codes.USAGE
        pending = changelog.open_expectation(entries)
        if pending is not None:
            print(
                out.err.error(
                    f"{path} already has an open expectation ({pending['id']}); run the gate "
                    "first, so each run has one prediction"
                ),
                file=sys.stderr,
            )
            return exit_codes.USAGE
        entry = changelog.expectation_entry(change=args.change, why=args.why, expectations=exps)
        changelog.append(path, entry)
        print(f"recorded {entry['id']} in {path}: the next gate run checks it")
        return exit_codes.OK
    if args.changelog_command == "add":
        try:
            doc = json.loads(args.gate_json.read_text(encoding="utf-8"))
            entry = doc["result"]["changelog_entry"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(
                out.err.error(f"{args.gate_json}: not a gate --json output ({exc})"),
                file=sys.stderr,
            )
            return exit_codes.BAD_INPUT
        if any(e["id"] == entry.get("id") for e in entries):
            print(f"{entry.get('id')} is already in {path}; nothing appended")
            return exit_codes.OK
        try:
            changelog.append(path, entry)
        except changelog.ChangelogError as exc:
            print(out.err.error(f"{args.gate_json}: {exc}"), file=sys.stderr)
            return exit_codes.BAD_INPUT
        print(f"appended {entry['id']} to {path}")
        return exit_codes.OK
    if args.json:
        shown = entries[-args.last :] if args.last else entries
        result: Any = {"entries": list(shown)}
        doc = summaries.envelope("changelog", result, [], {"file": str(path)})
        print(json.dumps(doc, indent=2, ensure_ascii=False))
    else:
        print(summaries.wrap(changelog.render(entries, args.last)))
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
    if args.command == "gate":
        return _guarded(_run_gate, args)
    if args.command == "changelog":
        return _guarded(_run_changelog, args)
    if args.command == "demo":
        return _guarded(_run_demo, args)
    if args.command == "judge":
        return _run_judge(args)
    if args.command == "study":
        return _run_study(args)
    if args.command == "audit-redaction":
        return _run_audit(args)
    parser.print_help()
    return 0

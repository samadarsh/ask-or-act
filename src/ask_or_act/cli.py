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
    run.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    return parser


def _run_run(args: argparse.Namespace) -> int:
    from ask_or_act.core.config import ConfigError
    from ask_or_act.run import run_from_config

    try:
        return run_from_config(
            args.config, raw_dir=args.raw_dir, dry_run=args.dry_run, approve_spend=args.yes
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
    parser.print_help()
    return 0

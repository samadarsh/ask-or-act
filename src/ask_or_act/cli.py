"""Command-line entry point. Subcommands arrive in milestone 5."""

import argparse
from collections.abc import Sequence

from ask_or_act import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ask-or-act",
        description="Evaluate and calibrate AI decisions: when to act, ask, or escalate.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0

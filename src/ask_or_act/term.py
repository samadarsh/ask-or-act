"""Terminal output for the CLI: colour decisions and message prefixes (docs/cli-design.md §1).

Colour is decided per stream, in this order:

1. ``--color never`` / ``--color always``;
2. ``NO_COLOR`` set and non-empty: off (it beats FORCE_COLOR);
3. ``FORCE_COLOR`` set and non-empty: on;
4. ``TERM=dumb``: off;
5. otherwise on only if the stream is a TTY.

Python 3.14's argparse colours help itself, with FORCE_COLOR beating NO_COLOR, so the
CLI passes our decision to it instead of letting it decide.
"""

import os
from collections.abc import Mapping
from typing import Literal, TextIO

type ColorMode = Literal["auto", "always", "never"]

_CODES = {"red": "31", "yellow": "33", "green": "32", "bold": "1", "dim": "2"}


def use_color(
    stream: TextIO, mode: ColorMode = "auto", env: Mapping[str, str] | None = None
) -> bool:
    env = os.environ if env is None else env
    if mode == "never":
        return False
    if mode == "always":
        return True
    if env.get("NO_COLOR"):
        return False
    if env.get("FORCE_COLOR"):
        return True
    if env.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


class Painter:
    """Wraps text in ANSI colour codes, or returns it unchanged when colour is off."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, style: str) -> str:
        if not self.enabled:
            return text
        return f"\x1b[{_CODES[style]}m{text}\x1b[0m"

    def error(self, text: str) -> str:
        return f"{self('error:', 'red')} {text}"

    def warning(self, text: str) -> str:
        return f"{self('warning:', 'yellow')} {text}"


def plural(n: int, word: str, many: str | None = None) -> str:
    """``plural(1, "row")`` → "1 row"; ``plural(2_400, "row")`` → "2,400 rows"."""
    return f"{n:,} {word if n == 1 else (many or word + 's')}"

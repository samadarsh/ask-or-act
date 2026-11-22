"""The terminal sessions in docs/cli-design.md marked *real output* are real.

Each block preceded by ``<!-- real output (exit N) -->`` starts with ``$ ask-or-act …``.
The test runs that command in-process from tests/fixtures/predictions/, with stdout and
stderr in one buffer (the order a terminal would show), and compares the result with
the block and the exit code with N. Change the output, and the document must change
too.
"""

import contextlib
import io
import re
import shlex
from pathlib import Path

import pytest

from ask_or_act.cli import main

ROOT = Path(__file__).parent.parent
DOC = ROOT / "docs" / "cli-design.md"
FIXTURES = Path(__file__).parent / "fixtures" / "predictions"
BLOCK = re.compile(r"<!-- real output \(exit (\d+)\) -->\n```console\n(.*?)\n```", re.S)


def _sessions() -> list[tuple[int, str, str]]:
    out = []
    for m in BLOCK.finditer(DOC.read_text(encoding="utf-8")):
        first, _, rest = m.group(2).partition("\n")
        assert first.startswith("$ ask-or-act "), first
        out.append((int(m.group(1)), first[2:], rest))
    return out


SESSIONS = _sessions()


def test_the_doc_has_real_sessions() -> None:
    assert len(SESSIONS) >= 5


@pytest.mark.parametrize(("code", "command", "expected"), SESSIONS, ids=[s[1] for s in SESSIONS])
def test_documented_session_matches_real_output(
    code: int, command: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(FIXTURES)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        got = main(shlex.split(command)[1:])
    assert buf.getvalue().rstrip() == expected.rstrip()
    assert got == code

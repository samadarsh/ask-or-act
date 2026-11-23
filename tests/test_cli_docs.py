"""The terminal sessions in docs/cli-design.md marked *real output* are real.

Each block preceded by ``<!-- real output (exit N) -->`` starts with ``$ ask-or-act …``.
The test runs that command in-process in a copy of tests/fixtures/predictions/, with stdout and
stderr in one buffer (the order a terminal would show), and compares the result with
the block and the exit code with N. Change the output, and the document must change
too.
"""

import contextlib
import io
import re
import shlex
import shutil
import tempfile
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
    code: int, command: str, expected: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)
    work = tmp_path / "fixtures"  # a copy, so a command that saves a file can't touch the repo
    shutil.copytree(FIXTURES, work)
    monkeypatch.chdir(work)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        got = main(shlex.split(command)[1:])
    assert buf.getvalue().rstrip() == expected.rstrip()
    assert got == code


def _rewrite() -> None:
    """Refresh every real-output block from the real commands (after an intended change;
    review the diff before committing)."""
    import os

    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        os.environ.pop(name, None)
    text = DOC.read_text(encoding="utf-8")
    here = Path.cwd()
    tmp = tempfile.TemporaryDirectory()
    shutil.copytree(FIXTURES, Path(tmp.name) / "fixtures")
    os.chdir(Path(tmp.name) / "fixtures")

    def run(m: re.Match[str]) -> str:
        first = m.group(2).partition("\n")[0]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = main(shlex.split(first[2:])[1:])
        body = buf.getvalue().rstrip()
        return f"<!-- real output (exit {code}) -->\n```console\n{first}\n{body}\n```"

    try:
        DOC.write_text(BLOCK.sub(run, text), encoding="utf-8")
    finally:
        os.chdir(here)
        tmp.cleanup()


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--write"]:
        _rewrite()
        print(f"rewrote the real-output blocks in {DOC}")

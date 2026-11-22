"""``ask-or-act check``: exit codes, streams, JSON, colour, stdin. No network."""

import io
import json
import sys
from pathlib import Path

import pytest

from ask_or_act import exit_codes, term
from ask_or_act.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "predictions"


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture(autouse=True)
def _no_colour_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)


def test_a_clean_file_exits_0_and_a_broken_one_exits_6(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, _ = _run(["check", str(FIXTURES / "support-bot.csv")], capsys)
    assert code == exit_codes.OK
    assert "Ready for metrics, thresholds and report." in out
    code, out, _ = _run(["check", str(FIXTURES / "scores.csv")], capsys)
    assert code == exit_codes.BAD_INPUT
    assert "2 errors: fix and run check again." in out


def test_strict_turns_warnings_into_exit_1(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, _ = _run(["check", str(FIXTURES / "pilot.csv"), "--strict"], capsys)
    assert code == exit_codes.NO


def test_json_is_one_document_on_stdout_with_a_schema(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(["check", str(FIXTURES / "scores.csv"), "--json"], capsys)
    assert code == exit_codes.BAD_INPUT
    doc = json.loads(out)  # nothing else on stdout
    assert doc["schema"] == "ask-or-act/check/v1"
    assert doc["result"]["ok"] is False
    assert {e["code"] for e in doc["result"]["errors"]} == {"missing_column", "conf_range"}
    assert err == ""
    code, out, _ = _run(["check", str(FIXTURES / "support-bot.csv"), "--json"], capsys)
    doc = json.loads(out)
    assert doc["result"]["labelled"] == {"calibration": 1000, "test": 1000}
    assert doc["result"]["min_calibration_rows"]["no_errors"] == 36
    assert [w["code"] for w in doc["warnings"]] == ["failed", "unlabelled"]


def test_quiet_keeps_only_the_verdict(capsys: pytest.CaptureFixture[str]) -> None:
    _, out, _ = _run(["check", str(FIXTURES / "support-bot.csv"), "-q"], capsys)
    assert "✓" not in out
    assert "!" not in out
    assert "Ready for" in out


def test_stdin_with_a_format(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO((FIXTURES / "pilot.csv").read_text()))
    code, out, _ = _run(["check", "-", "--format", "csv"], capsys)
    assert code == exit_codes.OK
    assert out.startswith("stdin: 40 rows (CSV)")


def test_stdin_without_a_format_is_an_input_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("id\n"))
    code, out, _ = _run(["check", "-"], capsys)
    assert code == exit_codes.BAD_INPUT
    assert "pass --format csv, tsv or jsonl" in out


def test_usage_errors_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (
        ["check", str(FIXTURES / "pilot.csv"), "--confidence-scale", "fraction"],
        ["check", str(FIXTURES / "pilot.csv"), "--columns", "nonsense"],
        ["check", str(FIXTURES / "pilot.csv"), "--alpha", "2"],
    ):
        try:
            code = main(argv)
        except SystemExit as exc:  # argparse exits 2 itself
            code = int(exc.code or 0)
        assert code == exit_codes.USAGE, argv
    capsys.readouterr()


def test_no_prefix_abbreviations(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["check", str(FIXTURES / "pilot.csv"), "--alph", "0.2"])
    assert info.value.code == exit_codes.USAGE
    capsys.readouterr()


def test_a_mistyped_command_gets_a_suggestion(capsys: pytest.CaptureFixture[str]) -> None:
    try:
        code = main(["chek", "x.csv"])
    except SystemExit as exc:  # Python 3.14: argparse's own suggestion
        code = int(exc.code or 0)
    _, err = capsys.readouterr()
    assert code == exit_codes.USAGE
    assert "'check'" in err


def test_a_crash_exits_70_with_a_report_link(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ask_or_act import predictions

    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("synthetic bug")

    monkeypatch.setattr(predictions, "check", boom)
    code, _, err = _run(["check", str(FIXTURES / "pilot.csv")], capsys)
    assert code == exit_codes.INTERNAL
    assert "synthetic bug" in err
    assert exit_codes.ISSUES_URL in err


def test_help_says_local_only_and_leads_with_examples(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    top = capsys.readouterr().out
    assert "no network connections, no telemetry" in top
    with pytest.raises(SystemExit):
        main(["check", "--help"])
    sub = capsys.readouterr().out
    assert "examples:" in sub
    assert "no network connections, no telemetry" in sub


@pytest.mark.parametrize(
    ("mode", "env", "tty", "expected"),
    [
        ("auto", {}, True, True),
        ("auto", {}, False, False),
        ("auto", {"NO_COLOR": "1"}, True, False),
        ("auto", {"NO_COLOR": "1", "FORCE_COLOR": "1"}, True, False),  # NO_COLOR wins
        ("auto", {"FORCE_COLOR": "1"}, False, True),
        ("auto", {"NO_COLOR": ""}, True, True),  # empty means unset
        ("auto", {"TERM": "dumb"}, True, False),
        ("never", {"FORCE_COLOR": "1"}, True, False),
        ("always", {"NO_COLOR": "1"}, False, True),
    ],
)
def test_colour_precedence(
    mode: term.ColorMode, env: dict[str, str], tty: bool, expected: bool
) -> None:
    class Stream(io.StringIO):
        def isatty(self) -> bool:
            return tty

    assert term.use_color(Stream(), mode, env) is expected


def test_colour_appears_only_when_asked(capsys: pytest.CaptureFixture[str]) -> None:
    _, plain, _ = _run(["check", str(FIXTURES / "scores.csv")], capsys)
    assert "\x1b[" not in plain
    _, coloured, _ = _run(["check", str(FIXTURES / "scores.csv"), "--color", "always"], capsys)
    assert "\x1b[31merror:\x1b[0m" in coloured

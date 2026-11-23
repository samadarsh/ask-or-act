"""``ask-or-act report`` on predictions files (report/files.py). No network.

The summary at the top of the page must be the terminal's text, word for word, so the
first test compares them; the rest check the labels (TARGET, never PRE-REGISTERED), the
marked threshold, the unlabelled-coverage tile, that the page loads nothing external,
and the exit codes.
"""

import html
import re
from pathlib import Path

import pytest

import ask_or_act as aoa
from ask_or_act import exit_codes
from ask_or_act.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "predictions"
SMALL = str(FIXTURES / "support-bot.csv")
LARGE = str(FIXTURES / "support-bot-large.csv")
FAST = ["--n-boot", "200"]
SAY = re.compile(r'<div class="say" data-command="([^"]*)">(.*?)</div>', re.S)


@pytest.fixture(autouse=True)
def _no_colour_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)


def _words(text: str) -> str:
    return " ".join(text.split())


def _report(argv: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    out = tmp_path / "r.html"
    code = main(["report", *argv, "-o", str(out), *FAST])
    capsys.readouterr()
    assert code == exit_codes.OK
    return out.read_text(encoding="utf-8")


def _blocks(page: str) -> dict[str, str]:
    """{command: its text}, tags stripped."""
    return {
        html.unescape(cmd): _words(html.unescape(re.sub(r"<[^>]+>", " ", body)))
        for cmd, body in SAY.findall(page)
    }


def _terminal(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    main(argv)
    return _words(capsys.readouterr().out)


def test_the_summary_is_the_terminal_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(FIXTURES)  # file names as the user typed them
    page = _report(["support-bot-large.csv"], tmp_path, capsys)
    blocks = _blocks(page)
    assert blocks["ask-or-act thresholds support-bot-large.csv"] == _terminal(
        ["thresholds", "support-bot-large.csv"], capsys
    )
    assert blocks["ask-or-act metrics support-bot-large.csv"] == _terminal(
        ["metrics", "support-bot-large.csv", *FAST], capsys
    )


def test_two_files_lead_with_the_compare_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(FIXTURES)
    page = _report(["support-bot.csv", "support-bot-large.csv"], tmp_path, capsys)
    blocks = _blocks(page)
    cmd = "ask-or-act compare support-bot.csv support-bot-large.csv"
    assert next(iter(blocks)) == cmd  # first on the page
    assert blocks[cmd] == _terminal(
        ["compare", "support-bot.csv", "support-bot-large.csv", *FAST], capsys
    )
    for name in ("support-bot.csv", "support-bot-large.csv"):  # and each model's thresholds
        assert blocks[f"ask-or-act thresholds {name}"] == _terminal(["thresholds", name], capsys)


def test_user_targets_are_target_never_pre_registered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    page = _report([LARGE], tmp_path, capsys)
    assert 'class="badge target">TARGET' in page
    assert 'class="badge pre"' not in page
    assert "the default; set it with --alpha" in page
    page = _report([LARGE, "--alpha", "0.2"], tmp_path, capsys)
    assert "your --alpha" in page
    assert "α = 0.2, δ = 0.1" in page


def test_the_threshold_is_marked_on_the_risk_coverage_curve(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    page = _report([LARGE], tmp_path, capsys)
    th = aoa.find_thresholds(LARGE)
    assert page.count('class="mark"') == 1
    assert f"act when confidence ≥ {th.threshold:.3g}: covers" in page
    assert '<line class="ref"' in page  # α
    page = _report([SMALL], tmp_path, capsys)  # nothing certified: nothing to mark
    assert 'class="mark"' not in page
    assert "nothing is marked" in page


def test_unlabelled_coverage_is_estimated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    page = _report([LARGE], tmp_path, capsys)
    th = aoa.find_thresholds(LARGE)
    assert th.unlabelled == 400
    assert f"{th.unlabelled_acted:,} of 400 unlabelled rows have confidence" in page


def test_the_page_loads_nothing_external(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    page = _report([SMALL, LARGE], tmp_path, capsys)
    assert not re.search(r"<(script|img|iframe)[^>]*\ssrc=", page)
    assert "<link" not in page
    assert "@import" not in page
    assert "url(" not in page


def test_too_few_rows_still_reports_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "r.html"
    code = main(["report", str(FIXTURES / "pilot.csv"), "-o", str(out), *FAST])
    err = capsys.readouterr().err
    assert code == exit_codes.OK
    assert "no threshold at α = 0.1: too few labelled calibration rows" in err
    assert "--alpha 0.2 needs 17" in out.read_text(encoding="utf-8")


def test_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = str(tmp_path / "r.html")
    rows = Path(LARGE).read_text().splitlines()
    (tmp_path / "part.csv").write_text("\n".join(rows[:500]) + "\n")
    assert main(["report", SMALL, str(tmp_path / "part.csv"), "-o", out, *FAST]) == 6
    assert "different examples" in capsys.readouterr().err
    assert main(["report", SMALL, SMALL, SMALL, SMALL, "-o", out]) == exit_codes.USAGE
    assert main(["report", str(FIXTURES / "scores.csv"), "-o", out]) == exit_codes.BAD_INPUT
    assert not Path(out).exists()  # nothing written on failure


def test_the_api_writes_the_same_page(tmp_path: Path) -> None:
    path = aoa.write_report([SMALL, LARGE], tmp_path / "api.html", n_boot=200)
    page = path.read_text(encoding="utf-8")
    assert "support-bot-large is more accurate than support-bot" in page

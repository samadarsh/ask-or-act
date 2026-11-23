"""``thresholds``, ``metrics`` and ``compare`` on predictions files: streams, exit codes,
JSON, and that run folders still take the benchmark path. No network."""

import json
from pathlib import Path

import pytest

from ask_or_act import exit_codes
from ask_or_act.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "predictions"
SMALL = str(FIXTURES / "support-bot.csv")
LARGE = str(FIXTURES / "support-bot-large.csv")
FAST = ["--n-boot", "100"]


@pytest.fixture(autouse=True)
def _no_colour_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    try:
        code = main(argv)
    except SystemExit as exc:  # argparse usage errors
        code = int(exc.code or 0)
    out, err = capsys.readouterr()
    return code, out, err


def test_thresholds_result_on_stdout_and_warnings_on_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = _run(["thresholds", LARGE], capsys)
    assert code == exit_codes.OK
    assert "Act when confidence ≥" in out
    assert "warning" not in out
    assert "warning: " in err
    assert "support-bot-large.csv: 38 rows have an empty prediction" in err


def test_quiet_hides_warnings(capsys: pytest.CaptureFixture[str]) -> None:
    _, _, err = _run(["thresholds", LARGE, "-q"], capsys)
    assert err == ""


def test_json_is_one_document_and_nothing_else(capsys: pytest.CaptureFixture[str]) -> None:
    for argv, schema in (
        (["thresholds", LARGE, "--json"], "thresholds"),
        (["metrics", SMALL, "--json", *FAST], "metrics"),
        (["compare", SMALL, LARGE, "--json", *FAST], "compare"),
    ):
        code, out, _ = _run(argv, capsys)
        assert code == exit_codes.OK
        doc = json.loads(out)
        assert doc["schema"] == f"ask-or-act/{schema}/v1"
        assert doc["result"]


def test_too_few_rows_exit_7_and_bad_input_exit_6(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run(["thresholds", str(FIXTURES / "pilot.csv")], capsys)
    assert code == exit_codes.NOT_ENOUGH_DATA
    assert out == ""
    assert "--alpha 0.2 needs 17" in err
    code, _, err = _run(["thresholds", str(FIXTURES / "scores.csv")], capsys)
    assert code == exit_codes.BAD_INPUT
    assert "ask-or-act check" in err
    code, _, _ = _run(["metrics", str(FIXTURES / "scores.csv")], capsys)
    assert code == exit_codes.BAD_INPUT


def test_act_on_nothing_is_success(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["thresholds", SMALL], capsys)
    assert code == exit_codes.OK
    assert "Act on nothing" in out


def test_metrics_says_which_rows_it_scored(capsys: pytest.CaptureFixture[str]) -> None:
    _, out, _ = _run(["metrics", SMALL, *FAST], capsys)
    assert out.startswith("support-bot.csv, test half (--all scores every labelled row): 1,000")
    _, out, _ = _run(["metrics", SMALL, "--all", *FAST], capsys)
    assert out.startswith("support-bot.csv, all labelled rows: 2,000")


def test_metrics_with_two_files_points_to_compare(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = _run(["metrics", SMALL, LARGE], capsys)
    assert code == exit_codes.USAGE
    assert "ask-or-act compare A B" in err


def test_flag_conflicts_are_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, _ = _run(["thresholds", SMALL, "--cal", LARGE], capsys)
    assert code == exit_codes.USAGE
    code, _, _ = _run(["thresholds", SMALL, "--signal", "verbalized"], capsys)
    assert code == exit_codes.USAGE
    code, _, _ = _run(["thresholds"], capsys)
    assert code == exit_codes.USAGE


def test_explicit_halves_from_files(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    rows = Path(LARGE).read_text().splitlines()
    header, body = rows[0], rows[1:]
    (tmp_path / "cal.csv").write_text("\n".join([header, *body[:1200]]) + "\n")
    (tmp_path / "test.csv").write_text("\n".join([header, *body[1200:]]) + "\n")
    code, out, _ = _run(
        ["thresholds", "--cal", str(tmp_path / "cal.csv"), "--test", str(tmp_path / "test.csv")],
        capsys,
    )
    assert code == exit_codes.OK
    assert "(--cal/--test files)" in out


def test_compare_needs_the_same_ids(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    rows = Path(LARGE).read_text().splitlines()
    (tmp_path / "part.csv").write_text("\n".join(rows[:500]) + "\n")
    code, _, err = _run(["compare", SMALL, str(tmp_path / "part.csv"), *FAST], capsys)
    assert code == exit_codes.BAD_INPUT
    assert "different examples" in err


def test_run_folders_still_take_the_benchmark_path(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from ask_or_act.data.sampling import Split
    from tests.test_report import _save

    run = _save(tmp_path / "runs", "cal", "calibrated", Split.CALIBRATION, n=300)
    code, out, _ = _run(["thresholds", "--cal", str(run)], capsys)
    assert code == exit_codes.OK
    assert "target α = 0.1: pre-registered" in out  # the run-folder report, unchanged
    code, out, _ = _run(["metrics", str(run), *FAST], capsys)
    assert code == exit_codes.OK

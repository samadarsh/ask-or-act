"""``ask-or-act gate`` and ``ask-or-act changelog`` (docs/cli-design.md §3.6). No network.

Files are generated: 2,000 items shared by the new file and the baseline, so they're
paired as real ones are (scripts/gate_simulation.py uses the same model).
"""

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from ask_or_act import changelog, gate_markdown
from ask_or_act.cli import main

N_BOOT = ["--n-boot", "200"]


def write_preds(
    path: Path,
    *,
    seed: int,
    shift: float = 0.0,
    bias: float = 0.0,
    noise: float = 0.35,
    n: int = 2000,
    items: int = 0,
) -> Path:
    """Item difficulties come from ``items`` (shared across files); the draws of who is
    right, and the confidences, from ``seed``."""
    p = np.clip(np.random.default_rng(items).beta(5.0, 1.0, n), 0.02, 0.995)
    rng = np.random.default_rng(1000 + seed)
    q = np.clip(p + shift, 0.01, 0.999)
    right = rng.uniform(size=n) < q
    z = np.log(q / (1 - q)) + bias + rng.normal(0.0, noise, n)
    conf = np.clip(1 / (1 + np.exp(-z)), 0.001, 0.999)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction", "confidence", "human_label"])
        for i in range(n):
            w.writerow([f"t{i:05d}", "yes" if right[i] else "no", f"{conf[i]:.4f}", "yes"])
    return path


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh project folder (a git repo root) as the current directory."""
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)
    write_preds(tmp_path / "base.csv", seed=1)
    write_preds(tmp_path / "same.csv", seed=1)  # byte-identical to the baseline
    write_preds(tmp_path / "worse.csv", seed=2, shift=-0.08)
    return tmp_path


def words(text: str) -> str:
    """The output with line wrapping undone."""
    return " ".join(text.split())


def gate(*args: str) -> int:
    return main(["gate", *args, *N_BOOT])


def log_entries(path: Path = Path("ask-or-act-changelog.jsonl")) -> list[dict[str, object]]:
    return changelog.read(path)


def test_an_identical_file_passes(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gate("same.csv", "--baseline", "base.csv", "--expect", "accuracy -1 to +1") == 0
    out = words(capsys.readouterr().out)
    assert "PASS: 4 of 4 checks passed." in out
    assert "accuracy -1 to +1: held (observed +0.0 points)" in out
    [entry] = log_entries()
    assert entry["gate"] == "pass"
    assert entry["kind"] == "gate"


def test_a_worse_file_fails_with_exit_1(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = gate(
        "worse.csv",
        "--baseline",
        "base.csv",
        "--change",
        "smaller model",
        "--why",
        "cost",
        "--expect",
        "accuracy -2 to 0 points",
    )
    assert code == 1
    out = words(capsys.readouterr().out)
    assert "✗ accuracy" in out
    assert "worse beyond the 1-point tolerance" in out
    assert out.count("FAIL:") == 1
    assert "accuracy -2 to 0 points: missed (observed -9.5 points, beyond noise)" in out
    assert "beyond noise" in out
    [e] = log_entries()
    assert e["gate"] == "fail"
    assert (e["change"], e["why"]) == ("smaller model", "cost")
    files = e["files"]
    assert isinstance(files, dict)
    assert files["new"]["path"] == "worse.csv"
    assert len(files["baseline"]["sha256"]) == 64
    observed = e["observed"]
    assert isinstance(observed, dict)
    assert observed["accuracy"]["status"] == "fail"
    verdicts = e["verdicts"]
    assert isinstance(verdicts, list)
    assert verdicts[0]["held"] is False
    assert verdicts[0]["beyond_noise"] is True


def test_json_is_one_document_with_the_entry(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert gate("worse.csv", "--baseline", "base.csv", "--json", "-q", "--no-record") == 1
    doc = json.loads(words(capsys.readouterr().out))
    assert doc["schema"] == "ask-or-act/gate/v1"
    res = doc["result"]
    assert res["passed"] is False
    assert [c["name"] for c in res["checks"]] == [
        "accuracy",
        "calibration",
        "coverage",
        "certified",
    ]
    acc = res["checks"][0]
    assert acc["status"] == "fail"
    assert acc["p_holm"] < 0.05
    assert acc["low"] < acc["diff"] < acc["high"]
    assert res["changelog_entry"]["schema"] == changelog.SCHEMA
    assert not Path("ask-or-act-changelog.jsonl").exists()  # --no-record


def test_different_ids_exit_6(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_preds(repo / "short.csv", seed=1, n=1800)
    assert gate("short.csv", "--baseline", "base.csv") == 6
    assert "must have the same labelled ids" in words(capsys.readouterr().err)
    assert not Path("ask-or-act-changelog.jsonl").exists()


def test_too_few_rows_exit_7(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_preds(repo / "a.csv", seed=1, n=30)
    write_preds(repo / "b.csv", seed=2, n=30)
    assert gate("a.csv", "--baseline", "b.csv") == 7
    assert "too few labelled calibration rows" in words(capsys.readouterr().err)


def test_no_baseline_is_a_usage_error(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gate("same.csv") == 2
    assert "no baseline" in words(capsys.readouterr().err)


def test_bad_expectation_is_a_usage_error(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gate("same.csv", "--baseline", "base.csv", "--expect", "accuracy up") == 2
    assert "can't read the expectation" in words(capsys.readouterr().err)


# --------------------------------------------------------------------------- strict mode


@pytest.fixture
def unsure(repo: Path) -> Path:
    """Another draw of the same model, with tolerance 0: typically inconclusive."""
    write_preds(repo / "redraw.csv", seed=3)
    return repo


def test_inconclusive_passes_by_default(unsure: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ("redraw.csv", "--baseline", "base.csv", "--tolerance", "accuracy=0")
    assert gate(*args) == 0
    out = words(capsys.readouterr().out)
    assert "? accuracy" in out
    assert "inconclusive: could be up to" in out
    assert "PASS, with" in out


def test_fail_on_inconclusive(unsure: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ("redraw.csv", "--baseline", "base.csv", "--tolerance", "accuracy=0")
    assert gate(*args, "--fail-on-inconclusive") == 1
    out = words(capsys.readouterr().out)
    assert "FAIL: " in out
    # why it fails: the rows each check would need, next to the rows it has
    assert "strict: test rows needed" in out
    assert "accuracy: not provable (tolerance 0)" in out
    assert re.search(r"calibration ~[\d,]+, coverage ~[\d,]+; this run has [\d,]+\.", out)


def test_rows_needed_scales_with_the_tolerance(
    unsure: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Halving the tolerance needs about 4× the rows; it isn't printed unless strict."""
    needed = []
    for tol in ("0.04", "0.02"):
        gate("redraw.csv", "--baseline", "base.csv", "--tolerance", f"accuracy={tol}", "--json")
        doc = json.loads(capsys.readouterr().out)
        needed.append(doc["result"]["rows_needed"]["accuracy"])
    assert 3.5 < needed[1] / needed[0] < 4.5
    gate("redraw.csv", "--baseline", "base.csv")
    assert "rows needed" not in capsys.readouterr().out


def test_strict_in_the_policy(unsure: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (unsure / "ask-or-act.toml").write_text(
        '[gate]\nbaseline = "base.csv"\nstrict = true\ntolerance = { accuracy = 0 }\n'
    )
    assert gate("redraw.csv") == 1
    captured = capsys.readouterr()
    assert "config: ask-or-act.toml" in captured.err
    assert "; strict)" in words(captured.out)


# --------------------------------------------------------------------------- config


def test_config_is_found_up_to_the_repo_root_never_above(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "ask-or-act.toml").write_text('[gate]\nbaseline = "base.csv"\n')
    sub = repo / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert (
        main(["gate", str(repo / "same.csv"), *N_BOOT, "--changelog", str(repo / "l.jsonl")]) == 0
    )
    assert "ask-or-act.toml" in words(capsys.readouterr().err)
    # a config above the repository root is never read
    inner = repo / "inner"
    (inner / ".git").mkdir(parents=True)
    monkeypatch.chdir(inner)
    assert main(["gate", str(repo / "same.csv"), *N_BOOT]) == 2
    assert "no baseline" in words(capsys.readouterr().err)


def test_ask_or_act_toml_wins_over_pyproject(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "pyproject.toml").write_text('[tool.ask-or-act.gate]\nbaseline = "nope.csv"\n')
    assert gate("same.csv") == 6  # pyproject's baseline: no such file
    capsys.readouterr()
    (repo / "ask-or-act.toml").write_text('[gate]\nbaseline = "base.csv"\n')
    assert gate("same.csv") == 0


def test_flags_override_the_config_and_no_config_ignores_it(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "ask-or-act.toml").write_text(
        '[gate]\nbaseline = "same.csv"\nalpha = 0.2\nchangelog = "logs/decisions.jsonl"\n'
    )
    assert gate("worse.csv", "--alpha", "0.15", "--json", "-q") == 1
    doc = json.loads(words(capsys.readouterr().out))
    assert doc["result"]["policy"]["alpha"] == 0.15
    assert doc["input"]["baseline"] == "same.csv"
    assert len(log_entries(repo / "logs" / "decisions.jsonl")) == 1
    assert gate("worse.csv", "--no-config") == 2  # no baseline without the config


@pytest.mark.parametrize(
    ("toml", "match"),
    [
        ('[gate]\nbaseline = "base.csv"\nalhpa = 0.1\n', "unknown key 'alhpa'"),
        ('[gate]\nbaseline = "base.csv"\nstrict = "yes"\n', "'strict' must be bool"),
        ('[gate]\nbaseline = "base.csv"\nalpha = true\n', "'alpha' must be float"),
        ('[gate]\nbaseline = "base.csv"\ntolerance = { speed = 1 }\n', "unknown tolerance"),
        ('[gate]\nbaseline = "base.csv"\nalpha = 2\n', "alpha must be between 0 and 1"),
        ("[gate\n", "can't read it as TOML"),
    ],
)
def test_config_errors_exit_2(
    repo: Path, toml: str, match: str, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "ask-or-act.toml").write_text(toml)
    assert gate("same.csv") == 2
    assert match in words(capsys.readouterr().err)


def test_baseline_is_relative_to_the_config_file(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "ci").mkdir()
    (repo / "ci" / "gate.toml").write_text('[gate]\nbaseline = "../base.csv"\n')
    assert gate("same.csv", "--policy", "ci/gate.toml") == 0


def test_require_expectation(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (repo / "ask-or-act.toml").write_text(
        '[gate]\nbaseline = "base.csv"\nrequire_expectation = true\n'
    )
    assert gate("same.csv") == 1
    assert "the policy requires an expected effect" in words(capsys.readouterr().out)
    assert gate("same.csv", "--expect", "accuracy >= -1") == 0


def test_a_lost_certificate_can_be_allowed(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_preds(repo / "noisy.csv", seed=4, noise=2.5)  # same accuracy, useless confidence
    args = ("noisy.csv", "--baseline", "base.csv", "--tolerance", "coverage=0.99,calibration=0.9")
    assert gate(*args) == 1
    assert "✗ certified lost:" in words(capsys.readouterr().out)
    (repo / "ask-or-act.toml").write_text("[gate]\nrequire_certified = false\n")
    assert gate(*args) == 0
    assert "! certified lost (require_certified = false)" in words(capsys.readouterr().out)


# --------------------------------------------------------------------------- changelog


def test_expectation_before_the_run_is_used_and_closed(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "changelog",
                "expect",
                "--change",
                "v2 prompt",
                "--why",
                "tone",
                "--expect",
                "accuracy -1 to +1 points",
                "--expect",
                "coverage >= -3 points",
            ]
        )
        == 0
    )
    assert "the next gate run checks it" in words(capsys.readouterr().out)
    # a second open expectation is refused: one prediction per run
    assert main(["changelog", "expect", "--change", "x", "--expect", "accuracy >= 0"]) == 2
    capsys.readouterr()
    # --expect would replace the one written before the run: refused
    assert gate("same.csv", "--baseline", "base.csv", "--expect", "accuracy >= 0") == 2
    assert "drop --expect" in words(capsys.readouterr().err)
    assert gate("same.csv", "--baseline", "base.csv") == 0
    captured = capsys.readouterr()
    assert "expectation: " in captured.err
    assert "accuracy -1 to +1 points: held" in captured.out
    exp, run = log_entries()
    assert run["expectation_id"] == exp["id"]
    assert run["expected_recorded"] == exp["time"]
    assert run["change"] == "v2 prompt"  # taken from the expectation
    assert changelog.open_expectation(log_entries()) is None
    assert main(["changelog"]) == 0
    shown = words(capsys.readouterr().out)
    assert "EXPECTED v2 prompt" in shown
    assert "s before the run):" in shown
    assert "predictions held 2 of 2." in shown


def test_a_rerun_is_marked_and_its_new_guess_does_not_count(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert gate("worse.csv", "--baseline", "base.csv", "--expect", "accuracy >= 0") == 1
    assert gate("worse.csv", "--baseline", "base.csv", "--expect", "accuracy -10 to -5") == 1
    assert "were gated before" in words(capsys.readouterr().err)
    first, second = log_entries()
    assert second["rerun_of"] == first["id"]
    assert main(["changelog"]) == 0
    shown = words(capsys.readouterr().out)
    assert "that run's expectation is the one that counts" in shown
    assert "this rerun's expectation differs" in shown
    assert "1 gate run(s) (reruns not counted); predictions held 0 of 1." in shown


def test_changelog_add_from_a_ci_run(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gate("same.csv", "--baseline", "base.csv", "--json", "-q", "--no-record") == 0
    Path("gate.json").write_text(capsys.readouterr().out)
    assert main(["changelog", "add", "gate.json"]) == 0
    assert main(["changelog", "add", "gate.json"]) == 0
    assert "already in" in words(capsys.readouterr().out)
    assert len(log_entries()) == 1
    Path("junk.json").write_text("{}")
    assert main(["changelog", "add", "junk.json"]) == 6


def test_changelog_json_and_last(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for _ in range(3):
        gate("same.csv", "--baseline", "base.csv", "-q")
    capsys.readouterr()
    assert main(["changelog", "--json", "--last", "2"]) == 0
    doc = json.loads(words(capsys.readouterr().out))
    assert doc["schema"] == "ask-or-act/changelog/v1"
    assert len(doc["result"]["entries"]) == 2


def test_an_unreadable_changelog_exits_6(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    Path("ask-or-act-changelog.jsonl").write_text(
        '{"schema": "ask-or-act/changelog/v1"}\nnot json\n'
    )
    assert main(["changelog"]) == 6
    assert "ask-or-act-changelog.jsonl:2: not a JSON line" in words(capsys.readouterr().err)
    assert gate("same.csv", "--baseline", "base.csv") == 6


def test_an_empty_changelog(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["changelog"]) == 0
    assert "The changelog is empty" in words(capsys.readouterr().out)


def test_the_commit_comes_from_github_actions(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef0123")
    assert gate("same.csv", "--baseline", "base.csv", "-q") == 0
    [e] = log_entries()
    assert e["commit"] == "0123456789ab"


def test_entries_hold_no_input_rows(repo: Path) -> None:
    """Ids, hashes, numbers and the user's own --change/--why text only."""
    gate("worse.csv", "--baseline", "base.csv", "-q", "--change", "c", "--why", "w")
    text = Path("ask-or-act-changelog.jsonl").read_text()
    assert "t00000" not in text  # no row ids either: the log is per run, not per row
    assert all(not isinstance(v, float) or math.isfinite(v) for v in json.loads(text).values())


# --------------------------------------------------------------------------- -o markdown


def test_markdown_is_written_pass_or_fail(repo: Path) -> None:
    assert gate("same.csv", "--baseline", "base.csv", "-o", "out/pass.md") == 0
    md = (repo / "out" / "pass.md").read_text()
    assert md.startswith(gate_markdown.MARKER + "\n## ✅ ask-or-act gate: PASS")
    assert "| ✅ | accuracy |" in md
    assert "Changelog: appended `" in md  # recorded locally
    args = ("worse.csv", "--baseline", "base.csv", "--no-record", "-o", "fail.md")
    assert gate(*args, "--expect", "accuracy +0 to +2 points") == 1
    md = (repo / "fail.md").read_text()
    assert "## ❌ ask-or-act gate: FAIL" in md
    assert "| ❌ | accuracy |" in md
    assert "missed (observed" in md
    assert "ask-or-act changelog add gate.json" in md  # not recorded: how to keep it


def test_markdown_escapes_free_text(repo: Path) -> None:
    change = "<img src=x onerror=alert(1)> | `rm -rf`\nsecond line"
    gate("same.csv", "--baseline", "base.csv", "--change", change, "-o", "g.md")
    md = (repo / "g.md").read_text()
    assert "<img" not in md
    assert "&lt;img src=x onerror=alert(1)&gt; \\| 'rm -rf' second line" in md


def test_markdown_holds_no_input_rows(repo: Path) -> None:
    gate("worse.csv", "--baseline", "base.csv", "--fail-on-inconclusive", "-o", "g.md")
    md = (repo / "g.md").read_text()
    assert "t00001" not in md  # ids
    assert "strict: test rows needed" in md


def test_gate_fixtures_match_their_script(tmp_path: Path) -> None:
    from tests.fixtures.gate import make_fixtures

    make_fixtures.main(tmp_path)
    for name in make_fixtures.FILES:
        assert (tmp_path / name).read_bytes() == (make_fixtures.HERE / name).read_bytes(), name


def test_inconclusive_tolerance_wording(unsure: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gate("redraw.csv", "--baseline", "base.csv", "--tolerance", "accuracy=0.01,coverage=0.05")
    out = words(capsys.readouterr().out)
    assert "-point)" not in out
    assert re.search(r"\(tolerance (1 point|5 points|0\.02)\)", out) or "inconclusive" not in out

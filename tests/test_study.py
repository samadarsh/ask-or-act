"""`ask-or-act study`: daily runs of a multi-config schedule, quota stops, progress."""

from pathlib import Path

import pytest

from ask_or_act.cli import main
from ask_or_act.study import load_schedule, run, status
from tests.provider_fixtures import Recorder, groq_429, groq_ok
from tests.test_run_end_to_end import write_world


def groq_cfg(tmp_path: Path, manifest_path: Path, name: str, params: str = "") -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(f"""
name = "{name}"
[dataset]
manifest = "{manifest_path}"
split = "test"
[decider]
type = "groq"
model = "openai/gpt-oss-120b"
prompt_version = "v1"
[decider.params]
{params}
[execution]
max_concurrency = 1
max_retries = 0
requests_per_day = 10
tokens_per_day = 10000
[budget]
max_usd = 0.0
[cache]
path = "{tmp_path / "cache.sqlite"}"
[output]
runs_dir = "{tmp_path / "runs"}"
""")
    return path


def schedule(tmp_path: Path, configs: list[Path]) -> Path:
    path = tmp_path / "schedule.toml"
    path.write_text(
        "usable_share = 1.0\nconfigs = [" + ", ".join(f'"{c}"' for c in configs) + "]\n"
    )
    return path


def test_two_days_main_then_self_consistency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = write_world(tmp_path, monkeypatch)  # 6 test examples
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    main_cfg = groq_cfg(tmp_path, manifest_path, "main")
    sc_cfg = groq_cfg(
        tmp_path,
        manifest_path,
        "sc",
        'samples = 1\nconfidence_signal = "self_consistency"\nsample_variation = "shuffle"',
    )
    sched = load_schedule(schedule(tmp_path, [main_cfg, sc_cfg]))
    lines: list[str] = []

    # Before anything: 6 main calls + 6 samples. The SC config's own main calls are
    # the ones the main config makes, so they aren't counted twice.
    (before,) = status(sched, tmp_path, out=lines.append)
    assert (before.done, before.total, before.calls_left) == (0, 12, 12)
    assert before.days_left(1.0) == pytest.approx(1.2)  # 12 calls at 10/day

    # Day 1: the main config finishes; the daily limit stops the SC config.
    day1 = Recorder([groq_ok("card_arrival", 80)] * 6 + [groq_429("requests per day (RPD)")])
    lines.clear()
    assert run(sched, tmp_path, approve=True, out=lines.append, transport=day1.transport) == 0
    assert len(day1.requests) == 7
    text = "\n".join(lines)
    assert "openai/gpt-oss-120b: daily limit reached during sc; continue tomorrow" in text
    assert "6/12 decisions done (50%); today 7/10 requests, 2,940/10,000 tokens" in text
    assert "~6 calls" in text

    # Day 2: only the samples are sent; main calls come from the cache.
    day2 = Recorder([groq_ok("card_arrival", 70)] * 6)
    lines.clear()
    assert run(sched, tmp_path, approve=True, out=lines.append, transport=day2.transport) == 0
    assert len(day2.requests) == 6
    assert all(b["temperature"] == 1.0 for b in day2.bodies())
    text = "\n".join(lines)
    assert "openai/gpt-oss-120b: all configs complete" in text
    assert "12/12 decisions done (100%)" in text

    # Day 3: nothing left; completed configs are skipped without any call.
    day3 = Recorder([])
    assert run(sched, tmp_path, approve=True, out=lines.append, transport=day3.transport) == 0
    assert day3.requests == []


def test_schedule_errors(tmp_path: Path) -> None:
    bad = tmp_path / "s.toml"
    bad.write_text('configs = ["nope.toml"]\n')
    with pytest.raises(ValueError, match="not found"):
        load_schedule(bad)
    bad.write_text("configs = []\n")
    with pytest.raises(ValueError, match="no configs"):
        load_schedule(bad)
    bad.write_text('configs = ["s.toml"]\nwhen = "daily"\n')
    with pytest.raises(ValueError, match="unknown keys"):
        load_schedule(bad)


def test_study_run_needs_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = write_world(tmp_path, monkeypatch)
    sched = schedule(tmp_path, [groq_cfg(tmp_path, manifest_path, "main")])
    assert main(["study", "run", str(sched), "--raw-dir", str(tmp_path)]) == 3
    assert main(["study", "status", str(sched), "--raw-dir", str(tmp_path)]) == 0

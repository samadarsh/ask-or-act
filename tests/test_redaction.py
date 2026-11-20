"""Regression test for the 2026-11-20 Aegis leak: model output that restates unsafe input
must never reach a saved file or the console.

A fake Groq server answers each Aegis example with a different kind of failure, and
every failure body carries MARKER (standing in for restated unsafe text). The inputs
themselves carry INPUT_MARKER. After a full `ask-or-act run`, neither string may appear
in any file the run wrote (run files, SQLite cache) or in anything it printed or logged.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from ask_or_act.core.runs import load_run
from ask_or_act.data import hub, manifest
from ask_or_act.data.manifest import Manifest
from ask_or_act.data.sampling import SampledExample, Split, fingerprint
from ask_or_act.data.sources import AEGIS
from ask_or_act.run import EXIT_QUOTA, run_from_config
from tests.provider_fixtures import GROQ_HEADERS, groq_response

MARKER = "ZQX-RESTATED-UNSAFE-TEXT-7731"
INPUT_MARKER = "ZQX-RAW-AEGIS-PROMPT-4402"
MODEL = "openai/gpt-oss-safeguard-20b"


def _error(status: int, code: str, *, generation: bool = False) -> httpx.Response:
    err: dict[str, Any] = {
        "message": f"jsonschema: '{MARKER}' is not one of ['safe', 'unsafe']",
        "type": "invalid_request_error",
        "code": code,
    }
    if generation:
        err["failed_generation"] = f"We need to decide. The user asks {MARKER}"
    return httpx.Response(status, json={"error": err})


def _content(content: str, *, finish: str = "stop", **message: Any) -> httpx.Response:
    body = groq_response("safe", 90, model=MODEL)
    body["choices"][0]["message"] = {"role": "assistant", "content": content, **message}
    body["choices"][0]["finish_reason"] = finish
    return httpx.Response(200, json=body, headers=GROQ_HEADERS)


def _ok() -> httpx.Response:
    return httpx.Response(200, json=groq_response("unsafe", 80, model=MODEL), headers=GROQ_HEADERS)


# case -> (main-call response, sample response). Every path that carries provider or
# model text: validation errors, other 4xx/5xx, rate limits, unparseable or invalid
# answers, refusals, truncation, unrequested extra fields and reasoning, sample errors,
# and the daily-limit stop (last, because it ends the run).
CASES: list[tuple[httpx.Response, httpx.Response]] = [
    (_error(400, "json_validate_failed", generation=True), _ok()),
    (_error(400, "output_parse_failed", generation=True), _ok()),
    (_error(400, "invalid_request_error"), _ok()),
    (httpx.Response(500, text=f"upstream crashed near {MARKER}"), _ok()),
    (_error(429, "rate_limit_exceeded"), _ok()),  # per-minute: retryable
    (_content(f"I think {MARKER} is unsafe"), _ok()),
    (_content(json.dumps({"answer": MARKER, "confidence": 50})), _ok()),
    (_content("", refusal=f"I can't help with {MARKER}"), _ok()),
    (_content(f'{{"answer": "safe", "conf {MARKER}', finish="length"), _ok()),
    (
        _content(
            json.dumps({"answer": "unsafe", "confidence": 80, "reason": MARKER}),
            reasoning=f"The user asks {MARKER}",
        ),
        _error(400, "json_validate_failed", generation=True),
    ),
    (_ok(), _content(f"prose {MARKER}")),
    (_ok(), _error(400, "invalid_request_error")),
    (_ok(), _content(json.dumps({"answer": MARKER, "confidence": 10}))),
    (_ok(), _ok()),  # control: a clean example
    (
        httpx.Response(
            429,
            json={
                "error": {
                    "message": f"Rate limit on requests per day (RPD) for {MARKER}",
                    "code": "rate_limit_exceeded",
                }
            },
        ),
        _ok(),
    ),
]


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    text = " ".join(m["content"] for m in body["messages"])
    m = re.search(r"case-(\d+)", text)
    assert m is not None
    main, sample = CASES[int(m.group(1))]
    return sample if body["temperature"] == 1.0 else main


def _aegis_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = [
        {
            "id": f"aegis-{i:02d}",
            "prompt": f"case-{i} {INPUT_MARKER} placeholder prompt",
            "prompt_label": "unsafe" if i % 2 else "safe",
            "prompt_label_source": "human",
        }
        for i in range(len(CASES))
    ]
    raw = tmp_path / "raw" / "test.json"
    raw.parent.mkdir()
    raw.write_text(json.dumps(rows))
    monkeypatch.setattr(hub, "fetch", lambda source, raw_dir: [raw])
    examples = tuple(
        SampledExample(str(r["id"]), Split.TEST, str(r["prompt_label"]), fingerprint(r))
        for r in rows
    )
    path = tmp_path / "aegis.json"
    manifest.write(Manifest("aegis", "toy", AEGIS, 1, examples), path)
    return path


def test_restated_unsafe_text_never_reaches_disk_or_console(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    manifest_path = _aegis_world(tmp_path, monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    out_dir = tmp_path / "out"
    cfg = tmp_path / "run.toml"
    cfg.write_text(f"""
name = "redaction"
[dataset]
manifest = "{manifest_path}"
split = "test"
[decider]
type = "groq"
model = "{MODEL}"
prompt_version = "v1"
[decider.params]
samples = 1
system_prompt = true
[execution]
max_concurrency = 1
max_retries = 0
[budget]
max_usd = 0.0
[cache]
path = "{out_dir / "cache.sqlite"}"
[output]
runs_dir = "{out_dir / "runs"}"
""")
    lines: list[str] = []
    outcome = run_from_config(
        cfg,
        raw_dir=tmp_path,
        out=lines.append,
        approve_spend=True,
        transport=httpx.MockTransport(_handler),
    )

    # The scenario really ran: every case was reached and the last one stopped the run.
    assert outcome.exit_code == EXIT_QUOTA
    assert outcome.run_dir is not None
    record = load_run(outcome.run_dir)
    by_id = {d.example_id: d for d in record.decisions}
    assert len(by_id) == len(CASES)
    assert by_id["aegis-13"].ok
    assert by_id["aegis-09"].ok  # extra field and reasoning dropped, answer kept
    assert record.metadata["status"] == "quota_stopped"
    assert all(not by_id[f"aegis-{i:02d}"].ok for i in range(9))
    assert by_id["aegis-00"].error == "invalid output: json_validate_failed"

    written = [p for p in out_dir.rglob("*") if p.is_file()]
    assert any(p.suffix == ".jsonl" for p in written)
    assert any("cache.sqlite" in p.name for p in written)
    printed = "\n".join(lines) + capsys.readouterr().out + capsys.readouterr().err + caplog.text
    for secret in (MARKER, INPUT_MARKER):
        for p in written:
            assert secret.encode() not in p.read_bytes(), f"{secret} leaked into {p.name}"
        assert secret not in printed

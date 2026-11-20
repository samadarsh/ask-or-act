"""Groq adapter (OpenAI-compatible Chat Completions) against mock HTTP. No network."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from ask_or_act.core.cache import DecisionCache
from ask_or_act.core.limits import RateLimiter, RetryPolicy
from ask_or_act.core.runner import ExecutionConfig, Runner
from ask_or_act.core.runs import load_run
from ask_or_act.core.types import DeciderConfig, Example
from ask_or_act.deciders import DeciderContext, build_decider
from ask_or_act.deciders.http import _excerpt
from ask_or_act.deciders.prompts import OTHER
from ask_or_act.run import EXIT_NEEDS_APPROVAL, EXIT_OK, EXIT_QUOTA, run_from_config
from tests.helpers import FakeClock
from tests.provider_fixtures import Recorder, groq_429, groq_ok
from tests.test_run_end_to_end import write_world

KEY = "gsk-test-SECRET-456"
ENV = {"GROQ_API_KEY": KEY}
BANK = ("card_arrival", "top_up", "lost_or_stolen_card")
NO_RETRY = ExecutionConfig(retry=RetryPolicy(max_retries=0))


def bank(n: int = 1) -> list[Example]:
    return [Example(f"test:{i}", f"where is my card {i}", BANK, "card_arrival") for i in range(n)]


def make(
    model: str,
    rec: Recorder,
    params: dict[str, Any] | None = None,
    *,
    dataset: str = "banking77",
    cache: DecisionCache | None = None,
) -> Any:
    ctx = DeciderContext(dataset=dataset, env=ENV, transport=rec.transport, sample_cache=cache)
    return build_decider(DeciderConfig("groq", model, "v1", params or {}), ctx)


def run(decider: Any, examples: list[Example], **kw: Any) -> Any:
    return asyncio.run(Runner(decider, execution=kw.pop("execution", NO_RETRY), **kw).run(examples))


def test_gpt_oss_request_shape() -> None:
    rec = Recorder([groq_ok("card_arrival", 85)])
    (d,) = run(make("openai/gpt-oss-120b", rec, {"reasoning_effort": "low"}), bank()).decisions
    assert d.ok
    req = rec.requests[0]
    assert str(req.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(req.content)
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["include_reasoning"] is False
    assert body["reasoning_effort"] == "low"
    assert body["max_completion_tokens"] == 512
    assert body["temperature"] == 0.0
    assert "logprobs" not in body
    assert "top_logprobs" not in body
    assert "n" not in body
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_qwen_hides_reasoning_with_reasoning_format() -> None:
    rec = Recorder([groq_ok("top_up", 60, model="qwen/qwen3.8-27b")])
    run(make("qwen/qwen3.8-27b", rec, {"reasoning_effort": "none"}), bank())
    body = rec.bodies()[0]
    assert body["reasoning_format"] == "hidden"
    assert "include_reasoning" not in body  # mutually exclusive with reasoning_format
    assert body["reasoning_effort"] == "none"


def test_bad_reasoning_effort_is_a_permanent_error_without_a_call() -> None:
    rec = Recorder([])
    (d,) = run(make("openai/gpt-oss-120b", rec, {"reasoning_effort": "none"}), bank()).decisions
    assert d.error is not None
    assert "reasoning_effort must be one of" in d.error
    assert rec.requests == []


def test_safeguard_uses_best_effort_schema_and_a_system_policy() -> None:
    rec = Recorder([groq_ok("unsafe", 90, model="openai/gpt-oss-safeguard-20b")])
    ex = [Example("aegis:1", "[redacted test prompt]", ("safe", "unsafe"), "unsafe")]
    d = make("openai/gpt-oss-safeguard-20b", rec, {"system_prompt": True}, dataset="aegis")
    (out,) = run(d, ex).decisions
    assert out.choice == "unsafe"
    body = rec.bodies()[0]
    assert body["response_format"]["json_schema"]["strict"] is False
    system, user = body["messages"]
    assert system["role"] == "system"
    assert "content-safety classifier" in system["content"]
    assert "[redacted test prompt]" not in system["content"]
    assert user["content"].endswith("[redacted test prompt]\n>>>")


def test_free_tier_charges_zero_but_records_the_list_price() -> None:
    rec = Recorder([groq_ok("card_arrival", 85, in_tokens=1_000_000, out_tokens=1_000_000)])
    decider = make("openai/gpt-oss-120b", rec)
    est = decider.estimate(bank())
    assert est.usd == 0.0
    assert est.list_usd is not None
    assert est.list_usd > 0
    outcome = run(decider, bank())
    (d,) = outcome.decisions
    assert d.cost_usd == pytest.approx(0.15 + 0.60)  # $/1M in + $/1M out
    assert d.charged_usd == 0.0
    assert outcome.charged_usd == 0.0
    assert decider.config.params["tier"] == "free"
    paid = make("openai/gpt-oss-120b", Recorder([]), {"tier": "paid"})
    assert paid.estimate(bank()).usd == pytest.approx(est.at_list_price)


def test_logprob_signals_are_unsupported() -> None:
    rec = Recorder([])
    with pytest.raises(ValueError, match="needs logprobs"):
        make("openai/gpt-oss-120b", rec, {"confidence_signal": "token_logprob"})
    with pytest.raises(ValueError, match="does not support logprobs"):
        make("openai/gpt-oss-120b", rec, {"logprobs": 5})
    rec = Recorder([groq_ok("card_arrival", 85)])
    (d,) = run(make("openai/gpt-oss-120b", rec), bank()).decisions
    assert set(d.signals) == {"verbalized"}
    assert d.raw["unsupported_signals"] == ["token_logprob", "verbalized_logprob"]


def test_rate_limit_headers_are_recorded_and_reasoning_dropped() -> None:
    rec = Recorder([groq_ok("card_arrival", 85, reasoning="the user asks about a card ...")])
    (d,) = run(make("openai/gpt-oss-120b", rec), bank()).decisions
    main = d.raw["main"]
    assert isinstance(main, dict)
    assert main["rate_limit"]["x-ratelimit-remaining-requests"] == "999"
    assert "x-request-id" not in main["rate_limit"]
    message = main["choices"][0]["message"]
    assert "reasoning" not in message
    assert message["reasoning_chars"] == len("the user asks about a card ...")
    assert KEY not in json.dumps(d.raw)
    assert "where is my card" not in json.dumps(d.raw)  # no input echo


def test_self_consistency_with_prompt_variation(tmp_path: Path) -> None:
    """Shuffled options and reworded instructions per sample; cached per sample."""
    responses = [groq_ok("card_arrival", 80)] + [groq_ok("card_arrival", 70)] * 3
    rec = Recorder(responses)
    cache = DecisionCache(tmp_path / "c.sqlite")
    params = {
        "samples": 3,
        "sample_variation": "shuffle+paraphrase",
        "other_option": True,
        "confidence_signal": "self_consistency",
    }
    (d,) = run(make("openai/gpt-oss-120b", rec, params, cache=cache), bank()).decisions
    assert d.confidence == 1.0
    bodies = rec.bodies()
    prompts = [b["messages"][0]["content"] for b in bodies]
    assert len(set(prompts)) == 4  # main + 3 differently worded/ordered samples
    for b in bodies[1:]:
        enum = b["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"]
        assert sorted(enum) == sorted([*BANK, OTHER])
        assert enum[-1] == OTHER
        listed = [ln[2:] for ln in b["messages"][0]["content"].splitlines() if ln.startswith("- ")]
        assert listed == enum  # prompt order matches the schema order
    assert all(b["temperature"] == 1.0 for b in bodies[1:])
    assert all("seed" not in b for b in bodies[1:])
    # The variation mode is part of the sample identity: another mode re-samples.
    a = make("openai/gpt-oss-120b", Recorder([]), params)._sample_identity()
    b = make("openai/gpt-oss-120b", Recorder([]), {**params, "sample_variation": "shuffle"})
    assert a != b._sample_identity()
    # ... but the number of samples and the reported signal are not.
    c = make("openai/gpt-oss-120b", Recorder([]), {**params, "samples": 5})
    assert a == c._sample_identity()


def test_per_minute_429_waits_for_retry_after() -> None:
    clock = FakeClock()
    rec = Recorder([groq_429("tokens per minute (TPM)", retry_after="9"), groq_ok("top_up", 50)])
    execution = ExecutionConfig(retry=RetryPolicy(max_retries=2, base_s=0.5))
    runner = Runner(
        make("openai/gpt-oss-120b", rec), execution=execution, clock=clock, sleep=clock.sleep
    )
    (d,) = asyncio.run(runner.run(bank())).decisions
    assert d.ok
    assert d.attempts == 2
    assert max(clock.sleeps) >= 9.0


def test_daily_limit_stops_the_run_cleanly() -> None:
    rec = Recorder(
        [groq_ok("card_arrival", 80), groq_429("requests per day (RPD)", retry_after="5000")]
    )
    outcome = run(make("openai/gpt-oss-120b", rec), bank(5))
    assert len(rec.requests) == 2  # nothing is attempted after the daily 429
    assert outcome.quota_stopped is not None
    assert "daily limit" in outcome.quota_stopped
    assert "org_[redacted]" in outcome.quota_stopped
    assert outcome.quota_retry_after_s == 5000.0
    assert outcome.decisions[0].ok
    assert outcome.n_skipped == 4
    assert not outcome.budget_stopped


def test_excerpts_redact_org_ids() -> None:
    assert _excerpt("limit in organization `org_01hx9ZZ` hit") == (
        "limit in organization `org_[redacted]` hit"
    )


def test_rate_limiter_counts_requests_bigger_than_the_bucket() -> None:
    """30 RPM holds 1 token; a 6-call batch must still cost 6, not 1."""
    clock = FakeClock()

    async def go() -> None:
        limiter = RateLimiter(30, clock=clock, sleep=clock.sleep)
        await limiter.acquire(6)  # bucket full: goes through, 5 in debt
        await limiter.acquire(6)  # waits until full again: 6 tokens at 0.5/s = 12 s
        limiter.charge(3)  # extra usage: next caller waits 6 s more

        await limiter.acquire(1)

    asyncio.run(go())
    assert clock.now == pytest.approx(12.0 + 12.0 + 6.0)


# ------------------------------------------------- `ask-or-act run`: quota stop + resume


def groq_config(tmp_path: Path, manifest_path: Path) -> Path:
    cfg = tmp_path / "groq.toml"
    cfg.write_text(f"""
name = "groq-e2e"
[dataset]
manifest = "{manifest_path}"
split = "test"
[decider]
type = "groq"
model = "openai/gpt-oss-120b"
prompt_version = "v1"
[decider.params]
samples = 1
[execution]
max_concurrency = 1
requests_per_day = 4
max_retries = 0
[budget]
max_usd = 0.0
[cache]
path = "{tmp_path / "cache.sqlite"}"
[output]
runs_dir = "{tmp_path / "runs"}"
""")
    return cfg


def test_run_stops_at_the_daily_limit_and_resumes_from_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = write_world(tmp_path, monkeypatch)  # 6 test examples
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    cfg = groq_config(tmp_path, manifest_path)
    lines: list[str] = []

    def call(transport: Recorder, **kw: Any) -> Any:
        return run_from_config(
            cfg, raw_dir=tmp_path, out=lines.append, transport=transport.transport, **kw
        )

    # A free-tier plan still needs --yes.
    assert call(Recorder([])).exit_code == EXIT_NEEDS_APPROVAL
    plan_text = "\n".join(lines)
    assert "12 API call(s)" in plan_text
    assert "$0.0000 charged" in plan_text
    assert "at the published paid price" in plan_text
    assert "daily quota: ~12 requests of 4/day (300%); needs ~3.0 days" in plan_text

    # Day 1: two examples (2 calls each), then the daily limit.
    day1 = Recorder(
        [groq_ok("card_arrival", 90)] * 4 + [groq_429("requests per day (RPD)", retry_after="3600")]
    )
    lines.clear()
    first = call(day1, approve_spend=True)
    assert first.exit_code == EXIT_QUOTA
    assert len(day1.requests) == 5
    assert any("STOPPED: daily limit reached" in ln for ln in lines)
    rec1 = load_run(first.run_dir)
    assert rec1.metadata["status"] == "quota_stopped"
    counts = rec1.metadata["counts"]
    assert isinstance(counts, dict)
    assert counts["skipped"] == 4

    # Day 2: only the missing calls are made. The third example's main call (request 5)
    # was refused, so it is redone; nothing that succeeded is repeated.
    day2 = Recorder([groq_ok("top_up", 70)] * 8)
    lines.clear()
    second = call(day2, approve_spend=True)
    assert second.exit_code == EXIT_OK
    assert len(day2.requests) == 8
    assert any("2 cached, 8 API call(s)" in ln for ln in lines)
    rec2 = load_run(second.run_dir)
    assert rec2.metadata["status"] == "completed"
    assert sum(d.cached for d in rec2.decisions) == 2
    assert all(d.ok for d in rec2.decisions)

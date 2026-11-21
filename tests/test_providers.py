"""Provider adapters against mock HTTP. No network, no keys, no cost."""

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from ask_or_act.core.cache import DecisionCache
from ask_or_act.core.limits import RetryPolicy
from ask_or_act.core.runner import ExecutionConfig, Runner
from ask_or_act.core.types import DeciderConfig, Example
from ask_or_act.deciders import DeciderContext, build_decider
from ask_or_act.deciders.http import ProviderError
from ask_or_act.deciders.pricing import PRICES, price_for
from tests.provider_fixtures import (
    Recorder,
    decisions_choice,
    decisions_noul,
    gemini_response,
    openai_response,
)

KEY = "sk-test-SECRET-123"
ENV = {"OPENAI_API_KEY": KEY, "GEMINI_API_KEY": KEY}
BANK = ("card_arrival", "top_up", "lost_or_stolen_card")
NO_RETRY = ExecutionConfig(retry=RetryPolicy(max_retries=0))


def bank_examples(n: int = 1) -> list[Example]:
    return [Example(f"test:{i}", f"where is my card {i}", BANK, "card_arrival") for i in range(n)]


def make(
    kind: str,
    model: str,
    rec: Recorder,
    dataset: str = "banking77",
    params: dict[str, Any] | None = None,
    cache: DecisionCache | None = None,
    refresh: bool = False,
    env: dict[str, str] = ENV,
) -> Any:
    ctx = DeciderContext(
        dataset=dataset, env=env, transport=rec.transport, sample_cache=cache, refresh=refresh
    )
    return build_decider(DeciderConfig(kind, model, "v1", params or {}), ctx)


def run(decider: Any, examples: list[Example], **kw: Any) -> Any:
    return asyncio.run(Runner(decider, execution=kw.pop("execution", NO_RETRY), **kw).run(examples))


# ---------------------------------------------------------------- OpenAI


def test_openai_request_shape_and_signals() -> None:
    rec = Recorder([openai_response("card_arrival", 80, answer_p=0.9)])
    out = run(make("openai", "gpt-6-luna", rec), bank_examples())
    (d,) = out.decisions
    (body,) = rec.bodies()
    assert body["store"] is False  # Responses API stores for 30 days unless told not to
    assert body["top_logprobs"] == 20
    assert body["include"] == ["message.output_text.logprobs"]
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["format"]["schema"]["properties"]["answer"]["enum"] == list(BANK)
    assert body["temperature"] == 0.0
    assert d.choice == "card_arrival"
    assert d.signals["verbalized"] == pytest.approx(0.80)
    assert d.signals["token_logprob"] == pytest.approx(0.9)
    assert d.signals["verbalized_logprob"] == pytest.approx((80 * 0.6 + 50 * 0.4) / 100)
    assert d.confidence == d.signals["verbalized"]  # default confidence_signal
    assert d.cost_usd == pytest.approx(price_for("gpt-6-luna").pricing.cost(120, 12))
    assert (d.input_tokens, d.output_tokens) == (120, 12)


def test_openai_key_is_sent_but_never_recorded() -> None:
    rec = Recorder([openai_response("top_up", 60)])
    out = run(make("openai", "gpt-6-luna", rec), bank_examples())
    assert rec.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in repr(out.decisions)


def test_without_logprobs_hybrid_falls_back_to_verbalized() -> None:
    rec = Recorder([openai_response("top_up", 60, logprobs=False)])
    params = {"confidence_signal": "verbalized_logprob"}
    (d,) = run(make("openai", "gpt-6-luna", rec, params=params), bank_examples()).decisions
    assert "verbalized_logprob" not in d.signals
    assert d.confidence == pytest.approx(0.60)
    assert d.raw["confidence_fallback"] is True


def test_invalid_answer_becomes_an_error_with_cost() -> None:
    rec = Recorder([openai_response("not_an_intent", 99)])
    (d,) = run(make("openai", "gpt-6-luna", rec), bank_examples()).decisions
    assert d.error == "invalid answer 'not_an_intent'"
    assert d.cost_usd > 0  # the call was still paid for


# ---------------------------------------------------------------- Gemini LLM


def test_gemini_llm_counts_thinking_tokens_as_output() -> None:
    rec = Recorder([gemini_response("card_arrival", 90, out_tokens=10, thoughts=25)])
    params = {"thinking_budget": 0}
    (d,) = run(
        make("gemini", "gemini-3.5-flash-lite", rec, params=params), bank_examples()
    ).decisions
    (body,) = rec.bodies()
    cfg = body["generationConfig"]
    assert cfg["responseLogprobs"] is True
    assert cfg["logprobs"] == 20
    assert cfg["responseMimeType"] == "application/json"
    assert cfg["thinkingConfig"] == {"thinkingBudget": 0}
    assert rec.requests[0].headers["x-goog-api-key"] == KEY
    assert d.output_tokens == 35
    assert d.signals["token_logprob"] == pytest.approx(0.8)
    assert d.signals["verbalized_logprob"] == pytest.approx((90 * 0.5 + 40 * 0.5) / 100)


def test_gemini_truncated_output_is_an_error() -> None:
    rec = Recorder([gemini_response("card_arrival", 90, finish="MAX_TOKENS")])
    (d,) = run(make("gemini", "gemini-3.5-flash-lite", rec), bank_examples()).decisions
    assert d.error is not None
    assert "MAX_TOKENS" in d.error


# ---------------------------------------------------------------- self-consistency


def test_self_consistency_samples_are_cached_individually(tmp_path: Path) -> None:
    cache_path = tmp_path / "c.sqlite"
    params = {"samples": 3, "sample_temperature": 1.0, "confidence_signal": "self_consistency"}
    rec = Recorder(
        [
            openai_response("card_arrival", 70),
            openai_response("card_arrival", 70),
            openai_response("top_up", 40),
            openai_response("card_arrival", 80),
        ]
    )
    with DecisionCache(cache_path) as cache:
        (d,) = run(
            make("openai", "gpt-6-luna", rec, params=params, cache=cache), bank_examples()
        ).decisions
    assert d.signals["self_consistency"] == pytest.approx(2 / 3)
    assert d.confidence == pytest.approx(2 / 3)
    sample_bodies = rec.bodies()[1:]
    assert all(b["temperature"] == 1.0 and "top_logprobs" not in b for b in sample_bodies)
    assert d.charged_usd == pytest.approx(d.cost_usd)  # everything was paid for now

    # k: 3 → 5. Main call + first 3 samples come from cache; only 2 new samples are paid.
    rec2 = Recorder([openai_response("card_arrival", 75), openai_response("card_arrival", 75)])
    params5 = {**params, "samples": 5}
    with DecisionCache(cache_path) as cache:
        (d5,) = run(
            make("openai", "gpt-6-luna", rec2, params=params5, cache=cache), bank_examples()
        ).decisions
    assert len(rec2.requests) == 2  # main call cached under its own key; 2 new samples
    assert d5.raw["main_cached"] is True
    assert d5.signals["self_consistency"] == pytest.approx(4 / 5)
    assert d5.charged_usd is not None
    assert d5.charged_usd < d5.cost_usd  # main + 3 samples were reused, not re-billed


def test_self_consistency_estimate_counts_every_call() -> None:
    d = make("openai", "gpt-6-luna", Recorder([]), params={"samples": 4})
    est = d.estimate(bank_examples(2))
    assert est.calls == 2 * 5
    assert est.usd > 0


# ---------------------------------------------------------------- Decisions API


def test_decisions_routing_with_other_option() -> None:
    rec = Recorder(
        [
            decisions_choice(
                {"card_arrival": 0.6, "other": 0.3, "top_up": 0.1, "lost_or_stolen_card": 0.0}
            )
        ]
    )
    (d,) = run(make("gemini_decisions", "gemini-decision-1", rec), bank_examples()).decisions
    (body,) = rec.bodies()
    q = body["questions"]["answer"]
    assert q["type"] == "choice"
    assert "other" in q["criteria"]
    assert body["input"] == "where is my card 0"
    assert d.choice == "card_arrival"
    assert d.confidence == pytest.approx(0.6)
    assert d.signals["p_other"] == pytest.approx(0.3)
    assert d.output_tokens == 0
    assert d.cost_usd == pytest.approx(300 * 0.10 / 1e6)


def test_decisions_safety_uses_yes_no() -> None:
    ex = [Example("a1", "some prompt", ("safe", "unsafe"), "unsafe")]
    rec = Recorder([decisions_noul(0.83)])
    (d,) = run(make("gemini_decisions", "gemini-decision-1", rec, dataset="aegis"), ex).decisions
    assert rec.bodies()[0]["questions"]["answer"]["type"] == "noul"
    assert d.choice == "unsafe"
    assert d.probabilities == {"unsafe": pytest.approx(0.83), "safe": pytest.approx(0.17)}


def test_decisions_safety_block_is_a_permanent_error() -> None:
    blocked = httpx.Response(400, json={"error": {"code": "safety", "message": "blocked"}})
    rec = Recorder([blocked])
    ex = [Example("a1", "bad prompt", ("safe", "unsafe"), "unsafe")]
    execution = ExecutionConfig(retry=RetryPolicy(max_retries=3, base_s=0))
    (d,) = run(
        make("gemini_decisions", "gemini-decision-1", rec, dataset="aegis"), ex, execution=execution
    ).decisions
    assert d.error is not None
    assert "HTTP 400" in d.error
    assert d.attempts == 1  # not retried
    assert len(rec.requests) == 1


# ---------------------------------------------------------------- HTTP errors & keys


def test_rate_limits_and_server_errors_are_retried() -> None:
    rec = Recorder(
        [
            httpx.Response(429, json={"error": "slow down"}),
            httpx.Response(503, text="overloaded"),
            openai_response("card_arrival", 70),
        ]
    )
    execution = ExecutionConfig(retry=RetryPolicy(max_retries=3, base_s=0))
    (d,) = run(make("openai", "gpt-6-luna", rec), bank_examples(), execution=execution).decisions
    assert d.ok
    assert d.attempts == 3


def test_exhausted_quota_is_not_retried() -> None:
    rec = Recorder([httpx.Response(429, json={"error": {"code": "insufficient_quota"}})])
    execution = ExecutionConfig(retry=RetryPolicy(max_retries=3, base_s=0))
    (d,) = run(make("openai", "gpt-6-luna", rec), bank_examples(), execution=execution).decisions
    assert d.error is not None
    assert "quota" in d.error
    assert len(rec.requests) == 1


def test_missing_key_names_the_variable_only() -> None:
    rec = Recorder([])
    (d,) = run(make("openai", "gpt-6-luna", rec, env={}), bank_examples()).decisions
    assert d.error is not None
    assert "OPENAI_API_KEY is not set" in d.error
    assert rec.requests == []


def test_unknown_model_price_is_refused() -> None:
    with pytest.raises(ValueError, match="no price on record"):
        make("openai", "gpt-99-mystery", Recorder([]))


def test_every_price_entry_is_dated_and_cited() -> None:
    for p in PRICES.values():
        assert p.source.startswith("https://")
        assert p.retrieved in ("2026-11-18", "2026-11-20")
        assert p.input_per_mtok > 0
    assert not PRICES["gemini-decision-1"].verified  # flagged until seen verbatim


def test_provider_error_is_permanent() -> None:
    assert not issubclass(ProviderError, RuntimeError)  # not a RetryableError either


# ---------------------------------------------------------------- gold & refresh


def test_real_deciders_never_receive_gold() -> None:
    from tests.helpers import ScriptedDecider

    scripted = ScriptedDecider()
    asyncio.run(Runner(scripted).run(bank_examples(3)))
    assert scripted.calls
    assert scripted.saw_gold is False


def test_refresh_bypasses_the_cache(tmp_path: Path) -> None:
    from tests.helpers import ScriptedDecider

    with DecisionCache(tmp_path / "c.sqlite") as cache:
        asyncio.run(Runner(ScriptedDecider(), cache=cache).run(bank_examples(4)))
        again = ScriptedDecider()
        asyncio.run(Runner(again, cache=cache, refresh=True).run(bank_examples(4)))
    assert again.n_examples_called == 4

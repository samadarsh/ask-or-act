"""Prompted-LLM deciders with confidence signals.

Providers: Gemini generateContent, OpenAI Responses, and Groq (OpenAI-compatible Chat
Completions at https://api.groq.com/openai/v1).

Per example:

1. **Main call** at ``temperature`` (default 0), requesting top-k logprobs. It yields
   the answer plus three signals: ``verbalized``, ``token_logprob`` and
   ``verbalized_logprob`` (see deciders/signals.py).
2. **Self-consistency** (``samples: k`` > 0): k more calls at ``sample_temperature``.
   With ``sample_variation`` the k prompts also differ: shuffled option order and/or
   reworded instructions (deciders/prompts.py). That is the fallback when temperature
   alone gives little diversity, or is ignored by the model.
   The ``self_consistency`` signal is the share of samples that agree with the main
   answer, i.e. confidence in the answer we actually give. Each sample is cached
   separately, keyed by (config, input, sample_index, temperature), so:
   - samples are never identical cache copies of each other;
   - raising k reuses the samples already paid for.

``Decision.confidence`` is the signal named by ``confidence_signal``. If that signal is
unavailable for an example (e.g. no logprobs), it falls back to ``verbalized`` and
records ``raw["confidence_fallback"]``. All available signals are always stored, so
milestone 3/7 can compare them on the same calls.

API facts used (docs and SDK types read 2026-11-18):

* **OpenAI Responses.** Logprobs come from ``top_logprobs`` (0–20) plus
  ``include=["message.output_text.logprobs"]``. ``store`` defaults to true (30-day
  retention), so we always send ``store: false``. Structured output uses
  ``text.format = json_schema`` with ``strict: true``. Usage fields: ``input_tokens``,
  ``output_tokens`` (reasoning included) and
  ``output_tokens_details.reasoning_tokens``.
* **Gemini generateContent.** Logprobs come from ``responseLogprobs`` + ``logprobs``
  (0–20); the response has ``logprobsResult.chosenCandidates`` / ``topCandidates``.
  Structured output uses ``responseMimeType`` + ``responseJsonSchema``. Usage:
  ``promptTokenCount``, ``candidatesTokenCount`` and ``thoughtsTokenCount``; thoughts
  are billed as output.
* **Groq Chat Completions** (docs read 2026-11-20). ``logprobs`` / ``top_logprobs``
  are "not yet supported by any of our models", so ``token_logprob`` and
  ``verbalized_logprob`` are unsupported there. Structured output uses
  ``response_format = json_schema``, ``strict: true`` for gpt-oss-20b/120b and
  qwen3.8-27b. gpt-oss-safeguard-20b supports best-effort json_schema only (no
  ``strict``). ``n`` must be 1. ``max_completion_tokens`` replaces ``max_tokens``. Rate-limit
  headers: ``x-ratelimit-*-requests`` are per *day*, ``x-ratelimit-*-tokens`` per
  *minute*. They are recorded per call in ``raw["rate_limit"]``. Reasoning text is
  hidden (``include_reasoning: false`` for gpt-oss, ``reasoning_format: hidden`` for
  Qwen). It isn't needed, and for safety data it could paraphrase unsafe input.
* **Which models actually return logprobs or honour temperature isn't documented**
  (OpenAI reference pages were unreadable; the Gemini docs are silent). The smoke test
  will establish it, and docs/deciders.md records the result.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from ask_or_act.core.decider import CostEstimate
from ask_or_act.core.types import JSON, DeciderConfig, Decision, PublicExample
from ask_or_act.deciders import DeciderContext
from ask_or_act.deciders.http import ProviderError, api_key, post_json_full
from ask_or_act.deciders.pricing import Price, price_for
from ask_or_act.deciders.prompts import (
    OTHER,
    PROMPT_VERSION,
    VARIATIONS,
    answer_schema,
    build_prompt,
    options_with_other,
    sample_prompt,
)
from ask_or_act.deciders.signals import (
    Token,
    answer_probability,
    parse_answer,
    verbalized_logprob,
)

SIGNALS = ("verbalized", "token_logprob", "verbalized_logprob", "self_consistency")
LOGPROB_SIGNALS = ("token_logprob", "verbalized_logprob")


@dataclass(frozen=True)
class Completion:
    text: str
    tokens: tuple[Token, ...] | None  # None when the provider returned no logprobs
    input_tokens: int
    output_tokens: int  # billed output, including reasoning/thinking tokens
    reasoning_tokens: int
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CallSpec:
    model: str
    prompt: str
    schema: dict[str, JSON]
    temperature: float | None
    logprobs: int  # 0 = don't request
    max_output_tokens: int
    seed: int | None = None
    reasoning_effort: str | None = None  # OpenAI, Groq
    thinking_budget: int | None = None  # Gemini (0 disables thinking where supported)
    system: str | None = None  # system message / instructions, if the prompt is split


class Provider(Protocol):
    name: str
    key_env: str
    supports_logprobs: bool
    default_tier: str  # "paid" or "free"
    record_headers: tuple[str, ...]  # response-header prefixes to keep in raw

    def request(self, spec: CallSpec, key: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        """(url, headers, body)."""
        ...

    def parse(self, response: dict[str, Any]) -> Completion: ...


def _messages(spec: CallSpec) -> list[dict[str, str]]:
    sys_msg = [{"role": "system", "content": spec.system}] if spec.system else []
    return [*sys_msg, {"role": "user", "content": spec.prompt}]


class OpenAILLM:
    name = "openai"
    key_env = "OPENAI_API_KEY"
    url = "https://api.openai.com/v1/responses"
    supports_logprobs = True
    default_tier = "paid"
    record_headers: tuple[str, ...] = ()

    def request(self, spec: CallSpec, key: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        body: dict[str, Any] = {
            "model": spec.model,
            "input": _messages(spec),
            "store": False,
            "max_output_tokens": spec.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "decision",
                    "strict": True,
                    "schema": spec.schema,
                }
            },
        }
        if spec.temperature is not None:
            body["temperature"] = spec.temperature
        if spec.logprobs:
            body["top_logprobs"] = spec.logprobs
            body["include"] = ["message.output_text.logprobs"]
        if spec.reasoning_effort:
            body["reasoning"] = {"effort": spec.reasoning_effort}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        return self.url, headers, body

    def parse(self, response: dict[str, Any]) -> Completion:
        if response.get("status") not in (None, "completed"):
            details = response.get("incomplete_details")
            raise ProviderError(f"response {response.get('status')}: {details}")
        texts: list[str] = []
        tokens: list[Token] = []
        has_logprobs = False
        for item in response.get("output") or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if part.get("type") == "refusal":
                    raise ProviderError(f"model refused: {str(part.get('refusal'))[:200]}")
                if part.get("type") != "output_text":
                    continue
                texts.append(str(part.get("text", "")))
                lps = part.get("logprobs")
                if lps:
                    has_logprobs = True
                    for lp in lps:
                        top = tuple(
                            (str(t["token"]), float(t["logprob"]))
                            for t in lp.get("top_logprobs") or []
                        )
                        tokens.append(Token(str(lp["token"]), float(lp["logprob"]), top))
        usage = response.get("usage") or {}
        raw = {k: v for k, v in response.items() if k not in ("instructions", "input")}
        return Completion(
            text="".join(texts),
            tokens=tuple(tokens) if has_logprobs else None,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            reasoning_tokens=int(
                (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
            ),
            raw=raw,
        )


class GeminiLLM:
    name = "gemini"
    key_env = "GEMINI_API_KEY"
    base = "https://generativelanguage.googleapis.com/v1beta/models"
    supports_logprobs = True
    default_tier = "paid"
    record_headers: tuple[str, ...] = ()

    def request(self, spec: CallSpec, key: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseJsonSchema": spec.schema,
            "maxOutputTokens": spec.max_output_tokens,
        }
        if spec.temperature is not None:
            config["temperature"] = spec.temperature
        if spec.logprobs:
            config["responseLogprobs"] = True
            config["logprobs"] = spec.logprobs
        if spec.seed is not None:
            config["seed"] = spec.seed
        if spec.thinking_budget is not None:
            config["thinkingConfig"] = {"thinkingBudget": spec.thinking_budget}
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": spec.prompt}]}],
            "generationConfig": config,
        }
        if spec.system:
            body["systemInstruction"] = {"parts": [{"text": spec.system}]}
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        return f"{self.base}/{spec.model}:generateContent", headers, body

    def parse(self, response: dict[str, Any]) -> Completion:
        candidates = response.get("candidates") or []
        if not candidates:
            block = (response.get("promptFeedback") or {}).get("blockReason")
            raise ProviderError(f"no candidates (blockReason={block!r})")
        cand = candidates[0]
        finish = cand.get("finishReason")
        if finish not in (None, "STOP"):
            raise ProviderError(f"generation stopped: finishReason={finish}")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(str(p.get("text", "")) for p in parts if not p.get("thought"))
        tokens: tuple[Token, ...] | None = None
        result = cand.get("logprobsResult")
        if result and result.get("chosenCandidates"):
            tops = result.get("topCandidates") or []
            toks = []
            for i, chosen in enumerate(result["chosenCandidates"]):
                alts = tops[i].get("candidates", []) if i < len(tops) else []
                top = tuple(
                    (str(a.get("token", "")), float(a.get("logProbability", 0.0))) for a in alts
                )
                toks.append(
                    Token(
                        str(chosen.get("token", "")), float(chosen.get("logProbability", 0.0)), top
                    )
                )
            tokens = tuple(toks)
        usage = response.get("usageMetadata") or {}
        thoughts = int(usage.get("thoughtsTokenCount", 0))
        return Completion(
            text=text,
            tokens=tokens,
            input_tokens=int(usage.get("promptTokenCount", 0)),
            output_tokens=int(usage.get("candidatesTokenCount", 0)) + thoughts,
            reasoning_tokens=thoughts,
            raw=response,  # candidates + usage + modelVersion; no input echo
        )


@dataclass(frozen=True)
class ChatModel:
    """What an OpenAI-compatible endpoint accepts for one model."""

    schema: str = "json_object"  # "strict" | "json_schema" (best effort) | "json_object"
    hide_reasoning: Mapping[str, Any] = field(default_factory=dict)  # body fields
    efforts: tuple[str, ...] = ()  # accepted reasoning_effort values; () = don't send


class ChatCompletionsLLM:
    """OpenAI Chat Completions wire format (``POST {base}/chat/completions``).

    Groq, and other OpenAI-compatible servers, speak this format. Our OpenAI decider uses
    the newer Responses API instead (see ``OpenAILLM``)."""

    name = "openai_chat"
    key_env = "OPENAI_API_KEY"
    base = "https://api.openai.com/v1"
    supports_logprobs = True
    default_tier = "paid"
    record_headers: tuple[str, ...] = ()
    models: ClassVar[Mapping[str, ChatModel]] = {}

    def model(self, name: str) -> ChatModel:
        return self.models.get(name, ChatModel(schema="strict"))

    def request(self, spec: CallSpec, key: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        caps = self.model(spec.model)
        body: dict[str, Any] = {
            "model": spec.model,
            "messages": _messages(spec),
            "max_completion_tokens": spec.max_output_tokens,
        }
        if caps.schema == "json_object":
            body["response_format"] = {"type": "json_object"}
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "decision",
                    "strict": caps.schema == "strict",
                    "schema": spec.schema,
                },
            }
        if spec.temperature is not None:
            body["temperature"] = spec.temperature
        if spec.seed is not None:
            body["seed"] = spec.seed
        if spec.logprobs:
            if not self.supports_logprobs:
                raise ProviderError(f"{self.name} does not support logprobs")
            body["logprobs"] = True
            body["top_logprobs"] = spec.logprobs
        if spec.reasoning_effort is not None:
            if caps.efforts and spec.reasoning_effort not in caps.efforts:
                raise ProviderError(f"{spec.model} reasoning_effort must be one of {caps.efforts}")
            body["reasoning_effort"] = spec.reasoning_effort
        body.update(caps.hide_reasoning)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        return f"{self.base}/chat/completions", headers, body

    def parse(self, response: dict[str, Any]) -> Completion:
        choices = response.get("choices") or []
        if not choices:
            raise ProviderError("no choices in response")
        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish not in (None, "stop"):
            raise ProviderError(f"generation stopped: finish_reason={finish}")
        message = choice.get("message") or {}
        if message.get("refusal"):
            raise ProviderError(f"model refused: {str(message['refusal'])[:200]}")
        tokens: tuple[Token, ...] | None = None
        content_lps = (choice.get("logprobs") or {}).get("content")
        if content_lps:
            tokens = tuple(
                Token(
                    str(lp["token"]),
                    float(lp["logprob"]),
                    tuple(
                        (str(t["token"]), float(t["logprob"])) for t in lp.get("top_logprobs") or []
                    ),
                )
                for lp in content_lps
            )
        usage = response.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return Completion(
            text=str(message.get("content") or ""),
            tokens=tokens,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),  # reasoning included
            reasoning_tokens=int(details.get("reasoning_tokens", 0) or 0),
            raw=_without_reasoning(response),
        )


def _without_reasoning(response: dict[str, Any]) -> dict[str, Any]:
    """Drop any reasoning text from the stored response, keeping only its length.
    It isn't needed for any metric, and on safety data it may restate unsafe input."""
    raw = dict(response)
    choices = []
    for c in response.get("choices") or []:
        c = dict(c)
        msg = dict(c.get("message") or {})
        if msg.get("reasoning"):
            msg["reasoning_chars"] = len(str(msg.pop("reasoning")))
        c["message"] = msg
        choices.append(c)
    raw["choices"] = choices
    return raw


_GPT_OSS = ChatModel(
    schema="strict", hide_reasoning={"include_reasoning": False}, efforts=("low", "medium", "high")
)


class GroqLLM(ChatCompletionsLLM):
    """Groq: OpenAI-compatible Chat Completions. Free tier by default. Model facts are
    from console.groq.com/docs (structured-outputs, reasoning, api-reference), read
    2026-11-20."""

    name = "groq"
    key_env = "GROQ_API_KEY"
    base = "https://api.groq.com/openai/v1"
    supports_logprobs = False  # "not yet supported by any of our models"
    default_tier = "free"
    record_headers: tuple[str, ...] = ("x-ratelimit-", "retry-after")
    models: ClassVar[Mapping[str, ChatModel]] = {
        "openai/gpt-oss-120b": _GPT_OSS,
        "openai/gpt-oss-20b": _GPT_OSS,
        # Best-effort json_schema only ("strict" isn't listed for it). Reasoning efforts
        # are per its model page ("low/medium/high reasoning effort").
        "openai/gpt-oss-safeguard-20b": ChatModel(
            schema="json_schema",
            hide_reasoning={"include_reasoning": False},
            efforts=("low", "medium", "high"),
        ),
        # Qwen takes reasoning_format; gpt-oss rejects it. "none" disables reasoning.
        "qwen/qwen3.8-27b": ChatModel(
            schema="strict",
            hide_reasoning={"reasoning_format": "hidden"},
            efforts=("none", "default", "low", "medium", "high"),
        ),
    }

    def model(self, name: str) -> ChatModel:
        return self.models.get(name, ChatModel(schema="json_object"))


KNOWN_PARAMS = frozenset(
    {
        "confidence_signal",
        "temperature",
        "logprobs",
        "samples",
        "sample_temperature",
        "max_output_tokens",
        "seed",
        "reasoning_effort",
        "thinking_budget",
        "other_option",
        "chars_per_token",
        "expected_output_tokens",
        "sample_variation",
        "system_prompt",
        "tier",
    }
)


def _opt_int(params: Mapping[str, JSON], key: str) -> int | None:
    v = params.get(key)
    return None if v is None else int(str(v))


class PromptedLLMDecider:
    def __init__(self, config: DeciderConfig, ctx: DeciderContext, provider: Provider) -> None:
        p = dict(config.params)
        unknown = set(p) - KNOWN_PARAMS
        if unknown:
            raise ValueError(f"unknown {provider.name} params: {sorted(unknown)}")
        if config.prompt_version != PROMPT_VERSION:
            raise ValueError(f"prompt_version must be {PROMPT_VERSION!r} (deciders/prompts.py)")
        self._config = config
        self.ctx = ctx
        self.provider = provider
        self.task = ctx.dataset
        self.signal = str(p.get("confidence_signal", "verbalized"))
        if self.signal not in SIGNALS:
            raise ValueError(f"confidence_signal must be one of {SIGNALS}")
        if not provider.supports_logprobs and self.signal in LOGPROB_SIGNALS:
            raise ValueError(
                f"confidence_signal={self.signal!r} needs logprobs, which {provider.name} "
                "does not return; use verbalized or self_consistency"
            )
        temp = p.get("temperature", 0.0)
        self.temperature = None if temp is None else float(str(temp))
        self.logprobs = int(str(p.get("logprobs", 20 if provider.supports_logprobs else 0)))
        if not 0 <= self.logprobs <= 20:
            raise ValueError("logprobs must be in 0..20 (both providers' limit)")
        if self.logprobs and not provider.supports_logprobs:
            raise ValueError(f"{provider.name} does not support logprobs; set logprobs = 0")
        self.samples = int(str(p.get("samples", 0)))
        self.sample_temperature = float(str(p.get("sample_temperature", 1.0)))
        if self.signal == "self_consistency" and self.samples < 1:
            raise ValueError("confidence_signal=self_consistency needs samples >= 1")
        self.variation = str(p.get("sample_variation", "none"))
        if self.variation not in VARIATIONS:
            raise ValueError(f"sample_variation must be one of {VARIATIONS}")
        self.system_prompt = bool(p.get("system_prompt", False))
        self.max_output_tokens = int(str(p.get("max_output_tokens", 512)))
        self.seed = _opt_int(p, "seed")
        effort = p.get("reasoning_effort")
        self.reasoning_effort = None if effort is None else str(effort)
        self.thinking_budget = _opt_int(p, "thinking_budget")
        self.other = bool(p.get("other_option", False))
        self.chars_per_token = float(str(p.get("chars_per_token", 4.0)))
        self.expected_output_tokens = int(str(p.get("expected_output_tokens", 40)))
        self.price: Price = price_for(config.model)
        self.tier = str(p.get("tier", provider.default_tier))
        if self.tier not in ("paid", "free"):
            raise ValueError("tier must be 'paid' or 'free'")
        if self.tier == "free" and not self.price.free_tier:
            raise ValueError(f"{config.model} has no free tier on record (deciders/pricing.py)")
        # Report resolved defaults, so changing a default changes the cache key.
        resolved: dict[str, JSON] = {
            "confidence_signal": self.signal,
            "temperature": self.temperature,
            "logprobs": self.logprobs,
            "samples": self.samples,
            "sample_temperature": self.sample_temperature,
            "sample_variation": self.variation,
            "system_prompt": self.system_prompt,
            "max_output_tokens": self.max_output_tokens,
            "seed": self.seed,
            "reasoning_effort": self.reasoning_effort,
            "thinking_budget": self.thinking_budget,
            "other_option": self.other,
            "tier": self.tier,
        }
        self._config = DeciderConfig(config.type, config.model, config.prompt_version, resolved)

    @property
    def config(self) -> DeciderConfig:
        return self._config

    @property
    def max_batch_size(self) -> int:
        return 1

    @property
    def needs_gold(self) -> bool:
        return False

    def _options(self, ex: PublicExample) -> tuple[str, ...]:
        return options_with_other(ex, self.other)

    def estimate(self, examples: Sequence[PublicExample]) -> CostEstimate:
        """Input ≈ prompt chars / chars_per_token (calibrate from a smoke run's real
        usage); output = expected_output_tokens per call. Self-consistency samples are
        assumed uncached, so the estimate is conservative. On a free tier ``usd`` is 0
        and ``list_usd`` is the same tokens at the published paid price."""
        calls = (1 + self.samples) * len(examples)
        tokens_in = 0
        for ex in examples:
            sys_msg, user = build_prompt(
                self.task, ex, self._options(ex), system=self.system_prompt
            )
            chars = len(user) + len(sys_msg or "")
            tokens_in += (int(chars / self.chars_per_token) + 1) * (1 + self.samples)
        tokens_out = calls * self.expected_output_tokens
        list_usd = self.price.pricing.cost(tokens_in, tokens_out)
        charged = 0.0 if self.tier == "free" else list_usd
        return CostEstimate(calls, tokens_in, tokens_out, charged, list_usd)

    def _spec(
        self,
        system: str | None,
        prompt: str,
        options: tuple[str, ...],
        *,
        temperature: float | None,
        logprobs: int,
        seed: int | None,
    ) -> CallSpec:
        return CallSpec(
            self._config.model,
            prompt,
            answer_schema(options),
            temperature,
            logprobs,
            self.max_output_tokens,
            seed,
            self.reasoning_effort,
            self.thinking_budget,
            system,
        )

    async def _call(self, spec: CallSpec) -> Completion:
        key = api_key(self.ctx.env, self.provider.key_env)
        url, headers, body = self.provider.request(spec, key)
        response, resp_headers = await post_json_full(
            url, body, headers=headers, transport=self.ctx.transport
        )
        completion = self.provider.parse(response)
        if self.provider.record_headers:
            completion.raw["rate_limit"] = {
                k: v
                for k, v in sorted(resp_headers.items())
                if k.startswith(self.provider.record_headers)
            }
        return completion

    def _cost(self, c: Completion) -> tuple[float, float]:
        """(cost at the published price, what was charged: 0 on a free tier)."""
        cost = self.price.pricing.cost(c.input_tokens, c.output_tokens)
        return cost, 0.0 if self.tier == "free" else cost

    async def decide_batch(self, examples: Sequence[PublicExample]) -> list[Decision]:
        (ex,) = examples
        return [await self._decide(ex)]

    async def _decide(self, ex: PublicExample) -> Decision:
        options = self._options(ex)
        system, prompt = build_prompt(self.task, ex, options, system=self.system_prompt)
        main = await self._call(
            self._spec(
                system,
                prompt,
                options,
                temperature=self.temperature,
                logprobs=self.logprobs,
                seed=self.seed,
            )
        )
        parsed = parse_answer(main.text)
        cost, charged = self._cost(main)
        raw: dict[str, JSON] = {"main": main.raw}
        if not self.provider.supports_logprobs:
            raw["unsupported_signals"] = list(LOGPROB_SIGNALS)
        if parsed.answer not in options:
            return Decision(
                ex.id,
                None,
                None,
                model=self._config.model,
                input_tokens=main.input_tokens,
                output_tokens=main.output_tokens,
                cost_usd=cost,
                charged_usd=charged,
                raw=raw,
                error=f"invalid answer {parsed.answer!r}",
            )

        signals: dict[str, float] = {}
        if parsed.confidence is not None and 0 <= parsed.confidence <= 100:
            signals["verbalized"] = parsed.confidence / 100
        if main.tokens:
            p = answer_probability(main.tokens)
            if p is not None:
                signals["token_logprob"] = p
            hybrid = verbalized_logprob(main.tokens)
            if hybrid is not None:
                signals["verbalized_logprob"] = hybrid.value
                raw["hybrid_mode"] = hybrid.mode
        if OTHER in options:
            raw["answered_other"] = parsed.answer == OTHER

        tokens_in, tokens_out = main.input_tokens, main.output_tokens
        if self.samples:
            answers, sample_records = await self._samples(ex, options)
            signals["self_consistency"] = sum(a == parsed.answer for a in answers) / len(answers)
            for rec in sample_records:
                cost += rec.cost_usd
                charged += rec.charged_usd or 0.0
                tokens_in += rec.input_tokens
                tokens_out += rec.output_tokens
            raw["samples"] = [
                {"answer": a, "cached": r.cached, "raw": dict(r.raw)}
                for a, r in zip(answers, sample_records, strict=True)
            ]

        confidence = signals.get(self.signal)
        if confidence is None:
            confidence = signals.get("verbalized")
            raw["confidence_fallback"] = True
        if confidence is None:
            return Decision(
                ex.id,
                None,
                None,
                signals=signals,
                model=self._config.model,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                cost_usd=cost,
                charged_usd=charged,
                raw=raw,
                error="no usable confidence signal",
            )
        return Decision(
            example_id=ex.id,
            choice=parsed.answer,
            confidence=max(0.0, min(1.0, confidence)),
            signals=signals,
            model=self._config.model,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=cost,
            charged_usd=charged,
            raw=raw,
        )

    # Params that do NOT change what a single sample call returns: how many samples,
    # which signal is reported, and the main call's own temperature/logprobs/seed
    # (samples use sample_temperature, which is in the cache key's ``extra``, no
    # logprobs and no seed). Everything else (including any future param) is part of
    # the sample key. Leaving params out is what lets a larger k reuse paid samples.
    SAMPLE_KEY_EXCLUDED = frozenset(
        {"confidence_signal", "samples", "sample_temperature", "temperature", "logprobs", "seed"}
    )

    def _sample_identity(self) -> DeciderConfig:
        params = {k: v for k, v in self._config.params.items() if k not in self.SAMPLE_KEY_EXCLUDED}
        return DeciderConfig(
            f"{self._config.type}:sample", self._config.model, self._config.prompt_version, params
        )

    async def _samples(
        self, ex: PublicExample, options: tuple[str, ...]
    ) -> tuple[list[str | None], list[Decision]]:
        identity = self._sample_identity()

        async def one(i: int) -> Decision:
            extra: dict[str, JSON] = {"sample_index": i, "temperature": self.sample_temperature}
            cache = self.ctx.sample_cache
            if cache is not None and not self.ctx.refresh:
                hit = cache.get(identity, ex, extra)
                if hit is not None:
                    return hit
            system, prompt, order = sample_prompt(
                self.task, ex, options, i, self.variation, system=self.system_prompt
            )
            c = await self._call(
                self._spec(
                    system,
                    prompt,
                    order,
                    temperature=self.sample_temperature,
                    logprobs=0,
                    seed=None,
                )
            )
            ans = parse_answer(c.text).answer
            cost, charged = self._cost(c)
            raw = dict(c.raw)
            if self.variation != "none":
                raw["variation"] = (
                    {"options_order": list(order)} if "shuffle" in self.variation else {}
                )
            d = Decision(
                ex.id,
                ans if ans in options else None,
                None,
                model=self._config.model,
                input_tokens=c.input_tokens,
                output_tokens=c.output_tokens,
                cost_usd=cost,
                charged_usd=charged,
                raw=raw,
                error=None if ans in options else f"invalid answer {ans!r}",
            )
            if cache is not None:
                cache.put(identity, ex, d, extra)
            return d

        records = list(await asyncio.gather(*(one(i) for i in range(self.samples))))
        return [r.choice for r in records], records

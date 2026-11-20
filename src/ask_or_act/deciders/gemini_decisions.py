"""Gemini Decisions API adapter (model ``gemini-decision-1``): native probabilities.

Docs read 2026-11-18: https://aistudio.google.com/docs/decisions-api and
https://aistudio.google.com/api-reference/decisions.

* ``POST /v1beta/decisions`` with ``{model, input, questions: {key: question}}``.
  Zero output tokens; billed on input only (shared input counted once per request).
* Question types: ``choice`` (2–255 options, a probability for every option),
  ``noul`` (yes/no, probability clamped to [0.01, 0.99]), and ``score``.
* Option keys are **sorted alphabetically** before evaluation. A choice question
  **always spreads probability over the options, even when none fits**, so for routing
  we add an ``other`` option (a plain key, not "none_of_the_above", per the docs). Its
  probability is recorded as the ``p_other`` signal: a natural escalate signal.

One request per example: the shared input *is* the example, so batching examples
wouldn't share anything.
"""

import json
from collections.abc import Sequence
from typing import Any

from ask_or_act.core.decider import CostEstimate, QuotaExhaustedError, RetryableError
from ask_or_act.core.types import JSON, DeciderConfig, Decision, PublicExample
from ask_or_act.deciders import DeciderContext
from ask_or_act.deciders.http import ProviderError, api_key, post_json, public_message
from ask_or_act.deciders.pricing import price_for
from ask_or_act.deciders.prompts import (
    OTHER,
    PROMPT_VERSION,
    SENSITIVE_TASKS,
    instructions,
    render_input,
)

URL = "https://generativelanguage.googleapis.com/v1beta/decisions"
KEY_ENV = "GEMINI_API_KEY"
QUESTION = "answer"

MT_BENCH_CRITERIA = {
    "model_a": "Assistant A's answer is better.",
    "model_b": "Assistant B's answer is better.",
    "tie": "Both answers are about equally good.",
}


class GeminiDecisionsDecider:
    def __init__(self, config: DeciderConfig, ctx: DeciderContext) -> None:
        params = dict(config.params)
        unknown = set(params) - {"other_option", "chars_per_token"}
        if unknown:
            raise ValueError(f"unknown gemini_decisions params: {sorted(unknown)}")
        if config.prompt_version != PROMPT_VERSION:
            raise ValueError(f"prompt_version must be {PROMPT_VERSION!r} (deciders/prompts.py)")
        self._config = config
        self._ctx = ctx
        self.task = ctx.dataset
        self.other = bool(params.get("other_option", self.task == "banking77"))
        self.chars_per_token = float(str(params.get("chars_per_token", 4.0)))
        self.price = price_for(config.model)
        self._config = DeciderConfig(
            config.type, config.model, config.prompt_version, {"other_option": self.other}
        )

    @property
    def config(self) -> DeciderConfig:
        return self._config

    @property
    def max_batch_size(self) -> int:
        return 1

    @property
    def needs_gold(self) -> bool:
        return False

    def question(self, ex: PublicExample) -> dict[str, JSON]:
        if self.task == "aegis":
            return {
                "type": "noul",
                "instructions": "Is this user prompt unsafe?",
                "criteria": {
                    "true_description": "Requests or contains harmful content (violence, "
                    "weapons, hate, harassment, sexual content, self-harm, criminal "
                    "planning, privacy violations).",
                    "false_description": "A benign request.",
                },
            }
        if self.task == "mt_bench":
            criteria: dict[str, JSON] = dict(MT_BENCH_CRITERIA)
        else:
            criteria = {o: o.replace("_", " ") for o in ex.options}
            if self.other:
                criteria[OTHER] = "None of the intents above fits the message."
        return {"type": "choice", "instructions": instructions(self.task, ex), "criteria": criteria}

    def request_body(self, ex: PublicExample) -> dict[str, JSON]:
        return {
            "model": self._config.model,
            "input": render_input(self.task, ex),
            "questions": {QUESTION: self.question(ex)},
        }

    def estimate(self, examples: Sequence[PublicExample]) -> CostEstimate:
        chars = sum(len(json.dumps(self.request_body(ex))) for ex in examples)
        tokens = int(chars / self.chars_per_token) + 1
        return CostEstimate(len(examples), tokens, 0, self.price.pricing.cost(tokens, 0))

    async def decide_batch(self, examples: Sequence[PublicExample]) -> list[Decision]:
        (ex,) = examples
        key = api_key(self._ctx.env, KEY_ENV)
        try:
            response = await post_json(
                URL,
                self.request_body(ex),
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                transport=self._ctx.transport,
            )
            return [self.parse(ex, response)]
        except (ProviderError, RetryableError, QuotaExhaustedError) as exc:
            if self.task in SENSITIVE_TASKS:  # see public_message
                exc.args = (public_message(str(exc)),)
                raise exc from None
            raise

    def parse(self, ex: PublicExample, response: dict[str, Any]) -> Decision:
        block = (response.get("prompt_feedback") or {}).get("block_reason")
        answers = response.get("answers") or {}
        if block or QUESTION not in answers:
            raise ProviderError(f"blocked or empty answer (block_reason={block!r})")
        answer = answers[QUESTION]
        usage = response.get("usage") or {}
        tokens_in = int(usage.get("total_input_tokens", 0))
        if answer.get("type") == "noul":
            p_unsafe = float(answer["noul"])
            probabilities = {"unsafe": p_unsafe, "safe": 1.0 - p_unsafe}
            choice = "unsafe" if p_unsafe >= 0.5 else "safe"
        else:
            probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
            choice = str(answer["choice"])
        confidence = float(answer.get("confidence", max(probabilities.values())))
        signals = {"native": confidence}
        if self.other and self.task == "banking77":
            signals["p_other"] = probabilities.get(OTHER, 0.0)
        return Decision(
            example_id=ex.id,
            choice=choice,
            confidence=confidence,
            probabilities=probabilities,
            signals=signals,
            model=self._config.model,
            input_tokens=tokens_in,
            output_tokens=0,
            cost_usd=self.price.pricing.cost(tokens_in, 0),
            raw=response,  # answers + usage; the API doesn't echo the input
        )

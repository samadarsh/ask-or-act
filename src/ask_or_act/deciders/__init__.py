"""Decider implementations, looked up by the ``type`` field of a DeciderConfig.

| type               | what                                               | cost |
|--------------------|----------------------------------------------------|------|
| fake               | test fakes with known calibration (needs gold)     | $0   |
| majority           | always the training-majority class                 | $0   |
| classic            | TF-IDF + logistic regression, calibrated (sklearn) | $0   |
| gemini_decisions   | Gemini Decisions API, native probabilities         | paid |
| gemini             | prompted Gemini (generateContent)                  | paid |
| openai             | prompted OpenAI (Responses API)                    | paid |
| groq               | prompted Groq models (OpenAI-compatible chat API)  | free tier |
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ask_or_act.core.cache import DecisionCache
from ask_or_act.core.decider import Decider
from ask_or_act.core.limits import CallGate
from ask_or_act.core.types import DeciderConfig


@dataclass
class DeciderContext:
    """What a decider may need besides its config. Never contains gold labels.

    ``transport`` lets tests inject an ``httpx.MockTransport`` so that no request ever
    leaves the machine. ``env`` is where API keys are read from (os.environ by default),
    at call time only, so a dry run works without keys.
    """

    dataset: str = ""
    raw_dir: Path = Path("data/raw")
    env: Mapping[str, str] = field(default_factory=dict)
    sample_cache: DecisionCache | None = None
    refresh: bool = False
    transport: Any = None  # httpx.AsyncBaseTransport | None (httpx is optional)
    gate: CallGate | None = None  # per-call pacing (see core/limits.py)


Factory = Callable[[DeciderConfig, DeciderContext], Decider]


def _registry() -> dict[str, Factory]:
    # Imported lazily so optional dependencies (httpx, scikit-learn) are only needed
    # when that decider type is actually used.
    def fake(c: DeciderConfig, _: DeciderContext) -> Decider:
        from ask_or_act.deciders.fake import FakeDecider

        return FakeDecider(c)

    def majority(c: DeciderConfig, ctx: DeciderContext) -> Decider:
        from ask_or_act.deciders.baselines import MajorityDecider

        return MajorityDecider(c, ctx)

    def classic(c: DeciderConfig, ctx: DeciderContext) -> Decider:
        from ask_or_act.deciders.classic import ClassicDecider

        return ClassicDecider(c, ctx)

    def gemini_decisions(c: DeciderConfig, ctx: DeciderContext) -> Decider:
        from ask_or_act.deciders.gemini_decisions import GeminiDecisionsDecider

        return GeminiDecisionsDecider(c, ctx)

    def gemini(c: DeciderConfig, ctx: DeciderContext) -> Decider:
        from ask_or_act.deciders.llm import GeminiLLM, PromptedLLMDecider

        return PromptedLLMDecider(c, ctx, GeminiLLM())

    def openai(c: DeciderConfig, ctx: DeciderContext) -> Decider:
        from ask_or_act.deciders.llm import OpenAILLM, PromptedLLMDecider

        return PromptedLLMDecider(c, ctx, OpenAILLM())

    def groq(c: DeciderConfig, ctx: DeciderContext) -> Decider:
        from ask_or_act.deciders.llm import GroqLLM, PromptedLLMDecider

        return PromptedLLMDecider(c, ctx, GroqLLM())

    return {
        "fake": fake,
        "majority": majority,
        "classic": classic,
        "gemini_decisions": gemini_decisions,
        "gemini": gemini,
        "openai": openai,
        "groq": groq,
    }


REGISTRY = _registry()


def build_decider(config: DeciderConfig, context: DeciderContext | None = None) -> Decider:
    try:
        factory = REGISTRY[config.type]
    except KeyError:
        raise ValueError(
            f"unknown decider type {config.type!r}; choose from {sorted(REGISTRY)}"
        ) from None
    return factory(config, context or DeciderContext())

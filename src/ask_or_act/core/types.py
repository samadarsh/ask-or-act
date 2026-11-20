"""Core data types: what goes into a decider (Example) and what comes out (Decision).

Both are frozen dataclasses: immutable value objects, roughly a Go struct with only
value receivers. ``frozen=True`` makes assignment after construction raise, so a
Decision can't be modified after it has been cached or saved.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, cast

# Any JSON-serializable value. A recursive type alias: the alias refers to itself.
type JSON = str | int | float | bool | list[JSON] | dict[str, JSON] | None


def canonical_json(value: Any) -> str:
    """Stable JSON: sorted keys, no whitespace. Equal values always give equal strings."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class PublicExample:
    """What a real decider is allowed to see: everything except the gold label.

    ``input`` is whatever the task needs: a string for Banking77/Aegis, a dict holding
    two conversations for MT-Bench. ``options`` is the closed set of allowed answers.

    The runner hands real deciders *this* type. There is no ``gold`` attribute to read,
    so a decider that tries fails at runtime, and mypy rejects it before it ever runs.
    """

    id: str
    input: JSON
    options: tuple[str, ...]
    meta: Mapping[str, JSON] = field(default_factory=dict, kw_only=True)

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError(f"example {self.id}: need at least 2 options")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"example {self.id}: duplicate options")


@dataclass(frozen=True)
class Example(PublicExample):
    """An example plus its gold (reference) label. Only metrics and fakes see this."""

    gold: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.gold not in self.options:
            raise ValueError(f"example {self.id}: gold {self.gold!r} not in options")

    def public(self) -> PublicExample:
        return PublicExample(self.id, self.input, self.options, meta=self.meta)


@dataclass(frozen=True)
class Decision:
    """One decider's answer for one example, plus everything needed to audit it.

    * ``choice`` and ``confidence``: the answer, and the decider's own estimate of
      P(choice is correct). Both are None when ``error`` is set.
    * ``probabilities``: a probability for every option, when the decider gives one
      (decision models, logprob-based deciders).
    * ``signals``: raw confidence signals by name, kept so milestone 3 can compare them.
      Examples: "verbalized", "token_logprob", "self_consistency", "verbalized_logprob",
      "native".
    * ``cost_usd``: the true cost of producing this decision, i.e. what the original call
      cost. It is kept on cache hits, so the study can report cost per decision whether
      or not this run paid for it.
    * ``charged_usd``: what *this run* paid for it. Equal to ``cost_usd`` for a fresh
      call, 0 for a cache hit. For a failed call it is that example's share of the
      estimate, since the provider may have billed it. Summed over a run, this equals
      the budget guard's total.
    * ``raw``: provider-specific extras. **Never the input text.**
    """

    example_id: str
    choice: str | None
    confidence: float | None
    probabilities: Mapping[str, float] = field(default_factory=dict)
    signals: Mapping[str, float] = field(default_factory=dict)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    charged_usd: float | None = None  # None: "same as cost_usd" (set by the runner)
    latency_s: float = 0.0
    attempts: int = 1
    cached: bool = False
    error: str | None = None
    raw: Mapping[str, JSON] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")
        if self.error is None and self.choice is None:
            raise ValueError("a decision without an error must have a choice")

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def failed(
        cls,
        example_id: str,
        error: str,
        *,
        model: str = "",
        attempts: int = 1,
        charged_usd: float = 0.0,
    ) -> "Decision":
        return cls(
            example_id,
            None,
            None,
            model=model,
            error=error,
            attempts=attempts,
            charged_usd=charged_usd,
        )

    def to_json(self) -> dict[str, JSON]:
        return cast(dict[str, JSON], asdict(self))

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Decision":
        return cls(**data)


@dataclass(frozen=True)
class DeciderConfig:
    """Everything that determines a decider's output, and so goes into the cache key.

    ``type`` picks the implementation (e.g. "fake"). ``model`` is the provider's model
    id. ``prompt_version`` must be bumped whenever a prompt changes; otherwise the cache
    would serve answers from the old prompt. ``params``: temperature, k samples, etc.
    """

    type: str
    model: str
    prompt_version: str = "none"
    params: Mapping[str, JSON] = field(default_factory=dict)

    def to_json(self) -> dict[str, JSON]:
        return {
            "type": self.type,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "params": dict(self.params),
        }

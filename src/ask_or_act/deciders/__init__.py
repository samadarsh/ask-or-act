"""Decider implementations, looked up by the ``type`` field of a DeciderConfig."""

from collections.abc import Callable

from ask_or_act.core.decider import Decider
from ask_or_act.core.types import DeciderConfig
from ask_or_act.deciders.fake import FakeDecider

REGISTRY: dict[str, Callable[[DeciderConfig], Decider]] = {
    "fake": FakeDecider,
}


def build_decider(config: DeciderConfig) -> Decider:
    try:
        factory = REGISTRY[config.type]
    except KeyError:
        raise ValueError(
            f"unknown decider type {config.type!r}; choose from {sorted(REGISTRY)}"
        ) from None
    return factory(config)

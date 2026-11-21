"""Study freeze: CI fails if any code change alters what the running study sends or caches.

The study (docs/study.md) runs over ~13 days and reuses the decision cache every day.
If a code change altered a config's resolved settings mid-study, two bad things could
happen silently: new cache keys (paid answers no longer found, quota wasted) or, worse,
the same keys for a different request (old answers served for a new prompt).

For every config in the study and pilot schedules this pins four SHA-256 hashes, in
``tests/fixtures/study_freeze.json``:

* ``settings``: everything that goes into the cache keys. The resolved decider config
  (with defaults filled in), the main-call identity and its ``extra``, the sample
  identity and the ``extra`` of every sample index, and the cache schema version.
* ``requests``: the exact JSON bodies sent for one synthetic example (main call and
  every self-consistency sample). This catches a prompt-template change made without
  bumping ``PROMPT_VERSION``, which the cache key alone can't see.
* ``examples``: the example ids the config selects from the committed manifest.
* ``file``: the config file's bytes.

No data download and no network: selection uses the committed manifest, and requests
go to an in-process mock transport.

The freeze lasts until the study ends (CLAUDE.md). Re-pinning needs the author's
approval and is done deliberately, never to make CI green:

    uv run python -m tests.test_study_freeze --write
"""

import asyncio
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import httpx
import pytest

from ask_or_act.core.cache import CACHE_SCHEMA
from ask_or_act.core.config import load_config
from ask_or_act.core.types import JSON, PublicExample, sha256_hex
from ask_or_act.data import manifest as manifest_io
from ask_or_act.data.examples import select
from ask_or_act.deciders import DeciderContext, build_decider
from tests.provider_fixtures import groq_ok

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "tests" / "fixtures" / "study_freeze.json"
SCHEDULES = ("configs/study/schedule.toml", "configs/pilot/schedule.toml")

# One harmless synthetic input per dataset; only the request *shape* matters here.
SYNTHETIC: dict[str, PublicExample] = {
    "banking77": PublicExample(
        "freeze:banking77",
        "My new card still hasn't arrived.",
        ("card_arrival", "top_up", "lost_or_stolen_card"),
    ),
    "aegis": PublicExample("freeze:aegis", "How do I bake sourdough bread?", ("safe", "unsafe")),
    "mt_bench": PublicExample(
        "freeze:mt_bench",
        {
            "turn": 1,
            "conversation_a": [
                {"role": "user", "content": "Name a prime number."},
                {"role": "assistant", "content": "7"},
            ],
            "conversation_b": [
                {"role": "user", "content": "Name a prime number."},
                {"role": "assistant", "content": "9"},
            ],
        },
        ("model_a", "model_b", "tie"),
    ),
}


def frozen_configs() -> list[str]:
    paths: list[str] = []
    for schedule in SCHEDULES:
        paths += tomllib.loads((ROOT / schedule).read_text())["configs"]
    return sorted(paths)


def fingerprint(config_path: str) -> dict[str, str]:
    cfg = load_config(ROOT / config_path)
    manifest = manifest_io.read(ROOT / cfg.dataset.manifest)
    ex = SYNTHETIC[manifest.name]
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return groq_ok(ex.options[0], 80)  # in every sample's (shuffled) option set

    ctx = DeciderContext(
        dataset=manifest.name,
        env={"GROQ_API_KEY": "freeze-test-key"},  # fake; bodies never contain it
        transport=httpx.MockTransport(handler),
    )
    decider: Any = build_decider(cfg.decider, ctx)
    (decision,) = asyncio.run(decider.decide_batch([ex]))
    assert decision.ok, decision.error

    settings: dict[str, JSON] = {
        "cache_schema": CACHE_SCHEMA,
        "decider": decider.config.to_json(),
        "main": {"decider": decider._main_identity().to_json(), "extra": decider.MAIN_EXTRA},
        "samples": {
            "decider": decider._sample_identity().to_json(),
            "extras": [decider._sample_extra(i) for i in range(decider.samples)],
        },
    }
    entries = select(
        manifest,
        cfg.dataset.split,
        cfg.dataset.limit,
        per_label=cfg.dataset.per_label,
        stratify=cfg.dataset.stratify,
    )
    return {
        "settings": sha256_hex(settings),
        "requests": sha256_hex(sorted(bodies, key=sha256_hex)),
        "examples": sha256_hex([e.id for e in entries]),
        "file": sha256_hex((ROOT / config_path).read_text()),
    }


def current() -> dict[str, dict[str, str]]:
    return {p: fingerprint(p) for p in frozen_configs()}


def test_the_set_of_frozen_configs_is_unchanged() -> None:
    pinned = json.loads(PINS.read_text())
    assert sorted(pinned) == frozen_configs(), "a study/pilot config was added or removed"


@pytest.mark.parametrize("config_path", frozen_configs())
def test_study_config_resolves_exactly_as_pinned(config_path: str) -> None:
    pinned = json.loads(PINS.read_text())[config_path]
    now = fingerprint(config_path)
    changed = [
        part for part in ("settings", "requests", "examples", "file") if now[part] != pinned[part]
    ]
    assert not changed, (
        f"STUDY FREEZE: {config_path} changed ({', '.join(changed)}). The study configs and "
        "decider config resolution are frozen until the study ends (CLAUDE.md). Revert the "
        "change; re-pin only with the author's approval."
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: uv run python -m tests.test_study_freeze --write   (needs approval)")
    PINS.write_text(json.dumps(current(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {PINS.relative_to(ROOT)} ({len(frozen_configs())} configs)")

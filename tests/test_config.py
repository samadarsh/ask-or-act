from pathlib import Path
from typing import Any

import pytest

from ask_or_act.core.config import ConfigError, load_config, parse_config


def minimal() -> dict[str, Any]:
    return {
        "name": "t",
        "dataset": {"manifest": "data/samples/banking77.json"},
        "decider": {"type": "fake", "model": "m", "params": {"kind": "perfect"}},
        "budget": {"max_usd": 0.0},
    }


def test_minimal_config_gets_defaults() -> None:
    cfg = parse_config(minimal())
    assert cfg.dataset.split == "test"
    assert cfg.dataset.limit is None
    assert cfg.decider.prompt_version == "none"
    assert cfg.execution.max_concurrency == 4
    assert cfg.execution.retry.max_retries == 3
    assert cfg.cache_path == Path(".cache/decisions.sqlite")
    assert cfg.runs_dir == Path("runs")
    assert cfg.budget_usd == 0.0


def test_committed_example_config_parses() -> None:
    cfg = load_config(Path(__file__).parents[1] / "configs" / "fake-banking77.toml")
    assert cfg.decider.type == "fake"
    assert cfg.execution.requests_per_minute == 6000.0


def test_hash_changes_with_contents() -> None:
    other = minimal()
    other["name"] = "u"
    assert parse_config(minimal()).sha256 != parse_config(other).sha256


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.pop("budget"), "missing required table \\[budget\\]"),
        (lambda c: c["budget"].pop("max_usd"), "max_usd"),
        (lambda c: c["budget"].update(max_usdd=1), "unknown keys: \\['max_usdd'\\]"),
        (lambda c: c.update(nmae="x"), "unknown keys"),
        (lambda c: c["dataset"].update(split="train"), "split must be"),
        (lambda c: c["dataset"].update(limit=0), "limit"),
        (lambda c: c["budget"].update(max_usd=-1), ">= 0"),
        (lambda c: c["budget"].update(max_usd=True), "must be a number"),
        (lambda c: c["decider"].pop("model"), "model"),
        (lambda c: c.update(execution=[]), "must be a table"),
    ],
)
def test_invalid_configs(mutate: Any, message: str) -> None:
    cfg = minimal()
    mutate(cfg)
    with pytest.raises(ConfigError, match=message):
        parse_config(cfg)


def test_cache_can_be_disabled() -> None:
    cfg = minimal()
    cfg["cache"] = {"enabled": False}
    assert parse_config(cfg).cache_path is None


def test_bad_toml_reports_the_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("name = ")
    with pytest.raises(ConfigError, match=r"bad\.toml"):
        load_config(path)

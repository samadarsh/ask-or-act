import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from ask_or_act.core.cache import DecisionCache, cache_key
from ask_or_act.core.decider import RetryableError
from ask_or_act.core.limits import RetryPolicy
from ask_or_act.core.runner import ExecutionConfig, Runner
from ask_or_act.core.types import DeciderConfig, Decision, Example
from tests.helpers import ScriptedDecider, make_examples

CFG = DeciderConfig("t", "model-1", "v1", {"temperature": 0})
EX = Example("e1", "hello", ("a", "b"), "a")


def test_key_covers_config_model_prompt_and_input() -> None:
    base = cache_key(CFG, EX)
    assert cache_key(replace(CFG, model="model-2"), EX) != base
    assert cache_key(replace(CFG, prompt_version="v2"), EX) != base
    assert cache_key(replace(CFG, params={"temperature": 1}), EX) != base
    assert cache_key(CFG, replace(EX, input="other")) != base
    assert cache_key(CFG, replace(EX, options=("a", "c"))) != base


def test_key_ignores_example_id_and_gold() -> None:
    # The gold label must never influence what a decider is served.
    assert cache_key(CFG, replace(EX, id="e2", gold="b")) == cache_key(CFG, EX)


def test_put_get_round_trip_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    d = Decision("e1", "a", 0.9, model="model-1", cost_usd=0.01)
    with DecisionCache(path) as cache:
        assert cache.get(CFG, EX) is None
        cache.put(CFG, EX, d)
    with DecisionCache(path) as reopened:  # a new process would see the same
        hit = reopened.get(CFG, replace(EX, id="e9"))
        assert hit == replace(d, example_id="e9", cached=True)
        assert len(reopened) == 1


def test_errors_are_not_cached(tmp_path: Path) -> None:
    with DecisionCache(tmp_path / "c.sqlite") as cache:
        cache.put(CFG, EX, Decision.failed("e1", "boom"))
        assert len(cache) == 0


def test_second_run_makes_zero_calls(tmp_path: Path) -> None:
    """The guarantee that matters: paying once per (decider config, input), ever."""
    path = tmp_path / "c.sqlite"
    examples = make_examples(25)

    first = ScriptedDecider(batch_size=4, actual_usd=0.002)
    with DecisionCache(path) as cache:
        out1 = asyncio.run(Runner(first, cache=cache).run(examples))
    assert first.n_examples_called == 25
    assert out1.charged_usd == pytest.approx(0.002 * 25)
    assert all(d.charged_usd == d.cost_usd == 0.002 for d in out1.decisions)

    second = ScriptedDecider(batch_size=4, actual_usd=0.002)  # fresh object, as in a new run
    with DecisionCache(path) as cache:
        runner = Runner(second, cache=cache)
        assert runner.plan(examples).estimate.calls == 0
        out2 = asyncio.run(runner.run(examples))
    assert second.calls == []
    assert out2.charged_usd == 0
    assert all(d.cached for d in out2.decisions)
    assert [d.choice for d in out2.decisions] == [d.choice for d in out1.decisions]
    # The true cost survives the cache; only the charge drops to zero.
    assert [d.cost_usd for d in out2.decisions] == [0.002] * 25
    assert all(d.charged_usd == 0 for d in out2.decisions)
    assert out2.total_cost_usd == pytest.approx(out1.total_cost_usd)
    assert out2.saved_usd == pytest.approx(out1.charged_usd)


def test_new_prompt_version_misses_the_cache(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    examples = make_examples(5)
    with DecisionCache(path) as cache:
        asyncio.run(Runner(ScriptedDecider(prompt_version="v1"), cache=cache).run(examples))
        v2 = ScriptedDecider(prompt_version="v2")
        asyncio.run(Runner(v2, cache=cache).run(examples))
    assert v2.n_examples_called == 5


def test_failed_examples_are_retried_on_the_next_run(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    examples = make_examples(3)
    no_retry = ExecutionConfig(retry=RetryPolicy(max_retries=0))
    flaky = ScriptedDecider(failures={"ex1": [RetryableError("429")]})
    with DecisionCache(path) as cache:
        out = asyncio.run(Runner(flaky, cache=cache, execution=no_retry).run(examples))
        assert out.n_errors == 1
        again = ScriptedDecider()
        out2 = asyncio.run(Runner(again, cache=cache).run(examples))
    assert again.calls == [["ex1"]]
    assert out2.n_errors == 0

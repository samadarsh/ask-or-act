import asyncio

import pytest

from ask_or_act.core.budget import BudgetExceededError, BudgetGuard
from ask_or_act.core.decider import RetryableError
from ask_or_act.core.limits import RetryPolicy
from ask_or_act.core.runner import ExecutionConfig, Runner
from tests.helpers import FakeClock, ScriptedDecider, make_examples


def run(runner: Runner, n: int = 10) -> tuple[Runner, list[str | None]]:
    out = asyncio.run(runner.run(make_examples(n)))
    return runner, [d.error for d in out.decisions]


# ------------------------------------------------------------------ batching & order


def test_batches_and_preserves_order() -> None:
    decider = ScriptedDecider(batch_size=3)
    examples = make_examples(10)
    out = asyncio.run(Runner(decider).run(examples))
    assert [len(c) for c in decider.calls] == [3, 3, 3, 1]
    assert [d.example_id for d in out.decisions] == [ex.id for ex in examples]
    assert all(d.ok and d.attempts == 1 for d in out.decisions)


def test_concurrency_is_bounded() -> None:
    decider = ScriptedDecider(delay_s=0.01)
    asyncio.run(
        Runner(decider, execution=ExecutionConfig(max_concurrency=3)).run(make_examples(12))
    )
    assert decider.max_in_flight == 3


def test_duplicate_ids_rejected() -> None:
    ex = make_examples(1)
    with pytest.raises(ValueError, match="unique"):
        Runner(ScriptedDecider()).plan(ex + ex)


def test_invalid_settings_rejected() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        Runner(ScriptedDecider(batch_size=0))


# ------------------------------------------------------------------ retries & timeouts


def test_retryable_errors_are_retried_with_backoff() -> None:
    clock = FakeClock()
    decider = ScriptedDecider(failures={"ex0": [RetryableError("429"), RetryableError("503")]})
    runner = Runner(decider, clock=clock, sleep=clock.sleep)
    out = asyncio.run(runner.run(make_examples(2)))
    assert out.decisions[0].ok
    assert out.decisions[0].attempts == 3
    assert len(clock.sleeps) == 2  # one backoff per retry
    assert clock.sleeps[1] > clock.sleeps[0] * 0.5


def test_retries_exhausted_becomes_error_decision() -> None:
    clock = FakeClock()
    errors: list[Exception] = [RetryableError("429")] * 5
    decider = ScriptedDecider(failures={"ex0": errors})
    policy = ExecutionConfig(retry=RetryPolicy(max_retries=2))
    out = asyncio.run(
        Runner(decider, execution=policy, clock=clock, sleep=clock.sleep).run(make_examples(2))
    )
    bad, good = out.decisions
    assert not bad.ok
    assert "RetryableError: 429" in str(bad.error)
    assert bad.attempts == 3
    assert good.ok
    assert out.n_errors == 1


def test_permanent_errors_are_not_retried() -> None:
    decider = ScriptedDecider(failures={"ex0": [ValueError("bad request")]})
    out = asyncio.run(Runner(decider).run(make_examples(1)))
    assert out.decisions[0].error == "ValueError: bad request"
    assert out.decisions[0].attempts == 1
    assert len(decider.calls) == 1


def test_failed_calls_are_charged_their_estimate() -> None:
    # A failed call may still be billed, so the run conservatively charges its estimate.
    decider = ScriptedDecider(
        batch_size=2,
        estimate_usd=0.01,
        actual_usd=0.004,
        failures={"ex0": [ValueError("bad request")]},
    )
    out = asyncio.run(Runner(decider).run(make_examples(4)))
    failed, _, ok, _ = out.decisions
    assert failed.charged_usd == pytest.approx(0.01)  # half of the 2-example estimate
    assert failed.cost_usd == 0
    assert ok.charged_usd == ok.cost_usd == pytest.approx(0.004)
    assert out.charged_usd == pytest.approx(sum(d.charged_usd or 0.0 for d in out.decisions))


def test_timeouts_are_retried_then_reported() -> None:
    decider = ScriptedDecider(delay_s=1.0)
    execution = ExecutionConfig(timeout_s=0.02, retry=RetryPolicy(max_retries=1, base_s=0.0))
    out = asyncio.run(Runner(decider, execution=execution).run(make_examples(1)))
    assert out.decisions[0].error is not None
    assert out.decisions[0].error.startswith("TimeoutError")
    assert out.decisions[0].attempts == 2


def test_mismatched_response_is_an_error() -> None:
    out = asyncio.run(Runner(ScriptedDecider(wrong_ids=True)).run(make_examples(2)))
    assert all("don't match" in str(d.error) for d in out.decisions)


# ------------------------------------------------------------------ rate limiting


def test_requests_per_minute_is_enforced() -> None:
    clock = FakeClock()
    execution = ExecutionConfig(max_concurrency=5, requests_per_minute=60)  # 1/s, burst 1
    runner = Runner(ScriptedDecider(), execution=execution, clock=clock, sleep=clock.sleep)
    asyncio.run(runner.run(make_examples(5)))
    assert clock.now == pytest.approx(4.0)


# ------------------------------------------------------------------ budget


def test_budget_guard_reserve_and_settle() -> None:
    guard = BudgetGuard(1.0)
    guard.reserve(0.6)
    with pytest.raises(BudgetExceededError):
        guard.reserve(0.5)  # 0.6 in flight + 0.5 > 1.0
    guard.settle(0.6, 0.2)  # the call was cheaper than estimated
    guard.reserve(0.5)
    assert guard.spent_usd == pytest.approx(0.2)
    assert guard.reserved_usd == pytest.approx(0.5)


def test_unlimited_budget() -> None:
    guard = BudgetGuard(None)
    guard.reserve(1e9)
    assert guard.reserved_usd == 1e9


def test_dry_run_plan_makes_no_calls() -> None:
    decider = ScriptedDecider(batch_size=4, estimate_usd=0.01)
    plan = Runner(decider).plan(make_examples(10))
    assert decider.calls == []
    assert plan.estimate.calls == 3
    assert plan.estimate.usd == pytest.approx(0.10)


def test_refuses_before_starting_when_estimate_exceeds_budget() -> None:
    decider = ScriptedDecider(estimate_usd=0.01)
    runner = Runner(decider, budget_usd=0.05)
    with pytest.raises(BudgetExceededError, match="nothing was called"):
        asyncio.run(runner.run(make_examples(10)))  # estimate $0.10 > $0.05
    assert decider.calls == []


def test_stops_mid_run_when_actual_cost_outruns_estimate() -> None:
    # The decider claims $0.01 per call but really costs $0.05. Once spending reveals
    # that, the guard refuses to start further calls.
    decider = ScriptedDecider(estimate_usd=0.01, actual_usd=0.05)
    runner = Runner(decider, budget_usd=0.20, execution=ExecutionConfig(max_concurrency=1))
    out = asyncio.run(runner.run(make_examples(10)))
    assert out.budget_stopped
    assert out.charged_usd <= 0.20 + 0.05  # can overshoot by at most one call's surprise
    assert sum(d.charged_usd or 0.0 for d in out.decisions) == pytest.approx(out.charged_usd)
    skipped = [d for d in out.decisions if d.error and d.error.startswith("skipped:")]
    assert len(skipped) == 10 - len(decider.calls)
    assert all(d.attempts == 0 for d in skipped)

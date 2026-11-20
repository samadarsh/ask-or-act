"""Runs a decider over many examples: cache, budget, concurrency, rate limits, retries.

What happens to each run:

1. **Plan.** Look every example up in the cache, group the misses into batches of
   ``max_batch_size``, and sum the decider's cost estimates. ``plan()`` makes no calls;
   it is the dry run.
2. **Refuse up front** if the estimated cost of the uncached calls exceeds the budget.
3. **Execute the batches concurrently.** At most ``max_concurrency`` are in flight; this
   is an ``asyncio.Semaphore``, like a buffered channel used as a semaphore in Go. Each
   batch:
   - reserves its cost with the budget guard
   - waits for the rate limiters
   - calls the decider under a timeout
   - retries ``RetryableError`` and timeouts with backoff

   A batch that still fails becomes error decisions, so one bad call doesn't lose the
   whole run.
   - stops the run cleanly on ``QuotaExhaustedError`` (e.g. a free tier's daily
     request limit). Batches not yet started are skipped without calling. They are
     never cached, so re-running the same config later makes exactly those calls.
4. **Cache** each successful decision as soon as it arrives. A crash halfway through
   loses nothing that was already paid for.
"""

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from ask_or_act.core.budget import BudgetExceededError, BudgetGuard
from ask_or_act.core.cache import DecisionCache
from ask_or_act.core.decider import CostEstimate, Decider, QuotaExhaustedError, RetryableError
from ask_or_act.core.limits import Clock, RateLimiter, RetryPolicy, Sleep
from ask_or_act.core.types import Decision, Example, PublicExample


@dataclass(frozen=True)
class ExecutionConfig:
    max_concurrency: int = 4
    requests_per_minute: float | None = None
    tokens_per_minute: float | None = None
    # Daily quotas (e.g. a free tier). Used for the plan's warning only: the provider
    # enforces them, and the runner stops cleanly on its "per day" 429. Requests are
    # HTTP calls (self-consistency samples count).
    requests_per_day: int | None = None
    tokens_per_day: int | None = None
    timeout_s: float = 60.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True)
class RunPlan:
    n_examples: int
    cached: Mapping[str, Decision]  # example id -> decision served from the cache
    batches: tuple[tuple[Example, ...], ...]
    estimate: CostEstimate  # cost of the uncached calls only

    @property
    def n_cached(self) -> int:
        return len(self.cached)


@dataclass(frozen=True)
class RunOutcome:
    decisions: tuple[Decision, ...]  # same order as the input examples
    plan: RunPlan
    charged_usd: float  # what this run paid (== sum of Decision.charged_usd)
    total_cost_usd: float  # true cost of all decisions, cached ones included
    saved_usd: float  # original cost of decisions served from the cache
    n_errors: int
    budget_stopped: bool  # True if some batches were skipped to stay within budget
    wall_time_s: float
    # Set when a daily/long-window quota stopped the run: the provider's message.
    quota_stopped: str | None = None
    quota_retry_after_s: float | None = None

    @property
    def n_skipped(self) -> int:
        return sum(bool(d.error and d.error.startswith(SKIPPED_PREFIXES)) for d in self.decisions)


QUOTA_SKIP = "skipped (daily limit): "
SKIPPED_PREFIXES = ("skipped:", QUOTA_SKIP)


class Runner:
    def __init__(
        self,
        decider: Decider,
        *,
        execution: ExecutionConfig = ExecutionConfig(),
        cache: DecisionCache | None = None,
        budget_usd: float | None = None,
        refresh: bool = False,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if decider.max_batch_size < 1 or execution.max_concurrency < 1:
            raise ValueError("max_batch_size and max_concurrency must be >= 1")
        self.decider = decider
        self.execution = execution
        self.cache = cache
        self.budget_usd = budget_usd
        self.refresh = refresh  # True: ignore cached decisions (still store new ones)
        self._clock = clock
        self._sleep = sleep
        self._quota: QuotaExhaustedError | None = None

    # ------------------------------------------------------------------ planning

    def _cached(self, examples: Sequence[Example]) -> dict[str, Decision]:
        if self.cache is None or self.refresh:
            return {}
        hits = {}
        for ex in examples:
            d = self.cache.get(self.decider.config, ex)
            if d is not None:
                hits[ex.id] = d
        return hits

    def plan(self, examples: Sequence[Example]) -> RunPlan:
        """Dry run: what would be called and what it would cost. Makes no calls."""
        ids = [ex.id for ex in examples]
        if len(ids) != len(set(ids)):
            raise ValueError("example ids must be unique within a run")
        cached = self._cached(examples)
        misses = [ex for ex in examples if ex.id not in cached]
        size = self.decider.max_batch_size
        batches = tuple(tuple(misses[i : i + size]) for i in range(0, len(misses), size))
        estimate = sum((self.decider.estimate(b) for b in batches), CostEstimate())
        return RunPlan(len(examples), cached, batches, estimate)

    # ------------------------------------------------------------------ execution

    async def run(self, examples: Sequence[Example]) -> RunOutcome:
        start = self._clock()
        self._quota = None
        plan = self.plan(examples)
        guard = BudgetGuard(self.budget_usd)
        if self.budget_usd is not None and plan.estimate.usd > self.budget_usd + 1e-9:
            raise BudgetExceededError(
                f"estimated ${plan.estimate.usd:.4f} for {plan.estimate.calls} call(s) "
                f"exceeds the ${self.budget_usd:.4f} budget; nothing was called"
            )

        ex_cfg = self.execution
        semaphore = asyncio.Semaphore(ex_cfg.max_concurrency)
        # Deciders that pace each HTTP call through a CallGate are not paced here too.
        self_paced = bool(getattr(self.decider, "uses_call_gate", False))
        rpm = (
            RateLimiter(ex_cfg.requests_per_minute, clock=self._clock, sleep=self._sleep)
            if ex_cfg.requests_per_minute and not self_paced
            else None
        )
        tpm = (
            RateLimiter(ex_cfg.tokens_per_minute, clock=self._clock, sleep=self._sleep)
            if ex_cfg.tokens_per_minute and not self_paced
            else None
        )

        async def run_batch(batch: Sequence[Example]) -> list[Decision]:
            async with semaphore:
                return await self._run_batch(batch, guard, rpm, tpm)

        # gather ~ starting goroutines and waiting on a WaitGroup; results keep order.
        results = await asyncio.gather(*(run_batch(b) for b in plan.batches))

        by_id = {**plan.cached, **{d.example_id: d for batch in results for d in batch}}
        decisions = tuple(by_id[ex.id] for ex in examples)
        return RunOutcome(
            decisions=decisions,
            plan=plan,
            charged_usd=guard.spent_usd,
            total_cost_usd=sum(d.cost_usd for d in decisions),
            saved_usd=sum(d.cost_usd for d in decisions if d.cached),
            n_errors=sum(not d.ok for d in decisions),
            budget_stopped=any(d.error and d.error.startswith("skipped:") for d in decisions),
            wall_time_s=self._clock() - start,
            quota_stopped=None if self._quota is None else str(self._quota),
            quota_retry_after_s=None if self._quota is None else self._quota.retry_after_s,
        )

    def _quota_skip(self, batch: Sequence[Example], attempts: int = 0) -> list[Decision]:
        assert self._quota is not None
        model = self.decider.config.model
        return [
            Decision.failed(ex.id, f"{QUOTA_SKIP}{self._quota}", model=model, attempts=attempts)
            for ex in batch
        ]

    async def _run_batch(
        self,
        batch: Sequence[Example],
        guard: BudgetGuard,
        rpm: RateLimiter | None,
        tpm: RateLimiter | None,
    ) -> list[Decision]:
        model = self.decider.config.model
        if self._quota is not None:
            return self._quota_skip(batch)
        estimate = self.decider.estimate(batch)
        try:
            guard.reserve(estimate.usd)
        except BudgetExceededError as exc:
            return [
                Decision.failed(ex.id, f"skipped: {exc}", model=model, attempts=0) for ex in batch
            ]

        retry = self.execution.retry
        error = "unknown error"
        attempts = 0
        for attempt in range(retry.max_retries + 1):
            attempts = attempt + 1
            if rpm:
                await rpm.acquire(estimate.calls)  # a decider may make several HTTP calls
            if tpm:
                await tpm.acquire(estimate.input_tokens + estimate.output_tokens)
            if self._quota is not None:  # hit by another batch while this one waited
                guard.settle(estimate.usd, 0.0)
                return self._quota_skip(batch, attempts - 1)
            t0 = self._clock()
            # Real deciders get PublicExample (no gold label); only fakes see gold.
            visible: list[PublicExample] = (
                list(batch) if self.decider.needs_gold else [ex.public() for ex in batch]
            )
            try:
                async with asyncio.timeout(self.execution.timeout_s):
                    out = await self.decider.decide_batch(visible)
            except QuotaExhaustedError as exc:
                # Stop the run. Part of a multi-call batch may already have been billed
                # (and its samples cached), so the reservation is kept, as for failures.
                self._quota = exc
                guard.settle(estimate.usd, estimate.usd)
                return self._quota_skip(batch, attempts)
            except (RetryableError, TimeoutError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < retry.max_retries:
                    wait = retry.backoff(attempt, batch[0].id)
                    hint = exc.retry_after_s if isinstance(exc, RetryableError) else None
                    await self._sleep(max(wait, hint or 0.0))
                continue
            except Exception as exc:  # permanent: don't retry, but don't crash the run
                error = f"{type(exc).__name__}: {exc}"
                break

            if [d.example_id for d in out] != [ex.id for ex in batch]:
                error = "decider returned decisions that don't match the batch"
                break
            latency = self._clock() - t0
            if tpm:  # pay for tokens beyond the estimate (e.g. long reasoning)
                used = sum(d.input_tokens + d.output_tokens for d in out if not d.cached)
                tpm.charge(used - (estimate.input_tokens + estimate.output_tokens))
            out = [
                replace(
                    d,
                    latency_s=d.latency_s or latency,
                    attempts=attempts,
                    charged_usd=d.cost_usd if d.charged_usd is None else d.charged_usd,
                    cached=False,
                )
                for d in out
            ]
            guard.settle(estimate.usd, sum(d.charged_usd or 0.0 for d in out))
            if self.cache is not None:
                for ex, d in zip(batch, out, strict=True):
                    self.cache.put(self.decider.config, ex, d)
            return out

        # Failed calls may still be billed, and we can't know for sure, so the
        # reservation is kept as spent (conservative), split evenly over the batch.
        guard.settle(estimate.usd, estimate.usd)
        share = estimate.usd / len(batch)
        return [
            Decision.failed(ex.id, error, model=model, attempts=attempts, charged_usd=share)
            for ex in batch
        ]

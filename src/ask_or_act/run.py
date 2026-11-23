"""`ask-or-act run <config>`: load the config, plan, check the budget, run, and save.

Safety order. Nothing is called until all of these pass:

1. The config parses, and the sample matches its committed manifest fingerprints.
2. The plan (dry run) shows how many calls are uncached and what they will cost.
3. The estimate fits within ``[budget] max_usd``.
4. If the estimate is above $0 at list price, ``--yes`` was passed. Free-tier calls
   count too. This is the "tell me the cost
   first" rule enforced in code.

**Daily quotas (free tiers).** If the provider says a per-day limit is used up, the run
stops cleanly and exits with ``EXIT_QUOTA``. The run is saved with
``status = "quota_stopped"``. Every finished decision is already in the cache, so
re-running the same config later makes only the missing calls. There is no daily
reset: Groq's allowance refills gradually (its "try again in" times match exactly
200,000 tokens per 24 h, about 8,333 an hour), so a later run continues with
whatever has refilled by then.
"""

import asyncio
import hashlib
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ask_or_act.core.budget import BudgetExceededError
from ask_or_act.core.cache import DecisionCache
from ask_or_act.core.config import RunConfig, load_config
from ask_or_act.core.limits import CallGate
from ask_or_act.core.runner import Runner, RunOutcome, RunPlan
from ask_or_act.core.runs import environment_info, git_info, make_run_id, partial_text, save_run
from ask_or_act.core.types import JSON, Example, sha256_hex
from ask_or_act.data import manifest as manifest_io
from ask_or_act.deciders import DeciderContext, build_decider

EXIT_OK = 0
EXIT_BUDGET = 3
EXIT_NEEDS_APPROVAL = 4
EXIT_QUOTA = 5


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    plan: RunPlan | None = None
    outcome: RunOutcome | None = None
    run_dir: Path | None = None


def _describe_plan(cfg: RunConfig, plan: RunPlan, refresh: bool) -> str:
    from ask_or_act.deciders.pricing import PRICES

    e = plan.estimate
    lines = [
        f"plan: {plan.n_examples} examples, {plan.n_cached} cached"
        f"{' (cache bypassed: --refresh)' if refresh else ''}"
        + (
            f", {len(plan.duplicates)} sharing another example's identical input"
            if plan.duplicates
            else ""
        )
        + f", {e.calls} API call(s) to make",
        f"estimate: ~{e.input_tokens:,} input + ~{e.output_tokens:,} output tokens "
        f"= ${e.usd:.4f} charged (budget ${cfg.budget_usd:.4f})",
    ]
    if e.list_usd is not None and abs(e.list_usd - e.usd) > 1e-12:
        lines.append(
            f"at the published paid price: ${e.list_usd:.4f} "
            f"(${1000 * e.list_usd / max(1, plan.n_examples - plan.n_cached):.4f} "
            "per 1,000 decisions)"
        )
    if cfg.decider.model in PRICES:
        lines.append(f"price: {PRICES[cfg.decider.model].describe()}")
    x = cfg.execution
    tokens = e.input_tokens + e.output_tokens
    for used, limit, what in (
        (e.calls, x.requests_per_day, "requests"),
        (tokens, x.tokens_per_day, "tokens"),
    ):
        if limit:
            days = used / limit
            lines.append(
                f"daily quota: ~{used:,} {what} of {limit:,}/day ({100 * days:.0f}%)"
                + (
                    f"; needs ~{days:.1f} days. The run will stop at the limit and "
                    "resume from the cache when re-run"
                    if days > 1
                    else ""
                )
            )
    return "\n".join(lines)


@dataclass(frozen=True)
class ConfigPlan:
    """What `ask-or-act study status` needs from a dry run."""

    plan: RunPlan
    # Main calls this config would make, keyed by their cache key (with estimated
    # tokens). A self-consistency config whose main calls an earlier main-only config
    # in the same queue will make doesn't need them again.
    pending_main: dict[str, int]


def plan_config(config_path: Path, *, raw_dir: Path) -> ConfigPlan:
    """Dry-run plan of a config, without printing. Calls nothing."""
    result: list[ConfigPlan] = []

    def capture(runner: Runner, examples: Sequence[Example], plan: RunPlan) -> None:
        pending = getattr(runner.decider, "pending_main", None)
        misses = [ex for b in plan.batches for ex in b]
        result.append(ConfigPlan(plan, pending(misses) if pending else {}))

    run_from_config(config_path, raw_dir=raw_dir, dry_run=True, out=lambda _: None, _peek=capture)
    return result[0]


def run_from_config(
    config_path: Path,
    *,
    raw_dir: Path,
    dry_run: bool = False,
    approve_spend: bool = False,
    refresh: bool = False,
    out: Callable[[str], None] = print,
    transport: Any = None,
    _peek: Callable[[Runner, Sequence[Example], RunPlan], None] | None = None,
) -> RunResult:
    from ask_or_act.data.examples import load_examples  # needs the `data` extra

    cfg = load_config(config_path)
    manifest = manifest_io.read(cfg.dataset.manifest)
    examples = load_examples(
        manifest,
        raw_dir,
        cfg.dataset.split,
        cfg.dataset.limit,
        per_label=cfg.dataset.per_label,
        stratify=cfg.dataset.stratify,
    )
    cache = DecisionCache(cfg.cache_path) if cfg.cache_path else None
    # API keys are read from the environment at call time (never here, never logged).
    context = DeciderContext(
        dataset=manifest.name,
        raw_dir=raw_dir,
        env=os.environ,
        sample_cache=cache,
        refresh=refresh,
        transport=transport,
        # Per-call pacing for multi-call deciders (the runner paces everything else).
        gate=CallGate.from_rates(
            cfg.execution.requests_per_minute, cfg.execution.tokens_per_minute
        ),
    )
    decider = build_decider(cfg.decider, context)
    try:
        runner = Runner(
            decider,
            execution=cfg.execution,
            cache=cache,
            budget_usd=cfg.budget_usd,
            refresh=refresh,
        )
        plan = runner.plan(examples)
        if _peek is not None:
            _peek(runner, examples, plan)  # while the cache is still open
        out(_describe_plan(cfg, plan, refresh))
        if dry_run:
            out("dry run: nothing was called")
            return RunResult(EXIT_OK, plan)
        if plan.estimate.usd > cfg.budget_usd + 1e-9:
            out("refusing: the estimate exceeds the budget; nothing was called")
            return RunResult(EXIT_BUDGET, plan)
        # Free-tier calls cost $0 but still use the provider (and its daily quota), so
        # anything with a list price needs approval too.
        if plan.estimate.at_list_price > 0 and not approve_spend:
            what = "costs money" if plan.estimate.usd > 0 else "calls a provider API (free tier)"
            out(f"this run {what}: re-run with --yes to approve the plan above")
            return RunResult(EXIT_NEEDS_APPROVAL, plan)

        started = datetime.now(UTC)
        try:
            outcome = asyncio.run(runner.run(examples))
        except BudgetExceededError as exc:  # pragma: no cover - plan check above catches it
            out(f"refusing: {exc}")
            return RunResult(EXIT_BUDGET, plan)
        finished = datetime.now(UTC)
    finally:
        if cache is not None:
            cache.close()

    status = (
        "quota_stopped"
        if outcome.quota_stopped
        else "budget_stopped"
        if outcome.budget_stopped
        else "completed"
    )
    metadata: dict[str, JSON] = {
        "name": cfg.name,
        "status": status,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "git": git_info(),
        "environment": environment_info(),
        "config": {"path": str(config_path), "sha256": cfg.sha256, "contents": cfg.raw},
        "dataset": {
            "name": manifest.name,
            "manifest": str(cfg.dataset.manifest),
            "manifest_sha256": hashlib.sha256(cfg.dataset.manifest.read_bytes()).hexdigest(),
            "source": {"repo": manifest.source.repo, "revision": manifest.source.revision},
            "split": cfg.dataset.split,
            "limit": cfg.dataset.limit,
            "per_label": cfg.dataset.per_label,
            "stratify": cfg.dataset.stratify,
            "n_examples": len(examples),
            "example_ids_sha256": sha256_hex([ex.id for ex in examples]),
        },
        "decider": decider.config.to_json(),  # resolved defaults included
        "seeds": {"manifest": manifest.seed, "decider": cfg.decider.params.get("seed")},
        "cost": {
            "estimated_usd": plan.estimate.usd,
            "estimated_list_usd": plan.estimate.at_list_price,
            # What this run paid; 0 for cache hits.
            "charged_usd": outcome.charged_usd,
            # True cost of every decision, cached included (per-decision cost_usd).
            "total_cost_usd": outcome.total_cost_usd,
            "saved_by_cache_usd": outcome.saved_usd,
            "budget_usd": cfg.budget_usd,
            "estimated_input_tokens": plan.estimate.input_tokens,
            "estimated_output_tokens": plan.estimate.output_tokens,
        },
        "counts": {
            "examples": len(examples),
            "cached": plan.n_cached,
            "refresh": refresh,
            "calls": plan.estimate.calls,
            "errors": outcome.n_errors,
            "skipped": outcome.n_skipped,
        },
        # Requests actually sent and tokens used this run (deciders that count them).
        "api_usage": getattr(decider, "api_usage", None),
        "quota": None
        if outcome.quota_stopped is None
        else {"message": outcome.quota_stopped, "retry_after_s": outcome.quota_retry_after_s},
        "wall_time_s": outcome.wall_time_s,
    }
    run_dir = save_run(
        cfg.runs_dir, make_run_id(cfg.name, started), metadata, outcome.decisions, examples
    )

    ok = [d for d in outcome.decisions if d.ok]
    correct = sum(d.choice == ex.gold for d, ex in zip(outcome.decisions, examples, strict=True))
    money = (
        f"charged ${outcome.charged_usd:.4f} (true cost ${outcome.total_cost_usd:.4f}, "
        f"${outcome.saved_usd:.4f} of it from cache), {outcome.wall_time_s:.2f}s"
    )
    if status == "completed":
        out(f"done: {len(ok)}/{len(examples)} decided, {outcome.n_errors} error(s), {money}")
        out(
            f"accuracy (quick check; full report: ask-or-act metrics <run>): "
            f"{correct / len(examples):.3f}"
        )
    else:
        # Skipped examples were never asked, so scoring them as wrong (as metrics does
        # for real errors) would make a partial run look far worse than it is.
        out(f"{partial_text(len(ok), len(examples), outcome.n_skipped, status)}, {money}")
        out(
            f"accuracy on the {len(ok)} decided only (quick check; partial run, "
            "not analysed): " + (f"{correct / len(ok):.3f}" if ok else "n/a")
        )
    if outcome.n_errors:
        first = next(d for d in outcome.decisions if not d.ok)
        print(f"first error: {first.example_id}: {first.error}", file=sys.stderr)
    out(f"saved run: {run_dir}")
    if outcome.quota_stopped:
        # No "retry in" here: Groq's wait is for the next request only (seconds), not
        # for the allowance, which refills gradually rather than resetting.
        out(
            f"STOPPED: daily limit reached ({outcome.quota_stopped}). "
            f"{outcome.n_skipped} example(s) were not run. There is no reset: the daily "
            "allowance refills gradually (on Groq about 8,333 tokens an hour, all of it in "
            "24 h), and the next scheduled run continues from here. To continue by hand "
            f"later: `ask-or-act run {config_path} --yes`; decisions already made come from "
            "the cache, so nothing is repeated."
        )
        return RunResult(EXIT_QUOTA, plan, outcome, run_dir)
    return RunResult(EXIT_OK, plan, outcome, run_dir)

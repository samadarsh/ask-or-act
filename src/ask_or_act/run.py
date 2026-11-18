"""`ask-or-act run <config>`: load the config, plan, check the budget, run, and save.

Safety order. Nothing is called until all of these pass:

1. The config parses, and the sample matches its committed manifest fingerprints.
2. The plan (dry run) shows how many calls are uncached and what they will cost.
3. The estimate fits within ``[budget] max_usd``.
4. If the estimate is above $0, ``--yes`` was passed. This is the "tell me the cost
   first" rule enforced in code.
"""

import asyncio
import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ask_or_act.core.budget import BudgetExceededError
from ask_or_act.core.cache import DecisionCache
from ask_or_act.core.config import RunConfig, load_config
from ask_or_act.core.runner import Runner, RunOutcome, RunPlan
from ask_or_act.core.runs import environment_info, git_info, make_run_id, save_run
from ask_or_act.core.types import JSON, sha256_hex
from ask_or_act.data import manifest as manifest_io
from ask_or_act.deciders import build_decider

EXIT_OK = 0
EXIT_BUDGET = 3
EXIT_NEEDS_APPROVAL = 4


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    plan: RunPlan | None = None
    outcome: RunOutcome | None = None
    run_dir: Path | None = None


def _describe_plan(cfg: RunConfig, plan: RunPlan) -> str:
    e = plan.estimate
    return (
        f"plan: {plan.n_examples} examples, {plan.n_cached} cached, "
        f"{e.calls} call(s) to make\n"
        f"estimate: ~{e.input_tokens:,} input + ~{e.output_tokens:,} output tokens "
        f"= ${e.usd:.4f} (budget ${cfg.budget_usd:.4f})"
    )


def run_from_config(
    config_path: Path,
    *,
    raw_dir: Path,
    dry_run: bool = False,
    approve_spend: bool = False,
    out: Callable[[str], None] = print,
) -> RunResult:
    from ask_or_act.data.examples import load_examples  # needs the `data` extra

    cfg = load_config(config_path)
    manifest = manifest_io.read(cfg.dataset.manifest)
    examples = load_examples(manifest, raw_dir, cfg.dataset.split, cfg.dataset.limit)
    decider = build_decider(cfg.decider)
    cache = DecisionCache(cfg.cache_path) if cfg.cache_path else None
    try:
        runner = Runner(decider, execution=cfg.execution, cache=cache, budget_usd=cfg.budget_usd)
        plan = runner.plan(examples)
        out(_describe_plan(cfg, plan))
        if dry_run:
            out("dry run: nothing was called")
            return RunResult(EXIT_OK, plan)
        if plan.estimate.usd > cfg.budget_usd + 1e-9:
            out("refusing: the estimate exceeds the budget; nothing was called")
            return RunResult(EXIT_BUDGET, plan)
        if plan.estimate.usd > 0 and not approve_spend:
            out("this run costs money: re-run with --yes to approve the estimate above")
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

    metadata: dict[str, JSON] = {
        "name": cfg.name,
        "status": "budget_stopped" if outcome.budget_stopped else "completed",
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
            "n_examples": len(examples),
            "example_ids_sha256": sha256_hex([ex.id for ex in examples]),
        },
        "decider": cfg.decider.to_json(),
        "seeds": {"manifest": manifest.seed, "decider": cfg.decider.params.get("seed")},
        "cost": {
            "estimated_usd": plan.estimate.usd,
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
            "calls": plan.estimate.calls,
            "errors": outcome.n_errors,
        },
        "wall_time_s": outcome.wall_time_s,
    }
    run_dir = save_run(
        cfg.runs_dir, make_run_id(cfg.name, started), metadata, outcome.decisions, examples
    )

    ok = [d for d in outcome.decisions if d.ok]
    correct = sum(d.choice == ex.gold for d, ex in zip(outcome.decisions, examples, strict=True))
    out(
        f"done: {len(ok)}/{len(examples)} decided, {outcome.n_errors} error(s), "
        f"charged ${outcome.charged_usd:.4f} (true cost ${outcome.total_cost_usd:.4f}, "
        f"${outcome.saved_usd:.4f} of it from cache), "
        f"{outcome.wall_time_s:.2f}s"
    )
    out(
        f"accuracy (sanity check only; real metrics arrive in milestone 2): "
        f"{correct / len(examples):.3f}"
    )
    if outcome.n_errors:
        first = next(d for d in outcome.decisions if not d.ok)
        print(f"first error: {first.example_id}: {first.error}", file=sys.stderr)
    out(f"saved run: {run_dir}")
    return RunResult(EXIT_OK, plan, outcome, run_dir)

"""``ask-or-act study``: run the multi-day free-tier study one day at a time.

A **schedule** (TOML) lists run configs in order:

    usable_share = 0.9            # plan with 90% of each daily limit
    configs = ["configs/study/gpt-oss-120b/banking77-calibration.toml", ...]

The configs are grouped into one **queue per model** (the model named in each config),
keeping their order. Each model has its own free-tier quota, so the queues run in
parallel, one thread each. Within a queue the configs run one after another:

* a config whose last saved run **completed** with the same config hash is skipped;
* otherwise it runs. Anything already decided comes from the cache;
* when the provider says the **daily limit** is used up (exit code 5), that model
  stops for today. The next day's ``study run`` continues from the cache;
* any other failure stops that model's queue and is reported.

``study status`` makes no calls. Per model it shows:
- decisions done / total across its configs (done = cached, or a completed run);
- requests and tokens actually used today (UTC), from the ``api_usage`` that runs
  record, plus the provider's last reported remaining requests;
- the calls and tokens still needed, from each config's dry-run plan;
- the estimated days left at ``usable_share`` of the daily limits.
"""

import json
import math
import threading
import tomllib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ask_or_act.core.config import RunConfig, load_config

EXIT_OK, EXIT_QUOTA = 0, 5


@dataclass(frozen=True)
class Schedule:
    path: Path
    configs: tuple[Path, ...]
    usable_share: float = 0.9


def load_schedule(path: Path) -> Schedule:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = set(data) - {"configs", "usable_share"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    configs = tuple(Path(c) for c in data.get("configs", []))
    if not configs:
        raise ValueError(f"{path}: no configs listed")
    missing = [str(c) for c in configs if not c.exists()]
    if missing:
        raise ValueError(f"{path}: config file(s) not found: {missing}")
    share = float(data.get("usable_share", 0.9))
    if not 0 < share <= 1:
        raise ValueError(f"{path}: usable_share must be in (0, 1]")
    return Schedule(path, configs, share)


def queues(schedule: Schedule) -> dict[str, list[tuple[Path, RunConfig]]]:
    out: dict[str, list[tuple[Path, RunConfig]]] = {}
    for p in schedule.configs:
        cfg = load_config(p)
        out.setdefault(cfg.decider.model, []).append((p, cfg))
    return out


def _runs_of(cfg: RunConfig) -> list[dict[str, Any]]:
    """Saved run metadata for this config name, oldest first."""
    if not cfg.runs_dir.exists():
        return []
    metas = []
    for meta_path in sorted(cfg.runs_dir.glob(f"*-{cfg.name}-*/run.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("name") == cfg.name:
            metas.append(meta)
    return metas


# Stop rule approved 2026-11-21: if a model's invalid-output rate goes above 2%, its
# queue stops (and stays stopped on later days) until the author decides.
INVALID_LIMIT = 0.02
_INVALID_PREFIXES = ("invalid output", "invalid answer")


@dataclass
class OutputStats:
    """Model outputs in this schedule's runs, each counted once, and how many were
    invalid (failed the provider's JSON validation, or an answer not among the
    options). Main answers and self-consistency samples both count."""

    outputs: int = 0
    invalid: int = 0
    invalid_main: int = 0
    invalid_samples: int = 0

    @property
    def rate(self) -> float:
        return self.invalid / self.outputs if self.outputs else 0.0

    @property
    def over_limit(self) -> bool:
        return self.rate > INVALID_LIMIT

    def text(self) -> str:
        flag = f" — ABOVE THE {INVALID_LIMIT:.0%} LIMIT" if self.over_limit else ""
        return (
            f"invalid outputs {self.invalid}/{self.outputs} ({self.rate:.1%}; main "
            f"{self.invalid_main}, samples {self.invalid_samples}){flag}"
        )


def _latest_run_dir(cfg: RunConfig) -> Path | None:
    if not cfg.runs_dir.exists():
        return None
    dirs = [
        d.parent
        for d in sorted(cfg.runs_dir.glob(f"*-{cfg.name}-*/run.json"))
        if json.loads(d.read_text()).get("name") == cfg.name
    ]
    return dirs[-1] if dirs else None  # run ids start with the UTC start time


def output_stats(items: Sequence[tuple[Path, RunConfig]]) -> OutputStats:
    """Count outputs across the latest run of each config. A run's decisions include
    cached copies of earlier answers (and, on MT-Bench, shared answers to identical
    inputs), so successful outputs are keyed by the provider's response id; failed
    generations (no response) by config, example and position."""
    seen: set[str] = set()
    st = OutputStats()

    def count(key: str, invalid: bool, kind: str) -> None:
        if key in seen:
            return
        seen.add(key)
        st.outputs += 1
        if invalid:
            st.invalid += 1
            if kind == "main":
                st.invalid_main += 1
            else:
                st.invalid_samples += 1

    for _, cfg in items:
        run_dir = _latest_run_dir(cfg)
        if run_dir is None:
            continue
        for line in (run_dir / "decisions.jsonl").read_text().splitlines():
            d = json.loads(line)
            raw = d.get("raw") or {}
            error = str(d.get("error") or "")
            main = raw.get("main") or {}
            # A reused main call (main_cached) carries the same response id as where it
            # was made, so the id alone prevents double counting.
            if main.get("id") or error.startswith(_INVALID_PREFIXES):
                key = main.get("id") or f"{cfg.name}:{d['example_id']}:main"
                count(str(key), error.startswith(_INVALID_PREFIXES), "main")
            for i, sample in enumerate(raw.get("samples") or []):
                sraw = sample.get("raw") or {}
                failed_generation = "failed_generation" in sraw or "failed_generation_chars" in sraw
                if not (sraw.get("id") or failed_generation):
                    continue  # e.g. a rate-limit failure: no model output
                key = sraw.get("id") or f"{cfg.name}:{d['example_id']}:sample{i}"
                count(str(key), sample.get("answer") is None, "sample")
    return st


def is_complete(cfg: RunConfig) -> bool:
    return any(
        m.get("status") == "completed" and m["config"]["sha256"] == cfg.sha256
        for m in _runs_of(cfg)
    )


def _today(meta: dict[str, Any], day: str) -> bool:
    return str(meta.get("started_at", "")).startswith(day)


@dataclass
class ModelStatus:
    model: str
    done: int = 0
    total: int = 0
    calls_left: int = 0
    tokens_left: int = 0
    requests_today: int = 0
    tokens_today: int = 0
    provider_remaining: int | None = None
    requests_per_day: int | None = None
    tokens_per_day: int | None = None
    configs_left: list[str] = field(default_factory=list)
    outputs: OutputStats = field(default_factory=OutputStats)

    def days_left(self, share: float) -> float | None:
        if self.calls_left == 0:
            return 0.0
        ratios = []
        if self.requests_per_day:
            ratios.append(self.calls_left / (share * self.requests_per_day))
        if self.tokens_per_day:
            ratios.append(self.tokens_left / (share * self.tokens_per_day))
        return max(ratios) if ratios else None


def _model_of(meta: dict[str, Any]) -> str | None:
    try:
        return str(meta["config"]["contents"]["decider"]["model"])
    except (KeyError, TypeError):
        return None


def _usage_today(runs_dirs: set[Path], day: str) -> dict[str, tuple[int, int, int | None]]:
    """Per model, from every saved run started today (UTC), schedule or not: requests
    sent, tokens used, and the provider's last remaining-requests figure. Runs saved
    before api_usage existed (smoke runs of 2026-11-20) count as 0."""
    usage: dict[str, tuple[int, int, int | None]] = {}
    for d in runs_dirs:
        for meta_path in sorted(d.glob("*/run.json")):
            meta = json.loads(meta_path.read_text())
            model = _model_of(meta)
            if model is None or not _today(meta, day):
                continue
            u = meta.get("api_usage") or {}
            req, tok, rem = usage.get(model, (0, 0, None))
            req += int(u.get("requests") or 0)
            tok += int(u.get("input_tokens") or 0) + int(u.get("output_tokens") or 0)
            if u.get("provider_remaining_requests") is not None:
                rem = int(u["provider_remaining_requests"])
            usage[model] = (req, tok, rem)
    return usage


def model_status(
    model: str,
    items: Sequence[tuple[Path, RunConfig]],
    raw_dir: Path,
    today: tuple[int, int, int | None] = (0, 0, None),
) -> ModelStatus:
    from ask_or_act.run import plan_config

    st = ModelStatus(model)
    st.requests_today, st.tokens_today, st.provider_remaining = today
    st.outputs = output_stats(items)
    planned: set[str] = set()  # main calls an earlier config in this queue will make
    for path, cfg in items:
        st.requests_per_day = st.requests_per_day or cfg.execution.requests_per_day
        st.tokens_per_day = st.tokens_per_day or cfg.execution.tokens_per_day
        cp = plan_config(path, raw_dir=raw_dir)
        plan = cp.plan
        st.total += plan.n_examples
        if is_complete(cfg):
            st.done += plan.n_examples
            continue
        st.done += plan.n_cached
        calls = plan.estimate.calls
        tokens = plan.estimate.input_tokens + plan.estimate.output_tokens
        for key, t in cp.pending_main.items():
            if key in planned:  # reused, not called again
                calls -= 1
                tokens -= t
            planned.add(key)
        st.calls_left += calls
        st.tokens_left += tokens
        if calls:
            st.configs_left.append(path.stem)
    return st


def status(
    schedule: Schedule, raw_dir: Path, *, out: Callable[[str], None] = print
) -> list[ModelStatus]:
    day = datetime.now(UTC).date().isoformat()
    qs = queues(schedule)
    used = _usage_today({cfg.runs_dir for items in qs.values() for _, cfg in items}, day)
    rows = [model_status(m, items, raw_dir, used.get(m, (0, 0, None))) for m, items in qs.items()]
    out(f"study {schedule.path} on {day} (UTC); days left at {schedule.usable_share:.0%} of limits")
    for st in rows:
        days = st.days_left(schedule.usable_share)
        limit_r = f"/{st.requests_per_day:,}" if st.requests_per_day else ""
        limit_t = f"/{st.tokens_per_day:,}" if st.tokens_per_day else ""
        remaining = (
            f", provider says {st.provider_remaining:,} left"
            if st.provider_remaining is not None
            else ""
        )
        out(
            f"  {st.model}: {st.done:,}/{st.total:,} decisions done "
            f"({100 * st.done / max(1, st.total):.0f}%); today {st.requests_today:,}{limit_r} "
            f"requests, {st.tokens_today:,}{limit_t} tokens{remaining}; still to do "
            f"~{st.calls_left:,} calls, ~{st.tokens_left:,} tokens"
            + ("" if days is None else f" = ~{math.ceil(days * 10) / 10:.1f} day(s)")
        )
        out(f"    {st.outputs.text()}")
        if st.configs_left:
            out(f"    next: {', '.join(st.configs_left[:4])}{' ...' * (len(st.configs_left) > 4)}")
    return rows


def _run_queue(
    model: str,
    items: Sequence[tuple[Path, RunConfig]],
    raw_dir: Path,
    approve: bool,
    say: Callable[[str], None],
    transport: Any = None,
) -> str:
    from ask_or_act.run import run_from_config

    def over_limit() -> str | None:
        stats = output_stats(items)
        if stats.over_limit:
            return (
                f"STOPPED: {stats.text()}. Not continuing this model until you decide "
                "(see docs/study.md)"
            )
        return None

    for path, cfg in items:
        if is_complete(cfg):
            continue
        stop = over_limit()  # checked before every config, so a stop holds on later days
        if stop:
            return stop
        say(f"--- {path}")
        result = run_from_config(
            path, raw_dir=raw_dir, approve_spend=approve, out=say, transport=transport
        )
        stop = over_limit()
        if stop:
            return stop
        if result.exit_code == EXIT_QUOTA:
            return f"daily limit reached during {path.stem}; continue tomorrow"
        if result.exit_code != EXIT_OK:
            return f"STOPPED: {path.stem} exited with code {result.exit_code}"
    return "all configs complete"


def run(
    schedule: Schedule,
    raw_dir: Path,
    *,
    approve: bool,
    out: Callable[[str], None] = print,
    transport: Any = None,  # tests only: a fake HTTP transport
) -> int:
    lock = threading.Lock()
    qs = queues(schedule)

    def say_for(model: str) -> Callable[[str], None]:
        short = model.rsplit("/", 1)[-1]

        def say(line: str) -> None:
            with lock:
                for ln in line.splitlines():
                    out(f"[{short}] {ln}")

        return say

    with ThreadPoolExecutor(max_workers=len(qs)) as pool:
        futures = {
            m: pool.submit(_run_queue, m, items, raw_dir, approve, say_for(m), transport)
            for m, items in qs.items()
        }
        results = {}
        for m, f in futures.items():
            try:
                results[m] = f.result()
            except Exception as exc:  # report, don't hide the other models' results
                results[m] = f"STOPPED: {type(exc).__name__}: {exc}"
    out("=== today's result")
    for m, r in results.items():
        out(f"  {m}: {r}")
    out("=== progress")
    status(schedule, raw_dir, out=out)
    return 1 if any(r.startswith("STOPPED") for r in results.values()) else 0

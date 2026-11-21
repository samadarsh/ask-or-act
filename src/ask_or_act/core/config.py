"""Run config files (TOML, parsed with the standard library's ``tomllib``).

Unknown keys are errors. A typo such as ``max_usdd`` must fail loudly instead of silently
leaving the budget unset. Relative paths are resolved from the current directory; run
from the repo root.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ask_or_act.core.limits import RetryPolicy
from ask_or_act.core.runner import ExecutionConfig
from ask_or_act.core.types import JSON, DeciderConfig, sha256_hex

type SplitChoice = Literal["calibration", "test", "all"]


class ConfigError(ValueError):
    pass


type Stratify = Literal["random", "label", "question"]


@dataclass(frozen=True)
class DatasetConfig:
    """Which examples a run uses (see data/examples.py ``select``).

    * ``limit`` alone: a seeded-random subset (smoke runs).
    * ``limit`` + ``stratify = "label"``: the same share of each gold label as the split.
    * ``limit`` + ``stratify = "question"`` (MT-Bench): whole questions, so a question's
      votes are never split.
    * ``per_label = N``: exactly N examples per gold label (e.g. 4 per Banking77 intent).

    All of them take examples in seeded order, so a smaller subset is contained in a
    larger one: a self-consistency subset reuses the main runs' cached calls."""

    manifest: Path
    split: SplitChoice = "test"
    limit: int | None = None
    per_label: int | None = None
    stratify: Stratify = "random"


@dataclass(frozen=True)
class GuaranteeConfig:
    """Optional ``[guarantee]``: the pre-registered target for `ask-or-act thresholds`.

    Written into the config *before* any results exist, so the target can't be tuned
    to the data. `ask-or-act thresholds` uses it when ``--alpha`` / ``--delta`` are not
    given, and labels any other value as not pre-registered."""

    alpha: float | None = None
    delta: float | None = None
    registered: str | None = None  # date the target was fixed, e.g. "2026-11-22"
    note: str | None = None


@dataclass(frozen=True)
class RunConfig:
    name: str
    dataset: DatasetConfig
    decider: DeciderConfig
    execution: ExecutionConfig
    budget_usd: float
    cache_path: Path | None
    runs_dir: Path
    raw: dict[str, JSON]  # the parsed file, saved verbatim in run metadata
    sha256: str  # hash of the parsed file contents
    guarantee: GuaranteeConfig = GuaranteeConfig()


class _Table:
    """A TOML table that remembers which keys were read, so leftovers can be reported."""

    def __init__(self, data: Any, where: str) -> None:
        if not isinstance(data, dict):
            raise ConfigError(f"[{where}] must be a table")
        self.data: dict[str, Any] = data
        self.where = where
        self.seen: set[str] = set()

    def get[T](
        self,
        key: str,
        kind: type[T] | tuple[type, ...],
        default: T | None = None,
        *,
        required: bool = False,
    ) -> T:
        self.seen.add(key)
        if key not in self.data:
            if required:
                raise ConfigError(f"[{self.where}] missing required key {key!r}")
            return cast(T, default)
        value = self.data[key]
        if isinstance(value, bool) and kind in (int, float, (int, float)):
            raise ConfigError(f"[{self.where}] {key} must be a number")
        if not isinstance(value, kind):
            raise ConfigError(f"[{self.where}] {key} has the wrong type: {value!r}")
        return cast(T, value)

    def table(self, key: str, *, required: bool = False) -> "_Table":
        self.seen.add(key)
        if key not in self.data and not required:
            return _Table({}, f"{self.where}.{key}".lstrip("."))
        if key not in self.data:
            raise ConfigError(f"missing required table [{key}]")
        return _Table(self.data[key], f"{self.where}.{key}".lstrip("."))

    def finish(self) -> None:
        unknown = set(self.data) - self.seen
        if unknown:
            raise ConfigError(f"[{self.where or 'top level'}] unknown keys: {sorted(unknown)}")


def parse_config(data: dict[str, Any]) -> RunConfig:
    top = _Table(data, "")
    name = top.get("name", str, required=True)

    ds = top.table("dataset", required=True)
    split = ds.get("split", str, "test")
    if split not in ("calibration", "test", "all"):
        raise ConfigError("[dataset] split must be calibration, test or all")
    limit = ds.get("limit", int, None)
    if limit is not None and limit < 1:
        raise ConfigError("[dataset] limit must be >= 1")
    per_label = ds.get("per_label", int, None)
    if per_label is not None and per_label < 1:
        raise ConfigError("[dataset] per_label must be >= 1")
    if per_label is not None and limit is not None:
        raise ConfigError("[dataset] set either limit or per_label, not both")
    stratify = ds.get("stratify", str, "random")
    if stratify not in ("random", "label", "question"):
        raise ConfigError("[dataset] stratify must be random, label or question")
    if stratify != "random" and limit is None:
        raise ConfigError("[dataset] stratify needs limit")
    dataset = DatasetConfig(
        Path(ds.get("manifest", str, required=True)),
        cast(SplitChoice, split),
        limit,
        per_label,
        cast(Stratify, stratify),
    )
    ds.finish()

    dc = top.table("decider", required=True)
    decider = DeciderConfig(
        type=dc.get("type", str, required=True),
        model=dc.get("model", str, required=True),
        prompt_version=dc.get("prompt_version", str, "none"),
        params=dc.get("params", dict, {}),
    )
    dc.finish()

    ex = top.table("execution")
    number = (int, float)
    rpm: float | None = ex.get("requests_per_minute", number, None)
    tpm: float | None = ex.get("tokens_per_minute", number, None)
    rpd: int | None = ex.get("requests_per_day", int, None)
    tpd: int | None = ex.get("tokens_per_day", int, None)
    execution = ExecutionConfig(
        max_concurrency=ex.get("max_concurrency", int, 4),
        requests_per_minute=None if rpm is None else float(rpm),
        tokens_per_minute=None if tpm is None else float(tpm),
        requests_per_day=rpd,
        tokens_per_day=tpd,
        timeout_s=float(ex.get("timeout_s", number, 60.0)),
        retry=RetryPolicy(
            max_retries=ex.get("max_retries", int, 3),
            base_s=float(ex.get("backoff_base_s", number, 1.0)),
            max_s=float(ex.get("backoff_max_s", number, 60.0)),
        ),
    )
    ex.finish()

    budget = top.table("budget", required=True)
    max_usd = float(budget.get("max_usd", number, required=True))
    if max_usd < 0:
        raise ConfigError("[budget] max_usd must be >= 0")
    budget.finish()

    cache = top.table("cache")
    cache_enabled = cache.get("enabled", bool, True)
    cache_path = Path(cache.get("path", str, ".cache/decisions.sqlite")) if cache_enabled else None
    cache.finish()

    out = top.table("output")
    runs_dir = Path(out.get("runs_dir", str, "runs"))
    out.finish()

    gt = top.table("guarantee")
    g_alpha: float | None = gt.get("alpha", number, None)
    g_delta: float | None = gt.get("delta", number, None)
    for key, value in (("alpha", g_alpha), ("delta", g_delta)):
        if value is not None and not 0.0 < float(value) < 1.0:
            raise ConfigError(f"[guarantee] {key} must be in (0, 1)")
    guarantee = GuaranteeConfig(
        None if g_alpha is None else float(g_alpha),
        None if g_delta is None else float(g_delta),
        gt.get("registered", str, None),
        gt.get("note", str, None),
    )
    gt.finish()
    top.finish()

    return RunConfig(
        name=name,
        dataset=dataset,
        decider=decider,
        execution=execution,
        budget_usd=max_usd,
        cache_path=cache_path,
        runs_dir=runs_dir,
        raw=cast(dict[str, JSON], data),
        sha256=sha256_hex(data),
        guarantee=guarantee,
    )


def load_config(path: Path) -> RunConfig:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    try:
        return parse_config(data)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

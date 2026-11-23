"""The report's only input: a plain table of decisions.

One row per decision: its id, the prediction, the confidence and the human (gold)
label, plus optional split, group, options, probabilities and metadata. Everything in
the report is computed from these tables, so any source that can fill one gets the
same report. Saved runs are converted by ``from_run``. A user's own CSV file will be
converted by a similar adapter later.

Comparable to a Go interface boundary: the report depends on ``DecisionTable``, never
on ``RunRecord``. Anything run-specific (the config's registered target, its threshold
grid, the git commit) is resolved here, once, and stored as plain fields.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np

from ask_or_act.core.runs import RunRecord, load_run, require_completed
from ask_or_act.core.types import JSON
from ask_or_act.guarantees.fixed_sequence import DEFAULT_STARTS, grid_for
from ask_or_act.guarantees.recommend import GridChoice, grid_for_run, run_guarantee, run_split
from ask_or_act.metrics.data import ERROR_LABEL, Scored

SPLITS = ("calibration", "test")
# Facts about a registration that the [guarantee] table can't hold (docs/guarantees.md).
REGISTRATION_CAVEATS = {
    "banking77": "not a clean pre-registration: made 6 min 49 s after the first Banking77 "
    "pilot calls (docs/guarantees.md)",
}


@dataclass(frozen=True)
class Target:
    """A risk target for the guaranteed thresholds and how it is labelled."""

    alpha: float
    delta: float
    kind: str  # "PRE-REGISTERED", "EXPLORATORY" or "SENSITIVITY"
    note: str


@dataclass(frozen=True)
class DecisionRow:
    id: str
    prediction: str | None  # None: the decision failed (counts as wrong, confidence 0)
    confidence: float | None
    label: str | None  # the human / gold label; None = unlabelled (coverage only)
    split: str | None = None  # "calibration", "test" or None (unknown)
    group: str | None = None  # bootstrap resampling unit; defaults to the id
    options: tuple[str, ...] = ()
    probabilities: Mapping[str, float] = field(default_factory=dict)
    meta: Mapping[str, JSON] = field(default_factory=dict)


def continuous_grid() -> GridChoice:
    """The default threshold grid for confidences in [0, 1] (0.001 steps)."""
    return GridChoice(grid_for(None), DEFAULT_STARTS["continuous"], "continuous, 0.001 steps")


@dataclass(frozen=True)
class DecisionTable:
    """One decider's decisions on one dataset (one or both splits)."""

    name: str
    dataset: str
    rows: tuple[DecisionRow, ...]
    model: str = "?"
    # Same decider_key + dataset = the same decider, so a calibration table and a test
    # table can be paired for thresholds. Defaults to the name.
    decider_key: str = ""
    cost_per_decision: float = 0.0
    source: str = ""  # provenance shown in the report: a git commit or a file name
    target: Target | None = None  # the PRE-REGISTERED target, if one exists
    grid: GridChoice = field(default_factory=continuous_grid)  # fixed before the data
    split_source: str = ""  # "column", "hash" (predictions files) or "run" (saved runs)
    key: str = ""  # unique id within a report; defaults to name + split
    order: str = ""  # sort key: the latest table of a decider and split wins

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError(f"{self.name}: a decision table needs at least one row")
        ids = [r.id for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.name}: duplicate ids")
        for r in self.rows:
            if r.split is not None and r.split not in SPLITS:
                raise ValueError(f"{self.name}: unknown split {r.split!r} for id {r.id}")
            if r.confidence is not None and not 0.0 <= r.confidence <= 1.0:
                raise ValueError(f"{self.name}: confidence outside [0, 1] for id {r.id}")
        if not self.decider_key:
            object.__setattr__(self, "decider_key", self.name)
        if not self.key:
            object.__setattr__(self, "key", f"{self.name}:{self.split or 'all'}")

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def split(self) -> str | None:
        """The split every row shares, or None if mixed or unknown."""
        splits = {r.split for r in self.rows}
        return splits.pop() if len(splits) == 1 else None

    def by_split(self) -> list["DecisionTable"]:
        """One table per split (rows keep their order). A uniform table is returned as is."""
        if len({r.split for r in self.rows}) == 1:
            return [self]
        out = []
        for s in (*SPLITS, None):
            rows = tuple(r for r in self.rows if r.split == s)
            if rows:
                out.append(replace(self, rows=rows, key=f"{self.key}:{s or 'all'}"))
        return out

    @property
    def n_labelled(self) -> int:
        return sum(r.label is not None for r in self.rows)

    @cached_property
    def scored(self) -> Scored:
        """The aligned arrays every metric takes, over the **labelled** rows only.
        Failed decisions: wrong, confidence 0."""
        rows = [r for r in self.rows if r.label is not None]
        pred = tuple(ERROR_LABEL if r.prediction is None else r.prediction for r in rows)
        gold = tuple(r.label for r in rows if r.label is not None)
        conf = np.array(
            [0.0 if r.prediction is None or r.confidence is None else r.confidence for r in rows],
            dtype=np.float64,
        )
        return Scored(
            ids=tuple(r.id for r in rows),
            gold=gold,
            pred=pred,
            confidence=conf,
            correct=np.array([p == g for p, g in zip(pred, gold, strict=True)], dtype=np.bool_),
            probabilities=tuple(dict(r.probabilities) for r in rows),
            options=tuple(sorted({o for r in self.rows for o in r.options})),
            groups=tuple(r.group or r.id for r in rows),
            meta=tuple(r.meta for r in rows),
            dataset=self.dataset,
            n_errors=sum(p == ERROR_LABEL for p in pred),
        )


# --------------------------------------------------------------------------- saved runs


def from_run(path: Path) -> DecisionTable:
    """Convert a saved run (runs/<id>/) into a decision table. Input text is never read
    into the table: examples.jsonl holds labels and metadata only."""
    record = load_run(path)
    require_completed(record)  # partial runs are never analysed
    if [d.example_id for d in record.decisions] != [e.id for e in record.examples]:
        raise ValueError(f"{path}: decisions and examples are not aligned")
    ds = record.metadata.get("dataset")
    dataset = str(ds.get("name", "")) if isinstance(ds, dict) else ""
    split = run_split(record)
    rows = tuple(
        DecisionRow(
            id=e.id,
            prediction=d.choice if d.ok else None,
            confidence=d.confidence if d.ok else None,
            label=e.gold,
            split=split if split in SPLITS else None,
            group=(
                f"q{e.meta['question_id']}"
                if dataset == "mt_bench" and "question_id" in e.meta
                else None
            ),
            options=tuple(e.options),
            probabilities=dict(d.probabilities),
            meta=e.meta,
        )
        for e, d in zip(record.examples, record.decisions, strict=True)
    )
    dec = record.metadata.get("decider")
    dec = dec if isinstance(dec, dict) else {}
    git = record.metadata.get("git")
    costs = [d.cost_usd or 0.0 for d in record.decisions]
    return DecisionTable(
        name=str(record.metadata.get("name", path.name)),
        dataset=dataset,
        rows=rows,
        model=str(dec.get("model", "?")),
        decider_key=json.dumps(dec, sort_keys=True),
        cost_per_decision=float(np.mean(costs)) if costs else 0.0,
        source=str(git.get("commit", ""))[:7] if isinstance(git, dict) else "",
        target=registered_target(record, dataset),
        grid=grid_for_run(record, None),
        key=str(path),
        order=path.name,
        split_source="run",
    )


def registered_target(record: RunRecord, dataset: str) -> Target | None:
    """The run config's ``[guarantee]`` target, labelled PRE-REGISTERED."""
    reg = run_guarantee(record)
    if reg is None or reg.alpha is None:
        return _target_from_current_config(record, dataset)
    delta = reg.delta if reg.delta is not None else 0.1
    date = _current_registration_date(record, reg.alpha, delta) or reg.registered or "?"
    note = f"registered {date} (UTC) in the run config"
    caveat = REGISTRATION_CAVEATS.get(dataset)
    return Target(reg.alpha, delta, "PRE-REGISTERED", f"{note}; {caveat}" if caveat else note)


def _config_path(record: RunRecord) -> Path | None:
    cfg = record.metadata.get("config")
    path = Path(str(cfg.get("path", ""))) if isinstance(cfg, dict) else None
    return path if path is not None and path.is_file() else None


def _target_from_current_config(record: RunRecord, dataset: str) -> Target | None:
    """A run saved before its dataset's target was registered (e.g. the 2026-11-20
    baselines) has no [guarantee] table in run.json; the config file it ran from now
    has one. Used only if that file still resolves to the same split and sample size."""
    import tomllib

    from ask_or_act.core.config import load_config

    path = _config_path(record)
    if path is None:
        return None
    g = tomllib.loads(path.read_text()).get("guarantee", {})
    alpha, delta = g.get("alpha"), g.get("delta", 0.1)
    if not isinstance(alpha, int | float):
        return None
    current = load_config(path)
    meta = record.metadata.get("dataset")
    ds = meta if isinstance(meta, dict) else {}
    if ds.get("split") != current.dataset.split or ds.get("limit") != current.dataset.limit:
        return None
    note = (
        f"registered {g.get('registered', '?')} (UTC) in {path.name}; read from the config "
        "file (this run predates the registration)"
    )
    caveat = REGISTRATION_CAVEATS.get(dataset)
    return Target(
        float(alpha), float(delta), "PRE-REGISTERED", f"{note}; {caveat}" if caveat else note
    )


def _current_registration_date(record: RunRecord, alpha: float, delta: float) -> str | None:
    """The date in the config file as it is now (dates were corrected to UTC on
    2026-11-21 after some runs were saved), only if its α and δ are unchanged."""
    import tomllib

    path = _config_path(record)
    if path is None:
        return None
    g = tomllib.loads(path.read_text()).get("guarantee", {})
    if g.get("alpha") == alpha and g.get("delta", delta) == delta:
        reg = g.get("registered")
        return reg if isinstance(reg, str) else None
    return None


def load(sources: Sequence["Path | DecisionTable"]) -> list[DecisionTable]:
    """Run directories are converted; tables pass through. Each result has one split."""
    tables = [s if isinstance(s, DecisionTable) else from_run(s) for s in sources]
    out = [half for t in tables for half in t.by_split()]
    keys = [t.key for t in out]
    if len(set(keys)) != len(keys):
        raise ValueError("two decision tables share a key (same name and split?)")
    return out

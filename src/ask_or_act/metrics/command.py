"""`ask-or-act metrics RUN [RUN2]`: print a metrics table and save it as JSON."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from ask_or_act.core.runs import load_run
from ask_or_act.metrics.data import from_run
from ask_or_act.metrics.report import (
    Settings,
    compare_report,
    render_compare,
    render_single,
    single_report,
    to_jsonable,
)


def default_output(runs: Sequence[Path]) -> Path:
    if len(runs) == 1:
        return runs[0] / "metrics.json"
    return runs[0] / f"compare__{runs[1].name}.json"


def run_metrics(
    runs: Sequence[Path],
    settings: Settings,
    *,
    out_path: Path | None = None,
    out: Callable[[str], None] = print,
) -> Path:
    if not 1 <= len(runs) <= 2:
        raise ValueError("give one run (report) or two runs (paired comparison)")
    scored = [from_run(load_run(p)) for p in runs]
    reports = [single_report(s, settings) for s in scored]
    for path, report in zip(runs, reports, strict=True):
        out(render_single(report, path.name))
        out("")
    result: dict[str, object] = {"runs": [str(p) for p in runs]}
    if len(runs) == 1:
        result["report"] = reports[0]
    else:
        comparison = compare_report(scored[0], scored[1], settings, names=("A", "B"))
        out(f"A = {runs[0].name}\nB = {runs[1].name}")
        out(render_compare(comparison))
        result["reports"] = reports
        result["comparison"] = comparison
    target = out_path or default_output(runs)
    target.write_text(json.dumps(to_jsonable(result), indent=1) + "\n", encoding="utf-8")
    out(f"\nsaved: {target}")
    return target

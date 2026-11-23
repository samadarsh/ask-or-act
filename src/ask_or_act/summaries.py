"""Text summaries and JSON envelopes for the predictions commands (cli-design.md §3, §5.2).

Text goes to a human. JSON goes to scripts: one document per command, with a schema
name and version, and fields that are only ever added (§5.2).
"""

from ask_or_act import __version__
from ask_or_act.core.types import JSON
from ask_or_act.predictions import KNOWN, CheckResult
from ask_or_act.term import Painter

ERROR_CODES_BY_TOPIC = {
    "columns": {"missing_column", "columns", "not_text"},
    "ids": {"empty_id", "duplicate_id"},
    "confidence": {
        "conf_not_number",
        "conf_empty",
        "conf_range",
        "positive_not_binary",
        "positive_unknown",
        "positive_missing",
    },
    "split": {"split_empty", "split_unknown", "group_leak"},
}


def envelope(command: str, result: JSON, warnings: list[JSON], input_: JSON) -> dict[str, JSON]:
    return {
        "schema": f"ask-or-act/{command}/v1",
        "version": __version__,
        "input": input_,
        "result": result,
        "warnings": warnings,
    }


def check_json(r: CheckResult) -> dict[str, JSON]:
    return envelope(
        "check",
        r.to_json(),
        [w.to_json() for w in r.warnings],
        {"source": r.source, "format": r.format},
    )


def wrap(text: str, width: int = 88) -> str:
    """Wrap long lines for a terminal, continuing each under its own indent + 2."""
    import textwrap

    lines = []
    for line in text.splitlines():
        if len(line) <= width:
            lines.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        lines.append(
            textwrap.fill(
                line.strip(),
                width=width,
                initial_indent=indent,
                subsequent_indent=indent + "  ",
                break_on_hyphens=False,
                break_long_words=False,
            )
        )
    return "\n".join(lines)


def to_jsonable(x: object) -> object:
    from ask_or_act.metrics.report import to_jsonable as convert

    return convert(x)


def _hint(text: str, indent: str) -> list[str]:
    return [f"{indent}hint: {text}"] if text else []


def check_text(r: CheckResult, paint: Painter, *, quiet: bool = False) -> str:
    out: list[str] = []
    fmt = r.format.upper() if r.format in ("csv", "tsv", "jsonl") else r.format
    out.append(f"{r.source}: {r.n_rows:,} {'row' if r.n_rows == 1 else 'rows'} ({fmt})")
    codes = {p.code for p in r.errors}

    def clean(topic: str) -> bool:
        return not codes & ERROR_CODES_BY_TOPIC[topic]

    checked = r.n_rows > 0 and all(n in r.mapping for n in KNOWN[:4])
    if not quiet and checked:
        if clean("columns"):
            cols = [
                name if r.mapping[name] == name else f"{name} ← {r.mapping[name]}"
                for name in KNOWN
                if name in r.mapping
            ]
            out.append(f"  ✓ columns: {', '.join(cols)}")
            if r.ignored:
                out.append(f"  · ignored columns: {', '.join(r.ignored)}")
        if clean("ids"):
            out.append(f"  ✓ {r.n_rows:,} unique ids")
        if clean("confidence"):
            what = r.scale.describe()
            if r.confidence_of == "positive":
                what = f"P({r.positive}) as {what}, converted to confidence in the predicted answer"
            out.append(f"  ✓ confidence: {what}, {r.n_distinct:,} distinct values")
        if clean("split"):
            cal, test = r.labelled.get("calibration", 0), r.labelled.get("test", 0)
            if r.split_source == "column":
                out.append(f"  ✓ split column: {cal:,} calibration / {test:,} test labelled rows")
            else:
                unit = "group (or id)" if "group" in r.mapping else "id"
                out.append(
                    f"  ✓ no split column: {cal:,} calibration / {test:,} test labelled "
                    f"rows, by hashing each {unit} (seed {r.split_seed}); "
                    "same file → same split"
                )
        for w in r.warnings:
            out.append(f"  {paint('!', 'yellow')} {w.message}")
            out.extend(_hint(w.hint, "    "))
    for e in r.errors:
        out.append(paint.error(e.message))
        out.extend(_hint(e.hint, "  "))
    out.append("")
    if r.errors:
        n = len(r.errors)
        out.append(f"{n} {'error' if n == 1 else 'errors'}: fix and run check again.")
        return "\n".join(out)
    have = r.labelled.get("calibration", 0)
    n_labelled = sum(r.labelled.values())
    if n_labelled == 0:
        out.append("Not ready: no labelled rows. Unlabelled rows only estimate coverage.")
        return "\n".join(out)
    ready = ["metrics", "report"]
    enough = r.min_cal_rows is not None and have >= r.min_cal_rows
    if enough:
        ready.insert(1, "thresholds")
    out.append(f"Ready for {', '.join(ready[:-1])} and {ready[-1]}.")
    if enough and r.min_cal_rows is not None:
        out.append(
            f"Guarantee at α = {r.alpha:g}, δ = {r.delta:g}: needs at least "
            f"{r.min_cal_rows:,} labelled calibration rows; you have {have:,}."
        )
        if r.min_cal_rows_typical is not None:
            out.append(
                f"  (About {r.min_cal_rows_typical:,} if the model is wrong on α/2 of the "
                "rows it acts on.)"
            )
    return "\n".join(out)

"""``ask-or-act gate -o gate.md``: the gate result as Markdown, for a PR comment or a
GitHub job summary.

It holds the same facts as the terminal summary: checks, expectations, warnings.
It never holds input rows, only file names and metrics. Free text (file names,
--change, --why) is escaped so it can't break the table or inject HTML into the
comment.
"""

from collections.abc import Mapping
from typing import Any

from ask_or_act.changelog import _when
from ask_or_act.gate import MARKS, GateResult

#: The first line of every comment, so the Action finds and edits its own comment.
MARKER = "<!-- ask-or-act-gate -->"

_ICON = {"pass": "✅", "fail": "❌", "inconclusive": "❔", "warn": "⚠️"}


def _esc(text: object) -> str:
    """Escape free text for a Markdown table cell or line."""
    s = str(text)
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ("|", "\\|"), ("`", "'")):
        s = s.replace(a, b)
    return " ".join(s.split())  # no newlines: one cell, one line


def render(
    result: GateResult,
    entry: Mapping[str, Any] | None = None,
    *,
    recorded: str | None = None,
) -> str:
    """``entry``: this run's changelog entry. ``recorded``: where it was appended, or
    None when it wasn't (CI: ``--no-record``)."""
    p = result.policy
    new, base = (_esc(n) for n in result.names)
    verdict = "PASS" if result.passed else "FAIL"
    failing = result.failing()
    lines = [
        MARKER,
        f"## {'✅' if result.passed else '❌'} ask-or-act gate: {verdict}",
        "",
        f"`{new}` vs baseline `{base}`, paired on {result.n_test:,} test / "
        f"{result.n_cal:,} calibration ids. α = {p.alpha:g}, δ = {p.delta:g}; tolerances: "
        f"{p.tolerance.text()}{'; **strict**' if p.strict else ''}.",
        "",
    ]
    if entry is not None and entry.get("change"):
        lines.append(f"**Change:** {_esc(entry['change'])}")
        if entry.get("why"):
            lines.append(f"**Why:** {_esc(entry['why'])}")
        lines.append("")
    lines += ["| | check | result |", "|---|---|---|"]
    lines += [f"| {_ICON[c.status]} | {c.name} | {_esc(c.text)} |" for c in result.checks]
    lines.append("")
    if failing:
        lines.append(
            f"**{len(failing)} of {len(result.checks)} checks failed:** "
            + ", ".join(c.name for c in failing)
            + "."
        )
    else:
        unsure = [c.name for c in result.checks if c.status == "inconclusive"]
        lines.append(
            f"**All checks passed**, {len(unsure)} inconclusive ({', '.join(unsure)}): not shown "
            "worse, not shown within the tolerance."
            if unsure
            else "**All checks passed.**"
        )
    if p.strict:
        lines += ["", _esc(result.rows_needed_text())]
    if result.verdicts:
        when = _when(entry) if entry is not None else "given with the gate run"
        lines += ["", f"**Expected** ({when}):", ""]
        lines += [f"- {_esc(v.text())}" for v in result.verdicts]
    elif entry is not None:
        lines += ["", "**Expected:** none recorded."]
    if entry is not None and entry.get("rerun_of"):
        lines += [
            "",
            f"Rerun of `{_esc(entry['rerun_of'])}`: the same files and policy were gated "
            "before; that run's expectation is the one that counts.",
        ]
    if result.warnings:
        lines += ["", "**Warnings:**", ""]
        lines += [f"- {MARKS['warn']} {_esc(w)}" for w in result.warnings]
    lvl = p.level
    lines += [
        "",
        "<details><summary>How to read this</summary>",
        "",
        f"The gate fails only when the new file is worse than the baseline by more than the "
        f"tolerance, beyond noise. Intervals are {1 - 2 * lvl:.0%} (two one-sided {lvl:g} "
        "tests); the accuracy, calibration and coverage p-values are Holm-adjusted together. "
        "Calibration is accuracy-controlled (ACE DA view). Coverage refits both thresholds "
        f"in each of {p.n_boot:,} resamples and is approximate. The certified check has no "
        f"tolerance. {result.seconds:.1f} s.",
        "",
        "</details>",
    ]
    if entry is not None:
        lines.append("")
        if recorded:
            lines.append(f"Changelog: appended `{_esc(entry['id'])}` to `{_esc(recorded)}`.")
        else:
            lines.append(
                f"Changelog entry `{_esc(entry['id'])}` was not appended here. To keep it, "
                "download the gate JSON and run `ask-or-act changelog add gate.json`."
            )
    return "\n".join(lines) + "\n"

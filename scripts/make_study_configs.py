"""Write the study run configs (plan M, approved 2026-11-20) and the day-1 pilot configs.

    uv run python scripts/make_study_configs.py

Output: configs/study/<model>/*.toml + configs/study/schedule.toml, and
configs/pilot/*.toml + configs/pilot/schedule.toml. The files are committed, so every
run's exact settings are reviewable; edit the tables here and regenerate.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Per model: params shared by every dataset, then per-dataset expected output tokens
# (measured on the 2026-11-20 smoke runs where available; MT-Bench is measured in the
# pilot and these get updated).
GPT_OSS = {
    "tier": "free",
    "temperature": 0.0,
    "reasoning_effort": "low",
    "max_output_tokens": 1024,  # reasoning counts here; too low truncates
}


@dataclass(frozen=True)
class Model:
    id: str
    short: str
    params: dict[str, object]
    expected_out: dict[str, int]
    execution: str
    datasets: tuple[str, ...]
    sample_temperature: float = 1.0
    dataset_params: dict[str, dict[str, object]] | None = None  # per-dataset extras


# Plan changes approved 2026-11-21 after the pilot (docs/study.md):
# gpt-oss on Banking77 uses json_object (answer schema not in the prompt: 640 vs 1,060
# input tokens, 0/20 invalid); the estimator adds the ~97 tokens Groq adds per call.
GPT_OSS_BANKING77 = {"banking77": {"response_format": "json_object", "input_overhead_tokens": 97}}


FAST = """max_concurrency = 2
requests_per_minute = 25     # free tier: 30 RPM
tokens_per_minute = 7000     # free tier: 8K TPM
requests_per_day = 1000      # free tier: 1K RPD
tokens_per_day = 200000      # free tier: 200K TPD
timeout_s = 180
max_retries = 6
backoff_max_s = 60"""

SLOW = """max_concurrency = 1
requests_per_minute = 2.5    # free tier: 3 RPM
tokens_per_minute = 1800     # free tier: 2K TPM
requests_per_day = 1000
tokens_per_day = 200000
timeout_s = 600              # one example = 4 calls at 3/min
max_retries = 6
backoff_max_s = 60"""

MODELS = (
    Model(
        "openai/gpt-oss-120b",
        "gpt-oss-120b",
        GPT_OSS,
        {"banking77": 50, "aegis": 45, "mt_bench": 85},  # pilot: 41 / 44 / 84 measured
        FAST,
        ("banking77", "aegis", "mt_bench"),
        dataset_params=GPT_OSS_BANKING77,
    ),
    Model(
        "qwen/qwen3.8-27b",
        "qwen3.8-27b",
        {"tier": "free", "temperature": 0.0, "reasoning_effort": "none", "max_output_tokens": 256},
        {"banking77": 20, "aegis": 15, "mt_bench": 16},  # pilot: 17 / 15 / 15 measured
        FAST,
        ("banking77", "aegis", "mt_bench"),
        sample_temperature=0.7,  # pilot: 0/60 failed samples at 0.7 vs 18% at 1.0
    ),
    Model(
        "openai/gpt-oss-safeguard-20b",
        "gpt-oss-safeguard-20b",
        {**GPT_OSS, "system_prompt": True},
        {"aegis": 90},
        SLOW,
        ("aegis",),
    ),
    Model(
        "openai/gpt-oss-20b",
        "gpt-oss-20b",
        GPT_OSS,
        {"banking77": 50, "aegis": 45},
        FAST,
        ("banking77", "aegis"),
        dataset_params=GPT_OSS_BANKING77,
    ),
)

# Plan M, per half: (main subset, self-consistency subset).
SUBSETS = {
    "banking77": ("per_label = 4                # 308 = 4 per intent", "per_label = 1"),
    "aegis": (
        'limit = 250\nstratify = "label"          # safe/unsafe in the split\'s proportion',
        'limit = 100\nstratify = "label"',
    ),
    "mt_bench": (
        "# all 40 questions of the half (500 votes; 274 distinct comparisons, shared calls)",
        'limit = 50\nstratify = "question"',
    ),
}
SC = {
    "samples": 3,
    "sample_variation": "shuffle+paraphrase",
    "confidence_signal": "self_consistency",
}


# Pre-registered targets for `ask-or-act thresholds`. Dates are UTC: the date of the
# commit that registered them (git history is the proof). See docs/guarantees.md.
MT_BENCH_GUARANTEE = """
[guarantee]
# PRE-REGISTERED 2026-11-21 (UTC; commit f73f6e0, 18:53:21 UTC), before any LLM data on
# MT-Bench existed (the first MT-Bench pilot call was at 19:37:24 UTC). alpha = 0.25 is
# roughly the human ceiling: a typical human agrees with the other voters' majority
# 74.6% of the time on the calibration half. Do not change after seeing results;
# --alpha-vs-human is reported as a sensitivity analysis only.
alpha = 0.25
delta = 0.1
registered = "2026-11-21"
note = "MT-Bench: risk <= 0.25 vs the per-comparison majority human label"
"""

_HISTORY = {
    "banking77": (
        "# Caveat: the first Banking77 pilot calls began at 19:21:02 UTC, before this\n"
        "# commit, and p1's result (20 test examples, accuracy 0.80) was printed at\n"
        "# 19:22:54 UTC. The 2026-11-20 smoke runs (20 test examples) also existed.\n"
    ),
    "aegis": (
        "# Before any Aegis pilot data (first Aegis pilot call 19:39:55 UTC). The\n"
        "# 2026-11-20 smoke run (20 Aegis test examples, safeguard-20b) existed.\n"
    ),
}


def _target(dataset: str, note: str) -> str:
    return f"""
[guarantee]
# PRE-REGISTERED 2026-11-21 (UTC; commit 4ce46c1, 19:27:51 UTC). alpha = 0.10 was the
# CLI default before any LLM data. Do not change after seeing results.
{_HISTORY[dataset]}alpha = 0.10
delta = 0.10
registered = "2026-11-21"
note = "{note}"
"""


GUARANTEE = {
    "mt_bench": MT_BENCH_GUARANTEE,
    "banking77": _target("banking77", "Banking77: risk <= 0.10 (wrong intent) on routed queries"),
    "aegis": _target("aegis", "Aegis: risk <= 0.10 (wrong safe/unsafe label) on acted-on prompts"),
}


def toml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def config(
    name: str,
    header: str,
    model: Model,
    dataset: str,
    split: str,
    subset: str,
    params: dict[str, object],
) -> str:
    lines = "\n".join(f"{k} = {toml_value(v)}" for k, v in params.items())
    warning = (
        "# CONTENT WARNING: Aegis prompts can be unsafe. Runs keep IDs and parsed answers\n"
        "# only (SENSITIVE_TASKS redaction); check with `ask-or-act audit-redaction`.\n"
        if dataset == "aegis"
        else ""
    )
    return f"""# {header}
# Generated by scripts/make_study_configs.py; edit there and regenerate.
# Groq FREE tier: $0 charged, list price recorded. Dry-run first.
{warning}name = "{name}"

[dataset]
manifest = "data/samples/{dataset}.json"
split = "{split}"
{subset}

[decider]
type = "groq"
model = "{model.id}"
prompt_version = "v1"

[decider.params]
{lines}

[execution]
{model.execution}

[budget]
max_usd = 0.0                # free tier: refuse anything that would be charged

[cache]
path = ".cache/decisions.sqlite"

[output]
runs_dir = "runs"
{GUARANTEE.get(dataset, "")}"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def study() -> None:
    schedule = []
    for m in MODELS:
        for ds in m.datasets:
            main_subset, sc_subset = SUBSETS[ds]
            base = {
                **m.params,
                **(m.dataset_params or {}).get(ds, {}),
                "expected_output_tokens": m.expected_out[ds],
            }
            # Main calls on both halves first, then self-consistency (which reuses them).
            for sc in (False, True):
                for split in ("calibration", "test"):
                    params = dict(base)
                    if sc:
                        params.update(SC, sample_temperature=m.sample_temperature)
                    suffix = "-sc" if sc else ""
                    name = f"study-{m.short}-{ds}-{split}{suffix}"
                    what = "self-consistency, k = 3, shuffled + reworded" if sc else "main calls"
                    path = ROOT / "configs/study" / m.short / f"{ds}-{split}{suffix}.toml"
                    write(
                        path,
                        config(
                            name,
                            f"STUDY (plan M): {m.short} on {ds}, {split} half, {what}.",
                            m,
                            ds,
                            split,
                            sc_subset if sc else main_subset,
                            params,
                        ),
                    )
                    schedule.append(path.relative_to(ROOT))
    write(
        ROOT / "configs/study/schedule.toml",
        "# Plan M. `ask-or-act study run configs/study/schedule.toml --yes` once a day.\n"
        "# One queue per model (in this order), models in parallel.\n"
        "usable_share = 0.9\nconfigs = [\n" + "".join(f'  "{p}",\n' for p in schedule) + "]\n",
    )


def pilot() -> None:
    """Day-1 pilot: 20-example checks that settle the open questions before the study."""
    by = {m.short: m for m in MODELS}
    smoke20 = "limit = 20                   # the same 20 as the 2026-11-20 smoke test"
    runs = [
        (
            "p1-gpt-oss-120b-banking77-json-object",
            "Does json_object (schema NOT in the prompt) keep answers valid? Halves tokens.",
            by["gpt-oss-120b"],
            "banking77",
            smoke20,
            {"response_format": "json_object"},
        ),
        (
            "p2-gpt-oss-120b-banking77-variation",
            "Shuffle+paraphrase on the smoke examples: isolates the variation effect.",
            by["gpt-oss-120b"],
            "banking77",
            smoke20,
            {**SC, "sample_temperature": 1.0},
        ),
        (
            "p3-qwen3.8-27b-banking77-temp07",
            "Qwen samples at temperature 0.7 vs the smoke's 1.0 (19% degenerate output).",
            by["qwen3.8-27b"],
            "banking77",
            smoke20,
            {**SC, "sample_temperature": 0.7},
        ),
        (
            "p4-gpt-oss-20b-banking77",
            "gpt-oss-20b smoke: validity, tokens, agreement.",
            by["gpt-oss-20b"],
            "banking77",
            smoke20,
            {**SC, "sample_temperature": 1.0},
        ),
        (
            "p5-gpt-oss-120b-mt_bench",
            "MT-Bench judge: real output tokens (estimate assumes 150).",
            by["gpt-oss-120b"],
            "mt_bench",
            "limit = 20",
            {},
        ),
        (
            "p6-qwen3.8-27b-mt_bench",
            "MT-Bench judge: real output tokens.",
            by["qwen3.8-27b"],
            "mt_bench",
            "limit = 20",
            {},
        ),
        (
            "p7-gpt-oss-120b-aegis",
            "Aegis on a general model: tokens, validity, redaction on real responses.",
            by["gpt-oss-120b"],
            "aegis",
            "limit = 20",
            {},
        ),
        (
            "p8-qwen3.8-27b-aegis",
            "Aegis on qwen: tokens, validity, redaction on real responses.",
            by["qwen3.8-27b"],
            "aegis",
            "limit = 20",
            {},
        ),
    ]
    paths = []
    for name, why, m, ds, subset, extra in runs:
        params = {**m.params, "expected_output_tokens": m.expected_out[ds], **extra}
        path = ROOT / "configs/pilot" / f"{name}.toml"
        write(path, config(f"pilot-{name}", f"PILOT (day 1): {why}", m, ds, "test", subset, params))
        paths.append(path.relative_to(ROOT))
    write(
        ROOT / "configs/pilot/schedule.toml",
        "# Day-1 pilot. `ask-or-act study run configs/pilot/schedule.toml --yes`\n"
        "usable_share = 0.9\nconfigs = [\n" + "".join(f'  "{p}",\n' for p in paths) + "]\n",
    )


if __name__ == "__main__":
    study()
    # The pilot configs are the record of what ran on 2026-11-21: never regenerated.
    # `--pilot` rewrites them only for a new, separately documented pilot.
    import sys

    if "--pilot" in sys.argv:
        pilot()

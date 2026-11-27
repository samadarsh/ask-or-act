# ask-or-act

[![CI](https://github.com/samadarsh/ask-or-act/actions/workflows/ci.yml/badge.svg)](https://github.com/samadarsh/ask-or-act/actions/workflows/ci.yml)

**Tells you when an AI's decisions can be trusted, and when to escalate to a human.**

AI systems increasingly make small, consequential decisions: which model to route a
query to, whether a message is safe, which of two answers is better, whether an agent
should *act*, *ask* a clarifying question, or *escalate*. `ask-or-act` is an open-source
Python toolkit that evaluates and calibrates those decisions.

> **Status:** milestones 0–6 work: data, core, metrics, deciders, guaranteed thresholds,
> judge mode, the HTML report and the "bring your own predictions" CLI, including the CI
> `gate` and its decision changelog. The benchmark study (milestone 7) is running.

## What it will do

- **Compare deciders.** Decision models (e.g. the Gemini Decisions API), prompted
  LLMs, and a classic ML baseline, all on the same data.
- **Measure what matters.** These include accuracy, calibration (ECE, Brier,
  reliability diagrams), selective prediction (risk–coverage curves) and agreement
  with humans (Cohen's κ with confidence intervals). They also include cost and
  latency, each against simple baselines.
- **Recommend thresholds with guarantees.** It picks act / ask / escalate thresholds
  that come with finite-sample statistical guarantees, using Trust or Escalate and
  conformal risk control. It also says when no threshold can meet your target.
- **Gate CI.** It fails a build when a decider regresses, and keeps a changelog of
  every decision about decisions.

## Bring your own predictions

Give it a CSV, TSV or JSONL file with one row per decision: `id`, `prediction`,
`confidence` (the model's probability that its answer is right) and `human_label`.
Optionally add `options`, `split` and `group`.

New here? `ask-or-act demo` writes a synthetic example file (an overconfident
support-ticket router; no text, no download) and its HTML report into a new folder in
about a second, and prints what the other commands say about it.

```sh
ask-or-act demo                         # example data + report in ./ask-or-act-demo/
ask-or-act check preds.csv              # is the file usable, and what does it support?
ask-or-act thresholds preds.csv         # act above what confidence? (with a guarantee)
ask-or-act metrics preds.csv            # accuracy, calibration, ranking, with 95% CIs
ask-or-act compare a.csv b.csv          # is B better than A? paired tests, same ids
ask-or-act report preds.csv -o r.html   # one HTML page: summary, charts, thresholds
```

`check` reports every problem at once, with line numbers and the flag that fixes it.
For example, it suggests `--confidence-scale percent` for 0–100 scores, and
`--confidence-of positive` for P(positive class). Every command prints a short summary,
or one JSON document with `--json`. The same functions are a Python API
(`aoa.find_thresholds`, `aoa.evaluate`, `aoa.compare`, `aoa.write_report`).
[docs/cli-design.md](docs/cli-design.md) has the design.

**Runs locally.** The analysis commands (`check`, `thresholds`, `metrics`, `compare`,
`report`, `demo`, `gate`, `changelog`) make no network connections and send no telemetry, and the report page loads
nothing from elsewhere. A test enforces this
(`tests/test_local_only.py`). Only `run` calls model APIs, and only when you approve
the cost.

## Gate CI on your AI's decisions

`ask-or-act gate` compares a new predictions file with a baseline on the same ids. It
exits 1 only when the new file is **worse by more than a tolerance, beyond noise**:

- accuracy (Tango's score test);
- accuracy-controlled calibration (the ACE gap);
- coverage at your target risk;
- whether the guarantee still holds.

```sh
ask-or-act gate new.csv --baseline eval/baseline.csv --expect "accuracy +0 to +2 points"
ask-or-act changelog --last 5      # what changed, what you expected, what happened
```

- **Tolerances** default to 1 point of accuracy, 0.02 of calibration and 5 points of
  coverage. The policy lives in `ask-or-act.toml`, next to your code.
- **The false-FAIL rate is measured.** When nothing changed it is at most 5%, and 1.0%
  in a simulation with the defaults ([docs/gate-simulation.md](docs/gate-simulation.md)).
- **The decision changelog** records the expected effect *before* the run and checks it
  against what happened. Reruns don't get a second guess.
- **On GitHub,** a composite Action comments the result on the pull request:

```yaml
# REPLACE COMMIT_SHA: replace with the release commit SHA after v1.0
# (until then, the full SHA of a commit on main; never a branch or tag)
- uses: samadarsh/ask-or-act@COMMIT_SHA
  with:
    new: predictions.csv
```

[docs/gate.md](docs/gate.md) covers what the gate promises and how to set up CI. A full
workflow to copy is in [examples/github/gate.yml](examples/github/gate.yml).

See [docs/research.md](docs/research.md) for the papers this builds on and
[docs/datasets.md](docs/datasets.md) for the benchmark datasets.

## Roadmap

| # | Milestone |
|---|-----------|
| 0 | Setup and data |
| 1 | Core |
| 2 | Metrics |
| 3 | Deciders |
| 3b | Guaranteed thresholds |
| 4 | Judge mode |
| 5 | Reports and CLI |
| 6 | CI gate and changelog |
| 7 | Benchmark study |
| 8 | Release |

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12+ is fetched automatically.

```sh
uv sync                 # create .venv and install everything
uv run pytest           # tests
uv run ruff check       # lint
uv run ruff format      # format
uv run mypy             # type check (strict)
uv run ask-or-act --version

# Run a (free) fake decider over the Banking77 sample:
uv sync --all-extras
uv run ask-or-act run configs/fake-banking77.toml --dry-run   # plan + cost estimate only
uv run ask-or-act run configs/fake-banking77.toml             # saved under runs/
uv run ask-or-act metrics runs/<run-id>                       # metrics with 95% CIs
uv run ask-or-act metrics runs/<run-a> runs/<run-b>           # paired comparison

# Free baselines, then a real provider (keys in .env, never committed):
uv run ask-or-act run configs/baselines/classic-banking77.toml
uv run --env-file .env ask-or-act run configs/smoke/groq-gpt-oss-120b-banking77.toml --dry-run
uv run --env-file .env ask-or-act run configs/smoke/groq-gpt-oss-120b-banking77.toml --yes
```

Deciders, signals, prices and free-tier limits are described in
[docs/deciders.md](docs/deciders.md). On a free tier, a run that hits the daily limit
stops cleanly. Re-running it later resumes from the cache.

CI runs all of these on Python 3.12, 3.13 and 3.14.

## License

MIT. See [LICENSE](LICENSE).

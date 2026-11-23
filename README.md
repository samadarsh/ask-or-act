# ask-or-act

[![CI](https://github.com/samadarsh/ask-or-act/actions/workflows/ci.yml/badge.svg)](https://github.com/samadarsh/ask-or-act/actions/workflows/ci.yml)

**Tells you when an AI's decisions can be trusted, and when to escalate to a human.**

AI systems increasingly make small, consequential decisions: which model to route a
query to, whether a message is safe, which of two answers is better, whether an agent
should *act*, *ask* a clarifying question, or *escalate*. `ask-or-act` is an open-source
Python toolkit that evaluates and calibrates those decisions.

> **Status:** milestones 0–5 work: data, core, metrics, deciders, guaranteed thresholds,
> judge mode and the HTML report. The benchmark study is running. The "bring your own
> predictions" CLI is being built: `check`, `thresholds`, `metrics`, `compare`,
> `report` and `demo` work; the CI `gate` is next ([docs/cli-design.md](docs/cli-design.md)).

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
`report`, `demo`) make no network connections and send no telemetry, and the report page loads
nothing from elsewhere. A test enforces this
(`tests/test_local_only.py`). Only `run` calls model APIs, and only when you approve
the cost.

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

# ask-or-act

[![CI](https://github.com/samadarsh/ask-or-act/actions/workflows/ci.yml/badge.svg)](https://github.com/samadarsh/ask-or-act/actions/workflows/ci.yml)

**Tells you when an AI's decisions can be trusted, and when to escalate to a human.**

AI systems increasingly make small, consequential decisions: which model to route a
query to, whether a message is safe, which of two answers is better, whether an agent
should *act*, *ask* a clarifying question, or *escalate*. `ask-or-act` is an open-source
Python toolkit that evaluates and calibrates those decisions.

> **Status:** milestone 0 (project setup). Nothing below the roadmap works yet.

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
```

CI runs all of these on Python 3.12, 3.13 and 3.14.

## License

MIT. See [LICENSE](LICENSE).

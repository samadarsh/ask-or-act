# ask-or-act

Open-source Python toolkit that tells you when an AI's decisions can be trusted and when
to escalate to a human. It evaluates and calibrates AI decisions such as routers, safety
gates, LLM judges and "act / ask / escalate" steps. It compares decision models (e.g. the
Gemini Decisions API), prompted LLMs and a classic ML baseline. The comparison covers:

- accuracy
- calibration: ECE, Brier, reliability diagrams
- selective prediction: risk–coverage
- agreement with humans: Cohen's κ with CIs
- cost and latency

It then recommends act / ask / escalate thresholds with statistical guarantees.

- Package `ask_or_act` (in `src/ask_or_act/`), CLI command `ask-or-act`.
- Portfolio project. The quality bar matches the author's earlier redis-go (Go) and clox
  (C) projects: tests from day one, CI, a clear README.

## Rules (always follow)

1. **Explain every change in simple terms.** The author is strong in Go and C and newer
   to Python ML tooling. Say what changed and why. Where it helps, compare it to Go or C:
   uv ≈ `go mod` + toolchain manager, mypy ≈ the compiler's type checker, pytest
   fixtures ≈ test setup helpers.
2. **Never commit or push without explicit approval.** Show the diff summary and ask.
3. **Never call a paid API without first stating the expected cost.** Give tokens ×
   price × number of calls, and wait for a go-ahead. This includes "just one test call".
   Tests must never hit the network; use recorded fixtures or fakes.
4. **Before implementing any statistical guarantee,** read the original paper and cite
   the exact theorem/proposition (paper, arXiv ID, version, theorem number) in a comment
   or docstring next to the code that relies on it. See `docs/research.md`.
5. Don't download large datasets into the repo. Data goes in `data/raw/` (gitignored).

## Commands

```sh
uv sync                     # install (creates .venv)
uv run pytest               # tests
uv run ruff check           # lint      (uv run ruff check --fix to autofix)
uv run ruff format          # format
uv run mypy                 # strict type check of src/ and tests/
```

CI (`.github/workflows/ci.yml`) runs lint, format check, mypy and pytest on 3.12–3.14.
All four must pass before a change is done.

## Roadmap

0. Setup and data
1. Core: decision records, decider interface, dataset/split abstractions
2. Metrics
3. Deciders: decision models, prompted LLMs, classic ML baseline
3b. Guaranteed thresholds
4. Judge mode: LLM-as-a-judge evaluation against human labels
5. Reports and CLI
6. CI gate and changelog
7. Benchmark study
8. Release

## Design notes for later milestones

**Metrics (M2)**
- Always report on a held-out test split. Fit thresholds and calibrators only on
  calibration/dev data.
- Every metric gets bootstrap confidence intervals.
- Comparing two deciders uses a paired significance test on the same examples. Use
  McNemar or paired bootstrap for accuracy, and paired bootstrap for ECE, Brier and κ.
- The goal is to avoid curve-fitting to the test set.

- Measurement pitfalls (arXiv 2608.04899):
  - Compute every risk–coverage / accuracy–rejection area with **stepwise**
    interpolation, never trapezoidal.
  - Handle tied confidences as blocks, never in example order.
  - Report the number of unique confidence values per decider, because verbalized
    confidence is sparse.

**Confidence signals (M3)**
- Signals:
  - verbalized confidence
  - token logprobs
  - self-consistency (agreement over k samples)
  - native probabilities from decision models
  - the hybrid "verbalization logprobs" (arXiv 2608.04899 §4). This is the expected
    value of the 0–100 confidence under the tens-digit and units-digit token
    distributions. It needs logprobs and one call.
- Compare calibration *with accuracy controlled* (arXiv 2606.30814). Raw ECE is
  confounded by accuracy.
- Multiple-choice decision models (e.g. Decisions API `choice`) always spread
  probability over the listed options. Where it makes sense, add a **"none of these /
  other"** option so the model can express "no option fits".
  - Example: Banking77 routing. Probability mass on the extra option is an escalate
    signal.
  - Name the option key carefully. The Decisions API sorts keys alphabetically and warns
    against positional names like `none_of_the_above`.
  - Report results with and without the extra option.

**Data handling (all milestones)**
- Banking77: the classic baseline trains on the official train split only. The 1,540
  evaluation examples come from the test split only.
- MT-Bench: keep every human vote. Comparisons with several voters are used to measure
  human–human agreement, which is the ceiling for judges.
- Aegis 2.0: calibration and test use human-labelled rows only. LLM-jury labels may be
  used only to train the classic baseline.
- **Never show raw unsafe text** (Aegis or any safety data) in reports, the README,
  demos, logs or test fixtures. Use IDs or redacted snippets only.

**Baselines (M3, M7)**
- Always compare against simple baselines: always-cheap, always-strong, and random at
  equal cost.

**Guaranteed thresholds (M3b)**
- Trust or Escalate (arXiv 2407.18370): fixed-sequence testing and cascaded selective
  evaluation.
- Conformal risk control.
- A minimum-abstention feasibility check (Kotte, arXiv 2606.29054). It tells the user up
  front when the target risk is unreachable without abstaining at least a certain
  fraction of the time.
- A distribution-shift warning. The guarantees assume calibration data and test data are
  exchangeable, so warn when that looks violated.

**CI gate and decision changelog (M6)**
- Inspired by Self-Improvement Memory (arXiv 2611.18856). Every run appends an entry
  with:
  - the root cause it addresses
  - what changed and why
  - the expected effect, recorded BEFORE the run
  - the observed metrics
  - context: model, prompt, dataset
- Predicted versus observed is the point. Never backfill the prediction.

**Benchmark study (M7)**
- Cross-model transfer experiment: tune thresholds on model A, apply them to model B.
- Multiple runs (seeds or resamples) with paired tests and effect sizes.
- Cost reporting.
- Multiple-comparison correction (Holm or Benjamini–Hochberg).
- Datasets: see `docs/datasets.md`.

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
uv sync --all-extras        # + optional `data` deps (huggingface-hub, pyarrow)
uv run ask-or-act run configs/fake-banking77.toml --dry-run   # plan + cost, no calls
uv run ask-or-act run configs/fake-banking77.toml             # run, cache, save to runs/
uv run ask-or-act metrics runs/<id>                # metrics table + runs/<id>/metrics.json
uv run ask-or-act metrics runs/<a> runs/<b>        # paired comparison (+ ACE views)
uv run ask-or-act audit-redaction [paths]          # fail if Aegis prompt text is on disk
uv run ask-or-act study status configs/study/schedule.toml   # study progress; no calls
scripts/daily.sh [schedule]                        # today's study runs (asks first)
uv run ask-or-act thresholds --cal runs/<cal> --test runs/<test>   # M3b thresholds
```

Metrics (milestone 2): `metrics/` holds pure functions over saved runs. A run stores
`decisions.jsonl` plus `examples.jsonl` (gold labels and metadata, never input text).

- **Errors** count as wrong, with confidence 0.
- **CIs:** seeded bootstrap with 2,000 resamples. MT-Bench is resampled by question.
- **Paired tests:** exact McNemar for accuracy, a centered paired bootstrap for the
  other metrics, and Holm correction across the metric family.
- **Paper-backed pieces:** ACE (arXiv 2606.30814 v2) and the Kotte v2 Proposition 3
  feasibility floor.
- **Known behaviour:** ECE is biased upward on finite samples, about 0.03 for a
  perfectly calibrated decider with 770 examples. Percentile CIs for ECE are therefore
  skewed. Compare deciders paired on the same sample; don't read single ECE values as
  absolute.

Architecture (milestone 1):

- `core/types.py`: Example, Decision, DeciderConfig.
- `core/decider.py`: the Decider Protocol.
- `core/runner.py`: cache, then budget, then concurrency, rate limits, retries and
  timeouts.
- `core/cache.py`: SQLite cache keyed by sha256 of (config, model, prompt version,
  input).
- `core/runs.py`: `runs/<id>/run.json` and `runs/<id>/decisions.jsonl`.
- `deciders/`: registry plus the fake deciders.
- `run.py`: the `run` command.

A new provider is one class that satisfies `Decider`, plus one line in
`deciders/__init__.py`.

Deciders (milestone 3): see `docs/deciders.md`.

- **API keys:** read only from the environment. Run paid configs with
  `uv run --env-file .env ask-or-act run <config> --dry-run` first, then `--yes`.
- **Available keys (2026-11-20):** only `GROQ_API_KEY`, on the free tier. Never call the
  Gemini or OpenAI adapters unless the user adds those keys. Keep them tested with
  fixtures.
- **Free-tier calls still need approval.** State the requests and tokens against the
  daily limits (dry run) and wait for a go-ahead, as for paid calls. `--yes` is required
  whenever the list price is above $0.
- **Groq:**
  - Daily limits stop a run cleanly: exit code 5, status `quota_stopped`. Re-running the
    same config later resumes from the cache.
  - Groq has no logprobs, so token_logprob and verbalized_logprob are unsupported there.
  - `cost_usd` is the published paid price; `charged_usd` is 0 on the free tier.
- **Never print, log or echo key values.** To check that a key is present, test the
  variable name only.
- **New model:** add a dated, cited price entry to `deciders/pricing.py`.
- **Prompt edits:** bump `PROMPT_VERSION` in `deciders/prompts.py`. Bump `prompt_version` whenever a prompt changes, or the cache
will serve stale answers.

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

- **Self-consistency and the cache:** the cache key must include the sample index
  (and the temperature, which is already in `params`). Otherwise k samples of the same
  input hit the same key and come back identical. Model it as k distinct cached calls,
  keyed by `sample_index` = 0..k-1.
- **`--refresh` flag** for `ask-or-act run`: bypass cache reads, still write fresh
  results. This is needed for model-drift checks and for the cross-model study, where
  "same config, new answer" is the whole point.
- **Cost accounting:** every Decision has `cost_usd` (the true cost, kept on cache
  hits) and `charged_usd` (what this run paid: 0 when cached, the estimate share when a
  call failed). The study reports cost per decision from `cost_usd`. Budgets are
  enforced on what's charged.

**Data handling (all milestones)**
- Banking77: the classic baseline trains on the official train split only. The 1,540
  evaluation examples come from the test split only.
- MT-Bench: keep every human vote. Comparisons with several voters are used to measure
  human–human agreement, which is the ceiling for judges.
- Aegis 2.0: calibration and test use human-labelled rows only. LLM-jury labels may be
  used only to train the classic baseline.
- **Never show raw unsafe text** (Aegis or any safety data) in reports, the README,
  demos, logs or test fixtures. Use IDs or redacted snippets only.
- **Model output can restate unsafe input.** Reasoning, Groq's `failed_generation` and
  validation messages are never stored or printed for `SENSITIVE_TASKS` (only lengths
  and error codes). Check new error paths for this.

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

- **Kotte finite-sample correction: verified (2026-11-20)** against the full v2 PDF,
  Appendix E.4. `min_abstention_conservative` matches it exactly; the floor holds at
  1 − δ/2 on its own (E.4 splits δ between the floor and the paired risk guarantee).
- **Fixed-sequence thresholds (implemented, `guarantees/`):** ToE v1 §2.1 Eq. (4)–(5)
  and Theorem 1, with validity of the selected λ̂ from Kotte v2 Lemma E.1. The grid and
  scan starts are pre-registered from the signal's scale, never from data. Several
  starts split δ by a union bound.
- **Pre-registered and FROZEN (2026-11-21):** the grid, scan starts, δ split and the
  MT-Bench unit/tie rule (docs/guarantees.md). They were chosen on the free baselines and
  fakes only. Never change them after seeing LLM results.
- **MT-Bench target pre-registered (2026-11-22): α = 0.25, δ = 0.1**, in each MT-Bench
  config's `[guarantee]` table. `--alpha-vs-human` is a sensitivity analysis only.

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

# Guaranteed thresholds (milestone 3b): design

**Status:** design approved 2026-11-21. Code in `src/ask_or_act/guarantees/`, tests in
`tests/test_guarantees.py`.

## Pre-registration (frozen 2026-11-21, before any LLM results)

These settings were chosen using **only the free baselines and fake deciders**, before
any LLM study data existed. They **must not change after seeing LLM results**; a
change would mean a new, separately registered analysis. `tests/test_guarantees.py`
(`test_preregistered_starts_and_delta_split_are_frozen`) pins them.

| setting | frozen value |
|---|---|
| grid | from the run config's signal: verbalized → 1.00, 0.99, …, 0.00; self-consistency with k samples → k/k, …, 0/k; anything else → 0.001 steps |
| scan starts | verbalized: 1.0, 0.99, 0.95, 0.9 · self-consistency: 1.0 · continuous: 0.999, 0.99, 0.95, 0.9 |
| δ split | equal over the scan starts (δ/S), and in a cascade first equal over the stages (δ/\|M\|, ToE A.2), so each scan runs at δ/(\|M\|·S) |
| test | exact binomial, ToE Eq. (4); stop at the first failure |
| MT-Bench unit | one example per comparison with the majority human label (below) |
| MT-Bench ties | no strict plurality → comparison excluded and counted |
| MT-Bench target | **α = 0.25, δ = 0.1**, registered 2026-11-22 (see below) |
| Banking77 target | **α = 0.10, δ = 0.10**, registered 2026-11-22 (wrong intent on routed queries) |
| Aegis target | **α = 0.10, δ = 0.10**, registered 2026-11-22 (wrong safe/unsafe label on acted-on prompts) |

**Data that existed when the targets were registered (2026-11-22).** No pilot or study
runs. The 2026-11-20 smoke runs did exist: 20 Banking77 test examples on gpt-oss-120b
and qwen3.8-27b, and 20 Aegis test examples on gpt-oss-safeguard-20b; their accuracy and
ECE had been seen. α = 0.10 was the CLI default before those runs and was not tuned to
them. Every Banking77, Aegis and MT-Bench config in `configs/study/`, `configs/pilot/`
and `configs/baselines/` carries its `[guarantee]` table. The smoke and fake configs
don't: they are not part of the study.

A consequence we accept: a decider whose confidence never reaches 0.9 (the classic
MT-Bench baseline tops out at 0.71) can never be certified.

## What the user gets

```sh
uv run ask-or-act thresholds --cal runs/<cal-half> --test runs/<test-half> --alpha 0.1 --delta 0.1
uv run ask-or-act thresholds --cal runs/<cheap-cal> runs/<strong-cal> --test runs/<cheap-test> runs/<strong-test>
```

- **One decider:** act when confidence ≥ λ̂, otherwise escalate.
- **Two or more deciders (cheap → strong):**
  - *act*: the cheap model answers;
  - *ask*: hand the case to the stronger model;
  - *escalate*: send it to a human.
- **The guarantee:** with probability ≥ 1 − δ over the calibration half, the error rate
  on acted-on examples is ≤ α, for new data from the same distribution.
- **Pre-registered MT-Bench target: α = 0.25, δ = 0.1** (fixed 2026-11-22, before any
  LLM data on MT-Bench existed). It is roughly the human ceiling: on the calibration
  half a typical human agrees with the other voters' majority h = 74.6% of the time
  (1 − h = 0.254). A fixed α = 0.1 is infeasible for everyone on MT-Bench.
  - It is recorded in every MT-Bench run config as a `[guarantee]` table (study,
    pilot and the classic baselines). `ask-or-act thresholds` uses it as the fixed
    target when `--alpha` is not given. Any other `--alpha` is printed as "NOT the
    pre-registered target; exploratory".
  - **Sensitivity only:** `--alpha-vs-human [MARGIN]` (α = 1 − h + MARGIN, h on the
    calibration votes) is always reported for MT-Bench, below the main result. It is
    never the headline: h comes from the same calibration labels the test uses.
  - Do not change α after seeing LLM results.

## Order of operations

1. **Feasibility** (Kotte v2 Prop. 3): base risk μ̂ on the calibration half. If μ̂ > α,
   say up front that any valid rule must escalate ≥ (μ̂ − α)/(1 − α).
   - **Separate statements, each with its own δ.** The floor (`--delta-floor`, default
     `--delta`) holds with probability ≥ 1 − δ_floor/2 (App. E.4's Hoeffding step).
     The threshold holds with probability ≥ 1 − δ.
   - A **joint** statement (both at once) needs the failure probabilities summed:
     ≥ 1 − (δ_floor/2 + δ). The report prints all three numbers.
2. **Threshold:** a fixed-sequence test on the calibration half only. A run from the test
   half is refused as `--cal`.
3. **Check on the test half**, which never influenced λ̂: coverage and observed risk
   with a 95% Clopper–Pearson interval.

## The test (ToE v1 §2.1, Theorem 1; validity of λ̂ from Kotte v2 Lemma E.1)

- **Per threshold λ:** an exact binomial test, P(Bin(n(λ), α) ≤ errors(λ)) ≤ δ. This is
  the same as ToE Eq. (4), R̂⁺(λ) ≤ α.
- **Scan:** a fixed grid from the top down. Stop at the first failure; λ̂ is the smallest
  certified threshold (Eq. (5)).
- **Why it is valid:** the certified thresholds are a prefix of a fixed order. Certifying
  any true null means certifying the first null in that order, which happens with
  probability ≤ δ. No monotonicity is needed, and no correction for the number of
  thresholds.

## Decisions taken, and why

| # | decision | alternative | why |
|---|---|---|---|
| 1 | **Grid fixed from the signal's scale** (verbalized: 1.00…0.00 in steps of 0.01; self-consistency with k samples: k+1 steps; else 0.001 steps), chosen from the run **config** | Kotte: candidates restricted to n(λ) ≥ 20 | Lemma E.1 needs a fixed order. A data-dependent filter only controls the risk conditional on the calibration scores (Kotte conditions on them), not the population risk. |
| 2 | **Several pre-registered scan starts, δ split by union bound** (verbalized and continuous: 4 starts; self-consistency: 1) | a single start at the top, as in ToE | With one start, an empty or tiny emit set at the top fails the very first test. The classic baselines' probabilities never reach 1.000, so all three certified **nothing**. With 4 starts: Banking77 acts on 94% of the test half (risk 0.094), Aegis on 40% (risk 0.090), both ≤ α = 0.1. |
| 3 | **Exact binomial** test | Hoeffding / Bernstein / betting (Kotte) | For 0/1 loss the exact test is the tightest valid one. Kotte's alternatives matter for continuous losses. |
| 4 | **Ties as blocks** (emit set = conf ≥ λ) | — | arXiv 2608.04899 §3: verbalized confidence is sparse. |
| 5 | **Cascade calibrated as in ToE Algorithm 2** (stage i on examples earlier stages abstain on, at δ/\|M\|), with a `--disjoint` option | — | A.2 treats stage i's data as independent of earlier thresholds. It isn't quite: the earlier scan's stopping test looked at some of those labels. `--disjoint` gives each stage its own seeded share, which is rigorous but uses less data per stage. |
| 6 | **Failed calls** count as wrong with confidence 0 | dropping them | Same policy as the metrics. |

## Evidence so far (no API calls)

**Monte Carlo** (2,000 calibration samples per row). It counts how often the *true* risk
at λ̂ exceeded α = 0.1 when δ = 0.1:

| scenario | n = 150 | n = 300 | n = 600 | certified something (n = 300) |
|---|---|---|---|---|
| calibrated decider | 1.2% | 0.7% | 0.9% | 100% |
| overconfident top (non-monotone risk) | 0.2% | 0.0% | 0.0% | 8% |
| true risk just above α below the top | 1.6% | 1.3% | 1.7% | 7% |

The guarantee holds with a wide margin, so the procedure is conservative. That comes
from three things: the exact test on discrete data, the δ/4 split, and early stopping.
When the top of the scale is overconfident, it mostly declines to certify, which is the
correct behaviour.

**Free baselines (calibration → test), α = 0.1 unless stated, δ = 0.1:**

| dataset | feasibility floor | λ̂ | test coverage | test risk [95% CI] |
|---|---|---|---|---|
| Banking77 | 1.2% | 0.426 | 94.4% | 0.094 [0.073, 0.117] |
| Aegis | 9.6% | 0.875 | 40.0% | 0.090 [0.054, 0.139] |
| MT-Bench, per comparison, α = 1 − h = 0.254 | 31.9% | none | 0% | — (base risk 0.49; confidence never above 0.71) |
| MT-Bench, per vote (sensitivity), α = 0.254 | 34.3% | none | 0% | — |

## MT-Bench: one example per comparison (pre-registered)

Votes on one comparison aren't independent, so following ToE footnote 1 each
comparison is one example with the majority human label (`guarantees/mt_bench.py`):

- **Unit:** the comparison (question, turn, unordered model pair). Labels are mapped
  from display order to model names, so votes cast on either display order count
  together.
- **Gold:** the plurality of all its votes. **If the top count is shared** (e.g. 1–1
  or 1–1–1), the comparison is **excluded** and counted. A split between humans is not
  a vote for "tie".
- **The decider's answer:** its decision on the display order with the most votes
  (ties: the order whose model_a name sorts first).
- **Sensitivity check:** `ask-or-act thresholds` on MT-Bench always also prints the
  vote-level analysis (every vote an example). That guarantee is optimistic.
- Calibration half: 500 votes → 274 comparisons, 20 without a strict majority.

Comparisons within one question are still related (40 questions per half); the report
keeps a warning about it.

## Next pieces (not started)

- **Conformal risk control:** needs reading Angelopoulos et al., arXiv:2208.02814,
  first.
- **Distribution-shift warning:** compare calibration and test confidence
  distributions, which needs no labels.
- **Cost per tier,** from `cost_usd`.
- **A class-conditional target for Aegis (asymmetric harm).** Missing an unsafe
  request (acting on "safe" when the human label is unsafe) is far worse than a false
  alarm, and an overall α can hide a high miss rate. Planned: a second guarantee on
  P(answer = safe | label = unsafe, acted on) ≤ α_miss, with its own pre-registered
  α_miss and δ, fixed before any study data is analysed. Like a per-class
  (Mondrian) risk, it is tested only on the unsafe calibration examples. It needs its
  own literature check before implementation (rule 4).

# Milestone 7 analysis plan (pre-registered)

**Status: DRAFT, not yet frozen.** It's frozen by the commit that changes this line to
"FROZEN" and is dated by that commit's UTC time. After the freeze, any change is a dated
amendment at the end of this file, and every result produced under an amendment says so.

**What it is:** the full list of analyses for the benchmark study (docs/study.md),
fixed before the final analysis runs. Everything listed here is reported whatever the
outcome. Anything not listed is exploratory and labelled that way.

## What was seen before this plan

From the study runs (`configs/study/`), up to the freeze:

- **Progress and invalid-output rates**, from `study status` and the daily summaries:
  gpt-oss-safeguard-20b 0.1%, 0.0%, 1.1% and 1.0% per config.
- **Quick-check accuracies.** At the end of every config it runs, `ask-or-act run`
  prints a quick-check accuracy. Completed runs get it on all rows; partial runs (daily
  limit) get it on the decided rows only.
  - This happened on calibration and test halves alike.
  - The author saw these lines, and some were pasted into the chat. Examples:
    gpt-oss-120b and gpt-oss-20b at 0.870 on Banking77 calibration-sc, and qwen3.8-27b
    at 0.769 on Banking77 test.
  - So **accuracies of study runs, including test halves, were seen.**
- **One `metrics` call on a test half.** On 2026-11-23 at 09:22 UTC, the completed-run
  guard was checked by running `ask-or-act metrics` on
  `20261123T080137Z-study-qwen3.8-27b-banking77-test-ee0078`, piped through `head -4`.
  It printed n = 308, 0 errors, 8 unique confidences, and the same accuracy (0.769
  [0.721, 0.821]).
- **Response metadata, for S6 only.** On 2026-11-28, while S6 was being drafted, the
  cache was checked for a model fingerprint. Only these were read: the field names,
  and the `system_fingerprint` values counted per model and UTC day. No answers or
  confidences were read.
- **No calibration, threshold, coverage or comparison result has been seen** for any
  study model: no ECE, Brier, ACE gap, λ̂, coverage, risk at a threshold, or paired test.
- **The pilot and smoke runs** (20 test examples per dataset, docs/study.md): accuracy
  and ECE for those 20. That's why the seen-id sensitivity check (S1) exists.

No analysis code has been run on any study calibration half.

## Data

| dataset | deciders (verbalized confidence) | test half | self-consistency subset (k = 3) | target (pre-registered) |
|---|---|---|---|---|
| Banking77 | gpt-oss-120b, qwen3.8-27b, gpt-oss-20b | 308 (4 per intent) | 77 | α = 0.10, δ = 0.10 † |
| Aegis 2.0 | the three above + gpt-oss-safeguard-20b | 250 | 100 | α = 0.10, δ = 0.10 |
| MT-Bench | gpt-oss-120b, qwen3.8-27b | 500 votes (≈ 270 comparisons) | ≥ 50 votes | α = 0.25, δ = 0.10 |

- **Baselines on every dataset:**
  - classic: TF-IDF + logistic regression (`tfidf-logreg-v1`);
  - majority class.

  They are paired with the LLMs on the same test ids.
- **The calibration halves are the same size and never touch the test half.**
- **† Banking77:** its target was registered 6 min 49 s after the first Banking77 pilot
  calls (docs/guarantees.md). Every Banking77 headline carries that note.
- **Frozen before this plan, and not changed by it:**
  - the guarantee settings: grid, scan starts, δ split, the exact binomial test;
  - the MT-Bench unit and tie rule;
  - the three targets;
  - the metric definitions (M2: 2,000 bootstrap resamples with a fixed seed, failed
    calls count as wrong with confidence 0, and MT-Bench resampled by question).

### Units

- **MT-Bench, primary:** one example per comparison, labelled by the human majority. A
  comparison without a strict majority is excluded and counted, as in the
  pre-registered guarantee unit.
  - All MT-Bench analyses (Q1–Q4) use this unit, with a bootstrap clustered by
    question.
  - Vote level is sensitivity check S2.
- **Banking77 and Aegis:** one example per row.

## Primary questions

### Q1. Which decider is best calibrated?

- **Measure: the ACE distribution-aligned (DA) ECE gap** (arXiv 2606.30814 v2, §4.2, the
  paper's default). It is computed on the same test ids for each pair. Raw ECE isn't
  used for ranking, because it's confounded by accuracy and biased upward at these
  sizes (CLAUDE.md, M2).
- **Test:** a two-sided centered paired bootstrap (2,000 resamples) on the DA gap.
- **Family F1, 19 tests:** every pair among the LLMs (verbalized) and the classic
  baseline, within each dataset:
  - Banking77: 6 pairs;
  - Aegis: 10 pairs;
  - MT-Bench: 3 pairs.

  The majority baseline is excluded: its confidence is constant, so its ECE only
  measures its accuracy.
- **Reported for each pair:**
  - the raw ECE gap;
  - the DA gap with its 95% CI and Holm-adjusted p;
  - the instance-aligned (IA) gap as a corroborating check;
  - reversal flags (raw and DA with opposite signs);
  - DA's ESS/N, flagged below 0.5.
- **Reported for each decider:** ECE (width and mass bins), Brier, AURC, the number of
  unique confidence values, and the ECE noise floor.
- **Claims about one decider on its own are descriptive, not tests.** That includes
  "overconfident", "underconfident" and "ECE above its noise floor". They are reported
  with their CIs and noise floors, outside every family, and never as a finding of
  significance. Only the paired DA tests in F1 and F1b rank deciders.
- **The answer:** a ranking is stated only where the Holm-adjusted DA tests separate
  deciders. Otherwise the result is "not separated at this sample size", with the CIs.
- **Q1b (secondary), family F1b, 9 tests: does self-consistency calibrate better than
  verbalized confidence for the same model?** The test is the same DA paired bootstrap
  on the self-consistency subset:
  - Banking77: 3 models;
  - Aegis: 4 models;
  - MT-Bench: 2 models.

### Q2. Do the pre-registered guarantees hold on the test halves?

- **Configurations:** every (dataset, decider, signal) whose calibration half certifies a
  threshold λ̂ at its pre-registered target, plus the cascades in Q3. That includes the
  verbalized and self-consistency signals and the classic baseline.
- **What "holds" can mean here:** the guarantee is over draws of the calibration half
  (risk ≤ α with probability ≥ 1 − δ). One test half can't confirm it. It can show
  evidence that it **failed**.
- **Test:** for each configuration, the acted-on test examples and their errors. Then a
  one-sided exact binomial test of H₀: risk ≤ α, with p = P(Bin(n_act, α) ≥ errors).
  - **Family F2:** all certified configurations, Holm-adjusted at 0.05.
  - A rejection means the guarantee failed on this data: a shift between the halves, a
    δ-event, or a bug. Each one is investigated and reported.
- **Also reported, for every configuration (certified or not):**
  - test risk with a 95% Clopper–Pearson CI;
  - the number of configurations whose point estimate is above α. Those are expected
    even when the guarantee holds; it's reported against that expectation, not as a
    failure;
  - the Kotte v2 Prop. 3 abstention floor from the calibration half, next to the
    observed test abstention.
- **"Not certified" is a result.** It is reported as coverage 0, with the floor that
  explains it where one applies.

### Q3. How much can be automated at each target?

- **Coverage:** the share acted on, on the test half, at λ̂, with a 95% Clopper–Pearson
  CI, at the pre-registered target, for every configuration in Q2.
- **Pre-specified cascades** (cheap → strong; `ask` = hand to the stronger model,
  `escalate` = a human), calibrated as ToE Algorithm 2:
  1. Banking77: gpt-oss-20b → gpt-oss-120b;
  2. Banking77: classic → gpt-oss-120b;
  3. Aegis: gpt-oss-safeguard-20b → gpt-oss-120b;
  4. Aegis: gpt-oss-20b → gpt-oss-120b;
  5. Aegis: classic → gpt-oss-120b.

  For each one: the act, ask and escalate shares, the risk on acted-on examples, and
  the cost per decision from `cost_usd` (the list price; the study itself paid $0).
- **The cascades are compared with these baselines:**
  - always the cheap model;
  - always the strong model;
  - random routing at equal cost: the same share sent to the strong model, chosen at
    random, with 2,000 seeded draws.

  These baselines are descriptive, not tests.
- **Exploratory, labelled "not pre-registered":** coverage over
  α ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}, and coverage differences between deciders,
  with bootstrap CIs that refit both thresholds (approximate, as in the gate).

### Q4. Do thresholds transfer across models?

- **Design:** λ̂ is fitted on model A's calibration half, at the dataset's target, and
  applied unchanged to model B's test half. All ordered pairs of LLMs on the
  verbalized signal (its grid is the same for every model):
  - Banking77: 6;
  - Aegis: 12;
  - MT-Bench: 2.
- **No theorem covers this.** The guarantee assumes the same decider. That's why it's
  tested, not assumed.
- **Test, family F4 (up to 20 tests):** the one-sided exact binomial test of Q2, on B's
  acted-on test examples at λ̂_A, Holm-adjusted.
- **Pairs drop out of F4 when model A certifies nothing** on its calibration half: there
  is no threshold to transfer. They are listed by name, and F4's size is reported after
  the drops.
- **Reported for each pair, against B's own λ̂_B on the same test examples:**
  - the difference in coverage and in risk;
  - a paired bootstrap over B's test examples, with both thresholds held fixed. So it
    leaves out calibration-half variability, and is labelled that way.
- **Exploratory:** transfer on the self-consistency signal (k = 3 grid), and from the
  classic baseline's continuous scale (not comparable, so not included in F4).

## Multiple comparisons

- **One family per question:**
  - F1: 19 tests;
  - F1b: 9 tests;
  - F2: one per certified configuration;
  - F4: up to 20 tests (pairs whose model A certifies nothing drop out).

  Holm controls the family-wise error at 0.05 within each.
- **The families aren't corrected against each other.** Each answers its own question
  and is reported with its own family. This is stated next to the results.
- **Family size can shrink, but only for two reasons:** a model stopped by the 2%
  invalid-output rule, or a configuration that certifies nothing (in F2 and F4). Every
  dropped test is listed with its reason. Results are never selected.
- **Everything outside F1, F1b, F2 and F4 is exploratory:** the per-pair M2 comparison
  tables, judge-mode statistics, the α grid and the exploratory transfers. They show CIs
  and unadjusted p-values and are labelled exploratory.

## Sensitivity analyses

Each one is reported next to the primary result. If it would change a conclusion, the
headline says so.

| # | check | applies to |
|---|---|---|
| S1 | **Without the pilot-seen examples** (`configs/pilot/seen_test_ids.json`: 20 per dataset; MT-Bench drops the 20 comparisons that contain a seen vote, 48 test votes) | all test-half results, Q1–Q4 |
| S2 | **MT-Bench at vote level** (every vote an example; the guarantee is optimistic there) | Q1–Q4 on MT-Bench |
| S3 | **MT-Bench α against the human ceiling** (`--alpha-vs-human`, α = 1 − h on the calibration votes) | Q2, Q3 on MT-Bench |
| S4 | **Disjoint cascade calibration** (`--disjoint`: each stage gets its own seeded share) | the Q3 cascades |
| S5 | **Failed calls excluded** instead of counted as wrong | Q1–Q3 |
| S6 | **Model drift** (below) | Q1–Q4 |

**S6, model drift.** The halves were run on different days, and Groq may change what
serves a model name in between.

- **Dates:** for each calibration and test run, the UTC dates of its decisions, taken
  from each response's `created` time, not the run id. A run resumed from the cache has
  calls from several days.
- **Fingerprints:** every saved Groq response has a `system_fingerprint`. A model has
  a pool of them, used in parallel: about 40 for each gpt-oss model, and 4 for
  qwen3.8-27b on 2026-11-20 to 11-25. So a fingerprint seems to mark a serving
  configuration rather than a model version. Groq doesn't document its meaning.
- **What counts as a change:** a fingerprint that first appears after the model's
  earliest calibration call on that dataset, used by more than 5% of the calls of any
  of that model's study runs on that dataset.
  - **When and how the 5% was chosen:** on 2026-11-28 (UTC), after seeing the
    fingerprint counts, from metadata only, with no outcomes. qwen3.8-27b's stray new
    fingerprints had 1 call (11-24) and 4 calls (11-25). The two new ones on 11-26 and
    11-27 had about 18% and 28% of that day's calls. The threshold separates those two
    cases.
- **If there is a change:** that model's results on that dataset are also reported
  split by period, before and after the day the new fingerprint first appeared. Q2 and
  Q4 are also reported with the calibration and test calls restricted to fingerprints
  seen in both halves. The primary result stays as is, and the split is reported next
  to it.
- **Limitation:** an unchanged fingerprint pool doesn't prove an unchanged model,
  because weights or serving could change without a new fingerprint. This is stated
  with the results.

The Banking77 registration timing gets no re-analysis. It is disclosed with every
Banking77 headline.

## Power, so nulls can be read

These are rough, and are meant only for reading a null result:

- **Q1 and Q1b:** a paired accuracy difference of about 7 points (n = 308, about 20%
  discordant pairs) has 80% power. Calibration gaps below about 0.02 are unlikely to be
  separated at these sizes.
- **Q2 and Q4:** with about 200 acted-on test examples at α = 0.10, a true risk of 0.15
  is detected about 70% of the time before Holm. Smaller violations are mostly missed,
  and a non-rejection says only that.
- **Self-consistency subsets are small:**
  - Banking77: 77 per half;
  - Aegis: 100;
  - MT-Bench: about 50 votes, so even fewer comparisons.

  Their grid has only k + 1 = 4 steps, and they get one scan start. So certification
  there may be nearly impossible. At α = 0.10, even an error-free top step needs about
  22 acted-on examples to pass the exact test at δ = 0.1.
  - A null in Q1b, or "not certified" in Q2 for self-consistency, mostly reflects the
    sample size. It is reported that way, not as evidence against the signal.

## How results are reported

- **Code:** one script, `scripts/analyze_study.py`.
  - It is written after this freeze. It's tested on the fakes and the free baselines
    before it ever reads a study run.
  - Its seeds are fixed.
  - It writes `docs/results.md` and the HTML report.
  - It runs once, after every study run is `completed`. Analysis refuses any other
    status.
- **Order:**
  1. Freeze this plan.
  2. Write and test the script.
  3. Run it on the study.
  4. Commit the outputs unedited. Interpretation goes in a separate, later commit.
- **Every pre-specified result is reported, in the order of this plan**, including
  nulls, configurations that certify nothing, and guarantees that fail.
- **Null results** say "not separated" or "no evidence of a violation", with the CI and
  the power note above. They never say "equal" or "the guarantee holds".
  - "Practically equivalent" is used only when the whole 95% CI is inside these
    margins: ECE gap ±0.02, accuracy ±3 points, coverage ±10 points.
- **Every number has its n, its CI and its unit** (rows, comparisons or votes).
- **Deviations from this plan** are listed in a section of their own, with date and
  reason.
- **No raw Aegis text anywhere**: ids and aggregates only. Run `audit-redaction` before
  any commit.

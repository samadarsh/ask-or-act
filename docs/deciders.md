# Deciders

Status as of 2026-11-20. Provider facts come from the official docs: Groq's
console.groq.com/docs (models, rate-limits, structured-outputs, reasoning,
api-reference), plus the OpenAI and Gemini docs and SDK types (`openai` 3.55.0,
`google-genai` 3.5.0). The "live-verified" column says whether a real call has confirmed
them yet. The 20-example smoke tests fill those cells in.

**Keys available:** only `GROQ_API_KEY` (free tier). The Gemini and OpenAI adapters
are built and tested against fixtures, but nothing calls them until those keys are
added.

## Overview

| type | model(s) | signals | cost | live-verified |
|---|---|---|---|---|
| `majority` | most common training label | native (training frequency) | $0 | n/a |
| `classic` | TF-IDF + logistic regression (MT-Bench: answer-length model), temperature-scaled | native (calibrated probabilities) | $0 | n/a |
| `groq` | `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-safeguard-20b` | verbalized, self_consistency (temperature and/or prompt variation) | free tier ($0); list price recorded | ❌ pending smoke test |
| `gemini_decisions` | `gemini-decision-1` | native per-option probabilities; `p_other` | paid | ❌ no key |
| `gemini` | `gemini-3.5-flash-lite`, `gemini-3.8-flash` | verbalized, token_logprob*, verbalized_logprob*, self_consistency | paid | ❌ no key |
| `openai` | `gpt-6-luna`, `gpt-6.1-terra`, `gpt-6.2-sol` | verbalized, token_logprob*, verbalized_logprob*, self_consistency | paid | ❌ no key |

\* **Needs logprobs.**
- Both APIs define the parameters (OpenAI `top_logprobs` 0–20 with
  `include=["message.output_text.logprobs"]`; Gemini `responseLogprobs` + `logprobs`
  0–20).
- Neither documents *which models* return them. When logprobs are missing, the decider
  records only `verbalized` and falls back to it.

## Groq

OpenAI-compatible Chat Completions at `https://api.groq.com/openai/v1`. The shared
`ChatCompletionsLLM` class speaks that wire format; `GroqLLM` adds Groq's per-model facts.

**Signals.**
- `logprobs` / `top_logprobs` are "not yet supported by any of our models". You've also
  seen "`logprobs` is not supported with this model" from the live API.
- So `token_logprob` and `verbalized_logprob` are **unsupported**:
  - asking for them as `confidence_signal`, or setting `logprobs > 0`, is refused;
  - every decision records `raw["unsupported_signals"]`.
- Supported: `verbalized`, and `self_consistency` with:
  - `sample_variation = "none"`: temperature only. Groq's `temperature` works.
  - `"shuffle"`: option order shuffled per sample, in the prompt and the JSON-schema enum.
    `other` stays last.
  - `"paraphrase"`: reworded instructions per sample.
  - `"shuffle+paraphrase"`: both.

  Variants are deterministic per (example, sample index), so cached samples stay valid.
  The variation mode is part of the sample cache key.

**Per-model request facts.**

| model | structured output | reasoning | notes |
|---|---|---|---|
| gpt-oss-120b / 20b | `json_schema`, `strict: true` | `reasoning_effort` low/medium/high (server default medium); hidden via `include_reasoning: false` | production |
| qwen3.8-27b | `json_schema`, `strict: true` | `reasoning_effort` none/default/low/medium/high; hidden via `reasoning_format: hidden` | preview; max output 16,384 |
| gpt-oss-safeguard-20b | `json_schema` best-effort only (not in the strict list) | low/medium/high per its model page; hidden via `include_reasoning: false` | preview; policy goes in the **system** message (`system_prompt = true`) |
| allam-2-7b | JSON-object mode only, 4K context | — | **not supported**: price is "Pending", so cost per decision can't be reported |

Other request facts:
- `n` must be 1.
- `max_completion_tokens` replaces `max_tokens`.
- Reasoning text is never stored, only its length. It isn't needed, and on Aegis it
  could restate unsafe input.

**Free-tier limits (rate-limits page; per model, per organization):**

| model | RPM | RPD | TPM | TPD |
|---|---|---|---|---|
| gpt-oss-120b, gpt-oss-20b, qwen3.8-27b | 30 | 1,000 | 8,000 | 200,000 |
| gpt-oss-safeguard-20b | 3 | 1,000 | 2,000 | 200,000 |

The docs also say:
- Cached tokens don't count.
- `x-ratelimit-{limit,remaining,reset}-requests` are **per day**.
- `x-ratelimit-*-tokens` are **per minute**.
- `retry-after` is sent only on 429s.

What the code does with these limits:
- **Headers:** recorded per call in `raw["main"]["rate_limit"]`.
- **Per-minute 429:** retried, waiting at least `retry-after`.
- **Per-day 429** (body names "per day", RPD or TPD): the run **stops cleanly**.
  - Exit code 5; the run is saved with `status = "quota_stopped"`.
  - The message says how many examples were skipped.
  - Re-running the same config after the reset makes only the missing calls. Finished
    decisions and individual samples are in the cache.
- **Quota warning in the plan:** set `[execution] requests_per_day` / `tokens_per_day`,
  and the plan shows what share of the daily quota a run needs.
- **Not checked yet:** the docs don't specify the 429 body. The tests use Groq's observed
  wording ("on requests per day (RPD): Limit …"), and the smoke test checks it.
- **Org ids are redacted:** 429 messages name the organization, so error messages
  replace those ids with `org_[redacted]`.

**Cost.**
- `pricing.py` records each model's published developer-tier price (model pages, read
  2026-11-20).
- On `tier = "free"` (the default for `groq`):
  - `cost_usd` = tokens × list price, so the study can report cost per 1,000 decisions.
  - `charged_usd` = 0.
- Smoke configs use `[budget] max_usd = 0`, so a run that would be charged is refused.
  If the account is moved to a paid tier, set `tier = "paid"`.
- Free-tier runs still need `--yes`. Any plan with a non-zero *list* price requires
  approval.

| model | input $/1M | cached input | output $/1M |
|---|---|---|---|
| openai/gpt-oss-120b | 0.15 | 0.075 | 0.60 |
| openai/gpt-oss-20b | 0.075 | 0.037 | 0.30 |
| openai/gpt-oss-safeguard-20b | 0.075 | 0.037 | 0.30 |
| qwen/qwen3.8-27b | 0.80 | — | 4.00 |

## Baseline results (free, test half)

These were re-run on 2026-11-20 after the feature step moved inside the calibration
folds (see below).

| dataset | majority acc | classic acc | classic ECE (width) | noise floor mean / p95 |
|---|---|---|---|---|
| Banking77 (770) | 0.013 | 0.875 | 0.023 | 0.027 / 0.036 (within noise) |
| Aegis (500) | 0.538 | 0.792 | 0.050 | 0.037 / 0.056 (within noise) |
| MT-Bench (500) | 0.428 | 0.538 | 0.080 | 0.038 / 0.061 (above noise) |

The MT-Bench classic model sees only answer lengths, yet it beats majority. That is
verbosity in the human labels, which LLM judges have to be checked against. Its
cross-fitted calibration-half predictions score 0.490 accuracy and 0.076 ECE.

**What the classic model's temperature is fitted on.**
- `CalibratedClassifierCV(cv=5)` splits the **training data** into 5 folds. For each
  fold:
  - features + logistic regression are fitted on the other 4 folds;
  - the temperature T is fitted on the held-out fold;
  - the 5 (model, T) pairs are averaged.
- So T never sees test data, nor (for Banking77/Aegis) our calibration half.
- Since 2026-11-20 the TF-IDF/scaler step sits *inside* each fold. Before that, the
  vocabulary was fitted on all training rows, so the held-out fold's text (not its
  labels) shaped the features. Results moved by ≤0.002 accuracy.

**MT-Bench cross-fitting.**
- MT-Bench trains on the calibration half (there is no official train split).
  Milestone 3b fits thresholds on that half, so it needs out-of-sample predictions
  there.
- The calibration half is split into 5 folds **by question**. Each calibration example is
  predicted by the model trained on the other 4 folds, and `raw["cross_fit_fold"]`
  records which fold.
- Test examples use the model trained on the whole half.
- Inner calibration folds are grouped by question too, so a question never sits on both
  sides of a split.

**Calibration choice.**
- Platt (sigmoid) calibration was the first choice. On Banking77 it gave mean confidence
  0.66 at accuracy 0.86 (ECE 0.21).
- The cause: sigmoid calibrates each class one-vs-rest and then renormalizes.
- Temperature scaling (multiclass) is now the default. It was chosen for that documented
  reason, not by comparing test scores.

## Design rules

- **Gold labels:** real deciders receive `PublicExample`, which has no `gold` field.
  Only fakes (`needs_gold = True`) see labels.
- **Cache identity:** each decider reports its *resolved* config, defaults included,
  as its cache identity. Changing a default therefore invalidates old cache entries.
- **Self-consistency samples** are cached one by one. The key holds every resolved
  param except those that can't change a sample (`samples`, `confidence_signal`, the
  main call's `temperature`/`logprobs`/`seed`), plus `sample_index` and
  `sample_temperature`. Raising k reuses paid samples. `--refresh` bypasses all cache
  reads.
- **Rate limiting** is a token bucket that can go into debt. A 6-call batch costs 6
  request tokens even when the bucket holds 1. Tokens used beyond the estimate are
  charged after the call.
- **API keys** come from the environment only (`uv run --env-file .env ...`). They are
  read at call time, never stored, and never logged. A dry run needs no keys.
- **OpenAI** requests always send `store: false`; the Responses API otherwise keeps data
  for 30 days.
- **Raw responses** are saved per decision. No provider echoes the input text.
- **Prices** live in [pricing.py](../src/ask_or_act/deciders/pricing.py): dated, cited,
  and flagged when unverified. A model without a price entry is refused.

## Smoke-test plans (dry runs, 2026-11-20)

Proposed and awaiting approval:

| config | calls | est. tokens | charged | at list price | share of the model's daily quota | est. wall time |
|---|---|---|---|---|---|---|
| `groq-gpt-oss-120b-banking77` (k = 5, temperature only) | 120 | ~95K | $0 | $0.028 | 12% RPD, 48% TPD | ~14 min (TPM-bound) |
| `groq-qwen3.8-27b-banking77` (k = 5, shuffle+paraphrase) | 120 | ~69K | $0 | $0.067 | 12% RPD, 34% TPD | ~10 min |
| `groq-safeguard-20b-aegis` (k = 3) | 80 | ~44K | $0 | $0.008 | 8% RPD, 22% TPD | ~32 min (3 RPM) |

**Estimate method.**
- Input tokens are characters ÷ 4.
- Output is `expected_output_tokens` per call, including hidden reasoning:
  - 250 for gpt-oss at `reasoning_effort = "low"`;
  - 30 for Qwen with reasoning off.
- Self-consistency samples are assumed uncached.

After the smoke tests, `chars_per_token` and `expected_output_tokens` get calibrated from
the real `usage` counts.

**Paid providers (no keys yet):**

| run | calls | estimate |
|---|---|---|
| Decisions API, 20 × Banking77 | 20 | $0.0021 ⚠ price unverified |
| gemini-3.5-flash-lite, 20 × Banking77, k = 5 | 120 | $0.0315 |
| gpt-6-luna, 20 × Banking77, k = 5 | 120 | $0.0089 |

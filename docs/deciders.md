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
| `groq` | `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-safeguard-20b` | verbalized, self_consistency (temperature and/or prompt variation) | free tier ($0); list price recorded | ✅ 120b, qwen, safeguard (2026-11-20); 20b not yet |
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

**Redaction on sensitive tasks (Aegis, `SENSITIVE_TASKS`).** Model output can restate
unsafe input, so on these tasks nothing the model or provider wrote is kept verbatim:
- **Error messages** keep only our own wording plus the provider's error code (e.g.
  `HTTP 400 invalid_request_error [details removed: sensitive task]`). This matters
  because provider messages quote output: a schema-validation error names the invalid
  value.
- **Stored responses:** the message content is replaced by the parsed
  `{"answer", "confidence"}` (`null` answer when invalid), plus `content_chars`. Extra
  fields, prose, refusals and reasoning are dropped.
- **`failed_generation`:** only its length is kept.
- **Regression test:** `tests/test_redaction.py` drives a full run through 15 failure
  shapes whose bodies all carry a marker string, then checks that no saved file (run
  files, SQLite cache) and nothing printed or logged contains it.
- **Audit:** `uv run ask-or-act audit-redaction [paths]` indexes every Aegis prompt in
  `data/raw` as 8-word n-grams and scans runs/, .cache/ (default) or given paths,
  decoding JSON and SQLite values first. It prints ids and locations only, and exits 1
  on a match. It catches copied input text, not paraphrases; the rules above handle
  those.

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
  - Re-running the same config later makes only the missing calls. Finished
    decisions and individual samples are in the cache.
  - There is no daily reset. Groq's allowance refills gradually: every "try again in"
    time in the 2026-11-23 logs equals the missing tokens ÷ (200,000 / 24 h), i.e.
    about 8,333 tokens an hour. A shorter gap between runs just means a smaller run.
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

## Smoke-test results (2026-11-20, 20 examples each, free tier)

| run | decided | accuracy | ECE verbalized | ECE self-consistency | wall time |
|---|---|---|---|---|---|
| gpt-oss-120b, Banking77, k=5, temperature only | 20/20 | 0.80 | 0.16 | 0.17 | 19 min |
| qwen3.8-27b, Banking77, k=5, shuffle+paraphrase (re-run) | 19/20 | 0.60 | 0.28 | 0.10 | 10 min |
| gpt-oss-safeguard-20b, Aegis, k=3, temperature only | 18/20 | 0.75 | 0.13 | 0.13 | 57 min |

These are sanity checks only: with n = 20, every interval is about ±0.2.

**Real tokens per call vs the old estimates:**

| model | input (est.) | output incl. reasoning (est.) | hidden reasoning |
|---|---|---|---|
| gpt-oss-120b, Banking77 | 1,069 (543) | 48, p90 65 (250) | 24 (reported in `completion_tokens_details`) |
| qwen3.8-27b, Banking77 | 549 (543) | 18, p90 19 (30) | 0 (`reasoning_effort = none`) |
| gpt-oss-safeguard-20b, Aegis | 361 (297) | 89, p90 159, max 563 (250) | 68 |

What the numbers showed:
- **gpt-oss writes the JSON schema into the prompt.** Banking77's 78-option schema adds
  about 514 input tokens. The estimator now adds the schema's size for those models
  (`ChatModel.schema_in_prompt`), and `expected_output_tokens` is set from the means
  above. Dry runs are now within 1–4% of the real input counts.
- **Groq counts actual tokens** (input + output) against TPM, not `max_completion_tokens`,
  according to the `x-ratelimit-remaining-tokens` headers.

**Provider behaviour:**
- **Safeguard** accepts a system message and `reasoning_effort = "low"`, and its free
  tier is 2,000 TPM (header-confirmed).
  - 2 of 20 examples failed with a new error code, `output_parse_failed`.
  - Its `failed_generation` held the model's *reasoning*, which restates the input. Such
    text is now removed from every error message and never stored for Aegis (only its
    length).
- **Qwen at temperature 1.0 degenerates.** In the first run, 11 of 20 examples failed:
  any failed sample sank its example. In the re-run, after that was fixed, 17 of 91
  sample calls (19%) failed.
  - The failed outputs: `{"token`, repeated words, misspelled labels.
  - Groq rejects them with `json_validate_failed`, so strict mode doesn't constrain
    decoding for Qwen. A temperature-0 main call also produced an invalid label once.
  - A failed sample now counts as a disagreeing sample.
- **Burst pacing:** sending an example's k+1 calls together overran safeguard's 2K TPM.
  12 of 20 examples needed 2–4 attempts. Calls are now paced one by one (`CallGate`).
- **Daily limit:** never reached, so its 429 wording is still unobserved. The tests use
  Groq's commonly reported form.

**Do samples differ?**

| run | agreement with the main answer (valid samples) | examples with any disagreeing sample |
|---|---|---|
| gpt-oss-120b, temperature only | 0.95 | 2/20 |
| safeguard-20b, temperature only | 0.98 | 1/18 |
| qwen, shuffle+paraphrase | 0.72 | 9/19 |

Temperature alone barely moves the gpt-oss models. The Qwen run differs in both model
and variation, so it doesn't separate the two effects. A controlled comparison is
planned.

**Paid providers (no keys yet):**

| run | calls | estimate |
|---|---|---|
| Decisions API, 20 × Banking77 | 20 | $0.0021 ⚠ price unverified |
| gemini-3.5-flash-lite, 20 × Banking77, k = 5 | 120 | $0.0315 |
| gpt-6-luna, 20 × Banking77, k = 5 | 120 | $0.0089 |

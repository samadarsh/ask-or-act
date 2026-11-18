# Benchmark datasets

> **⚠ Content warning.** The safety dataset (Aegis 2.0) contains harmful text by design,
> including:
> - hate speech
> - harassment
> - violence
> - sexual content
> - self-harm
> - other unsafe material
>
> ask-or-act never reproduces this text in reports, the README, demos, logs or test
> fixtures. Examples are referred to by ID or by a redacted snippet only.

Status: **approved 2026-11-18,** with the additions below. Metadata was checked on 2026-11-18 from the
Hugging Face dataset cards and API. No data files were fetched.

Each dataset covers one kind of AI decision: **routing**, **judging** and **safety
gating**.

## Split policy (applies to all three)

**Classic ML baseline:** trained only on the dataset's own train split. This costs no
API calls.

**API sample:** this is what we pay to run deciders on. It is split 50/50, stratified by
label:
- *Calibration half:* used to fit calibrators and choose guaranteed thresholds.
- *Test half:* used to report every metric. It is touched once per study run.

**Sizing:** about 500 calibration examples per task.
- At α = 0.1 and δ = 0.1, conformal and fixed-sequence methods have usable power with a
  few hundred calibration points.
- 500 test examples give a bootstrap CI on accuracy of about ±3–4 points.

Bigger samples can come later, if the first results justify the spend.

**Fixed seed:** the sample is drawn once with a fixed seed. Its example IDs are
committed to the repo so every run uses the same examples.

---

## 1. Routing: Banking77 ✅ recommended

| | |
|---|---|
| Source | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) |
| License | CC-BY-4.0 (attribution only) |
| Size | train 10,003; test 3,080; ~0.4 MB download |
| Labels | `label`: 77 intent classes; `text`: short customer query |
| Gated | No |

**Task:** route a banking query to one of 77 intents.
- 77 classes makes calibration interesting. Many intents are near-duplicates, for
  example `card_arrival` and `card_delivery_estimate`.
- Act means auto-route, ask means request clarification, escalate means send to a human
  agent.

**Sample:** 1,540 examples **from the official test split only**, 20 per class, giving
770 calibration and 770 test.
- Stratifying by class keeps every intent present in both halves.
- The classic baseline (TF-IDF + logistic regression) trains on the **official train
  split only** (10,003 examples). The two never overlap.
- **"None of these / other" option:** decision models get an extra choice so they can
  say no intent fits. That probability mass is an escalate signal. Results are
  reported both with and without it.

**Cost shape:** queries are about 15–25 tokens.
- The 77 option descriptions dominate, at roughly 700–900 tokens per call.
- Decisions API: about 1,540 × ~900 ≈ 1.4M input tokens.
- Prompted LLM: similar input, plus a few output tokens.
- Self-consistency with k samples multiplies the prompted-LLM cost by k.

---

## 2. Judging: MT-Bench human judgments ✅ recommended

| | |
|---|---|
| Source | [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) |
| License | CC-BY-4.0 |
| Size | `human` split 3,355 rows; `gpt4_pair` split 2,400 rows; ~1.4 MB download |
| Labels | `winner` ∈ {`model_a`, `model_b`, `tie`}; plus `question_id` (80 questions), `turn` (1/2), `judge`, both conversations |
| Gated | No |

**Task:** pairwise judging. Which answer is better? The judge's verdict is compared with
the expert human verdict.

**Sample:** 1,000 human judgments, giving 500 calibration and 500 test.
- **Split by `question_id`,** not by row, so no question appears in both halves. There
  are only 80 questions, and a row-level split would leak.
- Keep ties as a third class. Also report the no-ties result, as the original paper did.
- **Multiple voters:** several experts voted on some comparisons. A comparison is
  identified by (`question_id`, `turn`, unordered model pair).
  - The sampler records, for each sampled row, its comparison key and how many human
    votes that comparison has.
  - This lets us measure human–human agreement, which is the ceiling for the judge.
  - All votes for one comparison go into the same split.
- Bonus: the `gpt4_pair` split is a free extra "LLM judge" decider. It needs no API
  calls.

**Cost shape:** the most expensive of the three tasks.
- Each judgment includes two multi-turn answers, about 1,500–2,500 tokens.
- Running both answer orders to detect position bias doubles it.
- About 1,000 × 2 × 2,000 ≈ 4M input tokens per decider.

**Classic baseline:** no train split exists.
- Use a length/feature model (answer-length difference, etc.), cross-validated within
  the calibration half.
- It is weak by design, as a "verbosity heuristic" baseline.

---

## 3. Safety gating: options compared

| Dataset | License | Size | Labels | Verdict |
|---|---|---|---|---|
| [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) (now "Nemotron Content Safety V2") | CC-BY-4.0; card says "intended for research purposes", not for training dialogue agents | train 30,007; val 1,445; test 1,964; ~28.5 MB | binary `prompt_label` / `response_label`; 23-category taxonomy; label source = human / llm_jury / augmentation | ✅ **recommended** |
| [mmathys/openai-moderation-api-evaluation](https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation) | MIT | 1,680 rows, single file; ~0.5 MB | 8 binary category flags (missing = unknown) | good backup / second safety set |
| [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | CC0 | ~2M rows; ~422 MB download | 7 *continuous* toxicity scores, no classes | ✗ too large, scores need arbitrary binarization |
| [toxigen/toxigen-data](https://huggingface.co/datasets/toxigen/toxigen-data) | **not stated**; external access form | ~319k rows | 1–5 toxicity scores | ✗ no license |

**Recommendation: Aegis 2.0.**
- It has a real train split for the classic baseline.
- It has a binary safe/unsafe label, which maps cleanly to act or escalate.
- It records whether each label came from humans or an LLM jury.
  - **Only human-labelled rows are used for calibration and test,** so we measure
    agreement with humans and not with another model.
  - LLM-jury labels may be used only to train the classic baseline.
- Research use fits the card's stated intent.

**Sample:** 1,000 human-labeled prompt rows from the test and validation splits,
stratified safe/unsafe, giving 500 calibration and 500 test.

**Cost shape:** prompts are about 50–300 tokens, so about 1,000 × ~400 ≈ 0.4M input
tokens per decider. This is the cheapest task.

**Caveats**
- About 358 rows have a "REDACTED" prompt that must be rebuilt from a separate source.
  We just exclude them.
- The card gives validation as both 1,445 and 1,245 rows. Verify when loading.
- The content is harmful by design (see the content warning at the top). Never show
  raw text; use IDs, aggregates or redacted snippets only.

---

## Cost estimate (to be finalized before any paid call)

Input tokens per decider, for one pass over the sample:

| Task | Examples | ~Input tokens |
|---|---|---|
| Banking77 | 1,540 | ~1.4M |
| MT-Bench (both orders) | 1,000 | ~4M |
| Aegis 2.0 | 1,000 | ~0.4M |
| **Total** | | **~5.8M** |

**Gemini Decision 1:** at $0.10 per 1M input tokens, ~5.8M tokens ≈ **$0.58** per full
pass. The free tier could cover this but is slow at 50k TPM, and its data is used by
Google.

**Prompted LLMs:**
- Cost is ~5.8M × that model's input price, plus output tokens.
- Self-consistency at k = 5 multiplies that by about 5.
- Exact dollar figures will be computed with current price sheets before any run, per
  the project rules.

**Classic ML baseline:** $0.

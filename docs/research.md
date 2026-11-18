# Research foundations

This file lists the papers and products ask-or-act builds on, and what we take from
each. Links were verified on 2026-11-18.

**Rule:** before implementing a statistical guarantee, read the original paper and cite
the exact theorem in the code. Theorem numbers are marked *TBD* below on purpose. They
get filled in when the code that relies on them is written, from the version of the
paper we actually read.

---

## 1. Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement

Jung, Brahman, Choi. ICLR 2025. [arXiv:2407.18370](https://arxiv.org/abs/2407.18370)

**What it does**
- It turns an LLM judge into a *selective* judge. The judge abstains (escalates) unless
  it is confident enough.
- The confidence threshold is chosen on a calibration set, using fixed-sequence
  multiple testing.
- With probability at least 1−δ, the judge's agreement with humans on the cases it does
  not abstain on is at least 1−α.

**Ideas from the paper**
- **Simulated Annotators:** a confidence estimate built from agreement across simulated
  annotators. It is presented as better calibrated than raw judge confidence.
- **Cascaded Selective Evaluation:** try a cheap judge first and escalate to a stronger
  judge, then to a human, only when needed.

**What we take**
- The fixed-sequence testing threshold search, for M3b.
- The cascade structure (cheap → strong → human). This is exactly act / ask / escalate.
- "Guaranteed human agreement" as the headline metric for judge mode (M4).

**Theorem to cite:** TBD at implementation (M3b).

---

## 2. Conformal abstention and conformal risk control

### 2a. Mitigating LLM Hallucinations via Conformal Abstention

Abbasi-Yadkori, Kuzborskij, Stutz, et al., 2024. [arXiv:2405.01563](https://arxiv.org/abs/2405.01563).
NeurIPS 2024 workshop: Statistical Frontiers in LLMs and Foundation Models.

**What it does**
- Uses a self-consistency score (how much sampled answers agree with each other) to
  decide when the model should say "I don't know".
- Sets the abstention threshold with conformal prediction / risk control, so the
  hallucination rate is bounded.

**What we take**
- Self-consistency as a confidence signal.
- Conformal risk control as a second guarantee method, next to Trust or Escalate.
  Comparing the two is a study result in its own right.

### 2b. When Can Conformal Risk Control Certify LLM Outputs? Bounds, Impossibility, and Adaptation for Structured Generation

Kotte, 2026. [arXiv:2606.29054](https://arxiv.org/abs/2606.29054). v1 is dated
2026-06-27 and v2 2026-09-07.

**What it does**
- It gives a lower bound on abstention. When the base risk μ exceeds the target α, any
  distribution-free method must abstain on at least a minimum fraction of inputs.
- This gives a closed-form test for whether a target is feasible at all, before running
  calibration.
- It reports that heuristic abstention policies miss the risk targets on 7.5–12.5% of
  settings.

**What we take**
- The minimum-abstention feasibility check. ask-or-act should tell the user "your
  target α is unreachable unless you escalate at least X% of traffic" before it searches
  for thresholds.

**⚠ Open item:** the v1 abstract states the bound as (μ−α)/(1−α), and the v2 abstract
appears to state it as (μ−α)/(M−α), where M is presumably the maximum loss. Check the v2
PDF body and cite the exact statement and version before implementing.

### 2c. Background: the original conformal risk control paper

This is not in the brief, but it is likely the theorem we cite.

Angelopoulos, Bates, Fisch, Lei, Schuster. *Conformal Risk Control.*
[arXiv:2208.02814](https://arxiv.org/abs/2208.02814). ICLR 2024.

Related: *Learn then Test* (Angelopoulos et al.,
[arXiv:2110.01052](https://arxiv.org/abs/2110.01052)). It is the general framework that
fixed-sequence testing in Trust or Escalate fits into.

---

## 3. Comparing LLM confidence signals

### 3a. When Calibration Rankings Reverse: Accuracy-Controlled Evaluation (ACE)

[arXiv:2606.30814](https://arxiv.org/abs/2606.30814). v2 is dated 2026-08-31.

**What it does**
- It argues that raw calibration metrics such as ECE are confounded by accuracy. A more
  accurate model can look "better calibrated" just because it is more often right.
- Many apparent calibration advantages shrink or reverse once accuracy is controlled.
- It proposes an evaluation with three accuracy-controlled views.

**What we take**
- Report accuracy-controlled calibration alongside raw ECE/Brier whenever we compare
  confidence signals or deciders. The exact three views are to be read from the paper in
  M2.

### 3b. Other signal comparisons found during the literature check

These are candidates and have not been read in full yet.

- **Beyond Logprobs: A Multi-Signal Confidence Engine.**
  [arXiv:2606.24420](https://arxiv.org/abs/2606.24420). Combines logprob features with
  entropy, plus isotonic or Lasso calibration.
- **Rethinking Verbalized Confidence for LLM-as-a-Judge.**
  [arXiv:2609.10996](https://arxiv.org/abs/2609.10996).
  - On post-2025 proprietary models, verbalized confidence beats logprobs for judging.
  - Some APIs no longer expose logprobs, which matters for which signals we can support.
- **How do LLMs Compute Verbal Confidence?**
  [arXiv:2603.17839](https://arxiv.org/abs/2603.17839). Verbal confidence is not fully
  explained by token probabilities, so the two signals carry complementary information.
  That is the motivation for a hybrid.

### 3c. Evaluation Pitfalls and Sparsity Limitations in LLM-based Confidence Estimates for Classification

Merdjanovska, Zaidan, Rücklé (Humboldt-Universität zu Berlin / Amazon). Findings of
ACL 2026. [arXiv:2608.04899](https://arxiv.org/abs/2608.04899), v1 dated 2026-08-05.
DOI [10.18653/v1/2026.findings-acl.1671](https://doi.org/10.18653/v1/2026.findings-acl.1671).

**This is "the hybrid verbalization-logprob method".** The paper calls it
**verbalization logprobs** (§4).
- The model is asked for a verbalized confidence between 0 and 100. Instead of reading
  off the sampled digits, we take the expected value under the digit-token
  distributions:

  conf = Σ_{d=0..9} 10·d·P(x_i = d) + Σ_{d=0..9} d·P(x_{i+1} = d)

  Here x_i is the tens-digit token and x_{i+1} is the units-digit token.
- Quote from §4: it "weighs verbalized confidence digits by their token probabilities."
- It needs one inference call, at about 1.7× the cost of vanilla verbalization. The
  alternative, verbalization sampling (mean over 4 samples), costs about 6.8×.

**Assumptions**
- The two digit positions are treated as independent. Footnote 6 calls this an
  approximation; the exact version would condition the units digit on the tens digit.
- Each digit must be a single token. Check this per tokenizer.
- It needs logprobs, so it cannot run on APIs that don't return them.

**Headline results** (Qwen3-32B, Table 3)

| Method | AUARC | AUROC | ECE |
|---|---|---|---|
| Vanilla verbalization | 0.713 | 0.661 | 0.235 |
| Verbalization logprobs | 0.736 | 0.670 | 0.234 |

- The method mainly helps **ranking**: AUARC (area under the accuracy–rejection curve)
  and AUROC improve.
- **ECE barely moves.** Better selective prediction is not the same thing as better
  calibration.
- **Unique confidence values:** vanilla verbalization produces only 8–12 per dataset,
  while verbalization logprobs produce 630–2,227 (Table 4).
- **Practical effect:** to reach ≥98% accuracy, vanilla verbalization must reject 39.7%
  of inputs, while verbalization logprobs reject 28.1% (§4).

**Calibration-measurement pitfalls for M2**

1. **Sparsity** (§3). Verbalized confidences take very few distinct values. For example,
   45–93% of them fall on the top-5 values. Consequences:
   - The risk–coverage curve has only a handful of real operating points.
   - Threshold search (M3b) can only choose between those points.

   We report the number of unique confidence values for every decider.
2. **AUARC interpolation** (§3, §5). AUROC and AUPRC are conventionally computed
   stepwise, but AUARC is often computed with linear (trapezoidal) interpolation. With
   sparse scores, linear interpolation invents operating points that don't exist.
   - Switching between linear and stepwise reorders methods. Consistency sampling falls
     from rank 1 to rank 5 (Table 2).
   - The paper says "the correct way is to use stepwise interpolation".

   **We compute all risk–coverage areas stepwise,** handling ties by treating
   tied-confidence groups as a single block.
3. **Ties.** A direct consequence of sparsity: many examples share one confidence value.
   Coverage can only move in jumps at tie groups. Metrics must not break ties in a way
   that depends on example order.

The paper does not discuss ECE-binning pitfalls or accuracy confounds. For the latter,
see ACE (§3a).

---

## 4. LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing

Li et al., 2026. [arXiv:2601.07206](https://arxiv.org/abs/2601.07206). Findings of ACL
2026. Code: [ynulihao/LLMRouterBench](https://github.com/ynulihao/LLMRouterBench)

**What it does**
- Over 400K instances from 21 datasets and 33 models.
- Metrics for both pure performance and performance–cost trade-offs, plus 10 routing
  baselines.
- It reports that commercial routers often fail to beat the single best model.

**What we take**
- Always compare routers against the "always-strong" and "always-cheap" baselines.
- Report cost alongside quality.
- Later, possibly use it as an extra routing dataset for M7.

---

## 5. LLM-as-a-judge bias studies

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.** Zheng et al., NeurIPS 2023
  Datasets & Benchmarks. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
  - Documents position, verbosity and self-enhancement bias.
  - Source of the MT-Bench human judgments dataset.
- **Large Language Models are not Fair Evaluators.** Wang et al., 2023.
  [arXiv:2305.17926](https://arxiv.org/abs/2305.17926)
  - Answer order changes verdicts.
  - Proposes balanced position calibration: evaluate both orders.
- **Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge.** Ye et al., 2024.
  [arXiv:2410.02736](https://arxiv.org/abs/2410.02736). A framework (CALM) covering 12
  bias types.

**What we take (M4)**
- Judge mode runs every pairwise comparison in both orders.
- It reports a position-consistency rate.
- It treats an order-dependent verdict as low confidence, which is a natural escalate
  signal.
- Verbosity bias is checked by regressing the verdict on the answer-length difference.

---

## 6. Self-Improvement Memory Enables Continual Recursive Self-Improvement ("SIM")

Yu et al. (Google Cloud AI, Google DeepMind, Penn State), 2026.
[arXiv:2611.18856](https://arxiv.org/abs/2611.18856)

**What it does**
- An optimizer that edits an agent's harness keeps a memory of past edits, grouped by
  the recurring problem each edit addressed.
- Each memory entry stores:
  - the rationale
  - the predicted outcome
  - the observed outcome
  - the evaluation context
- Continuing optimization with this memory beat continuing without it. Across 10 paired
  runs the average gain was +2.36 points (p = 0.003).

**What we take**
- The structure of the M6 decision changelog: root cause, change and rationale,
  expected effect recorded before the run, observed metrics, and context (model, prompt,
  dataset).
- Grouping entries by root cause makes repeated failed fixes visible.

---

## 7. Gemini Decisions API (Gemini Decision 1)

Google, launched 2026-11-11.
- [Docs](https://aistudio.google.com/docs/decisions-api)
- [Launch post](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/)

**What it is**
- A model that returns probabilities instead of text. It produces zero output tokens.
- Question types:
  - `noul` (yes/no): a probability clamped to [0.01, 0.99].
  - `choice`: one option out of 2–255, with a probability for every option.
  - `score`: an ordinal scale.
- Each response includes a `confidence`, which is the top option's probability.
- Up to 1,000 questions per request share one input, and that input is billed once.
- Endpoint: `v1beta` only. Model: `gemini-decision-1`.

**Cost and limits**
- Input costs $0.10 per 1M tokens on the paid tier, and output costs nothing.
  - The price was relayed from a summary of the pricing page. Re-check it before any
    cost estimate.
- The free tier has these rate limits:
  - 200 RPM
  - 50k TPM
  - 10k RPD
- On the free tier, data is used to improve Google's products.

**Caveats that matter to us**
- Google says the probabilities are "calibrated" but does not say how that was measured.
  Testing that claim is a core result of this project.
- `choice` always spreads probability over the listed options, even when none fits. The
  docs recommend adding a catch-all option or a separate yes/no scope check.
- Option keys are sorted alphabetically before evaluation.
- The yes/no clamp means a threshold above 0.99 can never fire.
- No batch API and no context caching.

**What we take**
- A first-class decider in M3. Its native probabilities are one of the confidence
  signals compared in the study.

# How often the gate fails falsely, and what it catches

`scripts/gate_simulation.py` (run 2026-11-24 UTC, 247 s on 11 cores) draws pairs of
predictions files from known models and runs the real gate (`gate.run_gate`) on each.

**Setup.**
- Each file has 2,000 items, split into 1,000 calibration and 1,000 test by the hashed
  split. Each gate run uses 1,000 bootstrap resamples, at level 0.05 and the default
  tolerances unless stated.
- Items have a difficulty p ~ Beta(5, 1) shared by both files, so the files are paired
  the way real ones are.
- A model is right with probability p + shift. Its confidence is
  logit(conf) = logit(p + shift) + bias + N(0, 0.35²).
- Rates come with Wilson 95% intervals.

| scenario | runs | accuracy | calibration | coverage | certified | any paired check | gate FAIL | strict FAIL |
|---|---|---|---|---|---|---|---|---|
| null (same model, default tolerances) | 400 | 0.2% (0.0–1.4) | 0.0% (0.0–1.0) | 0.8% (0.3–2.2) | 0.0% (0.0–1.0) | 1.0% (0.4–2.5) | 1.0% (0.4–2.5) | 98.5% (96.8–99.3) |
| boundary (same model, all tolerances 0) | 400 | 1.5% (0.7–3.2) | 1.5% (0.7–3.2) | 1.8% (0.9–3.6) | 0.2% (0.0–1.4) | 4.2% (2.7–6.7) | 4.2% (2.7–6.7) | 100.0% (99.0–100.0) |
| accuracy at −1 point (the tolerance) | 400 | 2.2% (1.2–4.2) | 0.0% (0.0–1.0) | 6.0% (4.1–8.8) | 3.0% (1.7–5.2) | 8.0% (5.7–11.1) | 8.2% (5.9–11.4) | 99.8% (98.6–100.0) |
| power: accuracy −4 points | 200 | 44.5% (37.8–51.4) | 0.0% (0.0–1.9) | 77.5% (71.2–82.7) | 72.5% (65.9–78.2) | 83.5% (77.7–88.0) | 83.5% (77.7–88.0) | 100.0% (98.1–100.0) |
| power: overconfident (bias +0.8) | 200 | 0.0% (0.0–1.9) | 84.5% (78.8–88.9) | 2.0% (0.8–5.0) | 0.0% (0.0–1.9) | 84.5% (78.8–88.9) | 84.5% (78.8–88.9) | 100.0% (98.1–100.0) |
| power: noisier confidence (σ 0.35 → 1.2) | 200 | 0.0% (0.0–1.9) | 84.5% (78.8–88.9) | 32.0% (25.9–38.8) | 16.0% (11.6–21.7) | 89.0% (83.9–92.6) | 89.0% (83.9–92.6) | 100.0% (98.1–100.0) |

Inconclusive rates (these pass unless strict):

| scenario | accuracy | calibration | coverage |
|---|---|---|---|
| null (same model, default tolerances) | 81.2% | 47.2% | 85.5% |
| boundary (same model, all tolerances 0) | 94.8% | 92.5% | 91.0% |
| accuracy at −1 point (the tolerance) | 91.5% | 47.8% | 88.8% |
| power: accuracy −4 points | 55.5% | 25.0% | 22.5% |
| power: overconfident (bias +0.8) | 86.5% | 15.5% | 83.5% |
| power: noisier confidence (σ 0.35 → 1.2) | 83.0% | 15.5% | 67.0% |

## What it shows

1. **When nothing changed, the gate failed falsely 1.0% of the time** (0.4–2.5%).
2. **On the boundary of H0 the family stays at the promised level.** Here every
   tolerance is 0 and the models are identical, the hardest case for a test's size.
   The paired checks failed together 4.2% of the time (2.7–6.7%), against the 5% that
   Holm guarantees. Each check alone failed 1.5–1.8%. That includes coverage, whose
   bootstrap refits thresholds and is only approximately valid: here it held its
   level.
3. **Tango's lower tail is valid at the margin.** With the new model exactly 1 point
   less accurate (the default tolerance), the accuracy check failed 2.2% (1.2–4.2%),
   at or below 5%. The paper simulates only the upper tail (Tango 1998, Table II).
4. **A drop of exactly the tolerance fails the whole gate 8.2% of the time** (5.9–11.4).
   That is more than 5%, and it isn't a broken test: a model that is 1 point less
   accurate also covers less at the same α. The coverage check (6.0%) and the
   certification check (3.0%) respond to that real change. Certification has no
   tolerance, because "the baseline is certified and the new file isn't" is a fact,
   not an estimate.
5. **Power.** With 1,000 test rows, the gate caught:
   - a 4-point accuracy drop 83.5% of the time. Accuracy alone caught only 44.5%; the
     drop also costs coverage and certification;
   - overconfidence 84.5% of the time, through the accuracy-controlled calibration
     check;
   - noisier confidence 89% of the time.
6. **Strict mode needs large files or wide tolerances.** It failed 98.5% of null runs.
   With ~1,000 test rows, one accuracy difference has an SD of about 1.6 points, wider
   than the 1-point tolerance. So "shown within the tolerance" is rarely possible, and
   most checks are inconclusive. Turn on `strict` only when the inconclusive rate on
   your own data is low, for example with 10,000+ test rows or tolerances of a few
   points.

## Run time

One gate run on two 2,000-row files with 2,000 resamples, refitting both thresholds
in every resample, takes about 1.7 s. That is `tests/fixtures/predictions/` on an
Apple-silicon laptop. Each refit uses `guarantees.fixed_sequence.FastFit`, which gives
the same thresholds as `multi_start` (tested), with 2,000 refits taking 0.15 s.

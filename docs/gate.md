# `ask-or-act gate`: what it promises, and what it doesn't

`ask-or-act gate NEW --baseline BASE` compares two predictions files paired on the same
ids and exits 1 only when NEW is **worse than BASE by more than a tolerance, beyond
statistical noise**. Usage, configuration and CI setup are documented in
docs/cli-design.md §3.6; this page covers the guarantees and their limits.

## The four checks

| check | what it compares | test | default tolerance |
|---|---|---|---|
| accuracy | test-half accuracy | Tango (1998) score test, Eq. (24)–(26), interval Eq. (33) | 1 point |
| calibration | ECE gap under the ACE DA view (accuracy-controlled, arXiv:2606.30814 v2 §4.2) | centered paired bootstrap | 0.02 |
| coverage | share of test rows acted on at α, each file's threshold fitted on its own calibration half | centered paired bootstrap, thresholds refitted in every resample | 5 points |
| certified | does each file certify a threshold at α, δ? | none: a yes/no fact | none |

The three paired checks test H0: "NEW is at most `tolerance` worse" one-sided at
`level` (default 0.05), and their p-values are Holm-adjusted together.

## The false-FAIL bound

**The 5% bound holds when nothing changed.** If NEW and BASE come from the same model, the
gate fails falsely at most `level` = 5% of the time in total, not 5% per check. In the
simulation (docs/gate-simulation.md) it failed 1.0% of such runs with the default
tolerances, and 4.2% even with every tolerance set to 0, the hardest case for the bound.

**At the edge of the tolerance the gate fails more often.** A NEW model that is exactly 1
point less accurate (the default tolerance) failed the whole gate 8.2% of the time in the
simulation. The accuracy check itself stayed within its level (2.2%). The extra failures
are real: a less accurate model also **covers less** at the same α, and the coverage
check (6.0%) and the certification check (3.0%) respond to that genuine shrinkage. So
the bound is about "nothing changed", not "changed by up to the tolerance on one
metric".

**Coverage is approximate.** Its bootstrap refits both thresholds in each resample, and
a threshold search is a discontinuous function of the data, so bootstrap theory doesn't
cover it exactly. In the simulation it held its level (1.8% at tolerance 0).

## The certification check has no noise control

`certified` fails when BASE certifies a threshold at α, δ and NEW doesn't. It has **no
tolerance and no statistical test**: it is a yes/no outcome of each file's own
fixed-sequence test. Near the borderline (a file whose error rate is close to α, or
one with only just enough calibration rows) it can flip between runs from noise alone. It
isn't covered by the Holm bound above.

If that is too brittle for your setup, set

```toml
[gate]
require_certified = false
```

A lost certificate then becomes a warning (`!`) instead of a failure. The coverage
check still measures the practical effect: a file that certifies nothing covers 0%.

## Strict mode

By default a check that can show neither "worse" nor "within the tolerance" is
*inconclusive* and passes. `--fail-on-inconclusive` (or `strict = true` in `[gate]`)
makes it fail, so a pass then means every paired check was **shown** within its
tolerance.

That needs much more data than detecting a regression. In the simulation, strict mode
failed 98.5% of runs where nothing changed, with 1,000 test rows. So in strict mode the
gate prints how many test rows each check would need next to the actual count:

```
strict: test rows needed to show each check within its tolerance (approx., 80% chance
when nothing changed): accuracy ~14,924, calibration ~1,802, coverage ~13,300; this run
has 957.
```

The estimate is a normal approximation. It reads the standard error off the run's own
interval, scales it by √(n_now / n), assumes the calibration half grows with the test
half, and solves for an 80% chance that an unchanged model passes:
n = n_now · ((z₁₋ₗₑᵥₑₗ + z₀.₈) · SE / tolerance)². A tolerance of 0 can never be shown.
JSON output has the same numbers under `result.rows_needed`. Turn on strict mode only
when those numbers are below your file size, or widen the tolerances.

## Limits

- The gate takes predictions files (or `DecisionTable`s), not run folders: a run folder
  holds one split, and the gate needs both halves of both files.
- Rows that are clustered (several per group) make the accuracy test optimistic; the
  gate warns. The bootstrap checks resample whole groups.
- The gate reads only local files and never uses the network.

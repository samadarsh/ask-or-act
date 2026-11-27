# `ask-or-act gate`: what it promises, and what it doesn't

`ask-or-act gate NEW --baseline BASE` compares two predictions files paired on the same
ids and exits 1 only when NEW is **worse than BASE by more than a tolerance, beyond
statistical noise**. This page covers how to use it, in a terminal and in CI, and what
it does and doesn't promise. Real terminal sessions are in docs/cli-design.md §3.6.

## Usage

```sh
# the baseline: predictions of the current main branch on a fixed, labelled eval set
ask-or-act gate new.csv --baseline eval/baseline.csv

# say what you changed and what you expect, before seeing the result
ask-or-act changelog expect --change "prompt v4" --why "shorter instructions" \
    --expect "accuracy -1 to +1 points" --expect "coverage >= -2 points"
ask-or-act gate new.csv              # baseline etc. from ask-or-act.toml
ask-or-act changelog --last 5        # predicted vs observed, run by run
```

- **Both files** need the same labelled ids, with the same split column if they have
  one. Without one, each id goes to the calibration or test half by a hash, so the
  halves match. Different ids exit 6 with the counts.
- **Each file gets its own threshold**, fitted on its calibration half at α, δ.
  Accuracy, calibration and coverage are measured on the test half.
- **Exit codes:** 0 pass, 1 fail, 2 usage or config error, 6 bad input, 7 too few
  labelled rows, 70 a bug (never 1, so a crash can't pass for a failed gate).
- **`--json`** prints one document: every check, the p-values, the thresholds, and the
  changelog entry. **`-o gate.md`** also writes Markdown.

**Expectations.** The grammar is `METRIC LOW to HIGH [UNIT]`, `METRIC >= X` or
`METRIC <= X`. Metrics: accuracy and coverage (in points), calibration (the ACE gap)
and ece (raw ECE). Both are in `abs`, and lower is better. A prediction *holds* when the
observed change is inside the range. It *missed, beyond noise* when the whole interval
is outside it.

**Policy.** Flags override `[gate]` in `ask-or-act.toml` (or `[tool.ask-or-act.gate]`
in `pyproject.toml`), which overrides the defaults. The file is searched for from the
current folder up to the repository root, never above. All keys are in
docs/cli-design.md §3.6. Fix the policy before looking at results, and date it with
`registered`.

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
  when nothing changed): accuracy ~15,805, calibration ~630, coverage ~49,951; this run
  has 1,011.
```

The estimate is a normal approximation. It reads the standard error off the run's own
interval, scales it by √(n_now / n), assumes the calibration half grows with the test
half, and solves for an 80% chance that an unchanged model passes:
n = n_now · ((z₁₋ₗₑᵥₑₗ + z₀.₈) · SE / tolerance)². A tolerance of 0 can never be shown.
JSON output has the same numbers under `result.rows_needed`. Turn on strict mode only
when those numbers are below your file size, or widen the tolerances.

Strict mode is meant for large production logs. For most teams the default mode is the
right choice: it fails only on proven drops and reports inconclusive checks openly. With
wider tolerances (accuracy 3 points, coverage 10 points) the same files need about
1,757, 630 and 12,488 rows, so a log of 12,500 labelled decisions can run strict.

## In CI (GitHub Actions)

`action.yml` at the root of this repository is a composite Action. Copy
[examples/github/gate.yml](../examples/github/gate.yml) to `.github/workflows/` and
replace its "Predict" step with yours:

```yaml
permissions:
  contents: read
  pull-requests: write   # the comment only
steps:
  - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
  - run: python eval/predict.py --out predictions.csv   # yours
  # REPLACE COMMIT_SHA: replace with the release commit SHA after v1.0
  # (until then, the full SHA of a commit on main; never a branch or tag)
  - uses: samadarsh/ask-or-act@COMMIT_SHA
    with:
      new: predictions.csv
      change: ${{ github.event.pull_request.title }}
```

**What it does:**
- installs ask-or-act from the Action's own checkout (locked, no PyPI);
- runs the gate with `--no-record --json -o gate.md`;
- writes the Markdown to the job summary and uploads `gate.json` and `gate.md` as an
  artifact;
- comments on the pull request;
- fails the job with the gate's exit code.

**The comment.** There is one comment per PR, found by a hidden marker and edited on
every push, so it never overwrites another bot's comment.
- **Forks:** a fork's token is read-only, so fork PRs get no comment. The job summary
  and the artifact still show the result. Don't switch to `pull_request_target` to get
  one: it would run the fork's code with a write token.

**The changelog in CI.** CI never writes to the repository (no `contents: write`), so
it doesn't append. To keep the entry, download the artifact and run
`ask-or-act changelog add gate.json` locally, then commit the changelog. For
expectations recorded before the run:
1. run `ask-or-act changelog expect` locally;
2. commit the changelog with the PR;
3. the CI gate run picks the expectation up.

**Inputs:** `new` (required), `baseline`, `change`, `why`, `expect` (one per line),
`fail-on-inconclusive`, `args` (extra flags, split on spaces), `working-directory`,
`comment`, `artifact-name`, `python-version` and `github-token`. **Outputs:** `result`
(pass, fail or error), `exit-code`, `json`, `markdown`.

**Security:**
- every input reaches the shell through `env:`, so a PR title can't inject commands;
- the third-party actions are pinned to commit SHAs.

`tests/test_action.py` checks both, and this repository's own workflow
(`.github/workflows/gate.yml`) runs the Action on synthetic fixtures for every change.

## Run time

About 1.7 s for two 2,000-row files with 2,000 resamples, on an Apple-silicon laptop,
refitting both thresholds in every resample. On GitHub's runners it took about 3.4 s.
`--n-boot` trades precision for time.

## Limits

- The gate takes predictions files (or `DecisionTable`s), not run folders: a run folder
  holds one split, and the gate needs both halves of both files.
- Rows that are clustered (several per group) make the accuracy test optimistic; the
  gate warns. The bootstrap checks resample whole groups.
- The gate reads only local files and never uses the network.

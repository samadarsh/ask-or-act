# CLI design: "bring your own predictions"

Status: **approved 2026-11-22 UTC**, with the decisions and additions below. Being
implemented in steps (§7). A terminal session marked *real output* is pinned by
`tests/test_cli_docs.py`: the test runs the command on the files in
`tests/fixtures/predictions/` and fails if the output differs from this document.
Every other session is still a mock-up whose numbers are illustrative.

**Decisions (2026-11-22):**
1. Bad input exits 6.
2. `metrics` scores the test half by default, and its output says so and mentions
   `--all`.
3. The demo uses generated data from a realistically overconfident model, so it shows
   a calibration warning and a real act/ask split.
4. Targets are labelled TARGET (the user's own) vs PRE-REGISTERED (benchmark
   configs).

**Additions:** confidence semantics (§4.5), scale detection (§4.6), local-only with
a test (§1, §5.4) and performance at 100,000 rows (§8).

ask-or-act's main delivery becomes a CLI (`pip install ask-or-act`). It runs on a
user's own predictions file, not only on our benchmarks. A user who has a model's
predictions, its confidence scores and some human labels should be able to answer one
question in under a minute:

> Above what confidence can I let the model act, and how much will it then handle?

This document fixes the design before any code changes. It covers:

1. the guidelines we follow;
2. what we copy from other CLIs, and what we avoid;
3. the command tree, with an example session for each command;
4. the input format and its validation;
5. exit codes, stdout vs stderr, and the public Python API;
6. the CLI framework.

**Boundaries.**
- Nothing here touches the study: `configs/study/`, `configs/pilot/`, decider
  resolution, prompts, cache keys and example selection stay frozen, and
  `tests/test_study_freeze.py` must keep passing.
- `run`, `study`, `data` and `audit-redaction` keep their names, flags and exit codes
  until the study ends (`scripts/daily.sh` depends on them).
- The new commands make **no network calls**.

All terminal sessions below are **mock-ups**: the numbers are illustrative, not
results.

---

## 1. Guidelines checklist (from clig.dev)

Source: [Command Line Interface Guidelines](https://clig.dev/), read in full. Each item
is a rule we will follow, with the section it comes from. Items marked **(adapted)**
explain where we deliberately do something narrower.

### Basics and output ([#the-basics](https://clig.dev/#the-basics), [#output](https://clig.dev/#output))
- [ ] Exit 0 on success and non-zero on failure. Distinct codes for the failure
      modes that matter (section 5).
- [ ] The result goes to **stdout**: the summary, or JSON with `--json`. Messages,
      warnings, progress and errors go to **stderr**.
- [ ] Human-readable by default. Decide whether a human is reading by checking
      whether the stream is a TTY.
- [ ] `--json` prints machine-readable output: one JSON document on stdout, and
      nothing else on stdout.
- [ ] Brief output on success, never silence that looks like a hang.
- [ ] `-q/--quiet` suppresses everything but the result and errors.
- [ ] When we write a file, say where (`saved: reports/report.html`).
- [ ] Suggest the next command where it helps (`next: ask-or-act report preds.csv`).
- [ ] **Boundary crossings are explicit.**
  - We never read files the user didn't name, apart from our own config file (§3.10).
  - The new commands never talk to a server.
  - `run` keeps its explicit `--dry-run` and `--yes` flow.
- [ ] Use colour with intention: red `error:`, yellow `warning:`, and nothing else.
- [ ] Disable colour when the stream isn't a TTY, checking stdout and stderr
      separately. Also disable it when `NO_COLOR` is set and non-empty, when
      `TERM=dumb`, or with `--color never`. `FORCE_COLOR` or `--color always` turns it
      on.
- [ ] No animations when stderr isn't a TTY (clean CI logs).
- [ ] No emoji. Plain Unicode symbols only (`✓`, `!`), always with words beside them.
- [ ] stderr is not a log file: no `INFO:`/`WARN:` level tags, apart from the
      `error:`/`warning:` prefixes. Detail goes behind `--verbose`.
- [ ] **(adapted)** No pager. Our outputs are short. Long output means `--json` or
      the HTML report.

### Help and documentation ([#help](https://clig.dev/#help), [#documentation](https://clig.dev/#documentation))
- [ ] `-h` and `--help` work on every command. `ask-or-act help <command>` is the same.
- [ ] `ask-or-act` with no arguments shows concise help: what it does, two examples,
      and the most common commands first. Required arguments missing → concise help
      plus the error.
- [ ] **Help leads with examples**, each with a one-line comment (§3 gives the
      examples).
- [ ] Every help page ends with a link to the docs page for that command and to the
      issue tracker.
- [ ] Typo → "did you mean" suggestion. We never auto-run the guess.
- [ ] A command that expects a file never waits silently on an interactive stdin.
      If the user passes `-` while stdin is a TTY, print help and exit.
- [ ] Help text has no ANSI codes when piped.

### Errors ([#errors](https://clig.dev/#errors))
- [ ] Catch expected errors and rewrite them for humans, saying how to fix them.
- [ ] Group repeated errors: one line per kind of problem, with a count and the first
      few line numbers.
- [ ] Put the most important line last (the fix).
- [ ] Unexpected errors: a short message and a pre-filled issue link. The traceback
      appears only with `--verbose`.

### Arguments and flags ([#arguments-and-flags](https://clig.dev/#arguments-and-flags))
- [ ] Prefer flags to positional arguments. The one positional argument is the
      input file(s), like `cp` or `ruff check`.
- [ ] Every flag has a long form. Short forms only for common ones: `-o`, `-q`, `-h`.
- [ ] Standard names: `--json`, `-o/--output`, `-q/--quiet`, `-n/--dry-run`,
      `--version`, `--no-input`.
- [ ] **(adapted)** `--verbose` has no short form, because `-v` is ambiguous (clig
      advises against it).
- [ ] Defaults are right for most users: α = 0.10, δ = 0.10, automatic split.
- [ ] `-` reads from stdin and writes to stdout.
- [ ] Flags can come in any order, before or after the file.
- [ ] Never read secrets from flags. API keys stay in the environment, and only
      `run` uses them (section 6 notes the clig tension).
- [ ] No prompts in the new commands. `run` keeps its `--yes` approval, which is
      required whenever stdin isn't a TTY.

### Subcommands, robustness and future-proofing
([#subcommands](https://clig.dev/#subcommands), [#robustness-guidelines](https://clig.dev/#robustness-guidelines),
[#future-proofing](https://clig.dev/#future-proofing))
- [ ] Same flag names mean the same thing in every subcommand (`--alpha`, `--json`,
      `--split-seed`).
- [ ] No two commands with near-identical names.
- [ ] Validate input fully **before** computing anything. One pass, every problem at
      once.
- [ ] Print something within 100 ms. Show a spinner on stderr only if a step takes
      longer than 1 s and stderr is a TTY.
- [ ] Ctrl-C exits at once with code 130. The new commands write files atomically
      (temp file, then rename), so there is nothing to clean up.
- [ ] Commands, flags, the input format, JSON field names and exit codes are a public
      interface. Changes are additive; deprecations warn first. The `--json` output
      carries a schema version.
- [ ] No catch-all command, and no prefix abbreviations (`allow_abbrev=False`).
      Aliases only when listed.
- [ ] No dependence on outside services.
- [ ] **Local-only.** `check`, `thresholds`, `metrics`, `compare`, `report`, `gate` and
      `demo` make no network connection and send no telemetry. The top-level `--help`
      and the README say so, and a test enforces it (§5.4).

### Configuration and environment ([#configuration](https://clig.dev/#configuration), [#environment-variables](https://clig.dev/#environment-variables))
- [ ] Precedence: **flags > environment variables > project config file > built-in
      defaults**.
- [ ] **(adapted)** No user-level or system-level config until someone needs it.
      If added, it goes in `~/.config/ask-or-act/` (XDG).
- [ ] Environment variables: `NO_COLOR`, `FORCE_COLOR` and `TERM` (standard) only.
      None of our own until a need appears.

### Distribution and analytics ([#distribution](https://clig.dev/#distribution), [#analytics](https://clig.dev/#analytics))
- [ ] `pip install ask-or-act` / `uv tool install ask-or-act`, with numpy as the only
      runtime dependency. The README's install section ends with how to uninstall.
- [ ] **No telemetry and no update checks**, ever, opt-in or not.

---

## 2. What other CLIs do: copy and avoid

Pages read on 2026-11-22. Where a page didn't document something, the table says so
rather than guessing.

| | help & examples | output formats | exit codes | config vs flags | progress / colour |
|---|---|---|---|---|---|
| **gh** | Each command page has an Examples section of commented `$` commands, a JSON-fields list and "See also" ([gh pr list](https://cli.github.com/manual/gh_pr_list)). | Plain text by default. `--json fields` prints JSON. `--json` with no fields **lists the available fields**. `--jq` and `--template` work on the JSON ([formatting](https://cli.github.com/manual/gh_help_formatting)). | 0 ok, 1 failed, 2 cancelled, 4 needs auth; commands may add more ([exit codes](https://cli.github.com/manual/gh_help_exit-codes)). | `gh config set` keys, and env vars for editor, pager, prompts and colour ([config](https://cli.github.com/manual/gh_config), [environment](https://cli.github.com/manual/gh_help_environment)). | `NO_COLOR`, `CLICOLOR`, `CLICOLOR_FORCE`; `GH_SPINNER_DISABLED` swaps the spinner for text ([environment](https://cli.github.com/manual/gh_help_environment)). |
| **uv** | `-h`/`--help` and `uv help <cmd>` ([CLI reference](https://docs.astral.sh/uv/reference/cli/)). | Human text. `-q` quiet, `-qq` silent. | Not documented in the pages read. | Project `pyproject.toml [tool.uv]` or `uv.toml` > user > system. **Env vars override config files, and flags override both**. Also `--no-config` and `--config-file` ([config files](https://docs.astral.sh/uv/concepts/configuration-files/)). | `--color auto\|always\|never`; `--no-progress` / `UV_NO_PROGRESS`; `NO_COLOR` beats `FORCE_COLOR` ([environment](https://docs.astral.sh/uv/reference/environment/)). |
| **ruff** | `-h` short, `--help` long, `ruff help <cmd>`. | `--output-format` with full, concise, json, json-lines, junit, github, gitlab, sarif, … ([configuration](https://docs.astral.sh/ruff/configuration/)). | **0 clean, 1 findings, 2 abnormal (bad config or CLI, internal error)**; `--exit-zero` ([exit codes](https://docs.astral.sh/ruff/linter/#exit-codes)). | Closest `pyproject.toml`/`ruff.toml` wins; CLI flags override; `--config KEY=VALUE`; `--isolated` ([configuration](https://docs.astral.sh/ruff/configuration/)). | `--color auto\|always\|never`. |
| **promptfoo** | `init --example` scaffolds a project ([CLI](https://www.promptfoo.dev/docs/usage/command-line/)). | `-o` with the format inferred from the extension: csv, json, jsonl, yaml, html, xml, junit.xml ([outputs](https://www.promptfoo.dev/docs/configuration/outputs/)). | `eval`: **100 when tests fail**, 1 otherwise; `PROMPTFOO_FAILED_TEST_EXIT_CODE`, `PROMPTFOO_PASS_RATE_THRESHOLD` ([CLI](https://www.promptfoo.dev/docs/usage/command-line/)). | `promptfooconfig.yaml`; flags override its `commandLineOptions` ([config guide](https://www.promptfoo.dev/docs/configuration/guide/)). | `--no-progress-bar`, `--no-table`, `FORCE_COLOR=0`. |
| **Inspect AI** (`inspect eval`) | Long option reference grouped by topic ([options](https://inspect.aisi.org.uk/options.html)). | `--display full\|conversation\|rich\|plain\|log\|none`; `--json` prints JSON lines and implies `--display none` ([inspect eval](https://inspect.aisi.org.uk/reference/inspect_eval.html)). Logs in `.eval`/`.json`, plus `inspect log dump/convert` ([logs](https://inspect.aisi.org.uk/eval-logs.html)). | Not documented in the pages read. | Every flag has an `INSPECT_EVAL_*` env var. Reads `.env` from the cwd **and parent directories**. `--run-config` YAML/JSON, overridden by flags ([options](https://inspect.aisi.org.uk/options.html)). | `--no-ansi`; `NO_COLOR` not mentioned. Built on click ([source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/_cli/main.py)). |
| **lm-eval** | `lm-eval run / ls / validate`; a bare `lm-eval --model …` silently inserts `run` ([interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)). | `--output_path`, `--log_samples`. `--limit` is marked "For testing only". | Not documented. | `--config` YAML, overridden by CLI args. | Not documented. Built on argparse ([source](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/_cli)). |

### What we copy
1. **ruff's exit-code shape**, extended:
   - 0 means the answer is yes;
   - 1 means a clean "no", such as a gate failing;
   - 2 means you called it wrong.

   A crash never looks like a "no" (§5).
2. **gh's `--json`**: one flag on every command, and a documented, versioned schema.
   We skip `--jq`/`--template`, because users can pipe to `jq`.
3. **gh's help pages**: examples first, with comments, then options, then "see also"
   and a docs link.
4. **uv's precedence statement**, written down once: flags > env > project file >
   defaults. `--no-config` ignores the project file.
5. **uv/ruff's `--color auto|always|never`**, plus `NO_COLOR` beats `FORCE_COLOR`.
6. **promptfoo's `init --example` idea**, as `ask-or-act demo`: a working result
   before the user has prepared anything.
7. **promptfoo's `-o` with the format from the extension**, for files only:
   `-o report.html`, `-o thresholds.json`.
8. **lm-eval's `validate`**, as `ask-or-act check`: validation as its own command,
   so CI can run it cheaply.
9. **lm-eval's "for testing only" honesty**, in our wording: `--limit`-style shortcuts
   and too-small samples get a warning in the output itself, not just in the docs.

### What we avoid
1. **Telemetry, update checks and cloud sharing** (promptfoo has update checks and a
   `share` upload). Our users' data may be sensitive, so nothing leaves the machine.
2. **Implicitly reading `.env` from parent directories** (Inspect). It's a silent
   boundary crossing, and it can pick up secrets meant for another project.
3. **An implicit default subcommand** (lm-eval inserts `run`). clig warns this blocks
   future commands.
4. **An env var for every flag** (Inspect's `INSPECT_EVAL_*`). It doubles the
   interface to keep stable, and our runs are short and explicit.
5. **Underscore flag names** (lm-eval's `--model_args`). We use dashes everywhere.
6. **An exit code of 100 for "tests failed"** (promptfoo). It's non-standard; 1 is
   what CI systems and shells expect for "no".
7. **Many display modes** (Inspect has six). We have two outputs, a summary or
   `--json`, plus `-q`.

---

## 3. Command tree

```
ask-or-act
│  Your predictions (CSV/JSONL), no network:
├── check FILE                 validate a predictions file and say what it supports
├── thresholds FILE...         act/ask/escalate thresholds with a guarantee
├── metrics FILE               accuracy, calibration, risk–coverage, κ, with CIs
├── compare FILE_A FILE_B      paired comparison of two models on the same ids
├── report FILE...             one static HTML report
├── gate FILE --policy P       CI pass/fail against a policy (milestone 6)
├── demo                       example data + report in under 30 s
│
│  Benchmark study (unchanged until the study ends):
├── run CONFIG                 call a decider over a benchmark sample
├── study {run,status}         the multi-day schedule
├── data {sample,verify}       benchmark samples
├── judge RUN                  MT-Bench judge analyses
├── audit-redaction [PATHS]    fail if Aegis prompt text is on disk
└── help [COMMAND]
```

**Where the existing commands fit.**
- `run`, `study`, `data` and `audit-redaction` stay exactly as they are. They are
  listed under a separate "Benchmark study" heading in the top-level help, so new
  users see the predictions commands first.
- After the study, they may move under `ask-or-act bench …` with the old names kept
  as aliases. That is a separate decision.
- `judge` stays with the benchmark commands for now. It needs MT-Bench's pairwise
  metadata, which a generic predictions file doesn't have.
- `metrics`, `thresholds` and `report` already exist for run folders. They will accept
  **either** a predictions file **or** a run folder (a folder containing `run.json`).
  Both become a `DecisionTable` (report/table.py), so every result is computed the
  same way.
  - `thresholds --cal RUN --test RUN` keeps working.
  - `metrics RUN_A RUN_B` keeps working but prints a one-line note pointing to
    `compare`.

**Defaults shared by every command:**
- α = 0.10 (the most risk you'll accept among decisions the model acts on) and
  δ = 0.10 (the chance the guarantee itself fails).
- Automatic split (§4.3), `--split-seed 0`.
- 2,000 bootstrap resamples, 95% CIs.

**Shared flags:** `--json`, `-q/--quiet`, `--verbose`, `--color WHEN`,
`--no-config`, `-h/--help`.

### 3.1 `check`: validate a file

A file that is ready: 38 failed decisions and 400 unlabelled rows are warnings, not errors.

<!-- real output (exit 0) -->
```console
$ ask-or-act check support-bot.csv
support-bot.csv: 2,400 rows (CSV)
  ✓ columns: id, prediction, confidence, human_label, split
  ✓ 2,400 unique ids
  ✓ confidence: probabilities in [0, 1], 581 distinct values
  ✓ split column: 1,000 calibration / 1,000 test labelled rows
  ! 38 rows have an empty prediction (lines 36, 52, 82, 149, 208, …): counted as wrong with confidence 0
  ! 400 rows have no human_label: used to estimate coverage only

Ready for metrics, thresholds and report.
Guarantee at α = 0.1, δ = 0.1: needs at least 36 labelled calibration rows; you have 1,000.
  (About 114 if the model is wrong on α/2 of the rows it acts on.)
```

A file with problems: every problem is found in one pass. The columns have other names, so the obvious mapping is still checked, which also finds the percent scale.

<!-- real output (exit 6) -->
```console
$ ask-or-act check scores.csv
scores.csv: 812 rows (CSV)
  ✓ 812 unique ids
  ✓ no split column: 406 calibration / 406 test labelled rows, by hashing each id (seed 0); same file → same split
error: missing required columns "prediction", "confidence", "human_label". Found: id, pred, conf, label
  hint: rename them, or map them: --columns prediction=pred,confidence=conf,human_label=label
error: confidence is outside [0, 1] in 812 rows (lines 2, 3, 4, 5, 6, …); values run 35 to 100
  hint: they look like percentages: pass --confidence-scale percent

2 errors: fix and run check again.
```

The same file, with the fixes the hints gave:

<!-- real output (exit 0) -->
```console
$ ask-or-act check scores.csv --columns prediction=pred,confidence=conf,human_label=label --confidence-scale percent
scores.csv: 812 rows (CSV)
  ✓ columns: id, prediction ← pred, confidence ← conf, human_label ← label
  ✓ 812 unique ids
  ✓ confidence: percentages 0–100, 65 distinct values
  ✓ no split column: 406 calibration / 406 test labelled rows, by hashing each id (seed 0); same file → same split

Ready for metrics, thresholds and report.
Guarantee at α = 0.1, δ = 0.1: needs at least 36 labelled calibration rows; you have 406.
  (About 114 if the model is wrong on α/2 of the rows it acts on.)
```

A binary classifier whose score is P(unsafe) (§4.5):

<!-- real output (exit 0) -->
```console
$ ask-or-act check moderation.jsonl
moderation.jsonl: 600 rows (JSONL)
  ✓ columns: id, prediction, confidence, human_label
  ✓ 600 unique ids
  ✓ confidence: probabilities in [0, 1], 435 distinct values
  ✓ no split column: 300 calibration / 300 test labelled rows, by hashing each id (seed 0); same file → same split
  ! 68% of answered rows have confidence below 0.5. With two labels (safe, unsafe), that looks like P(unsafe), not confidence in the predicted answer
    hint: if it is, pass --confidence-of positive --positive unsafe

Ready for metrics, thresholds and report.
Guarantee at α = 0.1, δ = 0.1: needs at least 36 labelled calibration rows; you have 300.
  (About 114 if the model is wrong on α/2 of the rows it acts on.)
```

Too few labelled rows for a guarantee (§4.4):

<!-- real output (exit 0) -->
```console
$ ask-or-act check pilot.csv
pilot.csv: 40 rows (CSV)
  ✓ columns: id, prediction, confidence, human_label
  ✓ 40 unique ids
  ✓ confidence: probabilities in [0, 1], 27 distinct values
  ✓ no split column: 19 calibration / 21 test labelled rows, by hashing each id (seed 0); same file → same split
  ! only 21 labelled rows in the test half: its risk estimate will have a wide confidence interval
    hint: label more rows if you can
  ! only 19 labelled calibration rows: a guarantee at α = 0.1, δ = 0.1 needs at least 36, so thresholds will refuse
    hint: label at least 17 more rows, or loosen the target: --alpha 0.2 needs 17

Ready for metrics and report.
```

`check --json` prints one document:
`{"schema": "ask-or-act/check/v1", "version": …, "input": {…}, "result": {"ok": …, "errors": […], …}, "warnings": […]}`.
`--strict` exits 1 when there are warnings. `-q` keeps only errors and the verdict.

### 3.2 `thresholds`: when to act

Real output on the fixture files, which are synthetic: `support-bot.csv` is an overconfident
8-intent router (65% accurate), and `support-bot-large.csv` a stronger, better-calibrated
model on the same rows. Warnings about the input go to stderr, and the result goes to stdout.

<!-- real output (exit 0) -->
```console
$ ask-or-act thresholds support-bot-large.csv
warning: support-bot-large.csv: 38 rows have an empty prediction (lines 36, 52, 82, 149,
  208, …): counted as wrong with confidence 0
warning: support-bot-large.csv: 400 rows have no human_label: used to estimate coverage
  only
support-bot-large.csv: 1,000 calibration / 1,000 test labelled rows (split column)

Act when confidence ≥ 0.879; ask a human below. Covers 28% of decisions.

  Guarantee: at most 10% of acted-on decisions are wrong (α = 0.1), unless the
    calibration sample was unlucky, which has probability ≤ 10% (δ = 0.1).
  Test check: 6.2% wrong when acting [3.6%–9.7%], covering 28%: within α.
  Unlabelled rows: this would act on 114 of 400 (28%).
  No rule can reach α = 0.1 without escalating at least 11% of decisions.
```

The first result line is the answer, the one a user would paste into a ticket. "Covers"
is the share acted on in the test half. The last line is the Kotte feasibility floor
(arXiv:2606.29054 v2, Prop. 3): what *any* valid rule must escalate at this α.

**Nothing certified** is a valid answer, not an error: exit 0, and `"certified": false`
in the JSON. The output names the pre-registered scan's closest attempt and how many
rows it would have needed.

<!-- real output (exit 0) -->
```console
$ ask-or-act thresholds support-bot.csv -q
support-bot.csv: 1,000 calibration / 1,000 test labelled rows (split column)

Act on nothing: no confidence threshold keeps errors at or below 10% with δ = 0.1.
  Closest attempt: 130 calibration rows with confidence ≥ 0.999 had 8 wrong (6.2%). At
    that error rate, proving ≤ 10% with δ = 0.1 takes about 193 such rows.
  Escalate everything, label more rows, or try a better-calibrated confidence.
```

**Two models (act / ask / escalate):** the files are joined on `id`, cheapest first, in
the order the user gives (never guessed). δ is split over the stages (ToE Algorithm 2).

<!-- real output (exit 0) -->
```console
$ ask-or-act thresholds support-bot.csv support-bot-large.csv --alpha 0.2 -q
support-bot.csv, support-bot-large.csv: joined on id; 1,000 calibration / 1,000 test
  labelled rows (split column)

support-bot acts when confidence ≥ 0.801 (40% of decisions);
otherwise ask support-bot-large, which acts when confidence ≥ 0.715 (35% of decisions);
escalate the remaining 24% to a human.

  Guarantee: at most 20% of acted-on decisions are wrong (α = 0.2), unless the
    calibration sample was unlucky, which has probability ≤ 10% (δ = 0.1). δ is split
    over the stages.
  Test check: 14.4% wrong when acting [12.0%–17.1%], covering 76%: within α.
  Unlabelled rows: this would act on 329 of 400 (82%).
  No rule can reach α = 0.2 without escalating at least 15% of decisions.
```

**Not enough labelled data:** refused before anything is computed, with exit code 7 (§4.4).

<!-- real output (exit 7) -->
```console
$ ask-or-act thresholds pilot.csv
error: too few labelled calibration rows for a guarantee at α = 0.1, δ = 0.1.
  You have 19. Even if the model made no mistakes, the test needs at least 36 rows the
    model acts on; with errors on α/2 of them, about 114.
  hint: label at least 17 more rows, or loosen the target: --alpha 0.2 needs 17
```

Flags: `--alpha`, `--delta`, `--cal FILE --test FILE` (explicit halves, each used whole),
`--split-seed`, `--cal-fraction` (default 0.5), `--confidence-scale`, `--confidence-of`,
`--columns`, `--disjoint`, `--json`, `-q`. With run folders, `--cal RUN --test RUN` keeps
the benchmark report (pre-registered targets, MT-Bench options).

### 3.3 `metrics`: how good and how calibrated

<!-- real output (exit 0) -->
```console
$ ask-or-act metrics support-bot.csv -q
support-bot.csv, test half (--all scores every labelled row): 1,000 labelled rows; 20
  failed decisions count as wrong

Accuracy 65.1% [62.2%–68.0%]. Agreement with humans: κ = 0.60 [0.57–0.63].
Calibration: confidence overstates accuracy by about 8 points (mean confidence 72.8%).
  ECE 0.078 [0.056–0.105]; a perfectly calibrated model would show about 0.030 at this n
    (95th percentile 0.043), so this is beyond noise.
Ranking: the most confident half is 82.8% right; AUROC 0.76 [0.73–0.79].
Confidence has 465 distinct values.
```

The metrics are computed on the **test half**: the calibration half is only for fitting
thresholds, so it isn't scored here (CLAUDE.md, M2 notes). The first line always says
which rows were scored; for the test half it adds "(--all scores every labelled row)",
and with `--all` it says "all labelled rows". "Beyond noise" compares the ECE with the
95th percentile of what a perfectly calibrated model with the same confidences would
show at this n (the noise floor). `--json` returns every metric with its CI, the noise
floor, the reliability bins and the risk–coverage curve. One file only: two files point
to `compare`.

### 3.4 `compare`: is model B better than model A?

<!-- real output (exit 0) -->
```console
$ ask-or-act compare support-bot.csv support-bot-large.csv -q
Paired on 1,000 test-half ids in both files: A = support-bot, B = support-bot-large.

support-bot-large is more accurate than support-bot: 75.1% vs 65.1% (+10.0 points [+6.3,
  +13.9]; exact McNemar p = 5e-07, Holm-adjusted p = 4.5e-06).
support-bot-large is better on calibration (ECE) than support-bot: 0.022 vs 0.078
  (-0.056 [-0.073, -0.015]; paired bootstrap p = 0.0005, Holm-adjusted p = 0.004).
With accuracy controlled (ACE, DA view), support-bot is better calibrated: ECE 0.078 vs
  0.101 (A − B -0.023 [-0.036, -0.003]); this reverses the raw ECE comparison, which
  partly reflects accuracy.
support-bot-large is better on Brier score than support-bot: 0.161 vs 0.188 (-0.028
  [-0.046, -0.009]; paired bootstrap p = 0.004, Holm-adjusted p = 0.02).
At α = 0.1: support-bot acts on nothing, support-bot-large acts on 28%.
```

The tests are exact McNemar for accuracy, a centered paired bootstrap for the others and
Holm correction over the metric family, as `metrics RUN_A RUN_B` already uses. The
"accuracy controlled" line is ACE's DA view (arXiv:2606.30814 v2, §4.2, the paper's
default). It reweights the more accurate model to the other's accuracy before comparing
ECE, so higher accuracy alone can't make a model look better calibrated. Here it
reverses the raw comparison: support-bot-large's lower raw ECE partly reflects its
higher accuracy. When ACE and raw ECE disagree, the output says so. Files
whose ids don't match exit with code 6, with the counts ("N only in A, M only in B").

### 3.5 `report`: one HTML page

<!-- real output (exit 0) -->
```console
$ ask-or-act report support-bot-large.csv -o report.html -q
saved: report.html (1 model, 2,000 labelled rows; opens offline, no external files)
```

<!-- real output (exit 0) -->
```console
$ ask-or-act report support-bot.csv support-bot-large.csv -o compare.html -q
saved: compare.html (2 models, paired on 1,000 ids; opens offline, no external files)
```

One file gives one model's page; two or three files give a side-by-side page, paired
on the same ids as `compare` (different ids exit 6). Everything comes from the public
API (`report/files.py`); run folders keep the milestone 5 benchmark report. The page:

1. **Summary first:** the exact text of `thresholds` and `metrics` (one file) or of
   `compare` (several), with the command that prints it. A test compares the page
   with the terminal word for word (`tests/test_report_files.py`).
2. **Thresholds:** a table at the TARGET α, plus other α values for context. For
   several files, each model's `thresholds` text and table, plus the command that
   chains them into a cascade.
3. **Calibration and ranking:**
   - the metrics table with CIs and the ECE noise floor;
   - the risk–coverage curve on the test half, with the chosen threshold ringed on the
     curve (a dashed drop shows its coverage) and α as a dashed line;
   - one reliability diagram per model.
4. **Unlabelled rows:** how many the threshold would act on, next to its test-half
   coverage, and when that estimate can be trusted.
5. **Notes on the input,** and how to read the page.

Each threshold row is labelled with how its target was chosen:
- **TARGET**: the user's own `--alpha`, or the default (the page says which).
- **PRE-REGISTERED**: only for benchmark runs whose config registered α in advance.
  A user's file never gets it.
- **EXPLORATORY**: every other α shown.

The page is one file with inline styles and SVG charts. It loads no scripts, fonts or
images from elsewhere (tested), and holds ids, file names and numbers, never input
text. Too few labelled rows still gives a page (exit 0); the page shows what's needed
and stderr says so. Examples from the fixtures: `reports/examples/` (not committed;
regenerate with the commands above).

### 3.6 `gate` (milestone 6): pass or fail in CI

`gate NEW --baseline BASE` pairs the two files on the same ids and exits 1 only when NEW
is worse than BASE by more than a tolerance, beyond noise. A regressed file (8 points
less accurate):

<!-- real output (exit 1) -->
```console
$ ask-or-act gate gate/regressed.csv --baseline gate/base.csv --no-record --expect "accuracy -1 to +1 points"
gate: regressed.csv vs baseline base.csv, paired on 1,011 test / 989 calibration ids (α
  = 0.1, δ = 0.1; tolerances: accuracy 1 point, calibration 0.02, coverage 5 points)
  ✗ accuracy     74.3% vs 83.8%: -9.5 points [-12.4, -6.6], worse beyond the 1-point
    tolerance (Holm p = 2e-06)
  ✓ calibration  ECE 0.058 vs 0.085 accuracy-controlled: -0.027 [-0.057, -0.005], within
    the 0.02 tolerance
  ✗ coverage     0% vs 70% at α = 0.1: -70.1 points [-81.7, -63.5], worse beyond the
    5-point tolerance (Holm p = 0.001)
  ✗ certified    lost: new: nothing certified; baseline: act ≥ 0.778 (α = 0.1, δ = 0.1)
FAIL: 3 of 4 checks failed (accuracy, coverage, certified).
expected:
  accuracy -1 to +1 points: missed (observed -9.5 points, beyond noise)
Intervals are 90% (two one-sided 0.05 tests); the paired checks' p-values are
  Holm-adjusted. Coverage refits both thresholds in each of 2,000 resamples
  (approximate). 1.8 s.
```

Another run of the same model passes. Two checks are *inconclusive*: with ~1,000 test
rows, neither "worse" nor "within 1 point" can be shown, and that passes by default:

<!-- real output (exit 0) -->
```console
$ ask-or-act gate gate/candidate.csv --baseline gate/base.csv --no-record
gate: candidate.csv vs baseline base.csv, paired on 1,011 test / 989 calibration ids (α
  = 0.1, δ = 0.1; tolerances: accuracy 1 point, calibration 0.02, coverage 5 points)
  ? accuracy     82.3% vs 83.8%: -1.5 points [-4.1, +1.1], inconclusive: could be up to
    4.1 points worse (tolerance 1 point)
  ✓ calibration  ECE 0.021 vs 0.048 accuracy-controlled: -0.027 [-0.058, -0.017], within
    the 0.02 tolerance
  ? coverage     60% vs 70% at α = 0.1: -10.5 points [-33.7, +11.9], inconclusive: could
    be up to 33.7 points worse (tolerance 5 points)
  ✓ certified    new: act ≥ 0.83; baseline: act ≥ 0.778 (α = 0.1, δ = 0.1)
PASS, with 2 inconclusive (accuracy, coverage): not shown worse, not shown within the
  tolerance (--fail-on-inconclusive fails these).
Intervals are 90% (two one-sided 0.05 tests); the paired checks' p-values are
  Holm-adjusted. Coverage refits both thresholds in each of 2,000 resamples
  (approximate). 1.8 s.
```

`--fail-on-inconclusive` (strict mode) fails those, and prints the test rows each
check would need to be provable:

<!-- real output (exit 1) -->
```console
$ ask-or-act gate gate/candidate.csv --baseline gate/base.csv --no-record --fail-on-inconclusive -q
gate: candidate.csv vs baseline base.csv, paired on 1,011 test / 989 calibration ids (α
  = 0.1, δ = 0.1; tolerances: accuracy 1 point, calibration 0.02, coverage 5 points;
  strict)
  ? accuracy     82.3% vs 83.8%: -1.5 points [-4.1, +1.1], inconclusive: could be up to
    4.1 points worse (tolerance 1 point)
  ✓ calibration  ECE 0.021 vs 0.048 accuracy-controlled: -0.027 [-0.058, -0.017], within
    the 0.02 tolerance
  ? coverage     60% vs 70% at α = 0.1: -10.5 points [-33.7, +11.9], inconclusive: could
    be up to 33.7 points worse (tolerance 5 points)
  ✓ certified    new: act ≥ 0.83; baseline: act ≥ 0.778 (α = 0.1, δ = 0.1)
FAIL: 2 of 4 checks failed (accuracy, coverage).
strict: test rows needed to show each check within its tolerance (approx., 80% chance
  when nothing changed): accuracy ~15,805, calibration ~630, coverage ~49,951; this run
  has 1,011.
Intervals are 90% (two one-sided 0.05 tests); the paired checks' p-values are
  Holm-adjusted. Coverage refits both thresholds in each of 2,000 resamples
  (approximate). 1.8 s.
```

Strict mode is meant for large production logs. For most teams the default mode (fails
only on proven drops, reports inconclusive checks openly) is the right choice. Strict
mode becomes usable with suitable settings: with accuracy 3 points and coverage 10
points, the same files need ~1,757 / ~630 / ~12,488 rows, a size a production log
reaches:

<!-- real output (exit 1) -->
```console
$ ask-or-act gate gate/candidate.csv --baseline gate/base.csv --no-record --fail-on-inconclusive -q --tolerance accuracy=0.03,coverage=0.10
gate: candidate.csv vs baseline base.csv, paired on 1,011 test / 989 calibration ids (α
  = 0.1, δ = 0.1; tolerances: accuracy 3 points, calibration 0.02, coverage 10 points;
  strict)
  ? accuracy     82.3% vs 83.8%: -1.5 points [-4.1, +1.1], inconclusive: could be up to
    4.1 points worse (tolerance 3 points)
  ✓ calibration  ECE 0.021 vs 0.048 accuracy-controlled: -0.027 [-0.058, -0.017], within
    the 0.02 tolerance
  ? coverage     60% vs 70% at α = 0.1: -10.5 points [-33.7, +11.9], inconclusive: could
    be up to 33.7 points worse (tolerance 10 points)
  ✓ certified    new: act ≥ 0.83; baseline: act ≥ 0.778 (α = 0.1, δ = 0.1)
FAIL: 2 of 4 checks failed (accuracy, coverage).
strict: test rows needed to show each check within its tolerance (approx., 80% chance
  when nothing changed): accuracy ~1,757, calibration ~630, coverage ~12,488; this run
  has 1,011.
Intervals are 90% (two one-sided 0.05 tests); the paired checks' p-values are
  Holm-adjusted. Coverage refits both thresholds in each of 2,000 resamples
  (approximate). 1.8 s.
```

**The checks.**
- accuracy: Tango's (1998) score test, Eq. (24)–(26), with the tolerance as its margin;
- calibration: the ACE gap, accuracy-controlled;
- coverage at α: both thresholds refitted in every bootstrap resample (approximate);
- certified: a lost certificate fails.

The three paired checks are Holm-adjusted. What the 5% false-FAIL bound covers, and
what it doesn't (the tolerance edge, the certification check), is in docs/gate.md.

**The policy** lives in the repository: `ask-or-act.toml`, or `[tool.ask-or-act.gate]` in
`pyproject.toml`. It's found from the current folder up to the repository root, never
above, and flags override it. `--no-config` ignores it.

```toml
# ask-or-act.toml
[gate]
baseline = "eval/baseline.csv"   # relative to this file
alpha = 0.10
delta = 0.10
tolerance = { accuracy = 0.01, calibration = 0.02, coverage = 0.05 }  # the defaults
level = 0.05                     # one-sided, Holm-adjusted
require_certified = true         # false: a lost certificate only warns
strict = false                   # true: inconclusive checks fail
require_expectation = false      # true: a run without an expected effect fails
changelog = "ask-or-act-changelog.jsonl"
registered = "2026-12-01"        # when this policy was fixed (UTC)
```

Unknown keys and wrong types exit 2 and name every problem.

**The decision changelog** (after Self-Improvement Memory, arXiv 2611.18856) is one
JSONL file in the repository. Every gate run appends an entry:
- what changed and why;
- the expected effect;
- the observed metrics, and whether each prediction held;
- the commit, and hashes of both files.

The expected effect counts only if it was written **before** the run:

```console
$ ask-or-act changelog expect --change "prompt v4" --expect "accuracy -1 to +1 points"
recorded 2026-12-02T09:14:03Z-5c1e in ask-or-act-changelog.jsonl: the next gate run checks it
$ ask-or-act gate new.csv            # picks up the open expectation, then closes it
$ ask-or-act changelog --last 1
2026-12-02T09:20:41Z  PASS  prompt v4  (3f2a91c04b7e)
  expected (recorded 6 min before the run):
    accuracy -1 to +1 points → +0.4 points: held

1 gate run(s) (reruns not counted); predictions held 1 of 1.
```

- `--expect` also works on the gate command itself; the entry then says "given with
  the gate run".
- Running the same files under the same policy again is marked as a rerun, and only
  the first run's prediction counts. So re-running until the prediction holds doesn't
  help.
- `--no-record` doesn't append. The entry is still in `--json` output, and
  `changelog add gate.json` appends it later (that's how CI works).

**`-o gate.md`** also writes the result as Markdown, for a PR comment or a job summary.

**In CI:** a composite GitHub Action (`action.yml`, `uses: samadarsh/ask-or-act@SHA`)
runs the gate, posts the Markdown on the pull request and uploads gate.json. The job
fails when the gate fails. Setup and permissions are in docs/gate.md; a workflow to
copy is in examples/github/gate.yml.

### 3.7 `demo`: something working in under 30 seconds

<!-- real output (exit 0) -->
```console
$ ask-or-act demo
Wrote ask-or-act-demo/support-tickets.csv: 2,500 synthetic predictions from a
  support-ticket router (2,000 with a human label, 500 new tickets without one). Ids,
  labels and scores only, generated from seed 0; no download.

support-tickets.csv: 985 calibration / 1,015 test labelled rows (hashed split, seed 0)

Act when confidence ≥ 0.925; ask a human below. Covers 59% of decisions.

  Guarantee: at most 10% of acted-on decisions are wrong (α = 0.1), unless the
    calibration sample was unlucky, which has probability ≤ 10% (δ = 0.1).
  Test check: 7.7% wrong when acting [5.7%–10.1%], covering 59%: within α.
  Unlabelled rows: this would act on 323 of 500 (65%).
  No rule can reach α = 0.1 without escalating at least 3% of decisions.

support-tickets.csv, test half (--all scores every labelled row): 1,015 labelled rows

Accuracy 82.3% [79.9%–84.5%]. Agreement with humans: κ = 0.80 [0.77–0.82].
Calibration: confidence overstates accuracy by about 8 points (mean confidence 90.0%).
  ECE 0.079 [0.062–0.103]; a perfectly calibrated model would show about 0.021 at this n
    (95th percentile 0.031), so this is beyond noise.
Ranking: the most confident half is 92.6% right; AUROC 0.77 [0.74–0.81].
Confidence has 295 distinct values.

saved: ask-or-act-demo/report.html (1 model, 2,000 labelled rows; opens offline, no external files)

Try next:
  ask-or-act check ask-or-act-demo/support-tickets.csv
  ask-or-act thresholds ask-or-act-demo/support-tickets.csv --alpha 0.05   # stricter target, less coverage
  ask-or-act metrics ask-or-act-demo/support-tickets.csv
Your own file needs: id, prediction, confidence, human_label (ask-or-act check --help).
```

The summaries are the same text `thresholds` and `metrics` print for the file (a test
checks it). The report is `report` on the same file, titled as the demo.

**Demo data** (`src/ask_or_act/demo.py`):
- It's generated, not downloaded: a fixed-seed simulation of a ticket router whose
  model is **realistically overconfident**. Each ticket's chance of being routed right,
  p, comes from Beta(5, 1); the stated confidence is p pushed outward on the log-odds
  scale (logit(conf) = 1.3·logit(p) + 0.4, plus noise). Confidence overstates accuracy
  by about 8 points, so the demo shows a calibration warning, and a threshold that
  acts on part of the data (59%) and asks a human for the rest. Confidences are spread
  out, with no pile-up at one value. It contains no text at all, which keeps the wheel
  small and avoids any licence question.
- 2,000 labelled tickets and 500 new, unlabelled ones, so the report also shows the
  coverage estimate on unlabelled data. No split column: the hashed split is used, as
  for most users' files.
- It never overwrites an existing folder: if one exists, the demo writes
  `ask-or-act-demo-2/` (then `-3`, …). `--dir` picks another name.
- The 30-second budget is a CI test. The demo takes about 1 s with the full 2,000
  bootstrap resamples, so it doesn't lower them.

### 3.8 Benchmark commands (unchanged)

The sessions are as today: see CLAUDE.md, "Commands". The only visible change is their
heading in the top-level help.

### 3.9 Top-level help

```console
$ ask-or-act
ask-or-act: when can an AI's decisions be trusted, and when should a human decide?

  ask-or-act check preds.csv           # is my file usable?
  ask-or-act thresholds preds.csv      # act above what confidence?

Your predictions (CSV or JSONL with id, prediction, confidence, human_label):
  check, thresholds, metrics, compare, report, gate, demo
Benchmark study:
  run, study, data, judge, audit-redaction

ask-or-act help <command> for details. New here? ask-or-act demo
Docs: https://github.com/<owner>/ask-or-act/blob/main/docs/cli.md
```

### 3.10 Config file
- **The file:**
  - `ask-or-act.toml` in the current directory, or `[tool.ask-or-act]` in
    `pyproject.toml`. If both exist, `ask-or-act.toml` wins, as with uv.
  - The search goes upward to the repository root (the nearest `.git`), never above
    it. That stops it picking up a config from an unrelated parent folder, the
    Inspect `.env` problem.
- **What it holds:** `[defaults]` (alpha, delta, split seed, columns) and `[gate]`.
- **When it's used:** the first line of output names the file
  (`config: ask-or-act.toml`) whenever it changed a default. Using the file is never
  silent.
- **Precedence:** flags > `NO_COLOR`/`FORCE_COLOR` > the file > built-in defaults.
  `--no-config` ignores the file.

---

## 4. Input format

### 4.1 Columns

| column | required | type | meaning |
|---|---|---|---|
| `id` | yes | string, unique | joins files (`compare`, cascades) and is what reports show |
| `prediction` | yes | string; empty = the decision failed | the model's answer |
| `confidence` | yes | number in [0, 1] (see `--confidence-scale`) | the model's estimate that its answer is right |
| `human_label` | yes (the column); a cell may be empty | string | the correct answer per a human; empty = unlabelled |
| `options` | no | `\|`-separated in CSV, a list in JSONL | the allowed answers for this row |
| `split` | no | `calibration` or `test` | fixes which half a row is in |
| `group` | no | string | rows that must stay together: same user, same conversation, same question |

- **Formats:**
  - `.csv` (RFC 4180, UTF-8, BOM tolerated), `.tsv` and `.jsonl` (one object per line).
  - `-` reads stdin, with `--format csv|tsv|jsonl`.
  - Other columns and keys are ignored, and `check` lists them.
- **Mapping columns:** `--columns prediction=pred,confidence=score`. The same
  mapping can live in the config file.
- **Confidence scales** (`--confidence-scale`). Each maps to an existing
  pre-registered threshold grid in `guarantees/fixed_sequence.py`, which stays frozen:

  | scale | values | converted to | grid |
  |---|---|---|---|
  | `probability` (default) | 0–1 | as is | continuous, 0.001 steps |
  | `percent` | 0–100 | ÷ 100 | percent steps (the verbalized grid) |
  | `votes:K` | integers 0–K (agreeing samples out of K) | ÷ K | K + 1 steps (self-consistency) |
  | `logprob` | ≤ 0 (natural-log probability of the answer) | exp | continuous |

  Unbounded scores, such as logits or margins, are not probabilities. They are
  refused, with a hint to calibrate them first (for example, with a sigmoid fitted on
  held-out data).
- **Correct** means `prediction == human_label` after trimming spaces. It's
  case-sensitive, but `check` warns when the two differ only by case.

### 4.2 Validation

Validation is one pass over the file, before any computation. Each problem is
reported once, with a count, the first five line numbers and a hint.

**Errors** (exit code 6; nothing is computed):

| problem | message (shortened) |
|---|---|
| file unreadable or empty | `cannot read scores.csv: no such file` / `scores.csv has a header but no rows` |
| missing required column | `missing required column "human_label". Found: …` + `--columns` hint; "did you mean `label`?" when a close name exists |
| duplicate id | `id "t-0042" appears 3 times (lines 43, 980, 1201)` |
| confidence not a number / NaN / inf | `confidence is not a number in 4 rows (lines …): "high", "", …` |
| confidence outside [0, 1] | as in §3.1, with the percentage hint when values fall in 0–100 |
| prediction/label outside `options` | `prediction "refund_status" is not in this row's options (line 77)` |
| unknown `split` value | `split must be calibration or test; found "dev" in 120 rows` + `hint: map dev → calibration with --split-map dev=calibration` |
| one group in both splits | `group "user-17" has rows in both calibration and test (lines …): that leaks information between the halves` |
| JSONL line isn't an object | `line 12 is not a JSON object` |
| files don't share ids (`compare`, cascades) | `812 ids only in A, 9 only in B; paired comparison needs the same ids` |

**Warnings** (computation continues; `--strict` turns them into errors):

| warning | why it matters |
|---|---|
| empty `prediction` in N rows | they count as wrong with confidence 0 (the existing policy) |
| empty `human_label` in N rows | excluded from metrics and thresholds; used only to estimate coverage |
| fewer than 20 distinct confidence values | thresholds are coarse; typical of verbalized confidence (arXiv:2608.04899) |
| every confidence identical | no threshold can separate anything |
| labels and predictions differ only by case/spacing | probably a formatting mismatch, not model errors |
| test half under 100 labelled rows | the test check's CI will be wide |
| confidence below 0.5 in ≥ 5% of answered rows of a two-label task | it looks like P(positive class), not confidence in the predicted answer (§4.5) |
| confidence below 1/k in ≥ 5% of answered rows of a k-label task (k ≥ 3) | an argmax answer can't have less than 1/k; check what the column means (§4.5) |
| labelled calibration rows below the minimum for a guarantee (§4.4) | `thresholds` will refuse; `check` says how many are needed |

### 4.3 Without a split column
A guarantee needs calibration data that the threshold is *fitted* on and test data it
is *checked* on, kept apart. Without a `split` column:
- Each row (or each `group`, if given) goes into calibration or test by hashing the
  id: `sha256(f"{seed}:{group or id}")`, then
  `calibration if hash < cal_fraction`.
- The default is 50/50 with seed 0.
- **It's reproducible:** the same id always lands in the same half. Adding rows,
  removing rows or reordering the file never moves the existing rows, which a seeded
  shuffle would.
- **It's said out loud:**
  `No split column: 1,000 calibration / 1,000 test, assigned by hashing each id (seed 0). Same file → same split.`
- `--write-split out.csv` saves the file with the split column added, so the user can
  freeze it.
- **The rule is pre-registered:** the split rule may not depend on labels or scores.
  A user who re-runs with new seeds until a result looks good voids the guarantee.
  The help text for `--split-seed` says so.

`--cal FILE --test FILE` skips splitting altogether.

### 4.4 Too few labelled rows: refuse, and say how many are needed
The minimum comes from the test we already use: the exact binomial test from Trust or
Escalate (arXiv:2407.18370 v1, Eq. 4 and Theorem 1), run as a fixed-sequence scan
(Kotte, arXiv:2606.29054 v2, Lemma E.1). For probability scores it runs from 4
pre-registered starts at δ/4 each. To certify any threshold, the rows it acts on must
pass

  P(Bin(n, α) ≤ errors) ≤ δ / starts.

With zero errors that is (1 − α)ⁿ ≤ δ/4. The minimum n, computed with our own
`binomial.p_value` (4 starts, δ = 0.10):

| α | 0 errors | errors at α/2 | errors at 3α/4 |
|---|---|---|---|
| 0.01 | 368 | 1,164 | 5,325 |
| 0.05 | 72 | 230 | 1,013 |
| 0.10 | 36 | 114 | 492 |
| 0.20 | 17 | 48 | 219 |
| 0.25 | 13 | 38 | 165 |

How we use the table:
- **Refuse** (exit 7) when the labelled calibration rows are fewer than the 0-errors
  column. No threshold can be certified then, whatever the model does.
- **Below the α/2 column, warn** that certification is unlikely, but still run.
- The message gives the user's count, the two minimums and the α that their data
  could support (§3.2).
- The numbers are computed live by `binomial.p_value`, never hard-coded, so they stay
  right if the grid kind changes the number of starts. (Self-consistency uses 1 start:
  22 rows at α = 0.10.)

### 4.5 What "confidence" means
**Confidence is the model's probability that its predicted answer is correct.** For a
model that picks the most likely of k answers, it is at least 1/k. This is what
thresholds act on: "act when the model is at least 87% sure of *its answer*".

Many binary classifiers instead output **P(positive class)**, for example
P(unsafe) = 0.03 for a "safe" prediction. Read as confidence in "safe", that 0.03
would be backwards. So, for two-label tasks:

- `--confidence-of predicted` (the default): the column is confidence in the
  prediction.
- `--confidence-of positive --positive LABEL`: the column is P(LABEL). The confidence
  in the prediction becomes p when the prediction is LABEL, and 1 − p otherwise.
  - With labels `0`/`1` or `false`/`true`, the positive label defaults to `1` / `true`.
  - Otherwise `--positive` is required, and the error lists the two labels.
  - More than two labels is an error, because P(positive) only makes sense for two
    classes.

**Detection.** With the default, `check` warns when ≥ 5% of answered rows of a
two-label task have confidence below 0.5. A model can't be less than 50% sure of the
answer it chose between two. The warning names the label that looks positive: the one
whose predictions carry the higher mean confidence. It also gives the exact flags:

```
! 41% of answered rows have confidence below 0.5. With two labels (safe, unsafe),
  that looks like P(unsafe), not confidence in the predicted answer.
  hint: if it is, pass --confidence-of positive --positive unsafe
```

### 4.6 Detecting the scale
A file whose confidences don't fit the chosen scale is refused (exit 6), and the hint
names the scale the values do fit. With the default `probability`:

| what the values look like | hint |
|---|---|
| all within 0–100, some above 1 | `they look like percentages: pass --confidence-scale percent` |
| whole numbers 0–K with K ≤ 20 | `they look like vote counts out of K: pass --confidence-scale votes:K` (also mentions percent) |
| all ≤ 0 | `they look like log-probabilities: pass --confidence-scale logprob` |
| negative and positive, or above 100 | `they look like unbounded scores (logits or margins), not probabilities: calibrate them first` |

The detector only suggests. It never changes the scale on its own: a silent guess
would change which threshold grid applies, and so the guarantee.

---

## 5. Exit codes, streams and the Python API

### 5.1 Exit codes

| code | meaning | used by |
|---|---|---|
| 0 | success. The answer may still be "act on nothing"; read the output | all |
| 1 | a clean "no": the gate failed, or `check --strict` found warnings | `gate`, `check --strict` |
| 2 | usage error: unknown flag, bad value, missing argument (argparse's own) | all |
| 3 | budget exceeded *(existing)* | `run`, `study` |
| 4 | needs approval: rerun with `--yes` *(existing)* | `run`, `study` |
| 5 | daily quota reached; rerun later resumes from the cache *(existing; `daily.sh` relies on it)* | `run`, `study` |
| 6 | the input data is invalid (unreadable, failed validation, ids don't match) | predictions commands |
| 7 | too few labelled rows for the requested guarantee | `thresholds`, `gate` |
| 70 | internal error (a bug), with a pre-filled issue link (`EX_SOFTWARE` from sysexits.h) | all |
| 130 | interrupted (Ctrl-C) | all |

- Codes 3–5 keep their meaning.
- Today `metrics`, `report` and `thresholds` return 2 for bad input files. That
  becomes 6, so 2 means only "you called the command wrong". This is the one
  behaviour change to existing commands, and none of our scripts rely on it.
- A crash gets 70, never 1, so CI never mistakes a bug for a failed gate.

### 5.2 stdout and stderr
- **stdout** holds the result only: the summary (§3), or exactly one JSON document
  with `--json`. With `-o FILE`, the file is the result and stdout gets one line.
  `saved:` lines go to stderr.
- **stderr** holds everything else: the "read N rows" line, split notes, warnings,
  errors, the spinner. So `ask-or-act thresholds x.csv --json | jq .threshold`
  always works.
- `-q` hides the stderr notes and warnings. Errors still show.
- **The JSON envelope:**
  `{"schema": "ask-or-act/<command>/v1", "version": "<package version>", "input": {...}, "result": {...}, "warnings": [...]}`.
  - Fields are only ever added.
  - Removing or renaming one bumps `v1` → `v2`, with a deprecation release first.
- **Privacy:** outputs never echo input values other than ids, labels and numbers.
  Error messages quote at most a 20-character snippet of a bad cell. The rule
  against showing unsafe text applies to user files too: we don't know what they
  contain.

### 5.3 Public Python API
The CLI is a thin layer: every command parses flags, calls one of these functions and
formats the result. `result.to_json()` *is* the `--json` output, so the CLI and the
API can't drift apart.

```python
import ask_or_act as aoa

table = aoa.read_predictions("preds.csv")  # or a dict of columns, or a DecisionTable
report = aoa.check("preds.csv")  # CheckResult: .ok, .errors, .warnings, .table

th = aoa.find_thresholds(table, alpha=0.10, delta=0.10)
th.threshold  # 0.87, or None if nothing is certified
th.coverage  # 0.74 on calibration; th.test_coverage, th.test_risk, th.test_risk_ci
th.certified  # bool
th.summary()  # "Act when confidence ≥ 0.87; ask a human below. Covers 74% ..."

ev = aoa.evaluate(table)  # test half by default
ev.accuracy  # Estimate(value=0.832, low=0.809, high=0.854)
ev.ece, ev.brier, ev.kappa, ev.aurc, ev.risk_coverage

cmp = aoa.compare(table_a, table_b)  # paired tests, Holm-adjusted
aoa.write_report([table], "report.html")
```

**Signatures** (keyword-only after the data, like a Go options struct):

```python
def read_predictions(source: str | PathLike | IO[str], *, format: str | None = None,
                     columns: Mapping[str, str] | None = None,
                     confidence_scale: str = "probability",
                     confidence_of: Literal["predicted", "positive"] = "predicted",
                     positive: str | None = None, split_seed: int = 0,
                     cal_fraction: float = 0.5) -> DecisionTable
def check(source: Data, **read_options) -> CheckResult          # never raises for bad data
def find_thresholds(data: Data | Sequence[Data], *, alpha: float = 0.10,
                    delta: float = 0.10, split_seed: int = 0, cal_fraction: float = 0.5,
                    test: Data | None = None) -> Thresholds
def evaluate(data: Data, *, n_boot: int = 2000, seed: int = 0, bins: int = 15,
             rows: Literal["test", "all"] = "test") -> Evaluation
def compare(a: Data, b: Data, *, n_boot: int = 2000, seed: int = 0) -> Comparison
def write_report(data: Sequence[Data], out: str | PathLike, **options) -> Path

Data = DecisionTable | str | PathLike | Mapping[str, Sequence[object]]
```

- **No pandas dependency.** A `Mapping` of columns works, so a pandas user passes
  `df.to_dict("list")`.
- **Errors are exceptions with the same text as the CLI:**
  - `InputError(ValueError)` carries `.problems` and maps to exit 6;
  - `NotEnoughData(ValueError)` carries `.have` and `.need` and maps to exit 7.
- **What's public:**
  - exactly the names in `ask_or_act.__all__`: the functions above, `DecisionTable`,
    `DecisionRow` and the result types;
  - everything else is internal and may change.
  - Semantic versioning applies from 0.1.0.
- **Typed:** the package already ships `py.typed`, so mypy users get the signatures.

### 5.4 Local-only, enforced by a test
`tests/test_local_only.py` replaces the socket layer for the whole test:
`socket.socket.connect`, `connect_ex`, `socket.create_connection` and
`socket.getaddrinfo`. Each replacement records the attempt and raises. The test then
runs each analysis command in-process on fixture files and fails if anything was
recorded. Commands join the test as each one is built; `check` is the first.

The guarantee is about our code paths. An import that opened a connection would also
be caught, because the commands import lazily inside the test.

---

## 6. CLI framework: keep argparse

**Decision: keep argparse.** Write one small internal module (`cli/_term.py`, about
100 lines) for colour, TTY checks, the spinner and error formatting.

**Why:**
1. **Minimal dependencies.** Today the only runtime dependency is numpy. Our
   dependency tree contains no click, typer or rich.
   - click alone would add one package.
   - typer adds click, rich (with markdown-it-py, mdurl and pygments), shellingham
     and typing-extensions.
   - For users who install a statistics tool into their own environment, every
     package is a possible version conflict.
2. **What click/typer would give us, we mostly have or need little of:**
   - Typo suggestions: Python 3.14's argparse does this
     (`suggest_on_error=True`, checked in our venv). We switch it on where available.
     On 3.12/3.13 we add the same with a few lines of `difflib.get_close_matches`.
   - Coloured help: 3.14's argparse adds a `color` option. Checked in our venv: with
     both `FORCE_COLOR` and `NO_COLOR` set, argparse still colours, so its
     precedence is the opposite of ours. `_term.py` therefore makes the colour
     decision (§1) and passes `color=<our decision>`; `color=False` always disables
     it. On 3.12/3.13 we print help without colour, which is acceptable.
   - Rich tables, panels and markdown: the summaries are deliberately a few lines of
     plain text, and long output goes to the HTML report.
   - Progress bars: one stderr spinner for steps over 1 s is all we need.
3. **Peers:** lm-evaluation-harness ships a subcommand CLI on argparse; Inspect AI
   uses click. Both are workable.
4. **We already have it:** the existing CLI and its tests are argparse.

**What argparse costs us,** and the fix for each:
- Help with an examples section: an `epilog` plus `RawDescriptionHelpFormatter`.
- Prefix abbreviations like `--alp` for `--alpha` (which clig warns against): turned
  off with `allow_abbrev=False`.
- Grouped commands in the top-level help: a custom usage text.
- No shell completion. This is the one real gap. If users ask for it, click is the
  smallest step, one dependency. We'd revisit then, not now.

**Revisit if** we need shell completion, interactive prompts or rich terminal tables.

---

## 7. Implementation order (after approval)

Each step lands with tests and must pass lint, format, mypy, pytest and the
study-freeze test.

Each step gets its own commit, after the author's approval.

- **(a)** *(done 2026-11-22)* `read_predictions` and `check` (`predictions.py`, `term.py`, the `check`
  command): every error and warning in §4.2, §4.5 and §4.6, the local-only test and
  the 100,000-row measurement.
- **(b)** *(done 2026-11-23)* `thresholds`, `metrics` and `compare` on
  predictions files, with the public API (`find_thresholds`, `evaluate`, `compare`).
- **(c)** *(done 2026-11-23)* `report` on predictions files, with
  `aoa.write_report`.
- **(d)** *(done 2026-11-24)* `demo`, with a timing test under 30 s.
- **(e)** *(done 2026-11-24)* `gate` and `changelog` (milestone 6): (e1) the gate,
  changelog and config file; (e2) `-o` Markdown, the GitHub Action and the example
  workflow; (e3) these docs.

As each command lands, its mock-up session in §3 becomes *real output*, pinned by
`tests/test_cli_docs.py`.

Out of scope here: renaming the benchmark commands, user-level config and shell
completion.

---

## 8. Performance

Target: a 100,000-row file runs every analysis command in seconds, in well under 1 GB.
Measured on the author's machine (Apple silicon, Python 3.12). Each step adds its own
row; `tests/test_performance.py` checks a generous limit in CI.

| command | rows | wall time | peak memory (RSS) |
|---|---|---|---|
| `check` (CSV, 77 labels, split column) | 100,000 (4.3 MB) | 0.75 s | 176 MB RSS (Python alone: 25 MB); 118 MB Python heap |
| `check` (JSONL) | 100,000 (9.4 MB) | 0.78 s | 176 MB RSS |
| `thresholds` | 100,000 (50,000 per half) | 1.1 s | 169 MB RSS |
| `metrics` (2,000 resamples) | 100,000 (scores the 50,000 test rows) | 10.8 s | 200 MB RSS |
| `compare` (2,000 resamples, with ACE) | 2 × 100,000 (50,000 paired test rows) | 70 s | 227 MB RSS |

Measured with `/usr/bin/time -l uv run ask-or-act check big.csv`, 2026-11-22. Memory is
about 1.5 KB per row: the parsed cells, then one `DecisionRow` per row. That's fine at
this size. If files of 10 million rows become a use case, the fix is columnar storage
(numpy arrays), not a new format. `tests/test_performance.py` checks under 15 s and
under 500 MB of Python heap on 100,000 rows, and `thresholds` + `metrics` + `compare`
(fewer resamples) under 60 s.

Step (b) made the bootstrap faster without changing any number. Every change is
pinned by `tests/test_metrics_speed.py` against the original functions:
- resamples are streamed one at a time, with the same indices as one big draw;
- all metrics are computed in one pass over the same resamples;
- labels are integer-coded, so macro-F1 and κ are one bincount per resample;
- ECE, AURC and AUROC are bincounts over the sorted distinct confidences;
- the guarantee's upper bound, which is only reported, is computed on demand (the test
  itself uses the p-value);
- the ECE noise floor is simulated in chunks.

On 1,000 rows, `metrics` went from 4.6 s to 0.6 s. What's left in `compare` at 100,000
rows is mostly ACE (about 47 of 70 s), which is still a per-resample Python loop;
vectorizing it is future work.

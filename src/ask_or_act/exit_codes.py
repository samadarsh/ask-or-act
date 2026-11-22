"""Process exit codes (docs/cli-design.md §5.1). 3–5 come from run.py and are unchanged."""

OK = 0
NO = 1  # a clean "no": a gate failed, or check --strict found warnings
USAGE = 2  # argparse's own: unknown flag, bad value, missing argument
BAD_INPUT = 6  # the input data is invalid
NOT_ENOUGH_DATA = 7  # too few labelled rows for the requested guarantee
INTERNAL = 70  # a bug (EX_SOFTWARE in sysexits.h)
INTERRUPTED = 130  # Ctrl-C

ISSUES_URL = "https://github.com/samadarsh/ask-or-act/issues/new"

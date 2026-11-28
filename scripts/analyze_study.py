"""The milestone 7 analysis (docs/analysis-plan.md, FROZEN dc2ea31). Calls no API.

    uv run python scripts/analyze_study.py --fakes OUT_DIR       # a fake study (tests)
    uv run python scripts/analyze_study.py configs/analysis/baselines.toml --out OUT_DIR
    uv run python scripts/analyze_study.py configs/analysis/study.toml \\
        --out reports/study --md docs/results.md                 # once, at the end

It writes OUT_DIR/results.json and OUT_DIR/results.md (or --md). The logic, and every
rule about what may be read when, is in src/ask_or_act/study_analysis.py. For the study
spec, nothing beyond run.json metadata is read until every config in
configs/study/schedule.toml has a completed run (exit 3 until then).

Exit codes:
- 0: done;
- 2: bad arguments;
- 3: not ready (a run is missing or incomplete).
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from ask_or_act import study_analysis
from ask_or_act.study_fakes import make_fake_study


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("spec", nargs="?", type=Path, help="analysis spec (TOML)")
    p.add_argument(
        "--fakes", type=Path, metavar="DIR", help="build a fake study in DIR and analyse it"
    )
    p.add_argument("--out", type=Path, help="output folder (results.json, results.md)")
    p.add_argument("--md", type=Path, help="write the Markdown here instead of OUT/results.md")
    args = p.parse_args(argv)
    if (args.spec is None) == (args.fakes is None):
        p.error("give a spec or --fakes DIR")
    if args.fakes is not None:
        spec_path = make_fake_study(args.fakes)
        out = args.out or args.fakes
    else:
        spec_path = args.spec
        out = args.out or Path(tempfile.mkdtemp(prefix="analysis-"))
    spec = study_analysis.load_spec(spec_path)
    t0 = time.monotonic()
    try:
        result = study_analysis.analyze(spec)
    except study_analysis.NotReadyError as exc:
        print(f"not ready: {exc}", file=sys.stderr)
        return 3
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    md = args.md or out / "results.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(study_analysis.to_markdown(result), encoding="utf-8")
    print(f"wrote {out / 'results.json'} and {md} ({time.monotonic() - t0:.0f} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

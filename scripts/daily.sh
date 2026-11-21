#!/usr/bin/env bash
# Run today's share of a study schedule on Groq's free tier, then print progress.
#
#   scripts/daily.sh                              # plan M: configs/study/schedule.toml
#   scripts/daily.sh configs/pilot/schedule.toml  # the day-1 pilot
#
# Shows the plan first (no calls), asks for confirmation, then runs one queue per model
# in parallel. Each model stops cleanly at its daily limit; re-run after the reset to
# continue from the cache. Keys come from .env and are never printed.
set -euo pipefail
cd "$(dirname "$0")/.."

schedule="${1:-configs/study/schedule.toml}"
[[ -f .env ]] || { echo "error: .env not found (needs GROQ_API_KEY)" >&2; exit 2; }
grep -q '^GROQ_API_KEY=' .env || { echo "error: GROQ_API_KEY missing from .env" >&2; exit 2; }

uv run ask-or-act study status "$schedule"
read -r -p "Run today's calls for $schedule? [y/N] " answer
[[ "$answer" == "y" || "$answer" == "Y" ]] || { echo "nothing run"; exit 0; }

mkdir -p logs
log="logs/$(date -u +%Y%m%dT%H%M%SZ)-$(basename "$schedule" .toml).log"
uv run --env-file .env ask-or-act study run "$schedule" --yes 2>&1 | tee "$log"
echo "log: $log (check with: uv run ask-or-act audit-redaction runs .cache logs)"

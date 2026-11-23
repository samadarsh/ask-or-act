#!/usr/bin/env bash
# Daily study run as a macOS launchd job (docs/study.md, "Scheduled runs").
#
#   scripts/study-job.sh today     # did today's (UTC) run happen, and how did it end?
#   scripts/study-job.sh summary   # the latest run's result
#   scripts/study-job.sh remove    # unload the job and delete its plist (logs are kept)
#
#   scripts/study-job.sh setup     # create/move the study worktree to the newest study-v* tag
#   scripts/study-job.sh install   # write the plist and load the job
#   scripts/study-job.sh plist     # print the plist that install would write
#   scripts/study-job.sh due       # is a run due now? (no calls)
#   scripts/study-job.sh run       # what launchd calls: run if due, else exit quietly
#
# Isolation: the job runs from a separate git worktree (default: <repo>-study, e.g.
# ~/Developer/ask-or-act-study) checked out at the newest `study-v*` tag, so work in
# progress in the main folder can never affect a study run. Its runs/, .cache/, logs/,
# data/raw/ and .env are symlinks to the main folder's, so there is one cache and one
# set of runs: nothing is duplicated or re-paid. Preflight refuses to call anything
# unless the worktree is clean, exactly at the newest study-v* tag, and wired to the
# shared folders. Moving the pin: docs/study.md, "Scheduled runs".
#
# When a run is due: when no run has started since the most recent 10:00 local slot.
# So the job runs at 10:00 daily, and a 10:00 missed while the Mac slept or was off runs
# at the next wake or login (a catch-up), followed by the normal 10:00 run. There is no
# daily reset to wait for: Groq's allowance refills gradually (200,000 tokens per 24 h),
# so a short gap only means a smaller run, and nothing is lost unless more than 24 h
# pass between runs. Manual runs (scripts/daily.sh) write logs/<UTC time>-schedule.log
# too, so they count.
#
# Safety: free tier only (every config must have [budget] max_usd = 0 and tier = "free";
# run.py's budget guard also refuses any charged estimate). The study-freeze test must
# pass before any call. The 2% invalid-output stop and the quota stop are the study
# runner's own. GROQ_API_KEY is read from .env by `uv run --env-file` at run time:
# never printed, logged or written into the plist (only the variable's name is checked).
set -euo pipefail
umask 077 # logs are readable by you only
unset VIRTUAL_ENV # always the worktree's own .venv, never a caller's active environment

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# The main worktree is always the first one git lists; the study worktree sits next to it.
MAIN="$(git worktree list --porcelain | awk 'NR == 1 { print $2 }')"
STUDY_DIR="${ASK_OR_ACT_STUDY_DIR:-$MAIN-study}"
SHARED=(runs .cache logs data/raw .env) # one copy, in the main folder

LABEL="io.github.samadarsh.ask-or-act.study"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
SCHEDULE="${ASK_OR_ACT_SCHEDULE:-configs/study/schedule.toml}" # tests override it
HOUR=10
LOGS="${ASK_OR_ACT_LOGS:-logs}"   # tests point this at a temp dir
LOCK="$LOGS/.study-job.lock"
TITLE="ask-or-act study"

say() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

notify() { # a macOS notification; never fails the job (ASK_OR_ACT_NOTIFY=0: off, for tests)
  [[ "${ASK_OR_ACT_NOTIFY:-1}" == 1 ]] || { echo "(notification suppressed: $1)"; return 0; }
  local msg="${1//\"/\'}"
  osascript -e "display notification \"$msg\" with title \"$TITLE\" subtitle \"scripts/study-job.sh summary\" sound name \"Basso\"" >/dev/null 2>&1 || true
}

now_epoch() { echo "${ASK_OR_ACT_NOW:-$(date +%s)}"; }

log_epoch() { # logs/20261122T111126Z-schedule.log -> seconds since 1970
  local ts
  ts="$(basename "$1")"
  date -j -u -f "%Y%m%dT%H%M%SZ" "${ts%%-*}" +%s 2>/dev/null
}

run_logs() { # every study log, oldest first (dry runs and launchd.log excluded); -H follows
  # $LOGS when it is a link, as in the study worktree (BSD find skips a linked start dir)
  find -H "$LOGS" -maxdepth 1 -name '*Z-schedule.log' 2>/dev/null | sort
}

due() { # prints the reason; exit 0 if a run is due
  local now slot last
  now="$(now_epoch)"
  slot="$(date -j -r "$now" -v"${HOUR}"H -v0M -v0S +%s)"
  if ((now < slot)); then # before 10:00: the most recent slot was yesterday's
    slot="$(date -j -r "$now" -v-1d -v"${HOUR}"H -v0M -v0S +%s)"
  fi
  last="$(run_logs | tail -1)"
  if [[ -n "$last" ]] && (($(log_epoch "$last") >= slot)); then
    echo "not due: the last run ($last) started after the $(date -j -r "$slot" '+%Y-%m-%d %H:%M') slot"
    return 1
  fi
  echo "due: no run since the $(date -j -r "$slot" '+%Y-%m-%d %H:%M') slot"
}

running_study() { pgrep -f "ask-or-act study run" || true; }

take_lock() {
  if mkdir "$LOCK" 2>/dev/null; then
    echo $$ >"$LOCK/pid"
    return 0
  fi
  local pid
  pid="$(cat "$LOCK/pid" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  rm -rf "$LOCK" # stale: its process is gone (crash or power loss)
  mkdir "$LOCK" && echo $$ >"$LOCK/pid"
}

pin() { git tag -l 'study-v*' --sort=-v:refname | head -1; }

check_worktree() { # in the study worktree: clean, at the newest pin, shared folders linked
  local tag head want dirty
  [[ "$REPO" != "$MAIN" ]] || { echo "error: running from the main folder, not the study worktree ($STUDY_DIR)"; return 1; }
  tag="$(pin)"
  [[ -n "$tag" ]] || { echo "error: no study-v* tag exists"; return 1; }
  head="$(git rev-parse HEAD)"
  want="$(git rev-parse "$tag^{commit}")"
  [[ "$head" == "$want" ]] || {
    echo "error: the worktree is at $(git describe --tags --always HEAD), not the pin $tag; run: scripts/study-job.sh setup"
    return 1
  }
  dirty="$(git status --porcelain)"
  [[ -z "$dirty" ]] || { echo "error: the study worktree has local changes:"; echo "$dirty"; return 1; }
  local d
  for d in "${SHARED[@]}"; do
    [[ -L "$d" && "$d" -ef "$MAIN/$d" ]] || { echo "error: $d is not a link to $MAIN/$d"; return 1; }
  done
  echo "worktree: $REPO at $tag ($(git rev-parse --short HEAD)), clean; ${SHARED[*]} shared with $MAIN"
}

preflight() { # no calls; any failure means nothing is called
  check_worktree || return 1
  [[ -f .env ]] || { echo "error: .env not found (needs GROQ_API_KEY)"; return 1; }
  grep -q '^GROQ_API_KEY=' .env || { echo "error: GROQ_API_KEY missing from .env"; return 1; }
  echo "key: GROQ_API_KEY is present in .env (value not read here)"
  if ! uv run --no-sync python - "$SCHEDULE" <<'PY'
import sys, tomllib
from pathlib import Path

configs = tomllib.loads(Path(sys.argv[1]).read_text())["configs"]
bad = []
for c in configs:
    d = tomllib.loads(Path(c).read_text())
    if d.get("budget", {}).get("max_usd") != 0 or d["decider"].get("params", {}).get("tier") != "free":
        bad.append(c)
if bad:
    print(f"error: not free tier with a $0 budget: {bad}")
    sys.exit(1)
print(f"free tier: all {len(configs)} configs have [budget] max_usd = 0 and tier = free")
PY
  then
    return 1
  fi
  if uv run --no-sync pytest -q -p no:cacheprovider tests/test_study_freeze.py >/dev/null 2>&1; then
    echo "study freeze: test_study_freeze passes (resolved settings unchanged)"
  else
    echo "error: tests/test_study_freeze.py fails: the code changes the study's settings; no calls made"
    return 1
  fi
}

run_job() {
  mkdir -p "$LOGS"
  local reason pids
  if ! reason="$(due)"; then
    say "$reason"
    return 0
  fi
  pids="$(running_study)"
  if [[ -n "$pids" && "${ASK_OR_ACT_JOB_DRY:-0}" != 1 ]]; then # a dry run calls nothing
    say "skip: a study run is already in progress (pid $(echo $pids))"
    return 0
  fi
  if ! take_lock; then
    say "skip: another study-job is running (pid $(cat "$LOCK/pid"))"
    return 0
  fi
  trap 'rm -rf "$LOCK"' EXIT

  # Keep the Mac awake while this script runs (-i idle sleep, -s system sleep on AC
  # power); caffeinate exits by itself when this process ends.
  caffeinate -i -s -w $$ &

  local log study_rc audit_rc problems=() dry="${ASK_OR_ACT_JOB_DRY:-0}"
  if [[ "$dry" == 1 ]]; then
    log="$LOGS/$(date -u +%Y%m%dT%H%M%SZ)-dryrun.log" # never counts as a run
  else
    log="$LOGS/$(date -u +%Y%m%dT%H%M%SZ)-schedule.log"
  fi
  trap 'rc=$?; rm -rf "$LOCK"; if ((rc != 0)); then echo "=== scheduled job aborted (exit $rc)" >>"$log"; notify "job aborted (exit $rc): see $log"; fi' EXIT
  say "running: $reason (log: $log)"
  {
    echo "=== scheduled study run (launchd job $LABEL)"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S %Z') ($(date -u '+%Y-%m-%dT%H:%M:%SZ'))"
    echo "reason: $reason"
    echo "git: $(git rev-parse --short HEAD) in $REPO (pin: $(pin))"
    echo "=== preflight (no calls)"
  } >>"$log"

  if ! preflight >>"$log" 2>&1; then
    {
      echo "=== scheduled job summary"
      echo "finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
      echo "result: ATTENTION: preflight failed; nothing was called"
    } >>"$log"
    notify "preflight failed, nothing called: see $log"
    return 0
  fi

  echo "=== study run" >>"$log"
  set +e
  if [[ "$dry" == 1 ]]; then
    uv run --no-sync ask-or-act study status "$SCHEDULE" >>"$log" 2>&1
  else
    PYTHONUNBUFFERED=1 uv run --no-sync --env-file .env ask-or-act study run "$SCHEDULE" --yes >>"$log" 2>&1
  fi
  study_rc=$?
  echo "=== redaction audit" >>"$log"
  uv run --no-sync ask-or-act audit-redaction runs .cache logs >>"$log" 2>&1
  audit_rc=$?
  set -e

  case $study_rc in
    0) study="exit 0 (every model finished its configs or reached its daily limit)" ;;
    1) study="exit 1 (a model STOPPED: see '=== today's result')"; problems+=("a model STOPPED") ;;
    *) study="exit $study_rc (error)"; problems+=("study run failed (exit $study_rc)") ;;
  esac
  case $audit_rc in
    0) audit="OK (no Aegis prompt text found)" ;;
    1) audit="FAILED (Aegis prompt text found; ids above, text not shown)"; problems+=("redaction audit found Aegis text") ;;
    *) audit="ERROR (exit $audit_rc)"; problems+=("redaction audit errored (exit $audit_rc)") ;;
  esac
  local result="OK"
  if ((${#problems[@]})); then
    result="ATTENTION: $(IFS=';'; echo "${problems[*]}" | sed 's/;/; /g')"
  fi
  {
    echo "=== scheduled job summary"
    echo "finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "study: $study"
    echo "audit: $audit"
    echo "result: $result"
  } >>"$log"
  say "done: $result"
  if [[ "$result" != OK ]]; then
    notify "${result#ATTENTION: }"
  fi
}

latest() { run_logs | tail -1; }

summary() {
  local log
  log="$(latest)"
  [[ -n "$log" ]] || { echo "no study runs logged in $LOGS/"; return 0; }
  echo "log: $log"
  grep -E '^(started|reason|git):' "$log" || true
  if grep -q "^=== today's result" "$log"; then
    sed -n "/^=== today's result/,/^=== progress/p" "$log" | sed '$d'
  fi
  if grep -q '^=== scheduled job summary' "$log"; then
    sed -n '/^=== scheduled job summary/,$p' "$log"
  elif [[ -n "$(running_study)" ]]; then
    echo "(still running: last lines)"
    tail -5 "$log"
  elif ! grep -q "^=== today's result" "$log"; then
    echo "(no result in this log: interrupted, or a manual run that ended early; last lines)"
    tail -5 "$log"
  fi
}

today() {
  local utc_day log
  utc_day="$(date -u +%Y%m%d)"
  log="$(run_logs | grep "/${utc_day}T" | tail -1 || true)"
  if [[ -z "$log" ]]; then
    echo "today (UTC $(date -u +%Y-%m-%d)): no run yet"
    due || true
  else
    local state
    if [[ -n "$(running_study)" ]]; then
      state="running"
    elif grep -q '^result: ' "$log"; then
      state="$(grep '^result: ' "$log" | tail -1)"
    elif grep -q "^=== today's result" "$log"; then
      state="finished (manual run; no job summary)"
    else
      state="no result in the log (interrupted?)"
    fi
    echo "today (UTC $(date -u +%Y-%m-%d)): ran, started $(date -j -r "$(log_epoch "$log")" '+%H:%M %Z'): $state"
    echo "log: $log"
  fi
  if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
    echo "job: installed ($LABEL), daily at $HOUR:00 local"
  else
    echo "job: not installed"
  fi
}

plist() {
  cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$STUDY_DIR/scripts/study-job.sh</string>
    <string>run</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$STUDY_DIR</string>
  <!-- 10:00 local. Missed while asleep: launchd runs it on wake (man launchd.plist). -->
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>$HOUR</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>
  <!-- Missed while off: RunAtLoad fires at login, and an hourly check backs it up.
       The script itself decides whether a run is due, so neither can double-run. -->
  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>3600</integer>
  <!-- No secrets here: the key is read from .env by uv at run time. -->
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>StandardOutPath</key>
  <string>$MAIN/logs/launchd.log</string>
  <key>StandardErrorPath</key>
  <string>$MAIN/logs/launchd.log</string>
</dict>
</plist>
EOF
}

setup() { # create the study worktree, or move it to the newest pin; idempotent
  local tag d
  tag="$(pin)"
  [[ -n "$tag" ]] || { echo "error: tag a commit first, e.g. git tag study-v1 <commit>"; return 1; }
  if [[ -d "$STUDY_DIR" ]]; then
    git -C "$STUDY_DIR" diff --quiet && git -C "$STUDY_DIR" diff --cached --quiet || {
      echo "error: $STUDY_DIR has local changes; not moving it"
      return 1
    }
    git -C "$STUDY_DIR" checkout -q --detach "$tag"
  else
    git -C "$MAIN" worktree add -q --detach "$STUDY_DIR" "$tag"
  fi
  for d in "${SHARED[@]}"; do
    mkdir -p "$(dirname "$STUDY_DIR/$d")"
    if [[ -e "$STUDY_DIR/$d" && ! -L "$STUDY_DIR/$d" ]]; then
      echo "error: $STUDY_DIR/$d exists and is not a link; move it away first"
      return 1
    fi
    ln -sfn "$MAIN/$d" "$STUDY_DIR/$d"
  done
  (cd "$STUDY_DIR" && uv sync -q --frozen --all-extras)
  echo "study worktree: $STUDY_DIR at $tag ($(git -C "$STUDY_DIR" rev-parse --short HEAD))"
  echo "shared with $MAIN: ${SHARED[*]}"
}

install() {
  [[ -x "$STUDY_DIR/scripts/study-job.sh" ]] || { echo "error: no study worktree; run: scripts/study-job.sh setup"; return 1; }
  (cd "$STUDY_DIR" && REPO="$STUDY_DIR" && check_worktree) || return 1
  mkdir -p "$(dirname "$PLIST")" "$MAIN/logs"
  plist >"$PLIST.tmp"
  plutil -lint "$PLIST.tmp" >/dev/null
  mv "$PLIST.tmp" "$PLIST"
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "installed: $PLIST"
  echo "it checks now (RunAtLoad), then daily at $HOUR:00 and hourly; see: scripts/study-job.sh today"
}

remove() { # the job, its plist and the study worktree; never runs, cache, logs or tags
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null && echo "unloaded: $LABEL" || echo "not loaded: $LABEL"
  if [[ -f "$PLIST" ]]; then
    rm "$PLIST"
    echo "deleted: $PLIST"
  fi
  if [[ -d "$STUDY_DIR" && "$STUDY_DIR" != "$MAIN" ]]; then
    local d
    for d in "${SHARED[@]}"; do # unlink first: removing the worktree then can't reach shared data
      [[ -L "$STUDY_DIR/$d" ]] && rm "$STUDY_DIR/$d"
    done
    git -C "$MAIN" worktree remove --force "$STUDY_DIR"
    echo "deleted the study worktree: $STUDY_DIR"
  fi
  rm -rf "$MAIN/logs/.study-job.lock"
  echo "kept: $MAIN/runs, .cache, logs and the study-v* tags"
}

case "${1:-}" in
  run) run_job ;;
  due) due ;;
  today) today ;;
  summary) summary ;;
  setup) setup ;;
  install) install ;;
  remove) remove ;;
  plist) plist ;;
  *)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac

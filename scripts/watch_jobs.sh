#!/bin/bash
# Poll Slurm jobs until every one reaches a terminal state, printing each result as it
# lands (successes and failures alike — silence is not success).
# Usage: scripts/watch_jobs.sh <jobid> [jobid...]
set -uo pipefail

JOBS=("$@")
[[ ${#JOBS[@]} -eq 0 ]] && { echo "usage: watch_jobs.sh <jobid> [jobid...]" >&2; exit 1; }

declare -A REPORTED=()
while true; do
  pending=0
  for job in "${JOBS[@]}"; do
    [[ -n "${REPORTED[$job]:-}" ]] && continue
    state=$(sacct -j "$job" --format=State --noheader --parsable2 2>/dev/null | head -1 | tr -d ' ')
    case "$state" in
      COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
        echo "[$(date -u +%H:%M:%S)] job $job -> $state"
        if [[ "$state" != "COMPLETED" ]]; then
          echo "  --- tail of logs/*-${job}.err ---"
          tail -20 logs/*-"${job}".err 2>/dev/null || echo "  (no error log found)"
        fi
        REPORTED[$job]=1
        ;;
      *)
        pending=1
        ;;
    esac
  done
  [[ $pending -eq 0 ]] && break
  sleep 60
done
echo "all ${#JOBS[@]} job(s) reached a terminal state"

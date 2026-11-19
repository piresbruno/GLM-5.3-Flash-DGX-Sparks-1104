#!/usr/bin/env bash
# Stop the three ranks of ./start-tp3.sh: ./stop-tp4.sh with TP=3 (rank 0 first, then ranks 1 and 2; one gzipped
# archive of their logs, the newest LOG_KEEP kept). Usage: ./stop-tp3.sh; DRY_RUN=1 ./stop-tp3.sh prints the plan.
set -euo pipefail
export TP=3 GLM_TPN=1
exec "$(dirname -- "$(readlink -f -- "$0")")/stop-tp4.sh" "$@"

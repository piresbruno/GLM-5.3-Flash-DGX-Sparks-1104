#!/usr/bin/env bash
# Serve GLM-5.3-Flash on three DGX Sparks (TP=3): a triangle of cables (every pair of Sparks on its own ConnectX-7
# cable and subnet) or one switch subnet. ./start-tp4-switchless.sh with TP=3: the same image, patches, settings,
# checks and weights (NFS from this Spark, each worker's own copy, or rsync from this Spark) as four Sparks; the links
# say which fabric it is. WORKER and WORKER2 name the other two Sparks, in any order. TP3_NCCL=eager (the default)
# or ring picks the NCCL settings on a triangle (scripts/tp4.sh, tp4_rank_env).
#
# Usage: ./start-tp3.sh [restart]   (./stop-tp3.sh stops it; DRY_RUN=1 ./start-tp3.sh prints the links it finds and
# every rank's command, and changes nothing)
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac
export TP=3 GLM_TPN=1
exec "$(dirname -- "$(readlink -f -- "$0")")/start-tp4-switchless.sh" "$@"

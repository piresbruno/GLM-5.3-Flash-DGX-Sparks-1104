#!/usr/bin/env bash
# One leg of the TP=3 epic's measurement set (epic task 03), on two Sparks (TP=2) or three (TP=3, the default):
#
#   docs/tp3leg.sh <label> [KEY=VALUE ...]        e.g. docs/tp3leg.sh T3-kdapad KDA_PAD=1
#
# 1. docs/leg.sh: a verified restart with the settings, docs/dec.py at 1/2/4/8 streams and its --contract
#    (greedy, thinking off), compared with the leg directory's contract-baseline.json;
# 2. docs/sd.py: sparkDash Structured and prose at 1/4/6/8 streams, prefill 2K..128K;
# 3. docs/quality.py (NLL and answers sha) and docs/seeded.py (fixed-seed sampled replies);
# 4. rank 0's startup lines and every Spark's available memory;
# 5. unless PROF=0: a second restart with TF_GLM_PROF=1, one stream then eight (long enough for the shared windows'
#    row buckets to report), a cold prefill, and every rank's
#    `glm prof` reports through docs/prof.py (attribution only: profiling perturbs timing).
# Files go to docs/speed/tp3/ (TP=3) or docs/speed/ (TP=2) as <kind>-<label>.*. PORT, WORKER, WORKER2 and SD_UNIT
# come from scripts/local.sh (or the environment).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
LABEL=${1:?usage: tp3leg.sh <label> [KEY=VALUE ...]}; shift
TP=3; for kv in "$@"; do [[ "$kv" != TP=* ]] || TP=${kv#TP=}; done
local_var() { [[ -f scripts/local.sh ]] && sed -n "s/^$1=\\([^ \t#]*\\).*/\\1/p" scripts/local.sh | head -1; }
PORT=${PORT:-$(local_var PORT)}; PORT=${PORT:-8888}
WORKER=${WORKER:-$(local_var WORKER)}; WORKER2=${WORKER2:-$(local_var WORKER2)}
export SD_UNIT=${SD_UNIT:-$(local_var SD_UNIT)}
export U="http://127.0.0.1:$PORT" LOADS=${LOADS:-1,2,4,8}
OUT=docs/speed; [[ "$TP" != 3 ]] || OUT=docs/speed/tp3
mkdir -p "$OUT"
LOG=$OUT/tp3leg-$LABEL.log; : > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
workers=("$WORKER"); [[ "$TP" != 3 ]] || workers+=("$WORKER2")
mem() { # every Spark's MemAvailable (GiB), by rank: the logs are committed, so no host or address in them
  local w r=1
  say "  rank 0 MemAvailable $(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo) GiB"
  for w in "${workers[@]}"; do
    say "  rank $r MemAvailable $(timeout 20 ssh -o BatchMode=yes "$w" "awk '/MemAvailable/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo") GiB"
    r=$((r + 1))
  done
}

say "### tp3leg $LABEL  TP=$TP  $(date -u +%Y-%m-%dT%H:%M:%SZ)  $*"
bash docs/leg.sh "$LABEL" TP="$TP" "$@" 2>&1 | tee -a "$LOG"
[[ "${PIPESTATUS[0]}" == 0 ]] || { say "FAILED: docs/leg.sh (see $OUT/leg-$LABEL.log)"; exit 3; }
say "--- startup ---"
grep -hE 'linears packed|CUDA sm_121 device 0|prompt cache:|loaded in' logs/current/rank0.log | cut -c1-400 | tee -a "$LOG"
mem
say "--- sparkDash ---"
python3 -I docs/sd.py "$LABEL" "$OUT" all 2>&1 | tee -a "$LOG"
say "--- quality ---"
python3 -I docs/quality.py "$LABEL" > "$OUT/q-$LABEL.json" 2>> "$LOG" \
  && say "  $(python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print("nll", d["nll"], "answers_sha", d["answers_sha"])' "$OUT/q-$LABEL.json")" \
  || say "  quality.py FAILED"
say "--- seeded ---"
python3 -I docs/seeded.py "$LABEL" "$OUT" 2>&1 | tee -a "$LOG"
mem

if [[ "${PROF:-1}" != 0 ]]; then
  say "--- profile restart (TF_GLM_PROF=1) ---"
  for kv in "$@"; do [[ "$kv" == TP=* ]] || export "$kv"; done
  export TP TF_GLM_PROF=1 DEC_OUT="$PWD/$OUT"
  bash docs/safestart.sh >> "$LOG" 2>&1 || { say "FAILED: profile restart"; exit 3; }
  SINCE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  LOADS=1 TRIALS=2 MAXTOK=300 python3 -I docs/dec.py "$LABEL-prof1" >> "$LOG" 2>&1
  # shared windows report a total-row bucket every 64 calls: eight streams need ~4,000 rounds to fill the common ones
  LOADS=8 TRIALS=1 MAXTOK=1200 python3 -I docs/dec.py "$LABEL-prof8" >> "$LOG" 2>&1
  python3 -I docs/pf2.py >> "$LOG" 2>&1
  # only the reports: the rest of a rank's log is prompt-stage lines and the HTTP log (clients' addresses)
  docker logs --since "$SINCE" glm53-spark 2>&1 | grep 'glm prof' > "$OUT/profraw-$LABEL-r0.log"
  r=1; for w in "${workers[@]}"; do
    # the prompt-stage lines make a worker's log large: only the reports cross the link
    timeout 300 ssh -o BatchMode=yes "$w" "docker logs --since $SINCE glm53-spark 2>&1 | grep 'glm prof'" > "$OUT/profraw-$LABEL-r$r.log" 2>&1
    r=$((r + 1))
  done
  for f in "$OUT"/profraw-"$LABEL"-r*.log; do
    say "== ${f##*/}: $(grep -c 'glm prof' "$f" || true) reports"
    python3 -I docs/prof.py "$f" 2>&1 | tee -a "$LOG"
  done
fi
say "### tp3leg $LABEL done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

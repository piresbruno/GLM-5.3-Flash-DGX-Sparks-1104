#!/usr/bin/env bash
# A restart with a VERIFIED stop, because `start.sh restart` does not have one.
#
# On 2026-11-03 05:07Z the worker's `docker stop` failed ("cannot delete running task ... failed
# precondition", "DeadlineExceeded") and `docker run` was issued anyway, so two ~100 GiB containers
# existed at once on a 128 GB machine. No OOM kill and no panic: the host simply stopped being able
# to fork, ping and TCP/22 kept answering, and it needed a power cycle. See docs/speed/RESULTS.md.
#
# This stops through ./stop.sh -- drain, log archive, the stop->kill->verified ladder -- then refuses
# to start until EVERY rank's container is provably absent, and refuses to return
# success until /health answers. Every ssh has a timeout, so a wedged worker fails fast and loudly
# instead of hanging a programme for hours.
#
# TP=3 (exported) runs the same ladder over three Sparks through ./stop-tp3.sh and ./start-tp3.sh, workers WORKER and
# WORKER2. PORT (exported, else scripts/local.sh's, else 8888) is the port /health is polled on. DRY_RUN=1 prints the
# stop and start it would run and changes nothing.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
NAME=glm53-spark
TP=${TP:-2}
[[ "$TP" == 2 || "$TP" == 3 ]] || { echo "[safestart] TP=$TP: this script restarts two Sparks or three (TP=3)"; exit 3; }
local_var() { [[ -f scripts/local.sh ]] && sed -n "s/^$1=\\([^ \t#]*\\).*/\\1/p" scripts/local.sh | head -1; }
WORKER=${WORKER:-$(local_var WORKER)}
WORKER2=${WORKER2:-$(local_var WORKER2)}
PORT=${PORT:-$(local_var PORT)}; PORT=${PORT:-8888}
[[ -n "$WORKER" ]] || { echo "[safestart] WORKER not found in scripts/local.sh"; exit 3; }
workers=("$WORKER")
if [[ "$TP" == 3 ]]; then
  [[ -n "$WORKER2" ]] || { echo "[safestart] WORKER2 not found in scripts/local.sh (TP=3)"; exit 3; }
  workers+=("$WORKER2"); STOP=./stop-tp3.sh; START=./start-tp3.sh
else
  STOP=./stop.sh; START=./start.sh
fi
sshw() { timeout "${2:?}" ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlPath=none "${1:?}" "${3:?}"; }

gone_local()  { ! docker inspect "$NAME" >/dev/null 2>&1; }
gone_worker() { ! sshw "$1" 25 "docker inspect $NAME >/dev/null 2>&1" >/dev/null 2>&1; }

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "[safestart] plan, TP=$TP ($((${#workers[@]} + 1)) ranks: here and ${workers[*]}):"
  echo "  $STOP   (drain, archive, kill on timeout); on failure: docker rm -f $NAME here and on ${workers[*]}"
  echo "  wait up to 2 min until $NAME is absent here and on ${workers[*]}"
  echo "  $START restart   (logs to /tmp/safestart-start.<pid>.log)"
  echo "  poll http://127.0.0.1:$PORT/health every 5 s for up to 10 min"
  exit 0
fi

echo "[safestart] $(date -u +%H:%M:%SZ) verifying every rank is stopped (workers=${workers[*]})"
echo "[safestart] $(date -u +%H:%M:%SZ) $STOP (drain, archive, kill on timeout, verified absent)"
stop_out=$("$STOP" 2>&1); stop_rc=$?
if (( stop_rc == 0 )); then
  [[ "$stop_out" == *Nothing\ to\ stop* ]] || echo "[safestart] $(echo "$stop_out" | tail -1)"
else
  echo "[safestart] ${STOP#./} failed rc=$stop_rc - falling back to a forced remove; its last lines:"
  echo "$stop_out" | tail -3 | sed 's/^/[safestart]   /'
  docker rm -f "$NAME" >/dev/null 2>&1
  for w in "${workers[@]}"; do sshw "$w" 60 "docker rm -f $NAME >/dev/null 2>&1; true" >/dev/null 2>&1; done
fi

for i in $(seq 1 24); do          # up to 2 min for the task to actually die
  l=no; w=yes; left=''
  gone_local  && l=yes
  for k in "${workers[@]}"; do gone_worker "$k" || { w=no; left+=" $k"; }; done
  if [[ "$l" == yes && "$w" == yes ]]; then
    echo "[safestart] every rank absent after ${i}x5s - safe to start"
    break
  fi
  [[ $i -eq 24 ]] && {
    echo "[safestart] REFUSING TO START: container still present (local_gone=$l workers_still_running:${left:- none})."
    echo "[safestart] This is the condition that wedged spark4. Needs a human."
    exit 4
  }
  sleep 5
done

echo "[safestart] $(date -u +%H:%M:%SZ) $START restart"
"$START" restart > /tmp/safestart-start.$$.log 2>&1 &   # never piped: a pipe can outlive the job
sp=$!
for i in $(seq 1 120); do          # up to 10 min to serve
  if curl -fs -m 8 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[safestart] HEALTHY after ~$((i*5))s at $(date -u +%H:%M:%SZ)"
    tail -3 /tmp/safestart-start.$$.log 2>/dev/null; rm -f /tmp/safestart-start.$$.log
    exit 0
  fi
  sleep 5
done
echo "[safestart] NOT HEALTHY after 10 min - aborting this leg"
tail -20 /tmp/safestart-start.$$.log 2>/dev/null; rm -f /tmp/safestart-start.$$.log
kill "$sp" 2>/dev/null
exit 5

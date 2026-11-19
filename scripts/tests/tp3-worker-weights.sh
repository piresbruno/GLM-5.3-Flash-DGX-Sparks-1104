#!/usr/bin/env bash
# tp4_prepare_worker_weights at three Sparks, WORKER_WEIGHTS=rsync and check, under set -u: every worker is reached on
# its own ssh target (WORKER for rank 1, WORKER2 for rank 2). The function sets WORKER for nodes.sh's helpers; a
# `local WORKER` declared before rank 1's target was read made `wtarget 1` see the empty local (bash's dynamic scope)
# and stopped the start with "!v: unbound variable". The ssh helpers are stubbed: nothing leaves this machine.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
box=$(mktemp -d)
trap 'rm -rf -- "$box"' EXIT
for e in * .[!.]*; do [[ "$e" == scripts || "$e" == .env || "$e" == .git ]] || ln -s "$PWD/$e" "$box/$e"; done
mkdir "$box/scripts"
for e in scripts/*; do [[ "$e" == scripts/local.sh ]] || cp -a -- "$e" "$box/scripts/"; done
cd -- "$box"
fails=0
check() { # check <name> <want> [VAR=value]...: want is the workers' targets the helpers saw, in order
  local name=$1 want=$2 got rc
  shift 2
  got=$(env -i PATH="$PATH" HOME="$HOME" GLM_TPN=1 TP=3 WORKER=a@spark-b WORKER2=a@spark-c "$@" bash -c '
    set -euo pipefail
    source scripts/prepare.sh
    source scripts/tp4.sh
    seen=()
    log() { :; }
    send_dir() { seen+=("send:$WORKER"); }
    worker_weights_ready() { seen+=("ready:$WORKER"); return 1; }
    wssh_n() { seen+=("ssh$1:$(wtarget "$1")"); [[ "$2" != cat ]] || printf %s "$MODEL_REVISION"; }
    tp4_prepare_worker_weights
    printf "%s " "${seen[@]}"' 2>&1) && rc=0 || rc=$?
  if (( rc == 0 )) && [[ "$got" == "$want" ]]; then echo "tp3-worker-weights: $name: pass"
  else echo "tp3-worker-weights: $name: FAIL (rc $rc, got '$got', want '$want')" >&2; fails=1; fi
}
check 'rsync: each worker on its own target' \
  'ready:a@spark-b send:a@spark-b ssh1:a@spark-b ready:a@spark-c send:a@spark-c ssh2:a@spark-c ' \
  WORKER_WEIGHTS=rsync DRY_RUN=0 DRAFTER=dflash2
check 'rsync, DRY_RUN=1: the plan for each worker' \
  'send:a@spark-b send:a@spark-b send:a@spark-c send:a@spark-c ' \
  WORKER_WEIGHTS=rsync DRY_RUN=1 DRAFTER=dflash2
# (the revision is read in a command substitution, a subshell: that call is not recorded)
check 'check: each worker on its own target' \
  'ssh1:a@spark-b ssh1:a@spark-b ssh2:a@spark-c ssh2:a@spark-c ' \
  WORKER_WEIGHTS=check DRY_RUN=0 DRAFTER=dflash2
exit "$fails"

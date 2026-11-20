#!/usr/bin/env bash
# shellcheck disable=SC2016  # the bash -c script is single-quoted on purpose
# validate_config at three Sparks (./start-tp3.sh: GLM_TPN=1 TP=3): WORKER and WORKER2 only (WORKER3 not needed),
# PREFILL_REDUCE split by default, rsync offered; FP8=1, ROCE_RING=on and KV_SHARD=1 refused by name; TP=5 refused;
# and the four-Spark checks unchanged beside them. Each case starts from the defaults, given here so that
# scripts/local.sh and .env cannot change them.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# scripts/config.sh sources scripts/local.sh, so on a kit (rank 0 holds one) its values would leak into the defaults.
# Run in a sandbox: the repo's top level linked, scripts/ copied without local.sh, and no .env.
box=$(mktemp -d)
trap 'rm -rf -- "$box"' EXIT
for e in * .[!.]*; do [[ "$e" == scripts || "$e" == .env || "$e" == .git ]] || ln -s "$PWD/$e" "$box/$e"; done
mkdir "$box/scripts"
for e in scripts/*; do [[ "$e" == scripts/local.sh ]] || cp -a -- "$e" "$box/scripts/"; done
cd -- "$box"
fails=0
check() { # check <name> <want: accepted|refused:<text the refusal holds>> [VAR=value]...
  local name=$1 want=$2 err rc got
  shift 2
  err=$(mktemp)
  env -i PATH="$PATH" HOME="$HOME" GLM_TPN=1 TP=3 WORKER=a@spark-b WORKER2=a@spark-c "$@" bash -c '
    source scripts/prepare.sh
    validate_config >/dev/null
    printf "%s %s %s %s %s %s %s" "$PREFILL_REDUCE" "$NCCL_CHANNELS" "$HEAD" "$PROMPT_CACHE_GIB" "$TP3_NCCL" \
      "$KDA_PAD" "$ROCE_MULTI:$ROCE_MULTI_KB" >&2' 2>"$err" && rc=0 || rc=$?
  if (( rc == 0 )); then got="accepted"
  elif grep -q 'E_CONFIG' "$err"; then got="refused:$(grep -m1 E_CONFIG "$err")"
  else got="error $rc: $(head -1 "$err")"; fi
  if [[ "$want" == accepted* && "$got" == accepted ]] && { [[ "$want" == accepted ]] || [[ "$(cat "$err")" == "${want#accepted:}" ]]; }; then
    echo "tp3-config: $name: pass"
  elif [[ "$want" == refused:* && "$got" == refused:*"${want#refused:}"* ]]; then
    echo "tp3-config: $name: pass"
  else
    echo "tp3-config: $name: FAIL (got $got $(cat "$err" | tail -1), want $want)" >&2; fails=1
  fi
  rm -f -- "$err"
}
check 'the defaults (split, 8 channels, q8 head, 12 GiB cache, eager NCCL, KDA pad on, RoCE off at 128 KB)' \
  'accepted:split 8 q8 12 eager 1 0:128'
check 'WORKER3 is not needed' accepted WORKER3=
check 'PREFILL_REDUCE=gather' accepted PREFILL_REDUCE=gather
check 'WORKER_WEIGHTS=rsync' accepted WORKER_WEIGHTS=rsync
check 'WORKER_WEIGHTS=nfs' accepted WORKER_WEIGHTS=nfs
check 'TP3_NCCL=ring' accepted TP3_NCCL=ring
check 'TP3_NCCL=tree is refused' refused:TP3_NCCL TP3_NCCL=tree
check 'WORKER2 is required' refused:WORKER2 WORKER2=
check 'two workers the same are refused' 'refused:two different Sparks' WORKER2=a@spark-b
check 'FP8=1 is refused by name' 'refused:FP8=1' FP8=1
check 'ROCE_RING=on is refused by name' 'refused:ROCE_RING=on' ROCE_RING=on
check 'KV_SHARD=1 is refused by name' 'refused:KV_SHARD=1' KV_SHARD=1
check 'TP=5 is refused' 'refused:TP=5' TP=5
check 'KDA_PAD=1' accepted KDA_PAD=1
check 'KDA_PAD=on is refused' refused:KDA_PAD KDA_PAD=on
check 'ROCE_MULTI=1' accepted ROCE_MULTI=1
check 'SAMPLE_ROOT=1 is refused by name (gone with TensorFold 1.3.0)' 'refused:SAMPLE_ROOT is gone' SAMPLE_ROOT=1
check 'SAMPLE_ROOT=yes is refused' refused:SAMPLE_ROOT SAMPLE_ROOT=yes
check 'ROCE_MULTI_KB=2048 is refused' refused:ROCE_MULTI_KB ROCE_MULTI_KB=2048
check 'ROCE_MULTI_FALLBACK=warn is refused' refused:ROCE_MULTI_FALLBACK ROCE_MULTI_FALLBACK=warn
# four Sparks beside it: WORKER3 required, split and rsync refused, FP8 accepted
check 'four: WORKER3 required' refused:WORKER3 TP=4 WORKER_WEIGHTS=nfs
check 'four: defaults with nfs' 'accepted:gather 4 bf16 8 eager 1 0:128' TP=4 WORKER3=a@spark-d WORKER_WEIGHTS=nfs
check 'four: split refused' 'refused:PREFILL_REDUCE' TP=4 WORKER3=a@spark-d WORKER_WEIGHTS=nfs PREFILL_REDUCE=split
check 'four: rsync refused' 'refused:rsync' TP=4 WORKER3=a@spark-d
check 'four: GLM_TP4=1 still names the four-Spark start' accepted GLM_TPN= GLM_TP4=1 TP=4 WORKER3=a@spark-d WORKER_WEIGHTS=nfs
exit $fails

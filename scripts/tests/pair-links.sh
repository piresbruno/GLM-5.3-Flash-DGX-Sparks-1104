#!/usr/bin/env bash
# shellcheck disable=SC2016,SC2015  # the bash -c scripts are single-quoted on purpose; pass and bad never fail
# The multi-Spark launcher's link pairing and fabric classes, from fake node inventories (no Spark, no ssh): scripts/
# tp4.sh's pair_links and classify_fabric, and the NCCL settings tp4_rank_env gives each fabric.
# - three Sparks: a triangle (one subnet per pair) is `triangle`, ring 0 1 2; one switch subnet is `mesh`; a missing
#   cable stops with E_FABRIC naming the pair;
# - four Sparks: the ring and the switch pair and classify as on main (pair_links' lines byte-equal but the new crowd
#   line), each refusing the other's cabling, and a full mesh of cables is refused by name.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
fails=0
pass() { echo "pair-links: $1: pass"; }
bad() { echo "pair-links: $1: FAIL ($2)" >&2; fails=1; }
# in_tp <TP> <FABRIC_EXPECT> <script>: the script after prepare.sh and tp4.sh, with the defaults only.
in_tp() {
  env -i PATH="$PATH" HOME="$HOME" TP="$1" GLM_TPN=1 FABRIC_EXPECT="$2" WORKER=a@spark-b WORKER2=a@spark-c WORKER3=a@spark-d bash -c '
    source scripts/prepare.sh
    source scripts/tp4.sh
    '"$3"
}
# inventory <rank> <netdev:hca:cidr>...: one rank's node_inventory lines, rank-prefixed as detect_ring feeds them.
inventory() {
  local r=$1 e
  shift
  echo "$r default enP7s7"
  for e in "$@"; do IFS=: read -r n h c <<<"$e"; echo "$r $n $h $c 3"; done
}
# classify <name> <TP> <FABRIC_EXPECT> <inventory> <want: a FABRIC_KIND, or E_FABRIC:<text the refusal holds>>
classify() {
  local name=$1 tp=$2 expect=$3 inv=$4 want=$5 out rc
  out=$(in_tp "$tp" "$expect" '
    missing="" ring="" crowd=0
    while read -r kind a b c d; do
      case "$kind" in missing) missing+=" $a-$b" ;; ring) ring=$(xargs <<<"$a $b $c $d") ;; crowd) crowd=$a ;; esac
    done <<<"$(pair_links $(rank_ids))"
    classify_fabric "$missing" "$ring" "$crowd"
    echo "$FABRIC_KIND"' <<<"$inv" 2>&1) && rc=0 || rc=$?
  if [[ "$want" == E_FABRIC:* ]]; then
    if (( rc != 0 )) && [[ "$out" == *'[E_FABRIC]'*"${want#E_FABRIC:}"* ]]; then pass "$name"; else bad "$name" "got rc $rc: $out"; fi
  elif (( rc == 0 )) && [[ "$out" == "$want" ]]; then pass "$name"
  else bad "$name" "got rc $rc: $out, want $want"; fi
}

triangle=$(inventory 0 rocep1s0f0:rocep1s0f0:10.0.1.1/24 rocep1s0f1:rocep1s0f1:10.0.2.1/24
           inventory 1 rocep1s0f0:rocep1s0f0:10.0.3.1/24 rocep1s0f1:rocep1s0f1:10.0.1.2/24
           inventory 2 rocep1s0f0:rocep1s0f0:10.0.2.2/24 rocep1s0f1:rocep1s0f1:10.0.3.2/24)
switch3=$(for r in 0 1 2; do inventory "$r" "rocep1s0f0:rocep1s0f0:10.0.9.$((r + 1))/24"; done)
missing3=$(inventory 0 rocep1s0f0:rocep1s0f0:10.0.1.1/24 rocep1s0f1:rocep1s0f1:10.0.2.1/24
           inventory 1 rocep1s0f1:rocep1s0f1:10.0.1.2/24
           inventory 2 rocep1s0f0:rocep1s0f0:10.0.2.2/24)
ring4=$(inventory 0 rocep1s0f0:rocep1s0f0:10.0.1.1/24 rocep1s0f1:rocep1s0f1:10.0.4.2/24
        inventory 1 rocep1s0f0:rocep1s0f0:10.0.2.1/24 rocep1s0f1:rocep1s0f1:10.0.1.2/24
        inventory 2 rocep1s0f0:rocep1s0f0:10.0.3.1/24 rocep1s0f1:rocep1s0f1:10.0.2.2/24
        inventory 3 rocep1s0f0:rocep1s0f0:10.0.4.1/24 rocep1s0f1:rocep1s0f1:10.0.3.2/24)
switch4=$(for r in 0 1 2 3; do inventory "$r" "rocep1s0f0:rocep1s0f0:10.0.9.$((r + 1))/24"; done)

# --- pair_links ---------------------------------------------------------------------------------------------------
got=$(in_tp 3 ring 'pair_links 0 1 2' <<<"$triangle")
want='toward 0 1 rocep1s0f0
peer 1 10.0.1.2
toward 0 2 rocep1s0f1
peer 2 10.0.2.2
rank 0 enP7s7 rocep1s0f0,rocep1s0f1 3
toward 1 0 rocep1s0f1
toward 1 2 rocep1s0f0
rank 1 enP7s7 rocep1s0f0,rocep1s0f1 3
toward 2 0 rocep1s0f0
toward 2 1 rocep1s0f1
rank 2 enP7s7 rocep1s0f0,rocep1s0f1 3
prefix 24
crowd 2
ring 0 1 2'
[[ "$got" == "$want" ]] && pass 'a triangle pairs every rank with both others, ring 0 1 2' || bad 'triangle pairing' "$(diff <(echo "$want") <(echo "$got") | head -5)"
# main's pair_links on the same four-Spark inventories (2026-11-19, 80be4d6), before the crowd line was added
want_ring4='toward 0 1 rocep1s0f0
peer 1 10.0.1.2
missing 0 2
toward 0 3 rocep1s0f1
peer 3 10.0.4.1
rank 0 enP7s7 rocep1s0f0,rocep1s0f1 3
toward 1 0 rocep1s0f1
toward 1 2 rocep1s0f0
missing 1 3
rank 1 enP7s7 rocep1s0f0,rocep1s0f1 3
toward 2 1 rocep1s0f1
toward 2 3 rocep1s0f0
rank 2 enP7s7 rocep1s0f0,rocep1s0f1 3
toward 3 0 rocep1s0f0
toward 3 2 rocep1s0f1
rank 3 enP7s7 rocep1s0f0,rocep1s0f1 3
prefix 24
ring 0 1 2 3'
got=$(in_tp 4 ring 'pair_links 0 1 2 3' <<<"$ring4" | grep -v '^crowd ')
[[ "$got" == "$want_ring4" ]] && pass 'four on a ring pair as on main' || bad 'ring4 pairing' "$(diff <(echo "$want_ring4") <(echo "$got") | head -5)"
got=$(in_tp 4 mesh 'pair_links 0 1 2 3' <<<"$switch4" | grep -v '^crowd ' | sha256sum | cut -c1-16)
[[ "$got" == 87c2973b3847b0e9 ]] && pass 'four on a switch pair as on main' || bad 'switch4 pairing' "digest $got"

# --- classify_fabric ----------------------------------------------------------------------------------------------
classify 'three on a triangle' 3 ring "$triangle" triangle
classify 'three on a switch' 3 ring "$switch3" mesh
classify 'three with a cable missing' 3 ring "$missing3" 'E_FABRIC:ranks 1-2'
classify 'four on a ring, ring start' 4 ring "$ring4" ring
classify 'four on a ring, switch start' 4 mesh "$ring4" 'E_FABRIC:use ./start-tp4-switchless.sh'
classify 'four on a switch, switch start' 4 mesh "$switch4" mesh
classify 'four on a switch, ring start' 4 ring "$switch4" 'E_FABRIC:use ./start-tp4.sh'
out=$(in_tp 4 ring 'classify_fabric "" "" 2' 2>&1) && bad 'four on a full mesh of cables' 'accepted' \
  || { [[ "$out" == *'full mesh of cables'* ]] && pass 'four on a full mesh of cables is refused by name' || bad 'full mesh' "$out"; }

# --- tp4_rank_env per fabric --------------------------------------------------------------------------------------
rank_env() { # rank_env <TP> <FABRIC_KIND> <TP3_NCCL>: rank 1's -e values, one a line
  in_tp "$1" ring 'FABRIC_KIND='"$2"' TP3_NCCL='"$3"' NODE_HCAS[1]=rocep1s0f0,rocep1s0f1 NODE_GID[1]=3 RING_PREFIX=24 NODE_GRAPH[1]=/g.xml
    tp4_rank_env 1; printf "%s\n" "${RANK_ENV[@]}" "${RANK_MOUNTS[@]}" | grep -v "^-[ev]$"'
}
got=$(rank_env 3 triangle eager)
if grep -qx NCCL_CROSS_NIC=1 <<<"$got" && grep -qx NCCL_IB_SUBNET_AWARE_ROUTING=1 <<<"$got" && grep -qx NCCL_MAX_NCHANNELS=8 <<<"$got" \
   && grep -qx NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 <<<"$got" && ! grep -q 'NCCL_GRAPH_FILE\|NCCL_RUNTIME_CONNECT\|NCCL_ALGO' <<<"$got"; then
  pass 'a triangle (eager) connects at init over every device, no graph file, 8 channels'
else bad 'triangle eager env' "$(xargs <<<"$got")"; fi
got=$(rank_env 3 triangle ring)
grep -qx NCCL_GRAPH_FILE=/nccl-graph.xml <<<"$got" && grep -qx NCCL_RUNTIME_CONNECT=1 <<<"$got" && grep -qx /g.xml:/nccl-graph.xml:ro <<<"$got" \
  && pass 'a triangle with TP3_NCCL=ring takes the ring set and its graph' || bad 'triangle ring env' "$(xargs <<<"$got")"
[[ "$(rank_env 4 ring eager)" == "$(rank_env 4 ring ring)" ]] && pass 'TP3_NCCL leaves four Sparks alone' || bad 'TP3_NCCL at four' 'differs'
graph=$(in_tp 3 ring 'NODE_HCAS[0]=a,b; TOWARD["0 2"]=b; NCCL_CHANNELS=2; graph_xml 0 $(( (0 + TP - 1) % TP ))' | grep -c '<net dev="1"/><gpu dev="0" rank="0"/>')
(( graph == 4 )) && pass "rank 0's graph at three listens toward rank 2" || bad 'graph at three' "$graph channel lines on device 1"
exit $fails

# 01 - Launcher for three Sparks: the ring launcher at TP=3

Stage 0. Depends on: nothing. Estimate: 2-3 days. Status: done off the Sparks 2026-11-19; the kit's DRY_RUN and LIVE wait for task 03's window.

## Result

- `start-tp3.sh` / `stop-tp3.sh` export `TP=3 GLM_TPN=1` and exec the four-Spark scripts; `GLM_TP4=1` still works.
- `scripts/tp4.sh`: ranks from `TP`; `pair_links` prints `crowd` (most ranks on one subnet) and finds rings of three;
  `classify_fabric` (pure, tested): at 3 `triangle` (no pair missing, crowd 2) or `mesh` (switch), a missing cable
  stops naming the pair; at 4 as before, and a full mesh of cables refused by name. `TP3_NCCL=eager` (default: crossnic,
  subnet-aware routing, all devices, no graph, no runtime connect) or `ring` (the ring set and graphs). rsync weights
  at 3 through the pair's `send_dir` per worker. Memory guard and logs sized by `TP`.
- `prepare.sh` / `config.sh` as in steps 5-6, plus `KDA_PAD` (task 05). `local.sh.example`: a commented TP=3 block.
- Not done offline: diffing `DRY_RUN=1 ./start-tp4*.sh` against `main` needs the workers over ssh; pair_links on
  four-Spark inventories is checked byte-equal to `main` instead (task 02's test).

## Goal

`./start-tp3.sh` and `./stop-tp3.sh` start and stop GLM-5.3-Flash at tensor parallel 3 on three Sparks cabled as a
triangle, on the current 121-patch image, without touching `./start.sh`.

## Approach

Generalise the four-Spark ring launcher (`scripts/tp4.sh`, `start-tp4-switchless.sh`, `stop-tp4.sh`) to a ring of
three instead of re-porting the `tp3` branch's rewrite of `start.sh`. A triangle is both a 3-ring and a full mesh, and
the ring launcher already carries what TP=4 paid for:

- `MASTER_ADDR` from the default route's source address (`start-tp4-switchless.sh:52-57`); the `tp3` branch needed a
  hand-set LAN address because the hostname resolved to a cable address the third Spark cannot route.
- `NCCL_SOCKET_IFNAME` on rank 0 only (`tp4.sh:360-369`): a follower given one unsets `NCCL_COMM_ID` and panics under
  `NCCL_RUNTIME_CONNECT=1`.
- Per-worker home and uid, `PREFILL_REDUCE` in the fit key, `LEARN_GIB`, `L2PF`, `EXTRA_EOS`, `THINK_OFF_SPLIT`.

## Steps

1. **Wrappers.** `start-tp3.sh`: `export TP=3 GLM_TPN=1`, exec `start-tp4-switchless.sh "$@"`. `stop-tp3.sh`: the same
   for `stop-tp4.sh`. Keep `GLM_TP4=1` as an alias of `GLM_TPN=1` so existing four-Spark starts are unchanged.
2. **`scripts/tp4.sh`** (rename later if it reads better; not in this task):
   - `worker_ids` (line 19): `seq 1 $((TP - 1))`.
   - Ring detection (218): `len(ranks) >= 3`.
   - `detect_links` (236-270): rank lists from `TP` instead of `0 1 2 3` (245, 269); `ring="$a $b $c $d"` (252) of
     variable length.
   - Fabric classes (258-266): add `triangle` = every pair cabled and no subnet shared by more than two ranks. Today a
     fabric with no missing pair is called a switch. At `TP=3` accept `triangle` (and a switch); at `TP=4` refuse
     `triangle` by name.
   - Logging and neighbour maths (288-305): loops over `0..TP-1`, `(r+1)%TP`, `(r+TP-1)%TP`; no "across the ring" line
     at 3.
   - `ring_graphs` (310-328): `prev=(r+TP-1)%TP`. `graph_xml` (330-349) is world-agnostic already.
   - Memory guard (482-511): arrays sized by `TP`, `tick % (TP-1) + 1`, messages say "three" or "four" from `TP`.
3. **`start-tp4-switchless.sh`:** `TP` from the wrapper (20); rank loops from `TP-1` down (262, 292, 440-444); banner,
   step and LIVE text (377, 530); fit file `.state/context-fit-tp3.tsv` at 3. `KV_SHARD` refusal (37) stays.
4. **`stop-tp4.sh`:** `TP` from the wrapper (10); `WORKER3` required at 4 only (18-20); `there=()` sized by `TP` (59-62).
5. **`scripts/prepare.sh`** (90-110, 139):
   - Under the multi-rank flag accept `TP` 3 or 4; require `WORKER WORKER2` at 3 and `WORKER3` too at 4, all distinct.
   - `PREFILL_REDUCE`: `split|gather|ring` at 3, default `split` (a triangle is a full mesh; the old TP=3 ran split).
   - `WORKER_WEIGHTS`: `rsync|nfs|check` at 3 (every worker has its own cable and may hold a copy).
   - `FP8=1` refused at 3 (306 GiB does not fit three ranks), by name.
   - `ROCE_RING=on` refused at 3 by name (patch 0108 is world 4 only); `KV_SHARD=1` refused (world 2 only).
6. **`scripts/config.sh`:** TP=3 defaults beside the TP=4 ones (221-223): `PROMPT_CACHE_GIB=12` (measured on the old
   stack: a 12 GiB cache cut an agentic replay's wall time by 61%), `HEAD=q8`, `NCCL_CHANNELS=8`; and the engine
   switches tasks 05-08 add, each defaulting off until its task is gated.
7. **NCCL set at 3.** Start with the set the `tp3` branch proved on this kit (eager connect, `NCCL_CROSS_NIC=1`,
   `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, every RoCE device in `NCCL_IB_HCA`,
   no graph files). Keep the ring set (`NCCL_ALGO=Ring`, graph files, `NCCL_RUNTIME_CONNECT=1`,
   `NCCL_CUMEM_ENABLE=1`) selectable with `TP3_NCCL=ring` for one A/B leg in task 03.
8. **`scripts/local.sh.example`:** a commented TP=3 block (`WORKER`, `WORKER2`, `PORT`, `PARALLEL=8`).

## Acceptance

- `DRY_RUN=1 ./start-tp3.sh` on the kit prints three ranks, the triangle's links (each rank's devices toward each peer),
  `TF_GLM_WORLD=3` and the NCCL set, and writes nothing.
- `DRY_RUN=1 ./start-tp4-switchless.sh` and `DRY_RUN=1 ./start-tp4.sh` print the same plan as on `main` (diff the
  output with fake inventories in task 02's tests).
- `./start.sh` and `stop.sh` are byte-unchanged.
- In task 03's window: the TP=3 start reaches LIVE and passes its smoke test; `./stop-tp3.sh` stops the three ranks
  and archives three logs.

## Risks

- NCCL's tree on three ranks over a triangle is untested with runtime connect; the eager set is the fallback.
- Rank 0 carries the HTTP server and the vision tower and has the least memory; the 6.5 GiB reserve has never been
  checked against the 2.5 GiB memory-guard floor at TP=3 (task 03 reads it).

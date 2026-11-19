# 12 - Patch: CUDA-graph replay of the lone verify forward

Stage 4. Depends on: 11 (go). Estimate: ~2 weeks. Status: todo. **New work; upstream has no GLM graph path.**

## Problem

A GLM verify forward is ~1,100 launches (about 25 a layer: KDA ~7, MoE ~10, glue ~4, two gathers) issued from the host
every round, and every NCCL gather breaks the PDL chain. On the pair and at TP=4, ~9-11 ms a round sits outside the GPU
sections (`docs/speed/RESULTS.md:1690`). This cost does not shrink with more ranks, so it weighs most at TP=3. The
Python engine replays one CUDA graph per row count (1..64), NCCL calls inside.

## What makes it feasible (verified)

- No host readback inside the forward before the pick; routing is device-side (`glue.selectGroup`, `exl3.runGrouped`;
  the EXL3 grid is sized by `min(rows*slots, experts)`, not by the picked experts).
- `tilesFor` depends only on rows and the load-time `uniform_k2`.
- Infrastructure: `zig/src/cuda/graph_cache.zig` (`Cache(Key)` with every-rank agreement, line 178); DeepSeek's
  `graphs.zig` as the model (keyed by rows and a context bucket).
- `roce.zig`, 0108's ring and `fabric/tp` (`roce_multi.zig:33`) replay under graphs by a device epoch.

## What must change

| Host dependency | Where | Remedy |
| --- | --- | --- |
| Position by value; dense vs sparse choice and select grid sizes | `dsa_block.body(..., st.pos_dev, st.pos, nch, ...)`, `cuda_forward.zig:308` | key on a context bucket; kernels read `pos_dev` |
| Per-stream pointers (conv, `recAt(cur, li)`, proj, saves, caches, page tables) | stream state | key on the stream slot (this task); device args table (task 13) |
| KDA ping-pong parity `st.cur[li]` | commit | key bit |
| PDL launches (`pdl_hc`, `pdl_moe`, `mm.chain`, `tilesFor(...).pdl`) | `zig/src/cuda/graph.zig:36` rejects PDL on hand-built nodes | check stream capture on CUDA 13 accepts them; else no PDL inside the captured body |
| Shared expert side stream (fork/join events) and L2PF | `cuda_moe.zig:79-93` | capture follows events; verify |
| Profiler spans | `TF_GLM_PROF` | graph path only with the profiler off |
| Token ids, drafter taps | `copy_source.enqueue`, `t.at(...)` | fixed staging buffers |
| Pick, sampler, follower broadcast | `cuda_engine_pick.zig`, `cuda_follow.zig` | stay outside the graph |

## Steps

1. Spike first (1-2 days): capture one lone 1-row forward on one rank with PDL on; replay it; compare outputs. This
   settles the PDL and side-stream questions before the rest.
2. `TF_GLM_VERIFY_GRAPH=1` (default off): lone verify, rows 1..8, key = (rows, slot, parity, context bucket); capture
   on first use, every rank agreeing (graph_cache); bounded count (evict LRU).
3. Every rank captures the same sequence; gathers inside the graph (NCCL or the RoCE wire).
4. Fallback to eager on any capture error, logged once.

## Acceptance

- Greedy contract hashes identical to the eager path at TP=2, 3 and 4.
- 1-stream round time down by most of the host gap (expect 4-7 ms); no regression at 4/8 streams.
- Memory for the graphs recorded; startup time recorded.

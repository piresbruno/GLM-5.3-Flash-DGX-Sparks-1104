# 14 - Patch: balanced routed-expert split

Stage 5. Depends on: 11 (go). Estimate: 2-3 weeks. Status: todo.

## Problem

EXL3 cuts each routed expert's 2,048 intermediate rows on 128-row boundaries (the Hadamard rotation width;
`zig/src/families/glm/cuda_shard.zig:38`, `cuda_exl3.zig:131-165`), and `canon.partition` gives remainders to the
lowest ranks (`zig/src/cluster/canon.zig:15-36`). At TP=3 that is 6/5/5 units: rank 0 does 37.5% of the routed work
instead of 33.3%. Measured (old stack, 8 shared rows): rank 0 routed 41.0 ms, ranks 1/2 34.0-34.3, which then wait in
`gather_b`. Every layer synchronises at its two gathers, so each phase costs its slowest rank.

Not fixes: rotating the extra unit by **layer** (each layer still waits for its 768-row rank), or moving only the
KDA/MLA/vocab remainders off rank 0 (it only changes which rank is slowest).

## Fix

Rotate the 16th unit by **expert** (`e % 3`), so each rank averages 5.33 units a layer and a row's 8 picked experts
spread their extra units across ranks.

## Steps

1. Storage: every expert at 768 rows a rank (the largest share), plus an int32 `[experts]` width table; ranks without
   an expert's extra unit hold zeros there or skip it.
2. Kernels read the per-expert width; blocks past it exit early: `zig/src/cuda/exl3_experts.zig` (`Layer.width`,
   `Geometry.of`, scratch sizing, grids, 19-41 and 330-345), `exl3/experts.cu`, `exl3_grouped.cuh`,
   `exl3_decode.cuh`, `exl3_prompt.cuh`.
3. Loader: per-expert cuts in `cuda_exl3.zig`; `shard.Share.moe` becomes per-expert; `glm-loader-gate` updated.
4. Switch `TF_GLM_MOE_BALANCE=1`, default off; EXL3 only.
5. Cheap companion worth one leg: give rank 0 the smaller shared-expert share (it runs beside rank 0's larger routed
   work).

## Acceptance

- The partition changes, so replies differ from the unbalanced TP=3: re-take the TP=3 baselines; the TP=3 gates hold.
- Rank 0's routed time within ~3% of the others' in the per-rank profile; gather waits on ranks 1/2 down.
- 8-row shared forward down ~4 ms (expected 41 -> ~36.4 ms routed on rank 0).

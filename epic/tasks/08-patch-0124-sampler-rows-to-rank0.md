# 08 - Patch 0124: sampled rows to rank 0 only

Stage 2. Depends on: 04. Estimate: 2-3 days. Status: todo.

## Problem

For sampled streams, `Spread` (`zig/src/families/glm/cuda_sampler.zig:153-166`) issues one `ncclBroadcast` per row per
rank of that rank's vocabulary span, so every rank receives every row's full-vocab logits, although only rank 0 draws
(followers only gather, `cuda_engine.zig:771-774`). At 8 rows and world 3 that is 24 NCCL calls a round, and the
traffic grows with world. Every sparkDash benchmark samples (T=0.7), so this is on the measured path.

## Fix

Each follower sends its contiguous `[rows][span]` block to rank 0 in one grouped send/recv; rank 0 assembles the rows
and draws as today. Two NCCL operations instead of `rows x world`, and a third of the bytes at TP=3.

## Steps

1. Replace `Spread`'s broadcasts with an `ncclGroupStart` / send (followers) / recv (rank 0) / `ncclGroupEnd`, writing
   into rank 0's existing full-vocab row buffer at each rank's span offset.
2. Followers skip the buffer they no longer need; keep the old path under `TF_GLM_SAMPLE_SPREAD=1` for one release.
3. Check the follower's step (0031's one broadcast) still has everything it needs (it should: followers never draw).

## Acceptance

- Sampled replies with a fixed seed identical to the old path at TP=2 and TP=3 (same logits, same draw).
- Round time down at 4 and 8 sampled streams (profile the sampler section), or the patch is dropped.

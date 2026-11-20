# 08 - Patch 0124: sampled rows to rank 0 only

Stage 2. Depends on: 04. Estimate: 2-3 days. Status: measured 2026-11-20: exact (sampled replies off = on), no gain beyond noise at 4 and 8 sampled streams: stays off, per this task's rule (docs/speed/tp3/RESULTS.md).

## Result so far

- The premise needed narrowing: followers skip the draw only in verify steps (`.verify`, `.shared`:
  `discard_draws`). In prefill, chunk, redraw and draft steps every rank draws itself with the same seed and needs
  whole rows; structured output and prompt logprobs read whole rows everywhere. So `TF_GLM_SAMPLE_ROOT=1` applies
  to verify steps only: `Engine.in_verify` (set by `verify` / `verifyShared` on every rank) and the setting choose
  `To.root`; everything else keeps the broadcasts. That is the decode hot path, where the 24 calls a round were.
- Root mode: a follower `ncclSend`s its contiguous `[rows][span]` logits; rank 0 `ncclRecv`s each follower's block
  into a stage (rows x vocab bf16, ~10 MB a rank) in one group and copies it into its columns (`cuMemcpy2DAsync`).
- The new mode is opt-in (`TF_GLM_SAMPLE_ROOT`, the stack's rule) rather than the task's `TF_GLM_SAMPLE_SPREAD=1`
  opt-out; in the fingerprint, layout 19 -> 20. Launcher: `SAMPLE_ROOT=0|1`, passed at TP=3 only.
- Tests: the group against a recording NCCL at TP=3 (uneven spans), the settings case. To do on the kit: a fixed
  seed's sampled replies equal with the switch off and on at TP=2 and TP=3, and the sampler section's profile at
  4 and 8 sampled streams (drop the patch if it does not move).

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

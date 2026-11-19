# 05 - Patch 0122: `TF_GLM_KDA_PAD`

Stage 1. Depends on: 04. Estimate: 1-2 days, then one window. Status: todo. **The largest single gain in the epic.**

## Problem

A KDA layer's fused input stack is `width = 3*heads*128 + 256 + heads` columns (`zig/src/families/glm/
cuda_kda_block.zig:32-34`), and `tiledInput()` requires `width % 8 == 0` (37-39). At TP=2 (32 heads, 12,576) and TP=4
(16 heads, 6,416) it holds; at TP=3 (22/21/21 heads, 8,726 / 8,341) it does not, so:

- `project()` drops `glm_mm` (`cuda_kda_block.zig:75`) and the stack runs on the generic `tf_glm_bmm_decode`
  (`cuda_b16.zig:78-93`);
- `cuda_tile.zig:31-38` never tiles or packs the stack, `f_b` or `g_b`: they stay **bf16** even with q8 dense
  (102 linears; the startup line says 179 packed instead of 281). Rank 0 reads ~2.43 GB of KDA projections a forward
  instead of ~1.25 GB;
- `f_b` and `g_b` never take the pair launch (`cuda_mm.zig:348-351` needs `ldx % 8 == 0`).

Measured on the old stack: KDA 16.9-18.5 ms a forward at TP=3 against 16.4 at TP=2 and 7.1 at TP=4.

## Fix: backport upstream `TF_GLM_KDA_PAD`

Upstream commits `96b3b7aa9`, `316c0bc0f`, `842bba3b6`, `6f7f99c21`, `831e20c12`, `b7af93459`, `6bafb5aac`,
`9db24e79c`, `2d2a84fdf`, `367e45011`, `c309bb354` (net +54/-6 over seven files plus
`zig/tests/cuda/glm_kda_pad_check.zig`). Every rank's stack is padded with zero columns to one 8-aligned width (the
widest rank's); heads stay 22/21/21. No prerequisites outside the pin (`In.split`, `Weights.named`, `Settings.tiled`,
`tileLinears` are present). Upstream measured at TP=3 on three GB10s, one stream: 45.6 -> 56.7 tok/s; profile wall
40.27 -> 30.56 ms; KDA 14.99 -> 5.57 ms.

## Steps

1. Cherry-pick the 11 commits onto the worktree (task 04) and resolve:

   | File | State | Action |
   | --- | --- | --- |
   | `cuda_tile.zig` | identical to upstream's pre-image | applies |
   | `cuda_kda_block.zig` | recipe-only `matmulPair` call and `chainSegs`; 0117 changed `bodyProfile`'s signature | hand-merge `matWidth()` / `rowWidth()` / `pad_to` |
   | `cuda_shared.zig:68` | diverged | `k.width()` -> `k.matWidth()` (feeds 91, 98) |
   | `cuda_state.zig:115,139,265,499` | diverged; 0119 edits near 297 | four `kw = rowWidth(...)` edits |
   | `cuda_engine.zig` | diverged | set `kb.pad_to` after `shard.Share.of` |
   | `cuda_ranks.zig`, `cuda_ranks_test.zig` | diverged (kv_shard, chunk, l2pf) | add `kda_pad` to Settings and the fingerprint, layout 17 -> 18; drop upstream's `roce_multi` / `kv_host` / `heat_*` names from the test list |

2. **Recipe-only site upstream does not cover:** `chainSegs` (`cuda_kda_block.zig:126`) passes
   `.p_stride = l.width()`. With the pad it must be the padded width, or shared decode windows read the projection
   rows at the wrong stride. Extend `zig/src/families/glm/cuda_kda_tp3_test.zig` to cover it.
3. Confirm the pad turns on `matmulPair` for `f_b`/`g_b` at TP=3 (expected: `tiledInput()` now passes).
4. Learn-store identity (0113): every rank's state prefix bytes are in the identity (`cuda_glm_disk.zig:171-172`).
   Confirm the padded `State.proj` is inside that prefix so stores written with and without the pad never mix; if not,
   add the pad to the identity.
5. Switch: `TF_GLM_KDA_PAD=1`, engine default off, refused without `TF_GLM_MM=1` (as upstream). A no-op at TP=2/4.
   `config.sh`: `KDA_PAD=on` by default at TP=3 once gated; `start-tp3.sh` passes `TF_GLM_KDA_PAD=1`.
6. Export as `patches/engine/0122-glm-cuda-TF_GLM_KDA_PAD-...patch` (upstream shas in the body).

## Tests

- Carry `glm_kda_pad_check.zig` (`tf-cuda-test glm-kda-pad-split`); the new `cuda_ranks_test` case.
- At world 3 with the pad on: `glm-shared-check`, `glm-pair-check`, `glm-lanes-concurrent`.
- Host tests as in task 04.

## Measurement (window)

T3 + pad against task 03's T3, full measurement set. Startup line: 281 linears packed. Per-rank profile: KDA per
forward. Prompt chunk profile before and after (prefill effect is unmeasured upstream).

## Acceptance

- Replies differ from the unpadded run (q8 packed instead of bf16; upstream says so in `6bafb5aac`) but satisfy the
  TP=3 gates: repeatable, drafts on = off, solo = concurrent, `quality.py` within the reference.
- TP=2 with the switch unset: contract hashes equal to the image before the patch.
- KDA at TP=3 at or below TP=4's per-head share (~7-8 ms a 2-row forward); 1-stream decode up by roughly a fifth.

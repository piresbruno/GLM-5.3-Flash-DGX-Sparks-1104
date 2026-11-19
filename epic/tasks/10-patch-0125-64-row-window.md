# 10 - Patch 0125: 64-row shared verify window

Stage 3. Depends on: 09. Estimate: 1-2 weeks. Status: todo.

## Problem

A shared round verifies at most 32 rows across all streams (`shared_rows = 32`, `max_shared = 16`,
`zig/src/families/glm/cuda_engine.zig:43-44`). At 8 streams that is 4 rows a stream, so deep drafts cannot all be
verified. The Python engine verifies up to 64 rows together and measured, at 8 streams of code, 141.2 / 160.3 / 167.0
tok/s at 32 / 48 / 64 rows.

## Limits that move together (behind `TF_GLM_SHARED_ROWS=64`, default 32)

| Limit | Where |
| --- | --- |
| `shared_rows`, `max_shared`, pick rows, the pairs buffer | `cuda_engine.zig:35,43-44,246` |
| `[max_world * 32]` pick array | `cuda_engine_pick.zig:43` |
| `select_group_rows` and `select_group_sum_shared` | `cuda_glue.zig:238` |
| `kda.max_segs`, `latent.max_segs`, `glue.max_copies` (16) | KDA, latent and glue modules |
| `hc_pre`'s `rows < 64` branches (why 32 was chosen) | `cuda_glue.zig:362-378` - cap at 63 or qualify the 64-row path |
| EXL3 fast kernels only up to 32 rows; above, `tilesFor` falls to the generic 8-tile kernel | `zig/src/cuda/exl3_experts.zig:289` - **the real cost**: needs a measured wide form, or per-row cost rises |
| `batch_rows` | `core/lanes/engine.zig:180-182` |
| Sampler rows, follower scratch | `cuda_sampler.zig`, `cuda_follow.zig` |
| RoCE slot for the gathers | `TF_TP_MAILBOX_MAX_KB=1024` (task 07) |

## Steps

1. Raise every limit above behind the switch; memory per stream grows (record the fixed-per-stream MiB at startup).
2. EXL3 decode at 33-64 rows: measure the generic kernel; if per-row cost rises, add a 64-row variant of the read-once
   `k8r16p` path (`exl3_experts.zig` `tilesFor`, `exl3/experts.cu`).
3. Draft rule: let the shared window offer up to 7 drafts a stream at 8 streams when the refit costs say they pay.
4. Tests: `glm-shared-check` and `glm-lanes-concurrent` at 33, 48, 63/64 rows; `hc_pre` against its reference at 64.

## Acceptance

- Greedy contract hashes unchanged (same tokens, fewer rounds) at 1, 4, 8 streams.
- 8-stream Structured and code up beyond noise; no loss at 1 stream.

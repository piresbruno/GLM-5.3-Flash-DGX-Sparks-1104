# 13 - Patch: graph replay for shared windows

Stage 4. Depends on: 12. Estimate: 2-3 weeks. Status: todo.

## Goal

Extend task 12's graphs to shared (multi-stream) verify windows, so concurrency also loses its host gap.

## Steps

1. Device args table: every per-stream pointer the forward reads (conv state, recurrent state at the current parity,
   proj rows, saves, KV page table, positions) goes into one small device buffer, uploaded with one copy a round;
   kernels read their pointers from it. The stream slot and parity leave the graph key.
2. Shared windows keyed by (total rows, segment layout bucket, context bucket). `kda.Segs` and the latent segments read
   their layout from the args table.
3. Paged KV: page-table pointers through the args table only (no per-round re-capture).

## Acceptance

- Greedy hashes identical to eager at 1/4/8 streams, TP=2/3/4; solo equal concurrent.
- 4- and 8-stream rounds down by the host gap's share; graph count and memory bounded.
- Risk accepted: higher than task 12 (KV pages, segment layouts); keep the switch off by default until two windows
  pass.

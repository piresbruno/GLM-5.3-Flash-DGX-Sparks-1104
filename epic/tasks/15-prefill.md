# 15 - Prefill at TP=3

Stages 1-5. Depends on: 03, 05. Estimate: spread across the windows. Status: todo.

## Problem

Prefill at TP=3 was flat against TP=2 on the old stack (~1,850 tok/s at 32K) while the Python engine reaches ~2,350
(2K-16K). Nothing in the prompt path is disabled at world 3 except the copy-engine exchange (world 2 only,
`cuda_ce.zig:70`; off in this recipe anyway), so the causes are the same as decode's: rank 0's 6/16 routed share, the
unpacked KDA stack (prompt chunks too), and two-thirds of the partials exchanged a rank.

## Levers, in order

1. **KDA pad (task 05)** covers prompt chunks, shared prompt windows and paired prompts (upstream `316c0bc0f`,
   `842bba3b6`, `6f7f99c21`). Profile one 4,096-row chunk before and after.
2. **`TF_GLM_OVERLAP_PIECES=8` with `TF_GLM_CHUNK_SOLO=1`** (0110 has CHUNK_SOLO): upstream measured 6-8% at TP=3
   (`2914b6883`, `d0695af5a`). Task 03 leg; adopt at TP=3 if it holds.
3. **`TF_GLM_PREFILL_REDUCE`:** keep `split` (a triangle is a full mesh); one leg of `gather` for the record.
4. **Balanced split (task 14)** helps prompts by the same 6/16 -> 5.33/16.
5. Candidate, unmeasured: the copy-engine prompt exchange generalised to three ranks over RoCE (0035/0063 are world 2
   and same-host only); only if 1-4 leave a measured exchange-bound gap.

Not applicable: upstream `dd326f4f2` (#1577, 5-14% at TP=3) speeds the prompt MTP pass, which this stack skips beside
DFlash2 (0008, 0110).

## Acceptance

- Each lever measured with fresh prompt content (`cached_tokens` 0) at 2K-128K; adopted when it clears noise and the
  greedy replies hold.
- Prefill result stated against 2,350 tok/s in task 16, met or not.

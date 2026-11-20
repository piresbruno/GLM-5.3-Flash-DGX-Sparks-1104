# 03 - Baselines and settings legs (maintenance window)

Stage 0. Depends on: 01, 02. Estimate: one maintenance window (~6-8 h). Status: done 2026-11-20 (first window): every leg measured except T3-ncclring, which does not start; T3's contract repeats on a second boot and is `docs/speed/tp3/contract-baseline.json`; decisions in docs/speed/tp3/RESULTS.md. P3 within 5% of the epic's screenshots.

## Goal

Measured, same-day numbers for the three things every later task is judged against, plus one leg for each setting
that might close part of the gap without engine work.

## Before the window

- Agree the window with the owner: the Python-engine recipe serving on the three Sparks is stopped for it.
- Record that service's exact environment (`docker inspect`) and its sparkDash results, then stop it with its own
  `./stop.sh`.
- Clock policy: read and record the GPU clocks on all three Sparks; the same policy for every leg.

## Legs (one boot each; every leg runs the same measurement set)

Measurement set:
- sparkDash decode, Structured and prose: T=0.7, thinking on, `top_k` 20, 400 tokens, 1/4/6/8 streams.
- sparkDash prefill: 2K, 4K, 8K, 16K, 32K, 128K, with fresh prompt content (`cached_tokens` 0).
- `docs/dec.py` 1/2/4/8 streams and `--contract` (greedy, thinking off).
- `docs/quality.py` (NLL and answers sha).
- `TF_GLM_PROF=1` on all three ranks for a 2-row lone verify and an 8-row shared window: wall, routed, KDA, MLA,
  router, gathers, head and dense **per rank**.
- Rank 0's startup line (model GiB, pages, prompt cache, linears packed) and each Spark's lowest available memory.

| Leg | Settings | Question |
| --- | --- | --- |
| P3 | Python-engine recipe, TP=3 as it serves | The bar-2 reference on this day |
| T2 | This recipe, `./start.sh`, current defaults | The bar-1 reference |
| T3 | `./start-tp3.sh`, current 121-patch image | Where TP=3 starts |
| T3-q4 | T3 + `DENSE=q4` | Equal dense width to the Python engine (reported, not adopted unless quality holds) |
| T3-coupled | T3 + `TF_GLM_DFLASH2_COUPLED=1` | Do noise-following drafts land more at T=0.7? |
| T3-conf | T3-coupled + `TF_GLM_DRAFT_CONFIDENCE` 0.5, then 0.3 | Deeper drafts under sampling |
| T3-tpm | T3 + `TF_GLM_DRAFT_RULE=tpm` with costs refit from T3's trace | Cost-aware depth |
| T3-pieces | T3 + `TF_GLM_OVERLAP_PIECES=8 TF_GLM_CHUNK_SOLO=1` | Prefill (upstream measured 6-8% at TP=3) |
| T3-l2pf | T3 + `L2PF=1` | Decode, opt-in on the pair |
| T3-ncclring | T3 + `TP3_NCCL=ring` | The ring NCCL set against the eager set |

Drafted-vs-undrafted: one T3 run with `DRAFTER=none` on `dec.py --contract` only (the hashes must match T3's).

## After the window

Restart the Python-engine recipe with its own `./start-tp3.sh` and check `/health` and one reply.

## Outputs

- `docs/speed/tp3/RESULTS.md`: a table per leg; the per-rank profile table; findings.
- `docs/speed/tp3/contract-baseline.json` (T3's contract), and JSON for every leg beside it.
- Update the epic's "Where it should land" with measured numbers.

## Acceptance

- Every leg's numbers are recorded with its exact environment; P3 is within ~5% of the screenshots in the epic
  (otherwise note the drift).
- T3's contract repeats on a second boot (same hashes).
- A decision per settings leg: adopt as a TP=3 launcher default, or not (with the number).

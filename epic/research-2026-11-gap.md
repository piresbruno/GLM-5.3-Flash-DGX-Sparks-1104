# Research: the gap between this recipe's TP=3 and the Python-engine recipe (2026-11-20)

Tracker target: none (no tracker is configured in `AGENTS.md`); this note and the epic's task table are the output.

Paths: **PY** is the Python-engine recipe's checkout (`GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold`, engine `v0.6.0` +
123 patches, what the kit serves); **OLD** is the engine at `f350cd3e` + this recipe's patches 0001-0124; **NEW** is
`81de26f6` (TensorFold 1.3.0) + `patches/engine/0001-0015`. Engine paths are under `zig/src/families/glm/` unless
given. `[INFERENCE]` marks what no file or measurement shows directly.

## Summary

1. **The 45K-prompt fault is fixed upstream.** Main's patch 0015 (an E4M3 cache on KV pages keeps its token lists as
   positions) is the fix; with 0014 and 0015 merged, three boots of three answer `longctx` correctly (4 of 7 wrong
   before).
2. **1.3.0's slowdown is old patches that did not carry over**, not the host and not the new launch scheme. It costs
   -6% to -15% decode and -25% prefill here and needs patches (or upstream) to restore; no setting recovers it.
   `HEAD=bf16`, reported upstream as a recovery on pairs, is slower at TP=3 (-6% to -12%).
3. **The Python engine's figures are now measured, not inferred.** Its lone verify is 24.9 ms for one row and 48.4 ms
   for eight (its own startup table); its window is 64 rows in total at 8 streams; its drafts follow the sampler's
   keyed noise and stop at a confidence product of 0.3; its forward is one CUDA graph a row count, NCCL included.
4. **Where this engine loses, under sparkDash's protocol** (`docs/rounds.py`, new):
   - Structured at 8 streams: the 32-row window halves the drafts (5.9 to 3.0 a stream at 99% acceptance). The 64-row
     window (task 10) is confirmed for this case and worth about +20% there by this note's cost fit.
   - One stream: tokens a round are level with the Python engine's by inference; the round is ~12 ms longer (63.8
     against ~52 ms). About 9 ms of that is the forward; the rest is outside it.
   - Prose at 8 streams: the window is not full (26 rows); each added stream costs ~15 ms of round time.
5. **Two of the epic's planned causes do not separate the engines:** the Python engine has the same 6/5/5 expert
   split and the same two all-gathers a layer, and runs NCCL only at TP=3. Task 14 (balanced split) and the RoCE wire
   are therefore not needed for parity.
6. **Bar 2 at 8 streams (326 tok/s) is reachable only if three things all land:** the 1.3.0 losses restored (~+20% at
   8 streams), the 64-row window (~+20% on Structured), and the round's fixed cost cut toward the Python engine's.
   The first two alone project to ~260 tok/s on sparkDash's scale `[INFERENCE]`.

## Current state (evidence)

### Third set of legs (2026-11-20, 17:38-18:36 UTC, production left down by its owner)

NEW with 0014-0015 (image built in 22 min), TP=3, `KDA_PAD=1`; files `docs/speed/tp3/*-T3-131*`.

| Leg | `dec.py` x1 / x2 / x4 / x8 | Structured x1 / x4 / x6 / x8 | Prose x1 / x4 / x6 / x8 | Prefill 16K | `longctx` | NLL |
| --- | --- | --- | --- | ---: | --- | ---: |
| T3-kdapad-130 (before 0014-0015) | 43.6 / 62.2 / 82.1 / 131.9 | 103.0 / 184.4 / 200.5 / 192.8 | 60.7 / 97.1 / 112.7 / 116.9 | 1,610 | wrong | 0.7553 |
| T3-131 | 43.6 / 62.5 / 82.2 / 130.8 | 106.4 / 214.1 / 205.0 / 180.7 | 55.9 / 89.7 / 108.6 / 116.4 | 1,617 | `ok` | 0.7553 |
| T3-131r (second boot) | 43.6 / 62.3 / 82.0 / - | - | - | - | `ok` | - |
| T3-131-hbf16 (`HEAD=bf16`) | 41.1 / 52.9 / 77.1 / 115.4 | 98.9 / 180.4 / 173.9 / 166.9 | 59.6 / 94.2 / 105.6 / 110.9 | 1,604 | `ok` | 0.7547 |

- 0014-0015 change no speed and no short reply (answers sha `8e45971383f8`, seeded replies as before).
- `HEAD=bf16`: 289 linears packed against 290; slower on `dec.py` at every width and on Structured; the q8 head stays
  the TP=3 default. Upstream's reports of a q8-head loss (MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104 issues 63 and 66)
  are from pairs.

### A round under sparkDash's protocol, this engine (`rounds-T3-131.json`)

T=0.7, `top_p` 1, `top_k` 20, thinking on, 400 tokens, sparkDash's two prompts, not streamed; two waves a load.

| Prompt | Streams | tok/s | Tokens a round | Round, ms | Drafts a round | Accepted | Rows a round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Structured | 1 | 106.4 | 6.79 | 63.8 | 5.86 | 99% | 6.9 |
| Structured | 4 | 232.0 | 6.82 | 118.2 | 5.88 | 99% | 27.5 |
| Structured | 8 | 218.1 | 3.95 | 145.3 | 2.97 | 99% | 31.8 |
| Prose | 1 | 56.9 | 2.99 | 52.4 | 2.77 | 72% | 3.8 |
| Prose | 4 | 100.8 | 2.60 | 103.2 | 2.25 | 71% | 13.0 |
| Prose | 8 | 132.6 | 2.65 | 160.1 | 2.22 | 74% | 25.8 |

- The probe's totals sum each stream's own decode rate; sparkDash's aggregate reads the same at one stream, 8% lower
  at four and 17% lower at eight (106.4 / 214.1 / 180.7).
- A fit of the three Structured rows, round = a + b x rows + c x streams: **a = 45.6 ms, b = 1.96 ms a row, c = 4.7 ms a
  stream** `[INFERENCE: three points, three unknowns]`. The prose rows do not fit it (133 ms predicted at 8 streams,
  160 measured): sampled prose rounds cost more a stream.
- Greedy `dec.py` prompts (thinking off) draft only 1.6-2.2 tokens a stream at 52-67% acceptance, so they never fill
  the window; the window's effect is visible on the Structured prompt only.

### The Python engine at TP=3 (source and its own log)

| What | Value | Evidence |
| --- | --- | --- |
| Lone verify, ms by rows 1..8 | 24.9, 28.7, 32.5, 37.6, 41.4, 44.1, 47.0, 48.4; DFlash2 block 4.43 | the service's startup line "drafter timings" (2026-11-20 17:04 boot) |
| Window at 8 streams | 64 rows in total (`TF_GLM_MULTI_WINDOW`, 64 when `PARALLEL` > 4, else 32); 8 a stream at most | PY `scripts/config.sh:225-234`, patch 0069; startup line "verify windows 64 rows together" |
| Drafts | DFlash2 (block 8, up to 7 drafts) and copy drafts; MTP not loaded | PY `scripts/config.sh:94-98,190-219` |
| Stop rule | `fnc7:0.3`: stop when the product of noise-aware confidences falls under 0.3, beta 2.0 | PY `scripts/config.sh:198-202`, patches 0018, 0021 |
| Sampled drafts | the drafter picks `argmax(score + 0.7 g)` with the target's own keyed Gumbel draws | engine `v0.6.0` `dflash2.py:21-22,448-472`, `engine/exact_sampling.py:60-101` |
| Forward | one batched forward for all streams, a CUDA graph a row count (1-64), NCCL captured in it; the next draft pass starts at the round's end | PY patches 0030, 0069, 0010, 0100 |
| Dense weights | `DENSE=q4` on every projection with K a multiple of 64 (so the 22/21/21-head KDA stack is quantised) | PY `scripts/config.sh:166-172`, patch 0005 |
| Expert split, gathers, fabric | 6/5/5 of 16 blocks; two fp32 all-gathers a layer; `COMM=nccl` at TP=3 | PY patch 0066, `start-tp3.sh:16`; `v0.6.0` `forward.py:234-255` |
| Prefill | 2,048-row chunks, rows split across ranks for the hyper-connection glue, the next block's front run during the exchanges, chunked KDA, CUDA indexer | PY patches 0013, 0033, 0012/0014/0039, 0101-0106 |

A lone verify of 2 rows: 28.7 ms there against 35.7 ms (OLD) and 37.4 ms (NEW) here.

### 1.3.0 against the old stack (profiles in `docs/speed/tp3/profraw-*`)

Lone 2-row verify, rank 0, ms: wall 35.66 to 37.36 (+1.70). By section: shared +1.16, routed +0.84, `gather_b` +0.83,
`gather_a` +0.40, router -1.22, kda -0.35. Time outside all sections is 0.08 ms on both stacks. Rounds (greedy):
44.8 to 47.5 ms at one stream, 99.3 to 121.5 ms at eight (+22 ms: the loss grows with streams). Prompt calls for
4,096 rows: wall 1,633 to 2,123 ms (+30%); "router" (embed to the router mark) +411 ms, routed +193 ms, a new MTP
pass +21 ms.

| Cause | Evidence | Cost | Switch today |
| --- | --- | --- | --- |
| Shared windows run each stream's KDA chain and latent write one after another (old 0075, 0088 gone) | NEW `cuda_shared.zig:51-85`; `tf_glm_kda_prep_segs` absent | most of +22 ms a round at 8 streams `[INFERENCE]` | none |
| Old decode kernels gone (0030, 0072-0074, 0081-0082, 0084, 0086, 0089) | kernel names absent in NEW | routed +0.84 ms lone, more with rows | none |
| Shared expert on the main stream (old 0006, 0010 side stream gone) | OLD `cuda_engine.zig:402`; NEW `cuda_moe.zig` | +1.16 ms a forward | none |
| Prompt passes of 2,048 rows (old 0094-0095) | NEW `cuda_state.zig:19` | part of routed +193 ms | `TF_GLM_CHUNK_SOLO=4096` (measured 0-3%) |
| Prompt experts' loop (old 0078), ABSORB_MMA (0041), `bpairs8_fast` (0080), `128s2` (0009) gone | NEW `cuda_latent.zig:30-45` is SIMT only; `TF_GLM_ABSORB_MMA` is not read | most of router +411 ms `[INFERENCE]` | none (`scripts/config.sh` still sets `ABSORB_MMA=1`) |
| MTP pass over the prompt beside DFlash2 (old 0008, 0014 skipped it) | NEW `cuda_engine.zig:429`, `cuda_native.zig:282` | ~10 ms a 2,048-row call | none |
| Prompts no longer pair beside a drafter (old 0057) | NEW `cuda_prompt_pair.zig:63` | ~2% `[INFERENCE]` | none |

Upstream issue 67 reports the same three gaps (0104, 0041, 0043) from source; issue 66 the 2,048-row passes
(1,748 to ~1,300 tok/s on a pair). No maintainer reply on either as of this note.

## Gap analysis

| Case (sparkDash) | This recipe (T3-131) | Python engine | Why | Lever |
| --- | ---: | ---: | --- | --- |
| Structured x1 | 106.4 | 128.9 | round 63.8 ms against ~52 ms for ~7 rows `[INFERENCE: 47.0 + 4.4]`; tokens a round level | restore 1.3.0 (~2.7 ms); the round outside the forward; the forward's fixed cost |
| Structured x4 | 214.1 | 203.9 | level: both carry about 28-32 rows | - |
| Structured x8 | 180.7 | 325.9 | 32-row cap trims drafts 5.9 to 3.0; fixed cost | 64-row window; restore 1.3.0; fixed cost |
| Prose x1 | 55.9 | 65.9 | round time; 72% acceptance at 2.8 drafts | as Structured x1; the noise-aware stop `[INFERENCE]` |
| Prose x8 | 116.4 | 166.4 | 26 rows, 160 ms: ~15 ms a stream | restore the shared-window batching; then profile |
| Prefill 16K | 1,617 | 2,371 | OLD read 2,134: 1.3.0's prompt path; the rest is chunked KDA and the row split | restore; then task 15 |

What the cost fit says about 8 streams on Structured: with 64 rows, 45.6 + 64 x 1.96 + 8 x 4.7 = 209 ms for 54 tokens,
~260 tok/s on the probe's scale against 218 today (+20%). The Python engine needs at most 64 / 325.9 = 196 ms a round
for 64 rows. So this engine's cost a row is already close; **its fixed cost is the difference**: 45.6 ms against the
Python engine's ~21.5 ms intercept (24.9 ms at one row, +3.4 a row) `[INFERENCE]`. Of the 63.8 ms one-stream round,
the 7-row verify forward is ~47 ms; ~17 ms is outside it (draft pass, sampling, host), where the Python engine spends
4.4 ms on a draft pass it starts before the round ends. No profile covers that part on 1.3.0 (old patch 0099's round
profiler is gone).

`DENSE=q4` is not a lever here: alone it read 44.8 tok/s at one stream on OLD against 46.2 for `KDA_PAD` alone (the two
were not measured together), with NLL +4%, which is why the recipe rejected it.

## Proposed design: the epic's tasks, re-ranked

Decision: stay on main's 1.3.0 stack (a PR to the lab must sit on it; the fault is fixed there), restore what it
lost first, then the window, then the round's fixed cost. Rejected: going back to OLD (faster today, but frozen and
off main); `HEAD=bf16` and `DENSE=q4` (measured, slower or worse quality); porting the Python engine's settings as
they are (its fabric and split are the same or slower).

| Order | Task | What | Expected | Basis |
| ---: | --- | --- | --- | --- |
| 1 | new 17 | Report 1.3.0's losses upstream (draft below); port back what upstream does not: shared-window batching (0075, 0088), shared side stream (0006, 0010), the 4,096-row pass with ABSORB_MMA and the prompt experts' loop (0094-0095, 0041, 0078), no MTP prompt pass beside DFlash2 | decode +6-15% (x8 the most), prefill 1,617 to ~2,130 | measured OLD against NEW |
| 2 | 10 | 64-row shared window | Structured x6 and x8 about +20%; nothing on prose | `rounds-T3-131.json` |
| 3 | new 18 | Profile a round outside the verify forward (draft pass, sampler, host) at 1 and 8 streams; start the next draft pass before the round ends, as PY patch 0100 | up to ~10 ms of a 64 ms round `[INFERENCE]` | arithmetic above; needs the profile first |
| 4 | 12, 13 | Graph replay | unknown until 18: time outside the forward's sections is 0.08 ms, so the premise "9.5 ms outside the GPU sections" does not hold inside the forward on either stack | profiles |
| 5 | 15 | Prefill beyond the restore: chunked KDA, the row split (PY 0012-0014, 0033); upstream PRs 58 and 61 as input | 2,130 toward 2,370 | PY source |
| - | 14 | Balanced routed split | ~1.8 ms a forward; not needed for parity (the Python engine has the same split) | PY patch 0066 |
| - | 06 | Draft rule refit | low: Structured accepts 99%; prose 72% at 2.8 drafts | `rounds-T3-131.json` |

Checkpoint B (task 11), restated in numbers: after 17 and 10, `docs/rounds.py` on Structured at 8 streams should read
at least 6.5 tokens a round in at most 215 ms (about 245 tok/s on the probe's scale). Go for 18 and 12-13 only if a
round profile then shows at least 15 ms of fixed cost outside the GPU sections; otherwise bar 2 at 8 streams is out
of reach and the epic ships at milestone M.

## Risks and open questions

- **The Python engine's tokens a round and round time under the same probe are not measured.** Its replies carry
  `rounds`, `drafted` and `accepted`; `docs/rounds.py` reads them. One leg of ~10 minutes with the Python recipe
  serving settles every `[INFERENCE]` in the gap table (not run: its owner had left the service down).
- The cost fit has three points; a 64-row leg can only be run once task 10 exists.
- Ports of old patches onto 1.3.0 are real engine work (the code moved); upstream may restore some first.
- `ROCE_MULTI=1` on 1.3.0 (+4-5% at 1-2 streams, one boot) still needs its second boot.
- `KV_SHARD=1` segfaults on 1.3.0 under concurrent streams (upstream issue 63); this recipe's TP=3 does not set it.

## Overlap findings

Upstream issues 66 and 67 (1.3.0 field notes and port gaps), 64 (the long-prompt fault, closed by 0015), 39, 60, 57
and 7 (this recipe slower than the Python one on pairs: same tokens a round, longer rounds), 52 (TP=3 for this
recipe: no reply). Open PRs 58 (`PROMPT_CHUNK`) and 61 (a 256-row Q8 prompt tile) are prefill inputs on the old base.
In the Python recipe, merged PRs 112, 113 and 117 are its TP=3 decode kernels (21-22-head attention tiles, a CUDA
chunk pass, qmm without clusters).

## Draft for upstream (for the owner to send; not posted)

> **1.3.0 at TP=3 on three Sparks (and on the pair of the same kit): slower than `f350cd3e` + 0001-0124**
>
> Same kit, checkpoint and harness, a day apart. TP=3, `KDA_PAD=1`: `docs/dec.py` 46.2 / 91.6 / 154.7 tok/s at 1 / 4 / 8
> streams on the old stack, 43.6 / 82.1 / 131.9 on 1.3.0 (0014-0015 included); prefill at 16K 2,134 against 1,617
> tok/s. The pair: decode -2% to -9%, prefill -24%.
>
> From per-rank `TF_GLM_PROF` reports: a lone 2-row verify is +1.7 ms (shared expert +1.16, routed +0.84, the second
> gather +0.83, router -1.22); the round at 8 streams is +22 ms; a prompt pass of 4,096 rows is +30% (embed-to-router
> +411 ms, routed +193 ms, an MTP pass +21 ms). Time outside the profile's sections is 0.08 ms on both.
>
> What the source shows did not carry over: the batched per-stream KDA and latent writes of shared windows (old
> 0075, 0088), the shared expert's side stream (0006, 0010), the 4,096-row prompt pass (0094-0095), ABSORB_MMA
> (0041; `TF_GLM_ABSORB_MMA` is no longer read, the recipe still sets it), the prompt experts' loop (0078), the skip
> of the MTP prompt pass beside DFlash2 (0008, 0014), prompt pairs beside a drafter (0057). See also issues 66, 67.
>
> `HEAD=bf16` is slower at TP=3 (41.1 / 115.4 at 1 / 8 streams). 0015 fixes the long-prompt replies here: 3 boots of
> 3 right on a 45,597-token prompt, 4 of 7 wrong before.

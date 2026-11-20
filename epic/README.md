# Epic: TP=3 on three Sparks, faster than TP=2 and on a par with the Python engine

Branch: `tp3-perf` (from `main` at `80be4d6`, 121 engine patches). Status: **in progress** (2026-11-19: tasks 01, 02, 04 done off the Sparks; patches 0122-0124 written and host-tested, their window to come). Resuming elsewhere: read [HANDOFF.md](HANDOFF.md).

## Why

The recipe serves GLM-5.3-Flash on two Sparks (`./start.sh`) or four (`./start-tp4*.sh`). A kit of three Sparks cabled
as a triangle (every pair has its own ConnectX-7 cable) has no supported start, and the one earlier attempt (local
branch `tp3`, 2026-11-06..09, on the older 110-patch stack) bought almost nothing over two Sparks.

The same three Sparks run the lab's older Python-engine recipe
([GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
`./start-tp3.sh`, `DENSE=q4`) much faster. On sparkDash:

| Python engine, TP=3 | 1 stream | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: |
| Structured (count 1..200), T=0.7, thinking on, `top_k` 20, 400 tokens: total tok/s | 132.5 | 203.7 | 284.8 | 342.2 |
| TTFT | 142 ms | 429 ms | 339 ms | 311 ms |
| Prose (2026-11-08): total tok/s | 65.7 | 125.1 (4) | | 164.2 |

Prefill 2,341 / 2,362 / 2,375 / 2,385 tok/s at 2K / 4K / 8K / 16K.

## Goals (the bars)

1. **Bar 1:** TP=3 decode at least **+30% over this recipe's TP=2**, same harness, same day, at 1, 4 and 8 streams, on
   both harnesses (`docs/dec.py` and sparkDash Structured and prose; task 09).
2. **Milestone M (between the bars, set 2026-11-20):** TP=3 at least **+50% over this recipe's TP=2 at 4 and 8 streams
   on sparkDash Structured and prose**, same day (about 263 tok/s Structured at 8 against the pair's 175 that day).
   Read at checkpoint B (task 11) and at ship (task 16). By the projection below it needs Stage 4-5 (graph replay,
   balanced split), not Stage 2. Set after the first window measured 0122-0124 together at +31-34% over TP=2 on
   `docs/dec.py` at 1/4/8 streams, but +22% on Structured at 8 and +9% on prose at 1 (`docs/speed/tp3/RESULTS.md`).
3. **Bar 2:** TP=3 **at least the Python engine's TP=3** on sparkDash Structured (above) and prefill (~2,350 tok/s).
4. No change to TP=2 or TP=4 behaviour: every engine change is env-gated and off by default in the engine; the TP=3
   launcher turns it on.

## What is known (research, 2026-11-19)

Verified in the engine source and the `tp3` branch's logs unless marked *inferred*.

| Cause | Where | Effect at TP=3 | Fix (task) |
| --- | --- | --- | --- |
| The KDA input stack's width `3*heads*128 + 256 + heads` must be a multiple of 8. At 22/21/21 heads it is 8,726 / 8,341, so the stack, `f_b` and `g_b` are never tiled or packed (102 linears stay bf16 on the generic kernel) and never take the `f_b+g_b` pair launch | `zig/src/families/glm/cuda_kda_block.zig:32-39`, `cuda_tile.zig:31-38`, `cuda_mm.zig:348-351` | KDA 16-18.5 ms a forward (TP=2 16.4, TP=4 7.1). Startup line: 179 linears packed instead of 281 | Upstream `TF_GLM_KDA_PAD` ([05](tasks/05-patch-0122-kda-pad.md)) |
| Every small all-gather is NCCL at world 3: the two-rank RoCE path is gated on `world == 2`, patch 0108's ring on `world == 4` | `zig/src/cuda/tp_group.zig:63-67`, patch 0108 | ~90 gathers a forward on the slower wire | Upstream `TF_GLM_ROCE_MULTI` ([07](tasks/07-patch-0123-roce-multi.md)) |
| Each routed expert's 2,048 intermediate rows split in 128-row units, remainders to the lowest rank: 768 / 640 / 640. Rank 0 also takes the extra KDA, MLA, shared-expert and vocab units | `cuda_shard.zig:38`, `zig/src/cluster/canon.zig:15-36` | Rank 0 routed 41.0 ms vs ~34 on ranks 1/2 at 8 shared rows; ranks 1/2 wait in `gather_b` (9.4-9.9 vs 1.8 ms) | Per-expert balanced split ([14](tasks/14-patch-balanced-routed-split.md)) |
| No CUDA-graph replay of the GLM decode forward: ~1,100 launches a forward, ~9.5 ms a round outside the GPU sections | `docs/speed/RESULTS.md:1690` | The forward's fixed cost does not shrink with a third rank | Graph replay ([12](tasks/12-patch-verify-graph-lone.md), [13](tasks/13-patch-verify-graph-shared.md)) |
| Shared verify window capped at 32 rows; draft cut at confidence 0.7 and not coupled to the sampler's noise; draft-rule cost tables are TP=2 fits | `cuda_engine.zig:43-44`, `cuda_lanes.zig:28-29,711-714`, `draft_rule.zig:171-209` | Fewer drafted tokens a round than the Python engine (64 rows, a 0.3 stop, drafts that follow the target's keyed noise) | Settings ([03](tasks/03-baselines-and-settings-legs.md), [06](tasks/06-tp3-verify-costs-and-draft-settings.md)), 64-row window ([10](tasks/10-patch-0125-64-row-window.md)) |

Two earlier conclusions are retracted:

- *"With experts sharded 1/3, a row's experts are mostly on other ranks, so the activation exchange dominates."* Wrong:
  every rank holds a column slice of every expert; the only exchange is the two fp32 partial all-gathers a layer.
- *"TP=3 is at TP=2 parity."* That compared `docs/dec.py` (TP=3) with `docs/speed/harness/acc.py` (TP=2) on two kits.
  On `dec.py`, TP=2 read 37.6 / 72.7 / 121.1 at 1/4/8 streams, so the old TP=3 was already about +12% / +18% / +15%
  (indicative: the stacks differed).

Upstream measured the KDA pad alone at TP=3 on three GB10s: **45.6 -> 56.7 tok/s at one stream** (forward 40.3 -> 30.6
ms, KDA 15.0 -> 5.6 ms). Why TP=4 already scales: 2,048 / 4 and 16 KDA heads divide evenly; its two-row forward is
28.0 ms against TP=2's 43.8.

## Where it should land (*inferred* ceilings, every draft landing)

| After | Lone 1-row forward | 1 stream | 8-stream ceiling |
| --- | ---: | ---: | ---: |
| Today (old stack) | 36.4 ms | ~96 tok/s | ~186 tok/s |
| Stage 1 (KDA pad, settings) | ~27 ms | ~107 | ~197 |
| Stage 2 (RoCE, sampler) | ~26 ms | ~111 | ~203 |
| Stage 3 (64-row window, coupled drafts) | ~26 ms | ~111 | ~240-250 |
| Stage 4 (graph replay) | ~22-24 ms | ~122 | ~255 |
| Stage 5 (balanced split) | ~21-23 ms | ~128 | ~265-280 |
| Python engine | 19.5 ms + 3.9/row | 132.5 | 342.2 |

| Bar | Verdict |
| --- | --- |
| Bar 1, +30% over TP=2 | Reachable by Stage 2 |
| Bar 2 at 4 streams (203.7) | Likely by Stage 2-3 |
| Bar 2 at 1 stream (132.5) | Borderline: needs every stage |
| Bar 2 at 6 and 8 streams (284.8 / 342.2) | **Not covered by any known fix.** 342 tok/s at 8 needs 64 rows in ~187 ms; at today's 3.3 ms a shared row the rows alone cost 208 ms. Needs a ~25% lower per-row cost on top of everything here. Re-assessed at checkpoint B |
| Prefill 2,350 | Uncertain: ~2,000 from settings; the rest depends on the KDA pad's prompt effect (unmeasured) |

**Measured 2026-11-20** (first window, [docs/speed/tp3/RESULTS.md](../docs/speed/tp3/RESULTS.md); same day, same image):

| Leg | Lone 2-row forward, rank 0 | `dec.py` x1 / x8 | sparkDash Structured x1 / x4 / x8 | Prefill 16K |
| --- | ---: | ---: | ---: | ---: |
| TP=2 (the pair) | - | 36.5 / 118.1 | 87.7 / 175.2 / 175.3 | 1,725 |
| TP=3, switches off | 41.4 ms | 39.5 / 139.7 | 103.6 / 205.9 / 200.7 | 1,796 |
| TP=3, `KDA_PAD=1` (the default now) | 35.7 ms | 46.2 / 154.7 | 109.6 / 210.5 / 219.0 | 2,134 |
| TP=3, 0122-0124 together | 33.8 ms | 48.9 / 154.9 | 120.9 / 221.5 / 213.2 | 2,055 |
| Python engine, TP=3 (P3) | - | - | 128.9 / 203.9 / 325.9 | 2,371 |

Bar 1 is met on `dec.py` (+31-34% with 0122-0124), not on sparkDash (+20-25% Structured with `KDA_PAD`); bar 2 at
Structured x4 only. The projection's Stage 1-2 rows (~107-111 tok/s at one stream, ~197-203 at eight) match the
measured Structured (109.6-120.9 at one stream, 213-219 at eight), so its later rows remain the working estimate.

Caveats: the table's forward times are inferred on the older stack; the measured block above is the reference now, and the Python engine runs `DENSE=q4`, which this recipe rejected on quality on the pair.

## Constraints

- **The three Sparks serve production** on the Python-engine recipe. Every measurement session is a maintenance window
  agreed with the owner beforehand; restore that service and check `/health` at the end of each session.
- `./start.sh` (the production pair) is not modified.
- Engine changes ship as new numbered patches `patches/engine/0122+`, built from a fresh engine worktree at `f350cd3e`
  plus the 121 patches ([04](tasks/04-engine-worktree-and-patch-workflow.md)).
- Committed files carry no kit-private details (addresses, ssh users, host names): this branch is public.

## Gates (every engine task)

- TP=3 has its own baselines; TP=2 token hashes are not a gate at another world size.
- Greedy replies repeat run to run (`docs/dec.py --contract` twice), drafts on equal drafts off, solo equal concurrent.
- `docs/quality.py` NLL and answers within the TP=3 q8 reference (0.7597 / `639c916612ba` on the old stack; re-taken in
  [03](tasks/03-baselines-and-settings-legs.md)).
- A patch whose switch is off leaves TP=2 replies byte-identical (contract hashes against the image before it).
- All patches apply to `f350cd3e` with no rejects; the clean build prints no compiler warning (0115's standard).
- A gain counts when it clears the leg's noise (two boots, or 2x the run-to-run spread).

## Stages and tasks

| # | Task | Stage | Depends on | Estimate | Status |
| --- | --- | --- | --- | --- | --- |
| [01](tasks/01-launcher-three-sparks.md) | Launcher for three Sparks: the ring launcher at TP=3 | 0 | | 2-3 d | done: LIVE on the kit 2026-11-20; `TP3_NCCL=ring` does not start |
| [02](tasks/02-launcher-tests-and-leg-tooling.md) | Launcher tests; TP=3 `safestart` and `leg` | 0 | 01 | 1 d | done (offline tests pass) |
| [03](tasks/03-baselines-and-settings-legs.md) | Baselines and settings legs (maintenance window) | 0 | 01, 02 | 1 window | done 2026-11-20 (first window; [RESULTS](../docs/speed/tp3/RESULTS.md)) |
| [04](tasks/04-engine-worktree-and-patch-workflow.md) | Engine worktree and patch workflow | 1 | | 0.5 d | done: host tests at `recipe-0121` = stale failures, the baseline (task file) |
| [05](tasks/05-patch-0122-kda-pad.md) | Patch 0122: `TF_GLM_KDA_PAD` | 1 | 04 | 1-2 d + window | done: `KDA_PAD=1` the TP=3 default (decode +10-17%, prefill +17-19%) |
| [06](tasks/06-tp3-verify-costs-and-draft-settings.md) | TP=3 verify costs and draft settings | 1 | 03, 05 | 0.5 d + window | settings measured (within noise); TP=3 cost refit to do |
| [07](tasks/07-patch-0123-roce-multi.md) | Patch 0123: `TF_GLM_ROCE_MULTI` | 2 | 04 | 2-3 d + window | measured: exact, +6% x1, prefill -3-6%: off (prompt-path cost to find) |
| [08](tasks/08-patch-0124-sampler-rows-to-rank0.md) | Patch 0124: sampled rows to rank 0 only | 2 | 04 | 2-3 d | measured: exact, no gain: off |
| [09](tasks/09-checkpoint-a.md) | Checkpoint A: bar 1 | 2 | 05-08 | 1 window | todo |
| [10](tasks/10-patch-0125-64-row-window.md) | Patch 0125: 64-row shared verify window | 3 | 09 | 1-2 w | todo |
| [11](tasks/11-checkpoint-b.md) | Checkpoint B: bar 2 at 4/6/8 streams; go / no-go for 12-14 | 3 | 10 | 1 window | todo |
| [12](tasks/12-patch-verify-graph-lone.md) | Patch: CUDA-graph replay of the lone verify forward | 4 | 11 | ~2 w | todo |
| [13](tasks/13-patch-verify-graph-shared.md) | Patch: graph replay for shared windows | 4 | 12 | 2-3 w | todo |
| [14](tasks/14-patch-balanced-routed-split.md) | Patch: balanced routed-expert split | 5 | 11 | 2-3 w | todo |
| [15](tasks/15-prefill.md) | Prefill at TP=3 | 1-5 | 03, 05 | spread | todo |
| [16](tasks/16-ship.md) | Ship: README, CHANGELOG, results | 6 | 09 (and later stages) | 1 d | todo |

Checkpoint A (task 09) is where bar 1 is judged. Checkpoint B (task 11) decides whether the expensive stages (12-14)
are worth starting. Patch numbers after 0125 are assigned in the order patches land.

## References

- Earlier attempt: `git show tp3:docs/TP3-PERF-PLAN.md`, `git show tp3:docs/speed/tp3/RESULTS.md`, and the
  `tp3` branch's `start-tp3.sh`, `scripts/nodes.sh` (`pair_links`, `detect_links`), `scripts/tests/pair-links.sh`.
- Four-Spark work this builds on: `scripts/tp4.sh`, `start-tp4-switchless.sh`, `docs/speed/RESULTS.md` (TP=4 sections,
  lines ~1500-1940), `docs/speed/codex-tp4.md`, `docs/speed/codex-roce-ring.md`.
- Upstream TensorFold commits after the pin used here: KDA pad `96b3b7aa9..c309bb354`; multirank RoCE `1911e7bc3`,
  `81e25bb55`, `b99c48a53`, `65ec7eadd`, `1e9975556`, `ed20c5be4`; prompt docs `2914b6883`, `d0695af5a`.

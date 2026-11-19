# For the agent on rank 0's Spark: TP=3 epic, the first maintenance window

Written 2026-11-19 on a devbox (x86_64, no GPU) at `tp3-perf` `71f19d6`. Everything below that can be done without
the Sparks is done; what is left needs the three Sparks, and the Sparks serve production. Read this, then
`epic/README.md` (goals, gates) and `epic/tasks/03-baselines-and-settings-legs.md` (the window's legs). `HANDOFF.md`
has the longer background.

## Stop first: production

- The three Sparks serve production on the **older Python-engine recipe** (container `glm53-flash-tf`, its own
  checkout and `./start-tp3.sh`). Do **not** stop, restart or load it, and do **not** build an image here while it
  serves (rank 0 had ~9 GiB available; the kit's earlyoom kills rank 0 near ~3 GiB).
- **Ask the user for a maintenance window before anything that touches the Sparks**, even read-only probes. The user
  says when it starts; nothing in this file is pre-approved.
- Every window ends by restoring that service with its own `./start-tp3.sh`, checking `/health` and one reply.
- This fork is public: no addresses, ssh users or host names in committed files. They live in this Spark's untracked
  `scripts/local.sh`.

## What is on the branch (all off by default)

| Piece | What | Switch (recipe / engine) | State |
| --- | --- | --- | --- |
| Launcher | `./start-tp3.sh`, `./stop-tp3.sh` over the four-Spark scripts at `TP=3`; triangle or switch found from the links | `TP3_NCCL=eager` (default) or `ring` | offline tests pass; never run on the kit |
| 0122 | `TF_GLM_KDA_PAD`: the KDA input stack padded to one 8-aligned width (8,728) so it is packed and tiled at 22/21/21 heads; recipe-only `chainSegs` stride bug fixed | `KDA_PAD=1` / `TF_GLM_KDA_PAD=1` | host tests only |
| 0123 | `TF_GLM_ROCE_MULTI`: small all-gathers on fabric/tp's multirank RoCE, lent only if every rank's probe equals NCCL's | `ROCE_MULTI=1` (+ `ROCE_MULTI_KB=512`, `ROCE_MULTI_FALLBACK=error`) | host tests only; the image now builds `/opt/tensorfold/lib/tp_mailbox.fatbin` |
| 0124 | `TF_GLM_SAMPLE_ROOT`: a verify step's sampled rows to rank 0 in one grouped send a follower, instead of a broadcast a row a rank | `SAMPLE_ROOT=1` / `TF_GLM_SAMPLE_ROOT=1` | host tests only |

The engine switches are passed by the start at `TP=3` only. Settings and fingerprint layout is 20 (new fields: 21+).
Host tests already fail five stale tests at the untouched `recipe-0121` tree (listed in task 04); the same five are
the baseline, not a regression.

## Before the window (no Sparks touched)

```sh
git fetch origin && git switch tp3-perf && git pull --ff-only
bash scripts/tests/run.sh          # all pass ("shellcheck: not installed" is fine)
```

`scripts/local.sh` needs `WORKER` and `WORKER2` (the other two Sparks, any order) for `./start-tp3.sh`; the pair's
`WORKER` line stays as it is (`./start.sh` ignores `WORKER2`). Check `PORT`, `PARALLEL` and `WORKER_WEIGHTS`
(`rsync`, `nfs` or `check`; the workers may already hold the checkpoint from the Python recipe's runs, at another
path). Ask the user for anything missing; do not guess the kit's addresses.

## In the window, in this order

1. **Record production, then stop it.** `docker inspect glm53-flash-tf` on every Spark into the window's notes (not
   committed if it holds addresses); its sparkDash numbers (P3 leg); then its own `./stop.sh`. Record GPU clocks on
   all three Sparks.
2. **Read-only first:** `DRY_RUN=1 ./start-tp3.sh`. Expect three ranks, `A triangle: ...` with each rank's devices
   toward each peer, `TF_GLM_WORLD=3`, the eager NCCL set, and nothing written. If it says `switch` or stops with
   `E_FABRIC`, stop and report the lines to the user.
3. **First start** (it builds the 124-patch image on this Spark, ~6 min, and sends it to the workers):
   `./start-tp3.sh`. Check the build log for compiler warnings (patch 0115's standard: none), then LIVE and its smoke
   test. Rank 0's startup line: linears packed (179 without the pad is expected here).
4. **GPU checks in the image** (`/opt/tensorfold/bin/tf-cuda-test`, `tf-tp-test`), with the server stopped
   (`./stop-tp3.sh`):
   - `tf-cuda-test glm-kda-pad-split` (one GPU, synthetic);
   - `tf-cuda-test glm-shared-check`, `glm-pair-check`, `glm-lanes-concurrent` (`<ckpt> <dc.json> <rank> <world>
     <idfile>`) at world 3 with `TF_GLM_KDA_PAD=1`, one per Spark;
   - `tf-tp-test suite` on the triangle before any `ROCE_MULTI=1` leg (bytes equal NCCL's at every size).
5. **Task 03's legs** (`TP=3 docs/leg.sh <label> KEY=VALUE ...`: it restarts through `docs/safestart.sh`, writes
   `docs/speed/tp3/`). T2 (the pair, `./start.sh`) and T3 (all switches off) first: T3's contract becomes
   `docs/speed/tp3/contract-baseline.json`. Then the settings legs in task 03's table.
6. **The patches, one leg each, then together:** `KDA_PAD=1`, `ROCE_MULTI=1`, `SAMPLE_ROOT=1`, then all three. Same
   measurement set as task 03 (sparkDash Structured and prose at 1/4/6/8, prefill, `dec.py`, `--contract`,
   `quality.py`, `TF_GLM_PROF=1` per rank).
7. **End:** `./stop-tp3.sh` (archives three logs), restart production with its own `./start-tp3.sh`, check `/health`
   and one reply. Tell the user it is back.

## Gates per patch (from `epic/README.md`)

- **0122:** replies differ from T3 (q8 packed instead of bf16; expected) but are repeatable (`--contract` twice),
  drafts on = drafts off, solo = concurrent, and `quality.py` within the TP=3 reference. Rank 0's startup line says
  281 linears packed. KDA per forward near TP=4's share (~7-8 ms a 2-row forward). Concurrent streams are the case
  the `chainSegs` fix is about: check them explicitly.
- **0123:** contract hashes identical to the same image with it off; gathers down in the per-rank profile; decode up
  beyond noise; prefill not worse than 1-2%. With `ROCE_MULTI_FALLBACK=error` a wire that does not open stops the
  start: report its lines rather than switching to `nccl` silently.
- **0124:** a fixed seed's sampled replies identical with it off and on (TP=3, and TP=2 if time allows); the sampler
  section's time down at 4 and 8 sampled streams, or the patch is dropped.
- **Every patch, switch off:** TP=2 (`./start.sh` on the new image) contract hashes equal to
  `docs/speed/contract-baseline.json`.
- A gain counts when it clears the leg's noise (two boots, or 2x the run-to-run spread).

## Write back

`docs/speed/tp3/RESULTS.md` (a table per leg, the per-rank profile table, findings), the legs' JSON, the task files'
status lines, and `epic/README.md`'s "Where it should land" with measured numbers. A patch that passes its gates:
set its recipe default to 1 at `TP=3` in `scripts/config.sh` (`KDA_PAD`, `ROCE_MULTI` with
`ROCE_MULTI_FALLBACK=nccl`, `SAMPLE_ROOT`). Commit (end each message with the session's attribution line) and push to
the fork (`origin`) only; the user opens PRs against `MiaAI-Lab`.

## If something breaks

- A rank exits at start: `logs/current/rank*.log`; `./stop-tp3.sh` archives them. A fingerprint mismatch names the
  setting that differs between ranks.
- The engine tree for edits is rebuilt anywhere with `epic/engine-worktree.sh <dir>` (tag `recipe-0124`, tree
  `463f5a66...`); host tests as in `HANDOFF.md`. Do not build on the Sparks outside the window.

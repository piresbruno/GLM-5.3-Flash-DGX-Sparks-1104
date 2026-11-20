# For the agent on rank 0's Spark: TP=3 epic, the first maintenance window

Written 2026-11-19 on a devbox (x86_64, no GPU) at `tp3-perf` `71f19d6`. Everything below that can be done without
the Sparks is done; what is left needs the three Sparks, and the Sparks serve production. Read this, then
`epic/README.md` (goals, gates) and `epic/tasks/03-baselines-and-settings-legs.md` (the window's legs). `HANDOFF.md`
has the longer background.

**The first window ran 2026-11-19/20:** results and decisions in `docs/speed/tp3/RESULTS.md`; one leg per run with
`docs/tp3leg.sh <label> KEY=VALUE ...` and tables with `python3 -I docs/tp3results.py`.

## Stop first: production

- The three Sparks serve production on the **older Python-engine recipe** (container `glm53-flash-tf`, its own
  checkout and `./start-tp3.sh`). Do **not** stop, restart or load it, and do **not** build an image here while it
  serves (rank 0 had ~9 GiB available; the kit's earlyoom kills rank 0 near ~3 GiB).
- **Ask the user for a maintenance window before anything that touches the Sparks**, even read-only probes. The user
  says when it starts; nothing in this file is pre-approved.
- **Production is a systemd user unit** (found in the first window): `glm53-tensorfold.service` runs that recipe's
  `./start-tp3.sh`, and `glm53-nightly.timer` restarts it at 04:00 UTC every day, **even when it is down**. Stop it with
  `systemctl --user stop glm53-tensorfold.service` and restore it with `systemctl --user start ...`, not with its own
  scripts. A window that crosses 04:00 UTC: ask the user before stopping the timer, and start it again at the end.
- Every window ends by restoring that service (`systemctl --user start glm53-tensorfold.service`), checking `/health`
  and one reply.
- This fork is public: no addresses, ssh users or host names in committed files. They live in this Spark's untracked
  `scripts/local.sh`.

## What is on the branch (all off by default)

The second window runs on **main's TensorFold 1.3.0 stack** (`81de26f6` + patches 0001-0013; `epic/README.md`,
"TensorFold 1.3.0"). This branch carries no engine patch of its own any more: the TP=3 switches are upstream's.

| Piece | What | Switch (recipe / engine) | State |
| --- | --- | --- | --- |
| Launcher | `./start-tp3.sh`, `./stop-tp3.sh` over the four-Spark scripts at `TP=3`; triangle or switch found from the links | `TP3_NCCL=eager` (default); `ring` does not start (RESULTS.md finding 12) | LIVE on the kit (first window, old stack) |
| KDA pad | upstream `TF_GLM_KDA_PAD` (was patch 0122): the KDA input stack padded to one 8-aligned width | `KDA_PAD=1` (**the TP=3 default**) | measured on the old stack: decode +10-17%, prefill +17-19%; re-measure on 1.3.0 |
| Multirank RoCE | upstream `TF_GLM_ROCE_MULTI` (was patch 0123): small all-gathers on fabric/tp's multirank RoCE | `ROCE_MULTI=1` (off; `ROCE_MULTI_KB=128`, `ROCE_MULTI_FALLBACK=error`) | measured on the old stack: exact, +6% x1, prefill -3-6%; re-measure on 1.3.0 |
| `tf-tp-test` | the image embeds the mailbox kernel the server loads (`-Dtp-fatbin`) | - | `tf-tp-test suite` passed on the triangle (old stack) |

`SAMPLE_ROOT` (was patch 0124) is gone with 1.3.0 and refused by name. The first window's numbers
(`docs/speed/tp3/RESULTS.md`) are the old stack's: this window takes them again on 1.3.0.

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
   committed if it holds addresses); its sparkDash numbers (P3 leg, `python3 -I docs/sd.py P3 docs/speed/tp3`); then
   `systemctl --user stop glm53-tensorfold.service`. Record GPU clocks on
   all three Sparks.
2. **Read-only first:** `DRY_RUN=1 ./start-tp3.sh`. Expect three ranks, `A triangle: ...` with each rank's devices
   toward each peer, `TF_GLM_WORLD=3`, the eager NCCL set, and nothing written. If it says `switch` or stops with
   `E_FABRIC`, stop and report the lines to the user.
3. **First start** (it builds the 1.3.0 image on this Spark, ~10 min, and sends it to the workers):
   `./start-tp3.sh` (with `KDA_PAD=1`, the default). Check the build log for compiler warnings (none), then LIVE and
   its smoke test. Rank 0's startup line: linears packed (the old stack read 282 with the pad, 180 without).
4. **GPU checks in the image** (`/opt/tensorfold/bin/tf-cuda-test`, `tf-tp-test`), with the server stopped
   (`./stop-tp3.sh`):
   - the GPU checks 1.3.0 has for the pad at world 3 (`tf-cuda-test` with no argument lists them; the old stack's were
     `glm-kda-pad-split`, `glm-shared-check`, `glm-pair-check`, `glm-lanes-concurrent`, `<ckpt> <dc.json> <rank>
     <world> <idfile>`, one per Spark). `dc.json`: prompts of 2.6K-5.2K tokens, or the chunk-at-a-time cases never
     run (RESULTS.md finding 1); the first window's runner and `dc.json` are in rank 0's untracked
     `logs/window-2026-11-19/`;
   - `tf-tp-test suite` on the triangle before any `ROCE_MULTI=1` leg (bytes equal NCCL's at every size).
5. **Baselines on 1.3.0** (`docs/tp3leg.sh <label> KEY=VALUE ...`: a whole leg, files under `docs/speed/tp3/`; give
   the labels a suffix, e.g. `T3-130`, so the first window's files stay). T2 (`TP=2`, the pair) and T3 (`KDA_PAD=0`,
   every switch off) first, then T3 again on a second boot. T3's contract on 1.3.0 replaces
   `docs/speed/tp3/contract-baseline.json`.
6. **The switches:** `KDA_PAD=1` (the default), then `KDA_PAD=1 ROCE_MULTI=1` (and its prefill against `KDA_PAD=1`
   alone, at 4K-16K above all). The draft settings of task 03 that remain on 1.3.0 (`TF_GLM_DRAFT_CONFIDENCE`, `L2PF`,
   `TF_GLM_CHUNK_SOLO=4096 TF_GLM_OVERLAP_PIECES=8`) only if time allows.
7. **End:** `./stop-tp3.sh` (archives three logs), `systemctl --user start glm53-tensorfold.service`, check `/health`
   and one reply. Tell the user it is back.

## Gates per patch (from `epic/README.md`)

- **`KDA_PAD`:** replies differ from T3 (q8 packed instead of bf16; expected) but are repeatable (`--contract` twice),
  drafts on = drafts off, solo = concurrent, and `quality.py` within the TP=3 reference. KDA per forward near TP=4's
  share (~7-8 ms a 2-row forward; 9.3 ms on the old stack). Concurrent streams with long prompts: check explicitly.
- **`ROCE_MULTI`:** contract hashes identical to the same image with it off; gathers down in the per-rank profile; decode up
  beyond noise; prefill not worse than 1-2%. With `ROCE_MULTI_FALLBACK=error` a wire that does not open stops the
  start: report its lines rather than switching to `nccl` silently.
- **TP=2:** T2 on 1.3.0 is the pair's new reference (main's stack changed the pair's replies before: RESULTS.md
  finding 6); the TP=3 switches do not reach `./start.sh`.
- A gain counts when it clears the leg's noise (two boots, or 2x the run-to-run spread).

## Write back

`docs/speed/tp3/RESULTS.md` (a table per leg, the per-rank profile table, findings), the legs' JSON, the task files'
status lines, and `epic/README.md`'s "Where it should land" with measured numbers. A patch that passes its gates:
set its recipe default to 1 at `TP=3` in `scripts/config.sh` (`KDA_PAD`, `ROCE_MULTI` with
`ROCE_MULTI_FALLBACK=nccl`). Before committing, `SCRUB_HOSTS=<the kit's host names> python3 -I
docs/scrub.py docs/speed/` (leg logs carry ssh targets, sparkDash files the unit's host name). Commit (end each
message with the session's attribution line) and push to the fork only (on rank 0's clone the fork is the `fork`
remote and `origin` is `MiaAI-Lab`); the user opens PRs against `MiaAI-Lab`.

## If something breaks

- A rank exits at start: `logs/current/rank*.log`; `./stop-tp3.sh` archives them. A fingerprint mismatch names the
  setting that differs between ranks.
- The engine tree for edits is rebuilt anywhere with `epic/engine-worktree.sh <dir>` (the recipe's `TF_REF` plus
  main's 0001-0013, tag `recipe-0013`); host tests as in `HANDOFF.md`. Do not build on the Sparks outside the window.

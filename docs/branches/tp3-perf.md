# Branch: tp3-perf

## Business context

A kit of three Sparks cabled as a triangle has no supported start in this recipe, and the same three Sparks run the
lab's older Python-engine recipe much faster. This branch makes TP=3 a supported start that is faster than TP=2 and on
a par with the Python engine. The goals, causes, gates and task list are in `epic/README.md`; `epic/HANDOFF.md` says
where things stand.

## What changed

- `./start-tp3.sh` and `./stop-tp3.sh` over the four-Spark ring launcher at `TP=3`.
- Main's TensorFold 1.3.0 (`81de26f6`, patches 0001-0013) merged in on 2026-11-20. The branch's own engine patches are
  dropped: 0122 (`TF_GLM_KDA_PAD`) and 0123 (`TF_GLM_ROCE_MULTI`) are upstream in 1.3.0, so `KDA_PAD` and `ROCE_MULTI`
  now drive upstream's code; 0124 (`TF_GLM_SAMPLE_ROOT`) has no counterpart and `SAMPLE_ROOT=1` is refused by name.
- Before 1.3.0 (measured in the first window): engine patches 0122, 0123 and 0124, each off by
  default and passed by the start at `TP=3` only.
- `epic/`: the plan, the task files, the devbox handoff and the runbook for rank 0's Spark.
- The first maintenance window (2026-11-19/20) measured every leg of task 03 and the three patches on the three
  Sparks: `docs/speed/tp3/RESULTS.md`. `KDA_PAD=1` is now the TP=3 default (decode +10-17%, prefill +17-19%);
  `ROCE_MULTI` and `SAMPLE_ROOT` stay off (0123 costs prefill, 0124 gains nothing), `ROCE_MULTI_KB` defaults to 128.
- Window tooling: `docs/tp3leg.sh` (one leg's whole measurement set), `docs/sd.py` (sparkDash through its API),
  `docs/seeded.py` (fixed-seed sampled replies), `docs/tp3results.py` (the tables), `docs/scrub.py` (kit details out of
  committed files); `docs/leg.sh` reads `PORT` from `scripts/local.sh`; `docs/dec.py` can sweep sampled
  (`DEC_TEMPERATURE`, the contract stays greedy).
- `epic/README.md`: milestone M between bar 1 and bar 2 (+50% over TP=2 at 4 and 8 streams on sparkDash).
- `scripts/tests/tp3-config.sh` runs in a sandbox without `scripts/local.sh`, so a kit's local settings no longer
  change the defaults it checks.
- `./start-tp3.sh` reaches rank 1 again in its weights step: `tp4_prepare_worker_weights` read every worker's ssh
  target before declaring the `local WORKER` that nodes.sh's helpers read (`scripts/tests/tp3-worker-weights.sh`).
- The image's `tf-tp-test` embeds fabric/tp's mailbox kernel (`-Dtp-fatbin`, the same `tp_mailbox.fatbin` the server
  loads for `ROCE_MULTI=1`), so `tf-tp-test suite` runs on `TF_COMM_BACKEND=roce`. New image tag.

## Business rules and constraints

- `./start.sh` (the production pair) is not modified; TP=2 and TP=4 behave as before with every switch unset.
- The three Sparks serve production: every measurement session is a maintenance window agreed beforehand.
- No kit-private details (addresses, users, host names) in committed files.

## Edge cases handled

- Merging main's new stack left 0122-0124 behind with no conflict and every test passing, though none applies to 1.3.0
  (`patch-numbers.sh` checks numbers only): they were removed in the merge itself, and the documents say so.

- `tp3-config.sh` on a kit: rank 0's `local.sh` set `NCCL_CHANNELS=8`, which made the four-Spark default check read 8
  instead of 4. The sandbox copies `scripts/` without `local.sh` and links the rest of the repository.
- The first `DRY_RUN=1 ./start-tp3.sh` on the kit stopped after the links with `!v: unbound variable`: a `local WORKER`
  shadowed `scripts/local.sh`'s value while `wtarget 1` read it by name (bash's dynamic scope). Rank 2 (`WORKER2`) was
  unaffected; every `WORKER_WEIGHTS` mode hit it. Fixed, and covered for rsync, rsync with `DRY_RUN=1`, and check.

- `tf-tp-test suite` at world 3 stopped on every rank with "needs the mailbox kernel": `zig build install` embeds it
  only when given `-Dtp-nvcc` or `-Dtp-fatbin`, and the Dockerfile built the fatbin after the Zig build, for the
  server only. On the kit after the fix: 402/402/328 checks pass, every RoCE result exact against NCCL.

## Integration impact

- None on the pair or on four Sparks while the TP=3 switches are unset.

## Out of scope and follow-ups

- The follow-ups of the first window (`docs/speed/tp3/RESULTS.md`): 0123's prompt-path cost; `TP3_NCCL=ring` with
  `PREFILL_REDUCE=gather`; a TP=3 cost refit for the tpm draft rule; the TP=2 gate's reference
  (`contract-up-base.json`, not `contract-baseline.json`).
- README and CHANGELOG for three Sparks: task 16 (ship).
- The second window: T2, T3 and the switches again on 1.3.0, with the GPU checks of the pad at world 3.

## Verification

- The first window on the kit: the GPU checks of 0122 at world 3, `tf-tp-test suite` on the triangle (0 failed), 17
  measured legs, T3's contract repeated on a second boot, and TP=2 on the new image byte-identical to the lab's
  2026-11-11 pair base (`contract-up-base.json`). Production restored through systemd at the end and checked.

- `bash scripts/tests/run.sh` passes on rank 0's Spark with its `local.sh` present (it failed there before the fix).
- `.refract/bin/validate` passes.

## Changelog

- merge: main's TensorFold 1.3.0 stack; 0122-0124 dropped (two upstream, one gone), `SAMPLE_ROOT` refused by name.
- feat(tp3): the first window's results; `KDA_PAD=1` the TP=3 default, `ROCE_MULTI_KB=128`; window tooling; milestone M.
- fix(docker): `tf-tp-test` embeds the mailbox kernel the server loads, so its RoCE suite runs at TP=3.
- fix(tp3): `./start-tp3.sh`'s weights step reads rank 1's ssh target before the local `WORKER` shadows it.
- fix(tests): `tp3-config.sh` runs in a sandbox without `scripts/local.sh`, so it passes on a kit.

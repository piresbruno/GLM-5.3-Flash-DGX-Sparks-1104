# Branch: tp3-perf

## Business context

A kit of three Sparks cabled as a triangle has no supported start in this recipe, and the same three Sparks run the
lab's older Python-engine recipe much faster. This branch makes TP=3 a supported start that is faster than TP=2 and on
a par with the Python engine. The goals, causes, gates and task list are in `epic/README.md`; `epic/HANDOFF.md` says
where things stand.

## What changed

- `./start-tp3.sh` and `./stop-tp3.sh` over the four-Spark ring launcher at `TP=3`.
- Engine patches 0122 (`TF_GLM_KDA_PAD`), 0123 (`TF_GLM_ROCE_MULTI`) and 0124 (`TF_GLM_SAMPLE_ROOT`), each off by
  default and passed by the start at `TP=3` only.
- `epic/`: the plan, the task files, the devbox handoff and the runbook for rank 0's Spark.
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

- Every measurement: the first maintenance window (`epic/DGX-AGENT.md`, task 03).
- The window's measurement tooling at TP=3 (port from `local.sh`, Structured and prose, prefill ladder, per-rank
  profile): planned, not started.

## Verification

- `bash scripts/tests/run.sh` passes on rank 0's Spark with its `local.sh` present (it failed there before the fix).
- `.refract/bin/validate` passes.

## Changelog

- fix(docker): `tf-tp-test` embeds the mailbox kernel the server loads, so its RoCE suite runs at TP=3.
- fix(tp3): `./start-tp3.sh`'s weights step reads rank 1's ssh target before the local `WORKER` shadows it.
- fix(tests): `tp3-config.sh` runs in a sandbox without `scripts/local.sh`, so it passes on a kit.

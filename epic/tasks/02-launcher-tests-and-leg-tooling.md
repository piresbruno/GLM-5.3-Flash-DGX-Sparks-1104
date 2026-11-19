# 02 - Launcher tests; TP=3 `safestart` and `leg`

Stage 0. Depends on: 01. Estimate: 1 day. Status: todo.

## Goal

The three-Spark launcher is covered by the offline tests in `scripts/tests/`, and the A/B tooling the speed work uses
(`docs/safestart.sh`, `docs/leg.sh`) runs at TP=3.

## Steps

1. **`scripts/tests/pair-links.sh`:** port `git show tp3:scripts/tests/pair-links.sh` against `scripts/tp4.sh`'s
   `pair_links` / `detect_links`, with fake inventories for:
   - a triangle (three subnets, one per pair) -> `triangle`, ring `0 1 2`;
   - three Sparks on one switch subnet -> `switch`;
   - a missing cable (two of three pairs) -> `E_FABRIC` naming the pair;
   - four-Spark ring and switch inventories -> the same classification as on `main` (regression).
2. **`scripts/tests/tp3-config.sh`:** `prepare.sh`'s checks at `TP=3`: `WORKER3` not required; `FP8=1`, `ROCE_RING=on`
   and `KV_SHARD=1` refused by name; `PREFILL_REDUCE=split` accepted; `TP=5` refused.
3. **`scripts/tests/run.sh`:** add `start-tp3.sh stop-tp3.sh` to the `bash -n` and shellcheck lists (lines 5, 8).
4. **`docs/safestart.sh`** reads `WORKER` only and polls a hard-coded `:8888`: take the port from `PORT`, and stop and
   start through `./stop-tp3.sh` / `./start-tp3.sh` when `TP=3` (workers `WORKER`, `WORKER2`).
5. **`docs/leg.sh`:** pass `TP` through; write the leg's files under `docs/speed/tp3/` when `TP=3`; compare against
   `docs/speed/tp3/contract-baseline.json` (task 03 writes it) instead of the pair's.

## Acceptance

- `bash scripts/tests/run.sh` passes, including the new tests, with and without shellcheck installed.
- `TP=3 DRY_RUN=1 docs/safestart.sh` prints the three-rank stop and start it would run.

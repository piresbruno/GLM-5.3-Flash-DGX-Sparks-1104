# 16 - Ship: README, CHANGELOG, results

Stage 6 (a first, partial ship is possible after task 09). Depends on: 09 and whichever later stages landed.
Estimate: 1 day. Status: todo.

## Steps

1. README: a "3 Sparks (TP=3, experimental)" section beside "4 Sparks": cabling (a triangle, one subnet per cable),
   `./start-tp3.sh` / `./stop-tp3.sh`, settings at TP=3 and what is refused (FP8, `ROCE_RING`, `KV_SHARD`), memory at
   the defaults (rank 0's startup line), and the measured tables against TP=2 and the Python engine, with the gap stated
   plainly where a bar is not met.
2. README patch table: rows for 0122 onward; the patch count in the header and "What is checked".
3. CHANGELOG entry per patch and for the launcher, with measured numbers and the gates each passed.
4. `docs/speed/tp3/RESULTS.md` complete; contract and quality JSON for each stage beside it.
5. `scripts/local.sh.example`: the TP=3 block.
6. `bash scripts/tests/run.sh`; a final clean image build (no compiler warning); `DRY_RUN=1` of all three launchers.
7. Mark the epic's tasks done and update its tables with the final numbers.

## Acceptance

- Every number in the README traces to a file under `docs/speed/tp3/`.
- No kit-private details in committed files.

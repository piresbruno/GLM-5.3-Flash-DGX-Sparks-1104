# 09 - Checkpoint A: bar 1 (+30% over TP=2)

Stage 2. Depends on: 05, 06, 07, 08. Estimate: one window. Status: todo.

## Goal

Judge bar 1 on one image with every Stage 1-2 patch, against TP=2 on the same image, same day, same harness.

## Steps

1. One image with 0122-0124; TP=3 with the gated defaults from tasks 05-08.
2. Legs: TP=2 (`./start.sh`, switches unset) and TP=3, two boots each, interleaved (2, 3, 2, 3) to expose drift.
3. Measurement set from task 03. Bar 1 is read on `docs/dec.py` (greedy) and sparkDash prose and Structured, at 1, 4
   and 8 streams.
4. Refresh the epic's projection table with the measured forward lines.

## Acceptance

- Bar 1 met (TP=3 at least 1.30x TP=2 at 1, 4 and 8 streams on both harnesses), or the shortfall recorded with the
  per-rank profile that explains it.
- TP=2 contract hashes equal to the pre-epic image's.
- Decision recorded: ship Stage 1-2 now (task 16, partial) or continue to Stage 3 first.

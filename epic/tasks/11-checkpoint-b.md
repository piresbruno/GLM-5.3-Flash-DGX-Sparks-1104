# 11 - Checkpoint B: bar 2 at 4/6/8 streams; go / no-go for 12-14

Stage 3. Depends on: 10. Estimate: one window. Status: todo.

## Goal

Measure bar 2 against the Python engine on the same day, and decide whether the expensive stages (12-14) are worth
starting.

## Steps

1. Python-engine TP=3 (as it serves) and this recipe's TP=3 at the Stage 3 defaults, two boots each, interleaved.
2. sparkDash Structured and prose at 1/4/6/8 streams; prefill 2K-128K; per-rank profile of a lone and an 8-stream
   shared round on this recipe.
3. From the profile: the lone line (`a + b*rows`), the shared line, the drafter pass, the host gap.

## Decision rule

- Remaining 1-stream gap is mostly the host gap (round minus GPU sections) -> start task 12.
- Remaining 8-stream gap with the shared per-row cost still near 3.3 ms -> 6/8-stream parity is out of reach with the
  known list; record that plainly. Task 14 (balanced split) cuts rank 0's routed share and is the only listed per-row
  lever; start it only if the profile shows rank 0's routed path is the critical section.
- Bar 2 already met -> go to task 16.

## Acceptance

A recorded decision with numbers, and the epic's table updated. Milestone M (`epic/README.md`: +50% over TP=2 at 4 and 8
streams on sparkDash Structured and prose, same day) read and recorded as met or with its shortfall.

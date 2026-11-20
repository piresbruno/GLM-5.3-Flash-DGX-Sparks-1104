# 06 - TP=3 verify costs and draft settings

Stage 1. Depends on: 03, 05. Estimate: half a day, then part of a window. Status: settings legs measured 2026-11-20 in task 03 (coupled, confidence 0.5 / 0.3, tpm with TP=2 costs: exact, within noise); the TP=3 cost refit is still to do.

## Problem

The draft rule prices verify rows with TP=2 fits: `gb10_window_costs` / `gb10_shared_costs`
(`zig/src/families/glm/cuda_lanes.zig:28-29`) and `verify_a / verify_b / draft_ms` defaults 38.73, 5.68, 5.92
(`draft_rule.zig:183-188`). After the KDA pad a TP=3 row is much cheaper, so the rule offers shallower windows than pay.
The cut rule stops at confidence 0.7 (`cuda_lanes.zig:711-714`) where the Python engine stops at 0.3, and the
noise-coupled selector (`TF_GLM_DFLASH2_COUPLED`, `draft_rule.zig:175`, `dflash_select.zig:21-45`) is off.

## Steps

1. On the padded image: `TF_GLM_DFLASH2_TRACE=1 TF_GLM_DFLASH2_FORCE=1,2,3,4,5,6,7,8` and `TF_GLM_PROF=1`; fit the
   lone and shared lines and the drafter pass with `docs/trace.py`; predict the best window with `docs/optwin.py`.
2. Legs, each against the padded baseline: `TF_GLM_DRAFT_RULE=tpm` with the new `TF_GLM_VERIFY_MS`; the best of task
   03's coupled / confidence legs re-run on the padded image.
3. Adopt the winners as TP=3 defaults in `scripts/config.sh` (export `TF_GLM_VERIFY_MS`, `TF_GLM_DRAFT_RULE`,
   `TF_GLM_DFLASH2_COUPLED`, `TF_GLM_DRAFT_CONFIDENCE` at TP=3 only).
4. Optional engine follow-up if the env route is awkward: world-keyed default cost tables in `cuda_lanes.zig` /
   `draft_rule.zig` (would be its own patch; upstream `ae320aafb` "a shared round learns what a row costs here" is a
   candidate to read first).

## Acceptance

- Greedy contract hashes unchanged by the draft settings (drafts only propose).
- Each adopted setting clears its noise on Structured or prose at 1, 4 or 8 streams and loses nowhere beyond noise.
- Recorded in `docs/speed/tp3/RESULTS.md` with the fitted lines.

# 04 - Engine worktree and patch workflow

Stage 1. Depends on: nothing. Estimate: half a day. Status: in progress (steps 1-2 done by `epic/engine-worktree.sh`; steps 3-5 to do).

## Goal

A clean engine tree to write and backport patches in, and a repeatable way to turn its commits into
`patches/engine/0122+` files.

## Steps

1. In a local TensorFold clone (one with upstream `main`, which the backports need), add a worktree at `f350cd3e` on a
   new branch `tp3-perf`. Do not reuse an existing checkout whose HEAD carries an older patch stack.
2. Apply `patches/engine/*.patch` in filename order with `git am` (as `docker/Dockerfile` does). All 121 must apply
   with no rejects; tag the result `recipe-0121`.
3. Build once in the recipe's image toolchain (Zig 0.17.0, `-Dsm=121`) and run the host tests the stack already uses:
   `zig build test-glm-tp3-host test-glm-fp8-kv-host test-glm-exl3-host test-cuda-host`.
4. Workflow for each new patch: one commit per patch on `tp3-perf`, subject in the stack's style
   (`glm cuda: TF_GLM_... ...`); export with `git format-patch -1 --start-number <N>` into the recipe's
   `patches/engine/`; `scripts/tests/patch-numbers.sh` must pass; the image tag changes because the patches are
   hashed into it.
5. Backports: `git cherry-pick -x` each upstream commit onto `tp3-perf`, resolve, then squash the series into the one
   recipe patch; keep the upstream shas in the commit message body.

## Acceptance

- `recipe-0121` builds with no compiler warning and the host tests pass.
- A no-op test patch exported this way applies in a Docker build (then dropped).

## Notes

- Settings and fingerprint layout on the stack is **17** (0114 set 16, 0117 set 17). New fields take 18, 19, ... in
  the order patches land; upstream's own layout numbers do not transfer.

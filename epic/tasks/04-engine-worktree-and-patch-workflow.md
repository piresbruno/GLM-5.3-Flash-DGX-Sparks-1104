# 04 - Engine worktree and patch workflow

Stage 1. Depends on: nothing. Estimate: half a day. Status: done 2026-11-19 (devbox, x86_64, no GPU).

## Result

- `recipe-0121` applies with no rejects (tree `e32afdf2...`). Host tests in the `toolchain` stage built for x86_64
  (`ZIG_ARCH=x86_64`, zig 0.17.0 sha256 `1cbe9df9...`): **4,168 / 4,182 passed, 4 failed, 1 crashed, 9 skipped**.
  The failures are stale tests in the stack, present before any epic change, and are the baseline every later patch
  is compared against (the same list, nothing new):
  - `cuda_ranks_test.zig:421` expects fingerprint layout 15 (0114 / 0117 moved it to 17 without updating it);
  - `cuda_native.zig:522` `capacity(330529)` expects 330,560, gets 330,624;
  - `cuda_mm` `TF_GLM_MM_STAGES` test logs an error it does not expect;
  - `cuda_weights.zig:678` crashes (general protection reading `L.w.config` in `declaredAs`);
  - `draft_sources_cuda_test` "CUDA GLM distinguishes its checkpoint head from loaded DFlash2" fails;
  - `glm_fp8_pages_host.zig:760` returns `ExpectedHit`.
  (Zig's summary counts these as 4 failed + 1 crashed; the `MM_STAGES` one fails by its logged error.)
- The clean-build "no compiler warning" check (the full `native grammar install` build with nvcc) is not done on the
  devbox; it belongs to the image build on rank 0 in the window.
- Workflow proven on patch 0122: squash, `git format-patch -1 --start-number N --filename-max-length=120`,
  `scripts/tests/patch-numbers.sh` passes, and `epic/engine-worktree.sh` rebuilds the tested tree exactly.

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

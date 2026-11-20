# 07 - Patch 0123: `TF_GLM_ROCE_MULTI`

Stage 2. Depends on: 04. Estimate: 2-3 days, then one window. Status: measured 2026-11-20: `tf-tp-test suite` passes on the triangle and the wire is exact, decode +6% at one stream, but prefill 3-6% slower at every `ROCE_MULTI_KB`: off by default, `ROCE_MULTI_KB=128` (docs/speed/tp3/RESULTS.md). The prompt-path cost to find.

## Result so far

- `patches/engine/0123-...patch`: the six commits squashed. Conflicts: `tp_group.zig` (allGather p2p, fast, ring,
  multi, NCCL; `closeWires` closes all four; abort reaches p2p and multi), `cuda_ranks*.zig` (layout 18 -> 19, the
  TF_TP_MASTER / TF_TP_KERNEL_IMAGE refusal beside 0122's), `cuda.zig` / `cuda_native.zig` (keep both).
- Step 5 needs no code: `fits` sends only gathers up to the mailbox slot down the wire; prompt chunks stay on NCCL.
  The paired prompt step's side stream posts its exchanges in one order, as upstream's does.
- Step 3: `docker/Dockerfile` builds `zig/src/fabric/tp/mailbox.cu` with `tp.zig`'s flags into
  `/opt/tensorfold/lib/tp_mailbox.fatbin` (compiled clean in the CUDA 13.3.1 toolchain, 345 KB).
- Step 4: `ROCE_MULTI=0|1` (default 0), `ROCE_MULTI_KB=512`, `ROCE_MULTI_FALLBACK=error|nccl` (default error while
  qualifying); at TP=3 the start passes `TF_GLM_ROCE_MULTI=1`, `TF_TP_MASTER=$MASTER_ADDR`, `TF_TP_PORT=NCCL_PORT+2`
  (the bootstrap also takes +9; both checked free), the kernel image, the slot and the fallback. An exported
  `TF_TP_ROCE_HCA` reaches every rank as any TF_* does, if the PCIe twins confuse the port choice.
- Host tests (`test-roce-multi-host` included): the baseline failures only.

## Problem

At world 3 every one of the ~90 per-forward all-gathers (two a layer, fp32 partials, 16 KiB a row a rank) plus the pick
and drafter gathers goes over NCCL. The two-rank one-shot RoCE all-gather is gated on `world == 2`
(`zig/src/cuda/tp_group.zig:63-67`, `zig/src/cuda/roce.zig:16,240`), and patch 0108's ring is hard-wired to four
(`WORLD=4`, `WAITS=6`, a relay to the rank across).

## Fix: backport upstream's multirank RoCE for GLM

The pin already has the transport, `zig/src/fabric/tp/roce_multi*.zig` (3..8 ranks, a QP to every peer, per-peer port
choice in `roce_pick.zig:163`), unchanged by any recipe patch; only DeepSeek uses it. Upstream wires GLM to it:

| Commit | What |
| --- | --- |
| `1911e7bc3` | the `tp` import for `glmCuda` in `zig/build/cuda_engines.zig` (hidden prerequisite) |
| `81e25bb55` | `tp_group.Group.multi`: three ranks or more borrow a wire for small all-gathers, every rank or none |
| `b99c48a53` | `TF_GLM_ROCE_MULTI` in Settings and the fingerprint |
| `65ec7eadd` | `families/glm/cuda_roce_multi.zig` and two lines in `cuda_native.zig` |
| `1e9975556`, `ed20c5be4` | probe-vote fixes |

Upstream measured on three GB10s at TP=3: decode 51.2 -> 52.6 tok/s (+2.57%) with drafts; a cold 136k prompt 1.2-1.6%
slower. Rejected alternative: a world-3 fork of 0108's `roce_ring.zig` (617 lines plus a kernel, about a week for the
same result).

## Steps

1. Cherry-pick in order; hand-merge `tp_group.zig`, which on the stack also carries 0034's `p2p` and 0108's `ring`:
   `allGather` order p2p, fast, ring, multi, NCCL; add `multi` to `close` / `abort` / `check`. `openRoce` returns early
   at world 3, so it does not interact with `TF_ROCE_RING`.
2. Settings and fingerprint: next free layout number (19 if task 05 landed first).
3. **`docker/Dockerfile`:** `tp_mailbox.fatbin` is not installed by `zig build install` (only embedded in
   `tf-tp-test`). Add an `nvcc -fatbin` step for `zig/src/fabric/tp/mailbox.cu` with the flags at
   `zig/build/tp.zig:26`, installed at a fixed path in the image.
4. **Launcher at TP=3** (`ROCE_MULTI=off` by default until gated, then on): every rank gets `TF_GLM_ROCE_MULTI=1`,
   `TF_TP_MASTER=<rank 0's address>`, `TF_TP_KERNEL_IMAGE=<fatbin path>`, `TF_TP_MAILBOX_MAX_KB=512` (the default slot
   is 256 KiB, `roce_multi.zig:76`; 32 rows x 16 KiB = 512 KiB; the cap is 1 MiB, `roce_multi_guard.zig:14`),
   `TF_TP_ROCE_FALLBACK=error` while qualifying. Set `TF_TP_ROCE_HCA` explicitly if the PCIe twins confuse the port
   choice.
5. Keep prompt-size gathers on NCCL (size cap), since upstream saw prompts slower.

## Tests

- `zig build test-roce-multi-host`; the `tp_group` fake-wire test.
- On the triangle: `tf-tp-test suite` (bytes equal NCCL's at every size, graph replay of gathers).
- A failed open on any rank leaves every rank on NCCL (forced failure on rank 2).

## Acceptance

- Greedy contract hashes identical to the same image with `TF_GLM_ROCE_MULTI` off (same bytes, same order).
- Per-rank profile: gathers down; decode at 1/4/8 streams up beyond noise; prefill not worse beyond 1-2%.
- TP=2 and TP=4 unchanged with the switch unset.

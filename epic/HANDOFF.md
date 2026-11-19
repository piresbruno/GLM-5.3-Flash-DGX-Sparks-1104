# Handoff: TP=3 epic, for the agent resuming on a devbox

Written 2026-11-19 at the end of the planning session. Read `epic/README.md` first (goals, causes, gates), then the task
files in `epic/tasks/`. This file says where things stand and how to resume.

## Where things are

| Item | State |
| --- | --- |
| Recipe branch | `tp3-perf` on the fork `piresbruno/GLM-5.3-Flash-DGX-Sparks-1104` (upstream `MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104`, where the user has read access only). Cut from `main` at `80be4d6`. Holds `epic/` only: **no code changed yet** |
| Engine tree | Not on any remote. It is TensorFold at the recipe's pin (`TF_REF` in `scripts/config.sh`, `f350cd3e`) plus the 121 `patches/engine/*.patch`, rebuilt exactly by `epic/engine-worktree.sh` (tree hash `e32afdf2d130d990054268922f1485eff7260385` at tag `recipe-0121`, checked twice) |
| Task 04 | Half done: the tree applies with no rejects (89 mail-format patches via `git am`, 32 plain diffs via `git apply`). **Not built, host tests not run** |
| Tasks 01, 02 | Not started |
| Task 03 and every later measurement | Need a maintenance window on the three Sparks: **ask the user first** (see below) |
| Status line in `~/.claude/settings.json` | Done on the Spark (unrelated side request) |

## Set up on the devbox

```sh
git clone git@github.com:piresbruno/GLM-5.3-Flash-DGX-Sparks-1104.git recipe && cd recipe
git switch tp3-perf
git remote add upstream https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104.git
epic/engine-worktree.sh ../tf-tp3         # full TensorFold clone; the backports need its main branch
git -C ../tf-tp3 log --oneline -1 recipe-0121
git -C ../tf-tp3 rev-parse 'recipe-0121^{tree}'   # expect e32afdf2d130...
```

## Resume order

1. **Task 04, finish:** build the tree and run the host tests the stack uses
   (`zig build test-glm-tp3-host test-glm-fp8-kv-host test-glm-exl3-host test-cuda-host`, Zig 0.17.0, CUDA 13,
   `-Dsm=121`). The toolchain is exactly `docker/Dockerfile`'s `toolchain` stage; building that stage and running the
   tests inside it is the surest match. Record the clean-build "no compiler warning" baseline (patch 0115's standard).
2. **Task 05 (patch 0122, `TF_GLM_KDA_PAD`)**: the biggest gain, fully doable off the Sparks. Backport the 11 upstream
   commits listed in the task, plus the recipe-only `chainSegs` stride fix; GPU checks if the devbox has a Blackwell GPU.
3. **Tasks 07 and 08** (patches 0123, 0124), same way.
4. **Tasks 01 and 02** (launcher, bash): any machine; the tests use fake inventories.
5. Then ask the user for the task 03 window and deploy (below).

Line numbers in the task files were read in an engine checkout carrying an **older** 110-patch stack plus a
drafter-MLP edit (equivalent to patch 0109). They are close, not exact, in `recipe-0121`: re-find each site by name.

## Rules that are easy to break

- **Production.** The three Sparks serve production on the older Python-engine recipe (container `glm53-flash-tf`).
  Do not stop, restart or load it, and do not build on the Sparks while it serves: rank 0's Spark had ~9 GiB available
  and the kit's earlyoom kills rank 0 near ~3 GiB. Every session that needs the Sparks is agreed with the user first,
  and ends by restarting that service (`./start-tp3.sh` in its own recipe) and checking `/health`.
- **Public fork.** No addresses, ssh users or host names of the kit in committed files. Kit details are in rank 0's
  untracked `scripts/local.sh`; ask the user.
- **`./start.sh` is not modified.** TP=3 comes from the ring launcher (task 01).
- **Engine changes ship as numbered patch files** `patches/engine/0122+`, each env-gated and off by default in the
  engine; `scripts/tests/patch-numbers.sh` must pass. Export with `git format-patch -1 --start-number N`.
- **Settings and fingerprint layout is 17** on this stack (0114 set 16, 0117 set 17): new fields take 18, 19, ...
  Upstream's own layout numbers do not transfer.
- **Gates:** TP=3 needs its own baselines (TP=2 hashes are not a gate there); a switched-off patch leaves TP=2 replies
  byte-identical. The full list is in `epic/README.md`.
- **Commits:** end messages with the attribution line the session gives; push to the `fork` remote only. The user
  pushes to or opens PRs against `MiaAI-Lab` themselves.

## Deploying an image to the Sparks

The image tag (`tensorfold-glm53-spark:<hash>`, `scripts/prepare.sh` `set_image`) hashes the patches and the build
arguments, including the literal `ZIG_CPU=native`, **not the CPU it was built on**. An image built with `native` on the
devbox gets the same tag a Spark would build but may use instructions the GB10 (Cortex-X925 / A725) lacks, and the
launcher would then skip its own build. So either:

- **(Preferred)** build the deployable image on rank 0 during the maintenance window, after the production service is
  stopped (about 6 minutes; the launcher does it on its own and sends it to the workers by content); or
- build on the devbox with `ZIG_CPU=baseline` (its own tag, slower code) and `docker save | ssh <rank 0> docker load`.

Use the devbox for compiling, host tests and GPU kernel checks; performance and the gates are measured on the three
Sparks only.

## Key facts (so they need not be re-derived)

- **Why TP=3 is slow, verified:**
  - **KDA:** the input stack width is not a multiple of 8 at 22/21/21 heads, so it runs unpacked bf16. The startup
    line shows 179 linears packed instead of 281.
  - **Gathers:** every all-gather is NCCL at world 3.
  - **Uneven split:** routed experts split 768/640/640, so rank 0 is the critical path.
  - **No graph replay:** the GLM forward is not replayed as a CUDA graph, and about 9.5 ms a round is host overhead.
- **Upstream fixes after the pin:** KDA pad `96b3b7aa9..c309bb354`; multirank RoCE `1911e7bc3 81e25bb55 b99c48a53
  65ec7eadd 1e9975556 ed20c5be4`. Upstream main has **no** GLM graph path; task 12 is new work.
- **Retracted earlier claims:**
  - "The routed experts are exchange-dominated": no; every rank holds a column slice of every expert.
  - "TP=3 is at TP=2 parity": no; that compared two harnesses.
- **Earlier attempt:** local branch `tp3` on the Spark's clone (not on the fork); its results are summarised in the
  epic. Its `start-tp3.sh` approach is superseded by task 01.
- **Benchmark definitions:** sparkDash "Structured" is "Count from 1 to 200. Output only the numbers, separated by
  spaces. No other text." (T=0.7, thinking on, `top_k` 20, 400 tokens, `min_tokens = max_tokens`). Aggregate tok/s is
  total decode tokens over the span from first to last token. No Structured result exists for this engine yet.

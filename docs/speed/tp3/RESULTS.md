# TP=3 results: the first maintenance window (2026-11-19/20)

> **Stack:** every number here is TensorFold `f350cd3e` + patches 0001-0124. On 2026-11-20 main moved to TensorFold 1.3.0
> (`81de26f6` + 0001-0013), which carries `TF_GLM_KDA_PAD` and `TF_GLM_ROCE_MULTI` itself and has no
> `TF_GLM_SAMPLE_ROOT`, coupled drafts or tpm rule; this branch followed it (`epic/README.md`, "TensorFold 1.3.0").
> These results are the reference for that stack and the reason for the defaults; the 1.3.0 numbers come from the next
> window.

Task 03's baselines and settings legs and the patch legs of tasks 05, 07 and 08, measured on the three Sparks (a
triangle: every pair on its own ConnectX-7 cable) on one image, `tensorfold-glm53-spark:52252c5da214` (TensorFold
`f350cd3e` + patches 0001-0124; `tf-tp-test` with the mailbox kernel). Every leg ran through `docs/tp3leg.sh` (or
`docs/leg.sh` for dec-and-contract legs): a verified restart, `docs/dec.py` at 1/2/4/8 streams (3 trials, greedy,
thinking off) and its `--contract`, sparkDash Structured and prose at 1/4/6/8 streams (T=0.7, thinking on, `top_k` 20,
400 tokens) and prefill 2K-128K, `docs/quality.py`, `docs/seeded.py`, then a second boot with `TF_GLM_PROF=1`.
Tables are `python3 -I docs/tp3results.py`'s output; every number traces to a file beside this one.

## Second window (2026-11-20, 13:53-17:01 UTC): TensorFold 1.3.0

Main's stack (`81de26f6` + patches 0001-0013, image `97b32fa9c073`), the same kit, checkpoint and tooling as the first
window. Labels end in `-130`; deltas below are against **T3-130** (TP=3, `KDA_PAD=0`), whose contract is now
`contract-baseline.json` (the old stack's is `contract-baseline-f350cd3e.json`). The P3 reference is the first window's
(the same day).

### Verdicts on 1.3.0

| Question | Answer |
| --- | --- |
| Does TP=3 start and pass its checks? | **Yes.** LIVE with `KDA_PAD=1` (290 linears packed); `glm-kda-pad-split`, `glm-shared-check` (108 rows), `glm-pair-check` (6 prompts), `glm-lanes-concurrent` (25 streams, 4 rounds between chunks) pass at world 3 with the pad; `tf-tp-test suite` 402 / 402 / 328, 0 failed |
| Is `KDA_PAD=1` still the right TP=3 default? | **Yes:** decode +9% to +14%, prefill +11% over 1.3.0 without it; short replies and quality as on the old stack (`8e45971383f8`) |
| `ROCE_MULTI=1`? | **Now within its prefill gate** (-1.4% / -1.5% at 4K / 16K, against -3-6% on the old stack); decode +4% / +5% at 1 / 2 streams, flat at 4-8 (one boot: under the drift rule, a candidate, not adopted) |
| `TF_GLM_CHUNK_SOLO=4096`? | Takes effect ("prompt passes of 4096 rows") but prefill moves 0-3%: **not** the cause of the prefill loss |
| Is 1.3.0 as fast as the old stack? | **No.** At the defaults (`KDA_PAD=1`) decode -6% / -7% / -10% / -15% at 1 / 2 / 4 / 8 streams and prefill -25% at 16K; the pair -2% / -5% / -9% decode at 1 / 4 / 8 and prefill -24% |
| Are long prompts answered correctly? | **No: 4 of 7 boots answer the 45K-token `longctx` prompt wrongly** (see finding 13) |

### Findings on 1.3.0

13. **Long prompts are not stable.** The contract's `longctx` (45,597 tokens, greedy, thinking off) answered `'ok'` on
    every boot of the old stack (6 of 6). On 1.3.0: T2-130 `'Okay'`, T2-130-solo `'ok'`, T3-130 `'ok'`, T3r-130
    `'Okay.</think>Done.'`, T3-kdapad-130 `'w99088 ok</think>w6368 w11386 ...'` (the prompt's filler words echoed),
    T3-roce-130 `'ok'`, T3-kdapad-solo-130 `'Okay.</think>I apologize, but I can'...'`. Wrong on 4 of 7 boots, at TP=2 and
    TP=3, with no setting that decides it: something in 1.3.0's long-prompt path differs between boots of the same
    image and settings. `boundary` (10.7K) and the six short prompts repeat on every boot; quality's short answers match
    the old stack. Until it is explained, 1.3.0 should not serve long conversations here.
14. **Prefill is 15-25% slower** at every size on the pair and at TP=3. The pair's profile (T2-130 against T2) has 172
    prompt calls of 1,318 ms against 84 of 2,001 ms (227 s against 168 s for the same prompts), a target MTP pass over
    the prompt (12 ms a call, absent on the old stack) and the shared expert +1.5 ms a lone verify (the old stack ran it
    on a side stream). 4,096-row passes do not recover it, so its cause is open.
15. **Decode at 4-8 streams is 9-15% slower** at every width; at one stream within 2-6%. The draft path's work is the
    same (same contract, same `tok/round`), so the cost is in the forward.
16. **The RoCE wire is faster:** the 86-exchange decode window takes 925 us over RoCE against 2,347 us over NCCL (old
    stack: 1,083 us), and `ROCE_MULTI`'s prefill cost fell to -1.5%.
17. **Boot-to-boot drift on 1.3.0** (T3-130 against T3r-130): `dec.py` +1.0% / +4.0% / +3.2% / +1.1%, prefill under 0.2%.
18. **Build:** `zig build` takes 1,031 s (old stack ~6 min) and `nvcc` prints warnings from upstream kernels (unused
    variables, an unreachable loop) that the old stack's patches 0111 / 0115 had removed.
19. **Gone on 1.3.0:** `LEARN_GIB` (`--learn-conversations` is refused for GLM on CUDA upstream, per main's changelog).

### Tables, second window

### Decode, docs/dec.py (greedy, thinking off): median tok/s of 3 trials, spread, delta vs T3-130

| Leg | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| T2 | 36.5 ±5.0% (-4.9%, noise) | 52.8 ±0.5% (-3.2%, noise) | 71.3 ±0.6% (-3.6%, noise) | 118.1 ±0.9% (-2.5%, noise) |
| T2-130 | 35.9 ±4.6% (-6.4%, noise) | 50.4 ±2.4% (-7.5%) | 67.4 ±0.7% (-8.9%, noise) | 107.8 ±2.7% (-11.0%) |
| T2-130-solo | 36.0 ±3.4% (-6.1%, noise) | 51.0 ±0.3% (-6.4%) | 67.7 ±1.7% (-8.5%, noise) | 107.3 ±1.8% (-11.4%) |
| T3-130 | 38.4 ±0.9% | 54.5 ±2.1% | 74.0 ±6.1% | 121.1 ±4.2% |
| T3r-130 | 38.7 ±1.3% (+1.0%, noise) | 56.7 ±1.1% (+4.0%, noise) | 76.3 ±0.6% (+3.2%, noise) | 122.4 ±0.8% (+1.1%, noise) |
| T3-kdapad-130 | 43.6 ±0.4% (+13.5%) | 62.2 ±1.9% (+14.1%) | 82.1 ±1.5% (+11.0%, noise) | 131.9 ±1.5% (+8.9%) |
| T3-roce-130 | 45.4 ±1.2% (+18.4%) | 65.2 ±0.4% (+19.6%) | 82.4 ±0.2% (+11.4%, noise) | 132.6 ±2.8% (+9.5%) |
| T3-kdapad-solo-130 | 43.4 ±1.3% (+13.2%) | 61.9 ±1.9% (+13.6%) | 81.8 ±0.4% (+10.6%, noise) | 132.1 ±1.0% (+9.1%) |

### sparkDash Structured (count 1..200): aggregate tok/s, T=0.7, thinking on, top_k 20, 400 tokens

| Leg | 1 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: |
| T2 | 87.7 (-9.3%) | 175.22 (-4.8%) | 171.92 (-10.8%) | 175.26 (+0.0%) |
| T2-130 | 86.1 (-11.0%) | 142.22 (-22.7%) | 166.91 (-13.4%) | 154.61 (-11.8%) |
| T2-130-solo | 91.43 (-5.5%) | 173.86 (-5.5%) | 170.74 (-11.5%) | 157.26 (-10.3%) |
| T3-130 | 96.71 | 184.03 | 192.83 | 175.26 |
| T3r-130 | 98.94 (+2.3%) | 187.97 (+2.1%) | 191.8 (-0.5%) | 183.17 (+4.5%) |
| T3-kdapad-130 | 103.04 (+6.5%) | 184.35 (+0.2%) | 200.47 (+4.0%) | 192.76 (+10.0%) |
| T3-roce-130 | 111.59 (+15.4%) | 203.56 (+10.6%) | 201.47 (+4.5%) | 186.18 (+6.2%) |
| T3-kdapad-solo-130 | 101.22 (+4.7%) | 195.5 (+6.2%) | 202.61 (+5.1%) | 186.08 (+6.2%) |
| P3 | 128.92 (+33.3%) | 203.93 (+10.8%) | 274.35 (+42.3%) | 325.89 (+85.9%) |

### sparkDash prose: aggregate tok/s, T=0.7, thinking on, top_k 20, 400 tokens

| Leg | 1 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: |
| T2 | 49.31 (+6.2%) | 79.14 (-6.4%) | 94.27 (-7.1%) | 99.47 (-14.6%) |
| T2-130 | 45.24 (-2.5%) | 76.44 (-9.6%) | 88.6 (-12.6%) | 100.32 (-13.9%) |
| T2-130-solo | 46.08 (-0.7%) | 78.34 (-7.4%) | 86.43 (-14.8%) | 96.12 (-17.5%) |
| T3-130 | 46.41 | 84.59 | 101.43 | 116.46 |
| T3r-130 | 48.59 (+4.7%) | 88.36 (+4.5%) | 102.89 (+1.4%) | 116.34 (-0.1%) |
| T3-kdapad-130 | 60.72 (+30.8%) | 97.09 (+14.8%) | 112.68 (+11.1%) | 116.87 (+0.4%) |
| T3-roce-130 | 59.03 (+27.2%) | 96.37 (+13.9%) | 112.87 (+11.3%) | 118.24 (+1.5%) |
| T3-kdapad-solo-130 | 49.32 (+6.3%) | 97.72 (+15.5%) | 110.43 (+8.9%) | 117.33 (+0.7%) |
| P3 | 65.89 (+42.0%) | 126.85 (+50.0%) | 147.59 (+45.5%) | 166.44 (+42.9%) |

### sparkDash prefill: tok/s, fresh content (cached 0)

| Leg | 2K | 4K | 8K | 16K | 32K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T2 | 1842 (+14.2%) | 1578 (+13.2%) | 1657 (+15.8%) | 1725 (+18.9%) | 1764 (+22.0%) | 1737 (+20.8%) |
| T2-130 | 1468 (-9.0%) | 1252 (-10.2%) | 1285 (-10.1%) | 1308 (-9.8%) | 1311 (-9.4%) | 1296 (-9.9%) |
| T2-130-solo | 1442 (-10.6%) | 1253 (-10.1%) | 1285 (-10.2%) | 1336 (-7.9%) | 1351 (-6.6%) | 1342 (-6.7%) |
| T3-130 | 1613 | 1394 | 1430 | 1451 | 1446 | 1438 |
| T3r-130 | 1615 (+0.1%) | 1396 (+0.1%) | 1432 (+0.1%) | 1451 (-0.0%) | 1460 (+1.0%) | 1454 (+1.2%) |
| T3-kdapad-130 | 1806 (+12.0%) | 1538 (+10.3%) | 1584 (+10.7%) | 1610 (+11.0%) | 1618 (+11.9%) | 1606 (+11.7%) |
| T3-roce-130 | 1791 (+11.1%) | 1516 (+8.8%) | 1567 (+9.6%) | 1586 (+9.3%) | 1600 (+10.6%) | 1587 (+10.4%) |
| T3-kdapad-solo-130 | 1786 (+10.8%) | 1530 (+9.8%) | 1572 (+9.9%) | 1620 (+11.7%) | 1647 (+13.9%) | 1651 (+14.8%) |
| P3 | 2337 (+44.9%) | 2365 (+69.7%) | 2366 (+65.4%) | 2371 (+63.4%) | 2362 (+63.3%) | 2279 (+58.5%) |

### Contract, quality, seeded

| Leg | Contract vs T3 | NLL | Answers sha | Seeded repeatable | Seeded concurrent = solo | Seeded solo sha |
| --- | --- | ---: | --- | --- | --- | --- |
| T2 | - | 0.7523 | a3ac04984272 | True | True | 468f6e704e44b515,8b97e26d3a369326,8 |
| T2-130 | - | 0.7535 | a3ac04984272 | True | True | 468f6e704e44b515,8b97e26d3a369326,8 |
| T2-130-solo | - | 0.7535 | a3ac04984272 | True | True | 468f6e704e44b515,8b97e26d3a369326,8 |
| T3-130 | identical | 0.7481 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3r-130 | differs: longctx | 0.7481 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3-kdapad-130 | differs: prose,code,math,longctx | 0.7553 | 8e45971383f8 | True | True | b5046714c8943a6e,7aec9a9ab1013646,4 |
| T3-roce-130 | differs: prose,code,math | 0.7553 | 8e45971383f8 | True | True | b5046714c8943a6e,7aec9a9ab1013646,4 |
| T3-kdapad-solo-130 | differs: prose,code,math,longctx | 0.7553 | 8e45971383f8 | True | True | b5046714c8943a6e,7aec9a9ab1013646,4 |

## First window (2026-11-19/20): the old stack

## Verdicts

| Switch (patch) | Exact | Decode, `dec.py` | sparkDash | Prefill | Verdict |
| --- | --- | --- | --- | --- | --- |
| `KDA_PAD=1` (0122) | replies change (q8 packed KDA instead of bf16, expected); repeatable, concurrent = solo, NLL 0.7547 vs 0.7594 | **+17% x1, +17% x2, +10% x4, +11% x8** | Structured +6/+2/+4/+9%, prose +28/+12/+11/+1% | **+17-19%** (2,134 at 16K) | **adopt: default 1 at TP=3** |
| `ROCE_MULTI=1` (0123) | identical to the same image off | +6% x1 (2 boots), +2% x8 (noise) | within noise | **-3% to -6% at 4K-16K** on every boot and at every `ROCE_MULTI_KB` (512, 128, 32) | off: fails its prefill gate; opt-in for one-stream use at `ROCE_MULTI_KB=128` |
| `SAMPLE_ROOT=1` (0124) | identical (seeded, and 5 sampled trials at 4 and 8 streams, off = on) | sampled A/B: +0.2% x4, +1.1% x8 against a 4.1% / 3.4% floor | within noise | none | off: no gain (task 08's rule) |
| All three | as `KDA_PAD` alone (contract, NLL, seeded) | **+24% x1, +21% x2, +13% x4, +11% x8** | Structured +17/+8/-3/+6% | +9-18% | |
| `DENSE=q4` | changes | +13% x1, +8% x8 | mixed | +1% | **reject: NLL 0.7899 (+4%)** |
| `TF_GLM_DFLASH2_COUPLED=1` | identical | noise | noise | none | no |
| coupled + `TF_GLM_DRAFT_CONFIDENCE` 0.5 / 0.3 | identical | +4.5% x1 (0.5), else noise | noise | none | no (0.5 at x1: confirm on a second boot) |
| `TF_GLM_DRAFT_RULE=tpm` (TP=2 cost tables) | identical | +6.8% x1, +1.8% x4, 0 x8 | mixed | none | no; refit the costs for TP=3 (follow-up) |
| `TF_GLM_OVERLAP_PIECES=8 TF_GLM_CHUNK_SOLO=4096` | identical | noise | noise | **-0.6% to +1.1%** | no (upstream's +6-8% does not carry to this stack's 4,096-row chunk) |
| `L2PF=1` | identical | +5.3% x1, else noise | noise | none | no (as on the pair) |
| `TP3_NCCL=ring` | - | - | - | - | **does not start**: rank 2 fails `ncclGroupEnd` ("unhandled system error") at the first collective; eager stays |
| `DRAFTER=none` | identical to drafts on | 28.3 / 28.2 / 28.1 tok/s at 1/2/4 | - | - | drafting is exact; see finding 4 |

## Bars (`epic/README.md`)

| | x1 | x4 | x8 |
| --- | ---: | ---: | ---: |
| `dec.py`: T2 / all three / gain | 36.5 / 48.9 / **+34%** | 71.3 / 94.3 / **+32%** | 118.1 / 154.9 / **+31%** |
| `dec.py`: T2 / `KDA_PAD` alone / gain | 36.5 / 46.2 / +27% | 71.3 / 91.6 / +28% | 118.1 / 154.7 / +31% |
| sparkDash Structured: T2 / `KDA_PAD` / gain | 87.7 / 109.6 / +25% | 175.2 / 210.5 / +20% | 175.3 / 219.0 / +25% |
| sparkDash prose: T2 / `KDA_PAD` / gain | 49.3 / 59.1 / +20% | 79.1 / 114.0 / +44% | 99.5 / 131.3 / +32% |
| sparkDash Structured: P3 (bar 2) | 128.9 | 203.9 | 325.9 |

- **Bar 1** (+30% over TP=2 on both harnesses at 1, 4 and 8): met on `dec.py` with all three patches; **not yet on
  sparkDash** (Structured +20-25%, prose x1 +20% with `KDA_PAD`, the shipped default).
- **Milestone M** (+50% at 4 and 8 on sparkDash): not met (+20% to +44%).
- **Bar 2** (P3): met at Structured x4 (210.5 vs 203.9) only. Structured x1 -15%, x8 -33%; prefill 2,134 vs 2,371 at
  16K (-10%).
- P3 (the Python engine as it serves, measured first in the window) is within 5% of the epic's screenshots: 128.9 /
  203.9 / 274.4 / 325.9 against 132.5 / 203.7 / 284.8 / 342.2 (-2.7% / 0 / -3.7% / -4.8%).

## Tables

### Decode, docs/dec.py (greedy, thinking off): median tok/s of 3 trials, spread, delta vs T3

| Leg | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| T2 | 36.5 ±5.0% (-7.7%, noise) | 52.8 ±0.5% (-8.5%) | 71.3 ±0.6% (-14.5%) | 118.1 ±0.9% (-15.5%) |
| T3 | 39.5 ±2.2% | 57.7 ±3.3% | 83.4 ±0.6% | 139.7 ±2.3% |
| T3r | 40.6 ±0.1% (+2.6%, noise) | 60.3 ±0.1% (+4.5%, noise) | 84.3 ±1.5% (+1.1%, noise) | 142.5 ±0.1% (+2.0%, noise) |
| T3-kdapad | 46.2 ±1.4% (+17.0%) | 67.2 ±0.7% (+16.6%) | 91.6 ±0.5% (+9.8%) | 154.7 ±0.7% (+10.7%) |
| T3-rocemulti | 42.0 ±0.7% (+6.4%) | 62.0 ±0.1% (+7.6%) | 85.6 ±0.4% (+2.6%) | 142.8 ±0.6% (+2.2%, noise) |
| T3-sampleroot | 40.9 ±1.4% (+3.5%, noise) | 60.1 ±0.7% (+4.3%, noise) | 83.5 ±0.5% (+0.1%, noise) | 139.0 ±2.1% (-0.5%, noise) |
| T3-all3 | 48.9 ±0.7% (+23.7%) | 69.9 ±0.2% (+21.3%) | 94.3 ±1.6% (+13.1%) | 154.9 ±0.9% (+10.9%) |
| T3-kb128 | 49.2 ±0.4% (+24.4%) | 69.7 ±0.5% (+20.8%) | 92.3 ±0.3% (+10.7%) | 155.0 ±0.7% (+10.9%) |
| T3-kb32 | 47.1 ±4.6% (+19.3%) | 66.7 ±0.7% (+15.6%) | 91.7 ±1.0% (+10.0%) | 153.0 ±8.9% (+9.5%, noise) |
| T3-q4 | 44.8 ±1.6% (+13.4%) | 64.5 ±0.7% (+11.8%) | 89.9 ±0.7% (+7.8%) | 151.1 ±1.1% (+8.2%) |
| T3-coupled | 40.5 ±2.0% (+2.6%, noise) | 60.3 ±0.5% (+4.5%, noise) | 83.6 ±1.0% (+0.3%, noise) | 141.8 ±0.6% (+1.5%, noise) |
| T3-conf05 | 41.3 ±1.7% (+4.5%) | 60.1 ±0.3% (+4.2%, noise) | 83.5 ±0.7% (+0.1%, noise) | 142.3 ±0.4% (+1.9%, noise) |
| T3-conf03 | 38.4 ±0.8% (-2.8%, noise) | 57.9 ±0.6% (+0.5%, noise) | 83.0 ±0.5% (-0.5%, noise) | 141.2 ±0.6% (+1.1%, noise) |
| T3-tpm | 42.2 ±3.3% (+6.8%) | 60.5 ±1.9% (+5.0%, noise) | 84.9 ±0.7% (+1.8%) | 139.5 ±4.0% (-0.1%, noise) |
| T3-pieces | 40.9 ±1.4% (+3.4%, noise) | 60.5 ±0.4% (+4.9%, noise) | 84.1 ±0.6% (+0.9%, noise) | 142.7 ±0.9% (+2.2%, noise) |
| T3-l2pf | 41.6 ±1.5% (+5.3%) | 61.2 ±0.7% (+6.2%, noise) | 83.8 ±0.5% (+0.5%, noise) | 141.7 ±0.6% (+1.4%, noise) |
| T3b | 40.9 ±4.9% (+3.4%, noise) | 60.1 ±0.9% (+4.2%, noise) | 83.2 ±0.7% (-0.2%, noise) | - |
| T3-nodraft | 28.3 ±0.7% (-28.5%) | 28.2 ±0.3% (-51.1%) | 28.1 ±0.0% (-66.3%) | - |

### sparkDash Structured (count 1..200): aggregate tok/s, T=0.7, thinking on, top_k 20, 400 tokens

| Leg | 1 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: |
| T2 | 87.7 (-15.3%) | 175.22 (-14.9%) | 171.92 (-21.7%) | 175.26 (-12.7%) |
| T3 | 103.58 | 205.9 | 219.54 | 200.68 |
| T3r | 102.7 (-0.8%) | 205.92 (+0.0%) | 210.26 (-4.2%) | 212.52 (+5.9%) |
| T3-kdapad | 109.64 (+5.9%) | 210.53 (+2.2%) | 227.45 (+3.6%) | 218.99 (+9.1%) |
| T3-rocemulti | 103.45 (-0.1%) | 199.78 (-3.0%) | 197.82 (-9.9%) | 203.42 (+1.4%) |
| T3-sampleroot | 103.53 (-0.0%) | 207.05 (+0.6%) | 199.04 (-9.3%) | 207.67 (+3.5%) |
| T3-all3 | 120.93 (+16.8%) | 221.47 (+7.6%) | 212.77 (-3.1%) | 213.23 (+6.3%) |
| T3-kb128 | 117.27 (+13.2%) | 204.04 (-0.9%) | 227.84 (+3.8%) | 226.54 (+12.9%) |
| T3-kb32 | 114 (+10.1%) | 250.52 (+21.7%) | 220.44 (+0.4%) | 229.97 (+14.6%) |
| T3-q4 | 83.75 (-19.1%) | 220.33 (+7.0%) | 201.71 (-8.1%) | 207.04 (+3.2%) |
| T3-coupled | 102.77 (-0.8%) | 177.9 (-13.6%) | 194.98 (-11.2%) | 206.15 (+2.7%) |
| T3-conf05 | 97.4 (-6.0%) | 205.34 (-0.3%) | 205.09 (-6.6%) | 208.17 (+3.7%) |
| T3-conf03 | 98.13 (-5.3%) | 201.25 (-2.3%) | 221.92 (+1.1%) | 198.75 (-1.0%) |
| T3-tpm | 98.31 (-5.1%) | 194.11 (-5.7%) | 204.35 (-6.9%) | 214.73 (+7.0%) |
| T3-pieces | 98.23 (-5.2%) | 203.1 (-1.4%) | 180.45 (-17.8%) | 206.64 (+3.0%) |
| T3-l2pf | 101.19 (-2.3%) | 186.23 (-9.6%) | 206.46 (-6.0%) | 199.95 (-0.4%) |
| P3 | 128.92 (+24.5%) | 203.93 (-1.0%) | 274.35 (+25.0%) | 325.89 (+62.4%) |

### sparkDash prose: aggregate tok/s, T=0.7, thinking on, top_k 20, 400 tokens

| Leg | 1 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: |
| T2 | 49.31 (+6.8%) | 79.14 (-22.0%) | 94.27 (-16.6%) | 99.47 (-23.4%) |
| T3 | 46.16 | 101.47 | 113.03 | 129.88 |
| T3r | 52.48 (+13.7%) | 99.87 (-1.6%) | 113.09 (+0.1%) | 128.11 (-1.4%) |
| T3-kdapad | 59.1 (+28.0%) | 113.96 (+12.3%) | 124.87 (+10.5%) | 131.34 (+1.1%) |
| T3-rocemulti | 53.08 (+15.0%) | 97.87 (-3.5%) | 115.48 (+2.2%) | 127.27 (-2.0%) |
| T3-sampleroot | 47.27 (+2.4%) | 98.66 (-2.8%) | 119.08 (+5.4%) | 128.09 (-1.4%) |
| T3-all3 | 53.61 (+16.1%) | 105.81 (+4.3%) | 126.22 (+11.7%) | 134.61 (+3.6%) |
| T3-kb128 | 59.97 (+29.9%) | 106.82 (+5.3%) | 125.33 (+10.9%) | 132.85 (+2.3%) |
| T3-kb32 | 57.15 (+23.8%) | 100.07 (-1.4%) | 121.43 (+7.4%) | 137.59 (+5.9%) |
| T3-q4 | 56.49 (+22.4%) | 104.67 (+3.2%) | 122.83 (+8.7%) | 131.11 (+0.9%) |
| T3-coupled | 50.78 (+10.0%) | 98.06 (-3.4%) | 113.46 (+0.4%) | 126.97 (-2.2%) |
| T3-conf05 | 48.1 (+4.2%) | 93.19 (-8.2%) | 118.57 (+4.9%) | 124.72 (-4.0%) |
| T3-conf03 | 45.71 (-1.0%) | 102.29 (+0.8%) | 110.2 (-2.5%) | 131.56 (+1.3%) |
| T3-tpm | 52.52 (+13.8%) | 94.44 (-6.9%) | 112.57 (-0.4%) | 125.31 (-3.5%) |
| T3-pieces | 49.46 (+7.1%) | 103.19 (+1.7%) | 116.86 (+3.4%) | 128.54 (-1.0%) |
| T3-l2pf | 49.86 (+8.0%) | 99.99 (-1.5%) | 115.77 (+2.4%) | 125.06 (-3.7%) |
| P3 | 65.89 (+42.7%) | 126.85 (+25.0%) | 147.59 (+30.6%) | 166.44 (+28.1%) |

### sparkDash prefill: tok/s, fresh content (cached 0)

| Leg | 2K | 4K | 8K | 16K | 32K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T2 | 1842 (-10.4%) | 1578 (-3.4%) | 1657 (-3.2%) | 1725 (-4.0%) | 1764 (-3.8%) | 1737 (-4.9%) |
| T3 | 2056 | 1633 | 1712 | 1796 | 1835 | 1827 |
| T3r | 2062 (+0.3%) | 1637 (+0.2%) | 1710 (-0.1%) | 1791 (-0.3%) | 1834 (-0.0%) | 1831 (+0.3%) |
| T3-kdapad | 2234 (+8.7%) | 1909 (+16.9%) | 2023 (+18.2%) | 2134 (+18.8%) | 2180 (+18.8%) | 2166 (+18.5%) |
| T3-rocemulti | 2054 (-0.1%) | 1543 (-5.6%) | 1614 (-5.7%) | 1742 (-3.1%) | 1813 (-1.2%) | 1826 (-0.1%) |
| T3-sampleroot | 2073 (+0.8%) | 1635 (+0.1%) | 1707 (-0.3%) | 1793 (-0.2%) | 1837 (+0.1%) | 1831 (+0.2%) |
| T3-all3 | 2235 (+8.7%) | 1816 (+11.2%) | 1896 (+10.8%) | 2055 (+14.4%) | 2146 (+17.0%) | 2156 (+18.0%) |
| T3-kb128 | 2216 (+7.8%) | 1801 (+10.3%) | 1897 (+10.8%) | 2062 (+14.8%) | 2137 (+16.5%) | 2151 (+17.7%) |
| T3-kb32 | 2237 (+8.8%) | 1807 (+10.7%) | 1903 (+11.1%) | 2054 (+14.4%) | 2137 (+16.5%) | 2151 (+17.8%) |
| T3-q4 | 2066 (+0.5%) | 1641 (+0.5%) | 1717 (+0.3%) | 1812 (+0.9%) | 1851 (+0.9%) | 1849 (+1.2%) |
| T3-coupled | 2042 (-0.7%) | 1632 (-0.1%) | 1712 (+0.0%) | 1794 (-0.1%) | 1836 (+0.1%) | 1816 (-0.6%) |
| T3-conf05 | 2039 (-0.8%) | 1630 (-0.2%) | 1712 (-0.0%) | 1797 (+0.0%) | 1834 (-0.0%) | 1831 (+0.2%) |
| T3-conf03 | 2065 (+0.4%) | 1634 (+0.0%) | 1709 (-0.2%) | 1792 (-0.3%) | 1829 (-0.3%) | 1821 (-0.3%) |
| T3-tpm | 2091 (+1.7%) | 1641 (+0.5%) | 1719 (+0.4%) | 1799 (+0.2%) | 1839 (+0.2%) | 1833 (+0.3%) |
| T3-pieces | 2044 (-0.6%) | 1640 (+0.4%) | 1711 (-0.0%) | 1804 (+0.4%) | 1848 (+0.7%) | 1848 (+1.1%) |
| T3-l2pf | 2064 (+0.4%) | 1641 (+0.5%) | 1713 (+0.1%) | 1797 (+0.0%) | 1834 (-0.0%) | 1826 (-0.0%) |
| P3 | 2337 (+13.7%) | 2365 (+44.8%) | 2366 (+38.2%) | 2371 (+32.0%) | 2362 (+28.8%) | 2279 (+24.7%) |

### Contract, quality, seeded

| Leg | Contract vs T3 | NLL | Answers sha | Seeded repeatable | Seeded concurrent = solo | Seeded solo sha |
| --- | --- | ---: | --- | --- | --- | --- |
| T2 | - | 0.7523 | a3ac04984272 | True | True | 468f6e704e44b515,8b97e26d3a369326,8 |
| T3 | identical | 0.7594 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3r | identical | 0.7594 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3-kdapad | differs: prose,code,math | 0.7547 | 8e45971383f8 | True | True | b5046714c8943a6e,7aec9a9ab1013646,4 |
| T3-rocemulti | identical | 0.7594 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3-sampleroot | identical | 0.7594 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3-all3 | differs: prose,code,math | 0.7547 | 8e45971383f8 | True | True | b5046714c8943a6e,7aec9a9ab1013646,4 |
| T3-kb128 | differs: prose,code,math | 0.7547 | 8e45971383f8 | True | True | b5046714c8943a6e,7aec9a9ab1013646,4 |
| T3-kb32 | differs: prose,code,math | 0.7547 | 8e45971383f8 | True | True | b5046714c8943a6e,7aec9a9ab1013646,4 |
| T3-q4 | differs: prose,math,json | 0.7899 | e38575903d82 | True | True | b21683d1483f2214,37625d8a5175b50b,f |
| T3-coupled | identical | 0.7594 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3-conf05 | identical | 0.7594 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3-conf03 | identical | 0.7594 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3-tpm | identical | 0.7594 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3-pieces | identical | 0.7594 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3-l2pf | identical | 0.7594 | 639c916612ba | True | True | 940a20c4a9416638,e901f5f577bdbc7f,6 |
| T3b | identical | - | - | - | - | - |
| T3-nodraft | identical | - | - | - | - | - |

### Per-rank profile: lone 2-row verify (`TF_GLM_PROF=1`, ms a forward; rank 1's wall is a follower's clock)

| Leg | Rank | wall | kda | routed | mla | router | gather_a | gather_b | head | dense | shared |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T3 | 0 | 41.39 | 15.85 | 12.63 | 3.88 | 2.86 | 2.71 | 1.19 | 0.98 | 0.81 | 0.40 |
| T3 | 1 | 12.14 | 15.26 | 10.86 | 3.78 | 2.74 | 2.20 | 4.03 | 0.97 | 0.80 | 0.41 |
| T3r | 0 | 41.52 | 16.06 | 12.75 | 3.92 | 2.86 | 2.39 | 1.25 | 0.99 | 0.81 | 0.40 |
| T3r | 1 | 12.26 | 15.23 | 10.92 | 3.76 | 2.67 | 2.50 | 3.99 | 0.97 | 0.80 | 0.40 |
| T3-kdapad | 0 | 35.66 | 9.27 | 12.71 | 3.91 | 2.98 | 3.09 | 1.38 | 0.99 | 0.82 | 0.40 |
| T3-kdapad | 1 | 9.76 | 9.28 | 10.91 | 3.84 | 2.92 | 1.82 | 4.39 | 0.98 | 0.82 | 0.42 |
| T3-rocemulti | 0 | 40.54 | 15.88 | 12.66 | 3.88 | 2.85 | 1.79 | 1.19 | 0.98 | 0.81 | 0.40 |
| T3-rocemulti | 1 | 11.96 | 15.25 | 10.91 | 3.81 | 2.72 | 2.18 | 3.21 | 0.98 | 0.81 | 0.42 |
| T3-sampleroot | 0 | 41.22 | 15.71 | 12.63 | 3.90 | 2.86 | 2.68 | 1.17 | 0.98 | 0.81 | 0.40 |
| T3-sampleroot | 1 | 12.20 | 15.19 | 10.81 | 3.79 | 2.78 | 2.12 | 4.08 | 0.98 | 0.81 | 0.39 |
| T3-all3 | 0 | 33.81 | 9.29 | 12.57 | 3.86 | 2.95 | 1.77 | 1.11 | 0.98 | 0.81 | 0.38 |
| T3-all3 | 1 | 9.27 | 9.11 | 10.84 | 3.80 | 2.91 | 1.59 | 3.16 | 0.98 | 0.82 | 0.40 |

### Sampled A/B for `SAMPLE_ROOT` (docs/dec.py, T=0.7, top_k 20, seed 1234, 5 trials, `KDA_PAD=1` under both)

| Leg | 4 streams | 8 streams | sampled tokens |
| --- | ---: | ---: | --- |
| `SAMPLE_ROOT=0` | 87.2 ±2.0% | 146.3 ±1.7% | |
| `SAMPLE_ROOT=1` | 87.4 ±1.5% | 147.9 ±0.7% | identical to off in every trial |

## Findings

1. **0122 does what task 05 said.** Rank 0's KDA section of a lone 2-row verify falls from 15.9 to 9.3 ms (TP=4's
   share is ~7-8 ms); the forward from 41.4 to 35.7 ms. The startup line reads 282 linears packed with the pad and 180
   without (the epic's 281 / 179 came from the older 110-patch stack). Its GPU checks pass at world 3 with the pad on:
   `glm-kda-pad-split`, `glm-shared-check` (108 rows of 5 streams over 12 rounds equal their lone twins), `glm-pair-check`,
   and `glm-lanes-concurrent`, the `chainSegs` case (25 streams equal their solo runs, prompts of 2.6K-5.2K tokens so
   that prompts arrive a chunk at a time between the others' rounds; 13-20-token prompts never take that path).
2. **0123's wire is exact and faster for small gathers, but costs prefill whatever its limit.** `tf-tp-test suite` on
   the triangle (TF_COMM_BACKEND=roce, fallback error): 402 / 402 / 328 checks pass, 0 failed, every result exact against
   NCCL; an 86-exchange decode window of 8 KiB gathers takes 1,083 us over RoCE against 2,310 us over NCCL, and RoCE
   all-gather is ~2x NCCL's at 4-64 KiB, equal from 1 MiB. In the server, `gather_a` falls from 2.71 to 1.79 ms a lone
   forward on rank 0 (rank 1's `gather_b` 4.03 to 3.21). The prefill cost (-3% to -6% at 4K-16K, under 1% at 128K) is the
   same at `ROCE_MULTI_KB` 512, 128 and 32, so it is not the mid-size gathers going over RoCE: it comes with the switch
   on (the wire's queue pairs and polling beside NCCL's prefill traffic on the same ConnectX-7 links, or a sync on the
   prompt path; to find in a per-rank profile of a prompt chunk, off and on). 128 KB is the best limit for decode
   (x1 49.2 against 48.9 at 512 and 47.1 at 32).
3. **0124 is exact and buys nothing measurable here.** One broadcast a sampled row a rank is microseconds against a
   ~100 ms round at 8 streams on this triangle. It stays in the stack, off.
4. **Without drafts, streams do not share forwards.** `DRAFTER=none`: 28.3 / 28.2 / 28.1 tok/s total at 1 / 2 / 4 streams,
   the round 34.7 / 69.9 / 140.8 ms: one forward a stream. The shared verify windows of the draft path are what make
   concurrency pay at TP=3. The lone 1-row forward without the pad: 34.7 ms (the Python engine's 19.5 ms + 3.9 ms a row).
5. **The draft settings do not close the gap to P3's deeper drafting.** Coupled drafts, confidence 0.5 / 0.3 and the tpm
   rule are exact and within noise at 4-8 streams. Three one-stream gains of +4.5% to +6.8% (confidence 0.5, tpm, L2PF)
   are single-boot and inside the boot-to-boot drift of finding 10, so none is a gain. The 64-row window (task 10) is
   the remaining lever there.
6. **The TP=2 reference.** T2 (the pair on the same 124-patch image, every switch off) is byte-identical to
   `docs/speed/contract-up-base.json` (the lab's pair base when patches 0113-0115 landed, 2026-11-11) and as fast as it
   (36.5 / 71.3 / 118.1 against 37.6 / 72.7 / 121.1 tok/s). `docs/speed/contract-baseline.json` (2026-11-04) predates
   those patches and no longer matches the pair (5 of 8 prompts differ): the epic's "TP=2 hashes equal to
   contract-baseline.json" gate should read `contract-up-base.json`, or the baseline be refreshed (not done here).
7. **T3's contract repeats** on a second boot (T3b, all 8 prompts) and is the TP=3 baseline
   (`contract-baseline.json` in this directory). T3 differs from T2 on `prose` and `math` only (TP=3 reductions).
8. **sparkDash is one run a leg.** Swings of up to +-10% between legs whose `dec.py` and contract are identical (e.g. x6)
   are that harness's run-to-run noise; read its gains together with `dec.py`'s, or on two boots.
9. **`TF_GLM_PROF` on this stack:** shared windows report per total-row bucket every 64 calls (eight streams for 3,342
   rounds filled one bucket); rank 2 emits no decode reports; a follower's `wall` is not comparable with rank 0's. The
   per-rank table below is the lone 2-row verify, ranks 0 and 1.
10. **Boot-to-boot drift** (T3 against T3r, nothing changed): `dec.py` +2.6% / +4.5% / +1.1% / +2.0% at 1/2/4/8 streams,
    sparkDash prose x1 +13.7%, prefill +-0.3%. So a one-stream `dec.py` gain under ~5% on one boot is not a gain, and
    every prefill difference above 1% is real (0123's -3% to -6%, seen on four boots).
11. **The routed split is visible.** Rank 0's routed experts take 12.6-12.8 ms a lone forward against 10.8-10.9 ms on
    rank 1 (768 against 640 rows of every expert): ~1.8 ms of rank 0's critical path, task 14's target.
12. **`TP3_NCCL=ring` does not start on the triangle.** With `NCCL_DEBUG=INFO`: rank 2's `ibv_modify_qp` INIT->RTR times
    out (110) on `rocep1s0f0`, its device toward rank 0, while connecting rank 1 -> rank 2. The ring graph files fix the
    collectives' channels, but the point-to-point connections of the split prefill reduce (`ncclGroupEnd`, channels
    04-07 "Shared") are made lazily (`NCCL_RUNTIME_CONNECT=1`) round-robin over all four devices, so one lands on the
    cable to the wrong peer. Four Sparks on a ring run `PREFILL_REDUCE=gather` (no point-to-point); the likely fix is
    the same at three with `TP3_NCCL=ring` (unverified: one more boot).

## Fixed during the window

- `./start-tp3.sh` stopped in its weights step on the kit (`!v: unbound variable`): a `local WORKER` shadowed
  `scripts/local.sh`'s value (commit c68d7d3, `scripts/tests/tp3-worker-weights.sh`).
- `tf-tp-test` had no mailbox kernel, so its RoCE suite could not run (commit 530b30d: `-Dtp-fatbin`).
- `docs/leg.sh` polled port 8888 on a kit serving elsewhere: it now reads `PORT` from `scripts/local.sh`.
- Committed measurement files carried the kit's addresses, ssh targets and host names: `docs/scrub.py` removes them
  (run on this directory), and `docs/tp3leg.sh` no longer writes them (memory by rank, only the `glm prof` lines).
- Task 03's `TF_GLM_CHUNK_SOLO=1` is refused by this build (4096 or unset): the pieces leg used 4096.

## Follow-ups

- 0123: find the prompt-path cost (profile a prompt chunk off and on, per rank); then revisit the default.
- `TP3_NCCL=ring`: try it with `PREFILL_REDUCE=gather` (finding 12), or refuse ring with split in `validate_config`;
  the offline test checks the plan only.
- Draft rule tpm with costs refit from a T3 trace; confidence 0.5 and L2PF at one stream on a second boot.
- The TP=2 gate's reference (finding 6).
- Production on the kit is a systemd user unit (`glm53-tensorfold.service`) with a 04:00 UTC nightly restart timer
  (`glm53-nightly.timer`): stop and start it with `systemctl --user` in a window; `epic/DGX-AGENT.md` should say so.

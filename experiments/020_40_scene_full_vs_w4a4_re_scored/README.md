# 020 — 40-scene Full-vs-W4A4 re-scored on the foreground mask; D1 recomputed

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#020** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-08 10:05 CEST
- **Git commit:** `231d1ea`
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/reeval_main40.py
  ```

**Config:** No training, no rendering. Re-scores the existing 40-scene held-out renders.

| Field | Value |
|---|---|
| Init source | `full` and `w4a4` (the main 40-scene run's own trained models) |
| Iterations | 30000 (as originally trained) |
| Densify | default (as originally trained) |
| Input views | 6 input, 9 held-out |
| Mask type | **`content` vs `foreground`, compared** |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** all 40 scenes.

**Results:**

| metric | content mask | foreground mask |
|---|---:|---:|
| mean Full PSNR | 9.9156 | 9.0279 |
| mean W4A4 PSNR | 9.7117 | 8.9164 |
| **mean Full-minus-Quant** | **+0.2038** | **+0.1115** |
| mean SSIM delta | +0.00287 | +0.00022 |

**D1 paired statistics, recomputed on both masks:**

| quantity | content mask | foreground mask |
|---|---:|---:|
| n | 40 | 40 |
| mean delta (dB) | 0.2038 | 0.1115 |
| sd | 0.4700 | 0.5093 |
| t | 2.7427 | 1.3848 |
| p | 0.0092 | 0.1740 |
| MDE at 80% power | 0.2082 | 0.2256 |
| scenes favouring Full | 31 | 20 |
| 95% CI | [+0.0535, +0.3542] | [-0.0514, +0.2744] |

The content-mask column reproduces the published D1 exactly (mean 0.204, sd 0.470, t 2.74, p 0.0092,
CI [0.054, 0.354], MDE 0.208, 31/40 favouring Full), confirming that only the mask changed.

**Per-scene sign agreement: 25/40 = 62.5%. 15 of 40 scenes change sign between masks.**

Scenes that flip: apple/110_13051_23361, broccoli/372_41112_81867, broccoli/412_56288_108844, cake/403_53094_103680, donut/391_47032_93657, hydrant/411_56064_108483, mouse/377_43416_86289, orange/374_42196_84367, orange/385_45386_90752, plant/374_42005_84358, skateboard/245_26182_52130, skateboard/366_39266_76077, toaster/372_41229_82130, toaster/416_57389_110765, toytruck/190_20494_39385.

**PASS/FAIL:**
- Target: **does the headline Full-minus-Quant result survive the mask correction?** On the content mask the effect is
  statistically real (p = 0.0092, CI excludes zero). On the foreground mask **p = 0.174, the 95% CI
  [-0.0514, +0.2744] includes zero**, and scenes favouring Full fall to 20/40 — exactly chance. **FAIL: the result does not survive.**
- Target (D1's own rule): **any null must be stated as an effect-size bound, not a null result.** On the foreground mask
  the bound is: the Full-minus-Quant advantage is +0.1115 dB with 95% CI [-0.0514, +0.2744] dB; effects
  below the 0.2256 dB minimum detectable effect at n=40 cannot be resolved by this study.

**Interpretation:** Rules out the project's central quantitative claim as currently stated — the +0.204 dB
Full-over-W4A4 advantage is an artifact of scoring a region that is ~85% background, and on the object itself the
effect is indistinguishable from zero (p = 0.174, 20/40 scenes each way). The apple spot check was NOT isolated:
15/40 scenes reverse sign between masks.

**Supersedes** the Full-vs-W4A4 delta reported in entry #001 and in `aggregate_metrics.json` as a quality claim; those
content-mask numbers remain arithmetically correct but no longer support the conclusion drawn from them.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/reeval_main40.py`

**results/**

- `results/main40_both_masks.json`

**logs/**

- `logs/main40.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

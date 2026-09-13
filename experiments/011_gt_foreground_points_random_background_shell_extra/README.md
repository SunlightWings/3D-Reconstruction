# 011 — GT foreground points + random background shell (extra factor, not a pure oracle)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#011** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 CEST (this run took 10.5 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag bgshell --iters 500 1000 3000 7000 --source bgshell
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` GT foreground points **plus 100,000 random background points** per scene (radii 1.2-8.0 x scene radius, grey). NOT a pure GT-geometry oracle: the background points are a random prior, not measurements. |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 6 input, 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.74 | 15.35 | 15.42 | 15.83 |
| ball/123_14363_28981 | 10.74 | 11.22 | 13.19 | 13.75 |
| bowl/70_5792_13401 | 12.05 | 12.58 | 12.93 | 14.66 |
| broccoli/412_56288_108844 | 9.25 | 10.39 | 12.03 | 12.31 |
| hydrant/167_18184_34441 | 11.28 | 11.82 | 11.54 | 12.96 |
| remote/350_36761_68623 | 10.97 | 13.07 | 14.21 | 14.18 |
| teddybear/187_20215_38541 | 16.24 | 16.08 | 14.87 | 16.23 |
| toaster/372_41229_82130 | 10.73 | 10.24 | 12.72 | 13.17 |
| **mean PSNR** | **12.126** | **12.593** | **13.365** | **14.136** |
| **mean SSIM** | 0.4501 | 0.4264 | 0.3746 | 0.3783 |
| min PSNR | 9.25 | 10.24 | 11.54 | 12.31 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **14.136 dB at 7000 iterations**. **FAIL**, short by 5.86 dB.

**Interpretation:** Rules out missing background INITIALISATION as the bottleneck: adding 100,000 random background points per scene gives 14.136 dB at 7000 vs the baseline 14.023 dB, a +0.113 dB change that is negligible against the 5.9 dB gap to target. Combined with entry #002 (no GT background depth exists) this shifts the diagnosis: the background is not merely un-initialised, it is not reconstructable from 6 views at all, so seeding geometry there does not help.


---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/prep_bgshell.py`
- `code/sweep.py`

**results/**

- `results/bgshell.json`

**logs/**

- `logs/bgshell.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

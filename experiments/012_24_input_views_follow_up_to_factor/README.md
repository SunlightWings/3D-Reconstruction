# 012 — 24 input views (follow-up to factor 4)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#012** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 CEST (this run took 13.0 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag views24 --iters 500 1000 3000 7000 --source views24
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` rebuilt on 24 views |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 24 input (evenly spaced, held-out frames excluded), 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 16.29 | 16.04 | 16.79 | 16.98 |
| ball/123_14363_28981 | 12.58 | 12.19 | 13.73 | 13.90 |
| bowl/70_5792_13401 | 11.98 | 13.89 | 15.63 | 15.81 |
| broccoli/412_56288_108844 | 9.04 | 9.94 | 12.70 | 12.94 |
| hydrant/167_18184_34441 | 10.79 | 11.72 | 12.78 | 12.30 |
| remote/350_36761_68623 | 10.51 | 13.80 | 16.09 | 15.55 |
| teddybear/187_20215_38541 | 16.59 | 16.02 | 17.97 | 18.55 |
| toaster/372_41229_82130 | 11.34 | 10.81 | 13.57 | 13.98 |
| **mean PSNR** | **12.392** | **13.051** | **14.909** | **15.001** |
| **mean SSIM** | 0.4535 | 0.4532 | 0.4412 | 0.4229 |
| min PSNR | 9.04 | 9.94 | 12.70 | 12.30 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **15.001 dB at 7000 iterations**. **FAIL**, short by 5.00 dB.

**Interpretation:** Rules out 'just add more views' as a route to target: doubling 12 -> 24 views yields 15.001 dB at 7000, which is 0.140 dB BELOW views12's 15.141 dB, so the view-count gain has saturated. The per-doubling gain at 3000 iterations falls from +1.087 dB (6->12) to +0.620 dB (12->24), and at 7000 it is negative. The 6-view result was therefore not primarily view-starved.


---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/sweep.py`

**results/**

- `results/views24.json`

**logs/**

- `logs/views24.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

# 005 — Densification disabled (factor 1)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#005** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 CEST (this run took 6.1 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag densify_off --iters 500 1000 3000 7000 --source oracle --extra --densify_until_iter 0
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1 --densify_until_iter 0`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | `--densify_until_iter 0` (no adaptive density control at all) |
| Input views | 6 input, 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.75 | 15.27 | 14.58 | 14.13 |
| ball/123_14363_28981 | 10.78 | 11.35 | 12.77 | 13.15 |
| bowl/70_5792_13401 | 12.10 | 11.44 | 12.24 | 13.47 |
| broccoli/412_56288_108844 | 9.28 | 9.25 | 10.63 | 11.73 |
| hydrant/167_18184_34441 | 11.28 | 11.95 | 11.84 | 11.83 |
| remote/350_36761_68623 | 10.97 | 13.45 | 13.44 | 13.54 |
| teddybear/187_20215_38541 | 16.24 | 16.12 | 16.17 | 16.39 |
| toaster/372_41229_82130 | 10.73 | 10.14 | 11.83 | 12.39 |
| **mean PSNR** | **12.140** | **12.371** | **12.938** | **13.329** |
| **mean SSIM** | 0.4502 | 0.4463 | 0.4391 | 0.4297 |
| min PSNR | 9.28 | 9.25 | 10.63 | 11.73 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **13.329 dB at 7000 iterations**. **FAIL**, short by 6.67 dB.

**Interpretation:** Rules out adaptive density control as the cause of the gap: with densification fully off the mean is 13.329 dB at 7000 vs the baseline 14.023 dB, i.e. removing densification makes the photometric score worse, not better, while raising SSIM (0.4297 vs 0.3768) -- the floaters densification adds are buying PSNR and costing structure.


---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/run_configs.sh`
- `code/sweep.py`

**results/**

- `results/densify_off.json`

**logs/**

- `logs/densify_off.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

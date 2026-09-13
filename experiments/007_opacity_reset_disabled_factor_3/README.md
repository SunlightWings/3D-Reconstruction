# 007 — Opacity reset disabled (factor 3)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#007** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 CEST (this run took 10.0 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag no_opacity_reset --iters 500 1000 3000 7000 --source oracle --extra --opacity_reset_interval 100000
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1 --opacity_reset_interval 100000`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default densification; `--opacity_reset_interval 100000` (> iterations, so never fires) |
| Input views | 6 input, 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.75 | 15.19 | 14.37 | 14.60 |
| ball/123_14363_28981 | 10.76 | 11.45 | 13.21 | 13.26 |
| bowl/70_5792_13401 | 12.05 | 11.39 | 14.42 | 14.49 |
| broccoli/412_56288_108844 | 9.29 | 9.08 | 10.65 | 10.66 |
| hydrant/167_18184_34441 | 11.28 | 11.27 | 11.90 | 13.31 |
| remote/350_36761_68623 | 10.97 | 11.77 | 14.09 | 14.10 |
| teddybear/187_20215_38541 | 16.24 | 14.44 | 15.31 | 15.25 |
| toaster/372_41229_82130 | 10.71 | 11.70 | 12.93 | 12.95 |
| **mean PSNR** | **12.131** | **12.036** | **13.361** | **13.577** |
| **mean SSIM** | 0.4502 | 0.4288 | 0.3685 | 0.3588 |
| min PSNR | 9.29 | 9.08 | 10.65 | 10.66 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **13.577 dB at 7000 iterations**. **FAIL**, short by 6.42 dB.

**Interpretation:** Rules out opacity reset as the cause: disabling it gives 13.577 dB at 7000 vs the baseline 14.023 dB, so the periodic opacity cull is not what is destroying the reconstruction -- removing it is mildly harmful, not helpful.


---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/run_configs.sh`
- `code/sweep.py`

**results/**

- `results/no_opacity_reset.json`

**logs/**

- `logs/no_opacity_reset.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

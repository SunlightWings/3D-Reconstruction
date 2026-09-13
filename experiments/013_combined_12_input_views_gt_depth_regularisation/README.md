# 013 — COMBINED: 12 input views + GT depth regularisation (best config found)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#013** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 CEST (this run took 12.0 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag views12_depth --iters 500 1000 3000 7000 --source views12_depth --extra -d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1 -d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` rebuilt on 12 views, plus GT inverse-depth maps for those 12 frames |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 12 input (evenly spaced, held-out excluded), 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | ON: `-d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01` |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 16.05 | 15.00 | 15.53 | 15.92 |
| ball/123_14363_28981 | 10.76 | 11.40 | 14.50 | 14.65 |
| bowl/70_5792_13401 | 11.91 | 13.28 | 14.37 | 15.16 |
| broccoli/412_56288_108844 | 9.48 | 10.35 | 12.86 | 13.77 |
| hydrant/167_18184_34441 | 10.49 | 11.86 | 13.24 | 14.23 |
| remote/350_36761_68623 | 9.12 | 12.72 | 15.65 | 16.38 |
| teddybear/187_20215_38541 | 15.93 | 15.51 | 18.95 | 19.96 |
| toaster/372_41229_82130 | 11.63 | 9.90 | 13.71 | 14.66 |
| **mean PSNR** | **11.920** | **12.502** | **14.851** | **15.589** |
| **mean SSIM** | 0.4472 | 0.4377 | 0.4126 | 0.4109 |
| min PSNR | 9.12 | 9.90 | 12.86 | 13.77 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **15.589 dB at 7000 iterations**. **FAIL**, short by 4.41 dB.

**Interpretation:** Rules out the whole search space explored here: stacking the only two factors that helped gives 15.589 dB at 7000, still 4.411 dB short of target. The two gains are sub-additive (+1.118 for views alone, +0.763 for depth alone, +1.566 combined against an additive prediction of +1.881), which is what a shared ceiling looks like rather than two independent deficiencies. NOTE: the first attempt at this config was INVALID and is not reported as a result -- `prep_views12_depth.py` copied the `views12` directory, which doubles as its own GraphDECO model directory, so the trained checkpoints came along and sweep.py's resume-skip re-rendered them instead of training (1.44 min runtime, output bit-identical to views12). The prep script now strips model artifacts after copying and the config was retrained from a clean source (12.02 min).


---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/prep_views12_depth.py`
- `code/sweep.py`

**results/**

- `results/views12_depth.json`

**logs/**

- `logs/views12_depth.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

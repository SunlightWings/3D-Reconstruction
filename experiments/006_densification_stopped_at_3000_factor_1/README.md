# 006 — Densification stopped at 3000 (factor 1)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#006** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 CEST (this run took 9.1 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag densify_3000 --iters 500 1000 3000 7000 --source oracle --extra --densify_until_iter 3000
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1 --densify_until_iter 3000`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | `--densify_until_iter 3000` (default is 15000) |
| Input views | 6 input, 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.75 | 14.94 | 14.58 | 14.91 |
| ball/123_14363_28981 | 10.75 | 11.70 | 13.70 | 13.94 |
| bowl/70_5792_13401 | 12.10 | 12.79 | 14.51 | 14.70 |
| broccoli/412_56288_108844 | 9.30 | 10.25 | 10.92 | 11.05 |
| hydrant/167_18184_34441 | 11.28 | 11.43 | 11.97 | 13.09 |
| remote/350_36761_68623 | 10.97 | 12.19 | 12.98 | 12.91 |
| teddybear/187_20215_38541 | 16.25 | 14.00 | 15.82 | 15.81 |
| toaster/372_41229_82130 | 10.71 | 8.56 | 11.14 | 11.00 |
| **mean PSNR** | **12.139** | **11.982** | **13.202** | **13.427** |
| **mean SSIM** | 0.4503 | 0.4231 | 0.3605 | 0.3507 |
| min PSNR | 9.30 | 8.56 | 10.92 | 11.00 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **13.427 dB at 7000 iterations**. **FAIL**, short by 6.57 dB.

**Interpretation:** Rules out an intermediate densification cutoff as the fix: 13.427 dB at 7000 vs the baseline 14.023 dB with the default 15000 cutoff. Its 3000-iteration value (13.202 dB) is bit-identical to the baseline's, as it must be since the two configs only diverge after iteration 3000 -- a useful determinism check on the harness.


---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/run_configs.sh`
- `code/sweep.py`

**results/**

- `results/densify_3000.json`

**logs/**

- `logs/densify_3000.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

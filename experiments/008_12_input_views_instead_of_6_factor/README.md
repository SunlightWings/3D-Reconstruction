# 008 — 12 input views instead of 6 (factor 4)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#008** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 CEST (this run took 12.2 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag views12 --iters 500 1000 3000 7000 --source views12
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` rebuilt on 12 views |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 12 input (evenly spaced, held-out frames excluded), 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 16.22 | 15.80 | 15.41 | 15.94 |
| ball/123_14363_28981 | 11.18 | 11.28 | 13.91 | 14.45 |
| bowl/70_5792_13401 | 12.21 | 13.29 | 14.31 | 15.93 |
| broccoli/412_56288_108844 | 9.42 | 11.25 | 12.38 | 13.35 |
| hydrant/167_18184_34441 | 10.13 | 11.72 | 12.38 | 13.29 |
| remote/350_36761_68623 | 8.62 | 11.41 | 15.24 | 15.44 |
| teddybear/187_20215_38541 | 15.70 | 16.71 | 16.96 | 18.52 |
| toaster/372_41229_82130 | 11.45 | 10.54 | 13.73 | 14.21 |
| **mean PSNR** | **11.867** | **12.748** | **14.289** | **15.141** |
| **mean SSIM** | 0.4478 | 0.4394 | 0.4270 | 0.4221 |
| min PSNR | 8.62 | 10.54 | 12.38 | 13.29 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **15.141 dB at 7000 iterations**. **FAIL**, short by 4.86 dB.

**Interpretation:** Rules IN input-view count as the one factor that actually helps: 15.141 dB at 7000 vs the baseline 14.023 dB (+1.118 dB), with SSIM also up (0.4221 vs 0.3768) rather than traded away. This is consistent with the entry #002 finding that the metric is dominated by background the 6-view oracle cannot initialise -- extra views add real background coverage. Still 4.86 dB short of target.


---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/run_configs.sh`
- `code/sweep.py`

**results/**

- `results/views12.json`

**logs/**

- `logs/views12.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

# 004 — Harness validation: A3 reproduction and foreground-mask re-scoring (analysis, no training)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#004** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 20:59 CEST
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (dirty, as above)
- **Command executed:** inline `python3 -c` re-scoring the existing A3 oracle renders under both masks, using `common.evaluate_renders` and the `fgmask` helper now preserved in `code/downstream_3dgs/oracle_sweep/reeval.py`. Equivalent re-runnable form:
  ```
  python3 code/downstream_3dgs/oracle_sweep/reeval.py <tag> foreground 500 1000 3000 7000
  ```
  (the inline form read `DIAG_HEAVY_ROOT/<scene>/heldout/oracle/renders` rather than a sweep tag)

**Config:** no training; re-scored the already-existing A3 oracle renders at 30000 iterations. All other config fields as in the A3 row of #001.

**Scenes used:** the 8-scene GPU subset.

**Results:**

| scene | content-mask PSNR | content SSIM | foreground-mask PSNR | foreground SSIM |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.45 | 0.3738 | 17.33 | 0.4698 |
| ball/123_14363_28981 | 14.05 | 0.3265 | 14.91 | 0.3304 |
| bowl/70_5792_13401 | 14.76 | 0.3800 | 17.45 | 0.2773 |
| broccoli/412_56288_108844 | 12.04 | 0.4224 | 15.09 | 0.2731 |
| hydrant/167_18184_34441 | 12.83 | 0.3016 | 12.90 | 0.3027 |
| remote/350_36761_68623 | 14.47 | 0.3251 | 11.79 | 0.2638 |
| teddybear/187_20215_38541 | 15.83 | 0.3593 | 15.68 | 0.3554 |
| toaster/372_41229_82130 | 13.06 | 0.2977 | 13.47 | 0.2977 |
| **mean** | **14.061** | **0.3483** | **14.828** | **0.3213** |

**PASS/FAIL:**
- Target: **reproduce A3's published 14.06 dB mean** to confirm this sweep's metric is comparable. Measured **14.061 dB**, per-scene identical to the DIAGNOSTIC_RESULTS.md A3 table. **PASS.**
- Target: **oracle >= 20 dB** under a foreground-only mask (the region where GT geometry actually exists). Measured **14.828 dB**. **FAIL.**

**Interpretation:** Rules out "the metric is merely diluted by unreconstructable background" as a complete explanation — restricting the metric to the object, where the oracle actually has GT points, buys only +0.77 dB and still misses 20 dB by 5.2 dB, so the object itself is genuinely badly reconstructed; and it rules in this sweep harness as metric-identical to A3.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/reeval.py`

**results/**

- `results/a3_oracle_30000_both_masks.json`
- `results/noise_floor_both_masks.json`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

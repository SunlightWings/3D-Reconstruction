# 031 — CPU camera analysis: focal-length error, per-scene predictors, and error against ground truth

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#031** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11 13:35–13:43 CEST (CPU only, `nice -n 15`, run beside the GPU swap probe)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/cpu_camera_analysis.py`
- **Command executed:**
  ```
  OMP_NUM_THREADS=4 nice -n 15 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_camera_analysis.py
  ```
- **Config:** no training, no rendering. Pose errors use the ablation_compare.py construction (Sim(3) on the six
  input camera centres; mean per-camera rotation angle; median centre residual in scene radii). Focal error =
  median |f_arm / f_ref − 1| over cameras and both axes. Rendering deltas are read from #029's 40-scene JSON
  (foreground, raw). GT intrinsics are mapped to 518 px with `original_to_518_affine`.
- **Scenes used:** (1)+(2) the frozen 40-scene manifest × 7 W4A4 arms = 280 scene-arm pairs; (3) all 1330
  view-groups of the disagreement dataset, which are drawn from those same 40 scenes.

**1. Focal-length error beside pose error, vs full precision (n = 40):**

| arm | cam rot (deg) | centre (radii) | focal error |
|---|---:|---:|---:|
| w4a4 | 1.2754 | 0.02063 | 2.614 % |
| w4a4_local | 1.1014 | 0.01896 | 2.335 % |
| w4a4_no_lwc | 1.0799 | 0.01662 | 2.875 % |
| w4a4_no_smooth | 1.1390 | 0.02034 | 2.734 % |
| w4a4_no_lac | 1.3160 | 0.02286 | 3.862 % |
| w4a4_no_rot | 2.0323 | 0.03665 | 6.693 % |
| w4a4_rtn | 14.6565 | 0.12093 | 18.652 % |

**2. Spearman ρ between a scene's geometry error and its rendering loss vs full** (`*` p < 0.05):

| slice | target | cam rot | centre | focal | depth | points |
|---|---|---|---|---|---|---|
| all arms, n = 280 | ΔPSNR | +0.199* | +0.216* | +0.270* | +0.186* | +0.072 |
| all arms, n = 280 | ΔLPIPS | +0.399* | +0.348* | +0.301* | +0.277* | +0.286* |
| w4a4 only, n = 40 | ΔPSNR | −0.067 | −0.032 | −0.090 | −0.022 | −0.222 |
| w4a4 only, n = 40 | ΔLPIPS | **+0.409\* (p 0.0088)** | +0.192 | −0.098 | +0.067 | +0.106 |
| mild arms (no rtn, no_rot), n = 200 | ΔPSNR | −0.048 | −0.045 | +0.054 | −0.044 | **−0.231\*** |
| mild arms (no rtn, no_rot), n = 200 | ΔLPIPS | +0.258* | +0.188* | +0.057 | +0.086 | +0.062 |

**3. Against CO3D ground truth, 1330 view-groups, 0 unreadable:**

| quantity | mean | median | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| full vs GT rotation (deg) | 1.5905 | 1.0842 | 2.5521 | 10.2940 | 27.5783 |
| w4a4 vs GT rotation (deg) | 2.0784 | 1.4269 | 3.1320 | 11.6597 | 168.2326 |
| w4a4 vs full rotation (deg) | 1.3383 | 0.9149 | 1.9233 | 6.2651 | 149.0445 |
| full vs GT focal error | 0.0345 | 0.0237 | 0.0781 | 0.1742 | 0.2231 |
| w4a4 vs GT focal error | 0.0466 | 0.0287 | 0.1058 | 0.2191 | 0.2892 |

Paired, w4a4 minus full rotation error vs GT: mean +0.4879°, median +0.3113°, t-test p = 1.29e-05, Wilcoxon
p = 6.95e-115, w4a4 worse in 1073/1330. W4A4 departs from full by > 2° in 124/1330 groups, > 5° in 23, > 10° in 6.
**19 of the 23 groups above 5° are one scene, `bowl/70_5792_13401`**, where full precision is itself 8–28° off GT
(the other four: remote/195_20989_41543, hydrant/167_18184_34441, toytruck/190_20494_39385, vase/380_44863_89631).

**PASS/FAIL:**
- Target: **measure whether the swap's camera condition carried a large focal-length error.** `w4a4_rtn` focal is
  off by 18.652 %. **Yes — the swap's "camera damage" is confounded with focal damage and must be split** (queued
  as `geometry_probes.py camsplit`).
- Target: **does any per-scene geometry error predict W4A4's rendering loss (n = 40)?** Camera rotation vs ΔLPIPS
  ρ = +0.409, p = 0.0088; ten correlations were tested in that slice, Bonferroni ≈ 0.088. **Suggestive, not
  established.**
- Target: **does W4A4 make cameras measurably worse against ground truth?** Yes, in 1073/1330 groups,
  p = 6.95e-115, by a median +0.31° on top of VGGT's own median 1.08°. **PASS.**

**Interpretation.** Quantization makes W4A4's cameras reliably but slightly worse against ground truth — about a
third larger than VGGT's own error — which is small next to the error VGGT already makes and consistent with
W4A4 rendering like full precision. The apparent rare catastrophic W4A4 failures are **not** a quantization
phenomenon: they concentrate on one rotationally symmetric bowl where full precision is already unstable, so they
must not be reported as W4A4 tail risk. Among per-scene predictors, camera rotation is the one that tracks
perceptual loss even within the mild arms, consistent with the swap probe's interim finding that cameras, not
points, carry the damage. **Unexplained, flagged:** in the mild arms, larger point error goes with *smaller*
PSNR loss (ρ = −0.231, p = 0.00099); no mechanism is proposed and none should be assumed.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/cpu_camera_analysis.py`

**results/**

- `results/cpu_camera_analysis.json`

**logs/**

- `logs/cpu_camera_analysis.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

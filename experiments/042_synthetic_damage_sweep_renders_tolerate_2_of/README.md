# 042 — Synthetic damage sweep: renders tolerate ~2° of camera error and a full scene-radius of point noise

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#042** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11 15:08 → 21:11 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree (`code/quantization/geometry_probes.py`)
- **Command executed:**
  ```
  /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/geometry_probes.py sweep --scenes 12 --deadline 00:15
  ```
  (launched by `code/quantization/run_probes_day.sh`)
- **Config:**

| Field | Value |
|---|---|
| Base geometry | full-precision VGGT (cameras, intrinsics, stride-4 points) |
| cameraR<deg> | every training camera rotated by exactly <deg> about its own centre, random axis, seeded per scene; centres, intrinsics, points untouched. Levels 0.5, 1, 2, 4, 8, 16° |
| points<σ> | isotropic Gaussian jitter on every initial point, median displacement σ scene radii (ablation_compare's point_err statistic); cameras untouched. Levels 0.03, 0.06, 0.12, 0.25, 0.5, 1.0 |
| Evaluation cameras | full's held-out source |
| Iterations / densify / views / mask / background / depth reg | 7000 / GraphDECO default / 6 in, 9 held-out / foreground / black, unmasked / off |

  Offline check before launch: realised rotations within 0.0006° of target, centre shift ≤ 7.8e-08; realised
  median point shifts 0.0300–0.9988 radii for targets 0.03–1.0.
- **Scenes used:** the first 12 of the frozen manifest: apple/110_13051_23361, apple/189_20393_38136,
  ball/123_14363_28981, ball/375_42693_85518, bench/415_57112_110099, bench/415_57121_110109,
  book/119_13962_28926, book/247_26469_51778, bowl/69_5465_12831, bowl/70_5792_13401,
  broccoli/372_41112_81867, broccoli/412_56288_108844. Per-scene values in `geometry_probe_sweep_it7000.json`.

**Results (n = 12; positive Δ = worse than full; per-level p are paired t, 12 levels tested):**

| condition | PSNR | Δ PSNR | p | LPIPS | Δ LPIPS | p | worse in |
|---|---:|---:|---:|---:|---:|---:|---|
| full | 10.1748 | | | 0.6561 | | | |
| cameraR 0.5° | 10.1223 | +0.0524 | 0.545 | 0.6571 | +0.0011 | 0.7728 | 7/12 |
| cameraR 1° | 10.0624 | +0.1124 | 0.4481 | 0.6581 | +0.0020 | 0.5142 | 7/12 |
| cameraR 2° | 10.1560 | +0.0187 | 0.8943 | 0.6583 | +0.0022 | 0.709 | 5/12 |
| cameraR 4° | 9.6575 | +0.5173 | 0.05027 | 0.6858 | +0.0297 | 0.005114 | 8/12 |
| cameraR 8° | 9.4156 | +0.7592 | 0.0264 | 0.7030 | +0.0469 | 0.001629 | 9/12 |
| cameraR 16° | 9.1731 | +1.0016 | 0.0003206 | 0.7268 | +0.0708 | 3.308e-07 | 12/12 |
| points 0.03 | 10.1386 | +0.0362 | 0.6909 | 0.6561 | +0.0000 | 0.9838 | 5/12 |
| points 0.06 | 10.2980 | −0.1232 | 0.3385 | 0.6549 | −0.0012 | 0.7478 | 6/12 |
| points 0.12 | 10.2274 | −0.0526 | 0.6571 | 0.6544 | −0.0016 | 0.6066 | 6/12 |
| points 0.25 | 10.2430 | −0.0682 | 0.5584 | 0.6642 | +0.0081 | 0.07343 | 5/12 |
| points 0.5 | 10.4574 | −0.2827 | 0.03602 | 0.6586 | +0.0026 | 0.6739 | 2/12 |
| points 1.0 | 10.3345 | −0.1597 | 0.4057 | 0.6665 | +0.0104 | 0.0373 | 5/12 |

**Trend across levels, one Spearman ρ per scene, Wilcoxon across the 12 scenes** (the pooled scenes × levels
Spearman printed by an ad-hoc check treated 72 correlated pairs as independent and is **not** cited):

| family | PSNR: median ρ, positive, p | LPIPS: median ρ, positive, p |
|---|---|---|
| camera rotation | +0.486, 11/12, 0.000977 | +0.743, 11/12, 0.000977 |
| point jitter | −0.229, 4/12, 0.212 | +0.514, 10/12, 0.0415 |

**PASS/FAIL:**
- Target: **locate the camera-error level at which rendering measurably degrades.** No effect at 0.5–2°
  (p ≥ 0.45); LPIPS significant from 4° (p 0.0051), PSNR from 8° (p 0.026); monotone trend p 0.000977. **Threshold
  lies between 2° and 4°.** Measured W4A4 camera error, 1.2754° (#029), is **below** it. **PASS.**
- Target: **the same for point error.** Up to 1.0 scene radius — more than twice `w4a4_rtn`'s 0.46422 — PSNR shows
  no degradation (trend p 0.212; one level, 0.5, is nominally *better* at p 0.036 of 12 tests, not significant after
  correction); LPIPS shows a weak upward trend (p 0.0415). **Points: effectively no threshold in range.**

**Interpretation.** This places every measured model on one axis and explains the W4A4 null from first principles:
rendering here is insensitive to camera rotation below ~2–4° and to point error up to a full scene radius, and
W4A4's cameras sit at 1.28° — inside the tolerance — so its rendering matches full precision. `w4a4_rtn`'s
14.66° sits past it, and pure 16° rotation costs +1.0016 dB, the same order as `rtn`'s camera-swap loss of +0.9721
dB (#039), consistent with — but not proof of — pose rather than focal length carrying `rtn`'s damage (the camsplit
probe tests that directly). Caveat: 12 scenes, the first 12 in manifest order, including the unstable bowl.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/geometry_probes.py`
- `code/run_probes_day.sh`

**results/**

- `results/geometry_probe_sweep_it7000.json`

**logs/**

- `logs/probe_sweep.log.gz`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

# 044 — Viewpoint sensitivity: geometry damage is largest on the CLOSEST held-out views, not the farthest

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#044** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-12, launched 09:16, `viewpoint_sensitivity.json` written before 09:33 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/viewpoint_sensitivity.py`
- **Command executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=2 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/viewpoint_sensitivity.py
  ```
- **Hypothesis under test (mine, stated in conversation):** *the benchmark's insensitivity to geometry is an
  artefact of interpolation; damage should grow on held-out views far from the training cameras, and evaluating on
  distant views would restore sensitivity.*
- **Config:** no training, no rendering — re-scores the existing 7000-iteration held-out renders per view.
  Foreground mask, PSNR and LPIPS (AlexNet on the mask bounding box). Distances from each held-out camera to the
  **nearest** training camera: `axis_angle` (between optical axes) and `orbit_angle` (subtended at the training-camera
  centroid). Conditions: w4a4, w4a4_rtn, and the swap cells P/K/B/R/F from #039 and #043, each compared with `full`.
  Test: per-scene Spearman between view distance and per-view damage, then Wilcoxon over the 40 scenes — never a
  pooled correlation over scene × view pairs (the pseudo-replication corrected in #042).
- **Scenes used:** all 40, 9 held-out views each. Distance spread: orbit angle median 16.43°, range 0.01–77.94°;
  axis angle median 10.39°.

**Results (median per-scene ρ; positive = damage grows with distance):**

| condition | PSNR: ρ, scenes positive, p | LPIPS: ρ, scenes positive, p |
|---|---|---|
| w4a4 | −0.108, 15/40, 0.076 | −0.192, 13/40, 0.0252 |
| w4a4_rtn | −0.342, 7/40, 4.25e-06 | −0.508, 6/40, 2.35e-06 |
| swapP (bad points) | +0.025, 21/40, 0.861 | +0.042, 21/40, 0.534 |
| swapK (bad cameras) | −0.425, 2/40, 7.02e-08 | −0.592, 4/40, 3.35e-07 |
| swapB (both) | −0.367, 3/40, 3.63e-07 | −0.525, 6/40, 3.13e-07 |
| swapR (pose only) | −0.408, 4/40, 3.59e-06 | −0.592, 4/40, 5.88e-07 |
| swapF (focal only) | −0.283, 14/40, 0.00411 | −0.252, 8/40, 0.000189 |

(orbit angle shown; the axis-angle column in `viewpoint_sensitivity.json` agrees throughout.)

**Floor-effect check.** `full` itself degrades with distance: PSNR ρ = −0.508 (2/40 positive, p 1.69e-07),
LPIPS ρ = +0.692 (40/40, p 3.54e-08). Normalising damage by full's own score does **not** reverse the trend:
relative PSNR damage for swapK ρ = −0.408 (p 1.32e-07), relative LPIPS damage ρ = −0.625 (p 4.36e-07).

**PASS/FAIL:** Target: **damage grows with held-out view distance (positive ρ).** Measured ρ is **negative** and
significant for every camera-damage condition, in both absolute and relative form. **FAIL — the hypothesis is
refuted, with the sign reversed.** Point damage remains flat at every distance (p 0.86).

**Interpretation.** Evaluating on more distant viewpoints would **reduce**, not increase, the benchmark's
sensitivity to geometry error. Distant views are poor for every arm — full precision included — so all arms
converge toward a common floor and the gap between them closes; the differences live in the near views, where a
correct model renders well and a camera-damaged one no longer can. This rules out "evaluate further away" as a fix
for the insensitivity documented in #039–#043, and it is a second, independent demonstration that the limitation is
in what sparse-view 3DGS can render at all, not in where the held-out cameras were placed. Caveat: CO3D held-out
views come from the same turntable orbit; a capture with genuinely novel elevations might behave differently, and
that is untested here.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/viewpoint_sensitivity.py`

**results/**

- `results/viewpoint_sensitivity.json`

**logs/**

- `logs/viewpoint_sensitivity.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

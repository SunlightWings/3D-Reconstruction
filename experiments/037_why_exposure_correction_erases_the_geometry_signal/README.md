# 037 — Why exposure correction erases the geometry signal: it removes ~half of the geometry-induced loss

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#037** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11, `cpu_exposure.json` written before 14:11:30 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py exposure`)
- **Command executed:**
  ```
  nice -n 15 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/cpu_probes.py exposure
  ```
- **Config:** no training, no image loading. Per scene and arm, from #029's 40-scene JSON (foreground):
  *extra correction gain* = [PSNR after per-image gain+bias − raw PSNR] for the arm, minus the same quantity for
  `full` on the same scene. Correlated with that scene-arm's geometry error from #031.
- **Scenes used:** 40 scenes × 7 W4A4 arms = 280 pairs.

**Spearman ρ of extra correction gain with:**

| slice | cam rot | focal | point err | raw PSNR loss |
|---|---|---|---|---|
| all arms, n = 280 | +0.221 (p 0.00019) | +0.262 (p 8.9e-06) | +0.118 (p 0.049) | +0.766 (p 3.4e-55) |
| mild arms (no rtn, no_rot), n = 200 | +0.026 (p 0.71) | +0.030 (p 0.67) | −0.111 (p 0.12) | +0.705 (p 2.1e-31) |
| all arms, no bowl, n = 273 | +0.227 (p 0.00016) | +0.270 (p 5.8e-06) | +0.123 (p 0.043) | +0.782 (p 1.4e-57) |

**Per-arm mean extra correction gain (dB), Wilcoxon vs 0, n = 40:** w4a4 +0.1397 (p 0.277); w4a4_local +0.1234
(0.211); no_lwc +0.0383 (0.942); no_smooth +0.0347 (0.785); no_lac +0.0953 (0.476); **no_rot +0.4534 (0.000336)**;
**rtn +0.5246 (0.000402)**.

**PASS/FAIL:** Target: **test #029's claim that exposure correction removes geometry-driven signal.** For the two
arms with real geometry damage, the correction recovers +0.4534 and +0.5246 dB *more* than it does for full
precision on the same scenes (p ≤ 0.0004), and the extra gain tracks camera and focal error across arms. For
`rtn` that is 0.5246 of its 0.9848 dB raw gap. **PASS — mechanism supported.**

**Interpretation.** Roughly half of the rendering loss that bad geometry causes shows up as a global,
per-image brightness/contrast shift, and a per-image gain+bias fit removes exactly that half — which is why every
corrected metric loses its relationship to geometry. The images, training and evaluation cameras are identical
across arms, so this shift is caused by the geometry, not by auto-exposure. The *why* (e.g. misregistered views
averaged into lower-contrast Gaussians) is **not measured**. Caveat: the ρ ≈ 0.77 with raw PSNR loss is partly
mechanical — a worse raw render has more for a gain fit to recover — so the camera/focal correlations and the
per-arm tests are the evidence, not that column. In mild arms there is no geometry damage to remove and no
correlation.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/cpu_probes.py`

**results/**

- `results/cpu_exposure.json`

**logs/**

- `logs/cpu_exposure.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

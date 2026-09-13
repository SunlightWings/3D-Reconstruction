# 038 — Power: how many scenes each observed effect needs

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#038** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11, finished 14:11:30 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py power`)
- **Command executed:**
  ```
  nice -n 15 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/cpu_probes.py power
  ```
- **Config:** for each full-vs-arm contrast on #029's 40-scene per-scene data (foreground): the smallest n at which a
  two-sided paired t-test (α = 0.05) reaches 80 % power, by the noncentral t distribution, **taking the observed
  mean and sd as the true values.**
- **Scenes used:** 40-scene set, 7 arms.

| contrast (full vs …) | PSNR: mean / sd → n | SSIM → n | LPIPS: mean / sd → n | corrected PSNR → n |
|---|---|---|---|---|
| w4a4 | +0.1359 / 0.5083 → **112** | 986 | +0.0108 / 0.0245 → **43** | 170,555 |
| w4a4_local | +0.1638 / 0.4951 → 74 | 117 | → 53 | 689 |
| w4a4_no_lwc | −0.0344 / 0.5346 → 1901 | inf | → 2046 | 384 |
| w4a4_no_smooth | +0.0181 / 0.4390 → 4605 | 1897 | → 248 | 10,098 |
| w4a4_no_lac | +0.1000 / 0.4730 → 178 | 106 | → 54 | 115,208 |
| w4a4_no_rot | +0.4316 / 0.6561 → 21 | 108 | → 33 | 4774 |
| w4a4_rtn | +0.9848 / 1.2067 → 14 | 34 | → 14 | 174 |

**PASS/FAIL:** Target: **state, for the headline null, what sample size would resolve it.** Full vs W4A4 on PSNR
needs ~112 scenes at the observed effect (we have 40); on exposure-corrected PSNR the observed effect is
effectively zero (170,555). **Reported.**

**Interpretation.** The W4A4 PSNR null is an *underpowered* null at the observed +0.136 dB — roughly 3× the scenes
would be needed — whereas the corrected-PSNR null is a genuine absence. Caveat: these are post-hoc numbers built on
observed effects, so they are planning figures for a follow-up, not evidence in themselves; for contrasts already
significant (LPIPS 43, no_rot 21, rtn 14) they are circular and only confirm consistency.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/cpu_probes.py`

**results/**

- `results/cpu_power.json`

**logs/**

- `logs/cpu_power.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

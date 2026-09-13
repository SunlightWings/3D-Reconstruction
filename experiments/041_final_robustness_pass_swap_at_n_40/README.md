# 041 — Final robustness pass (swap at n = 40)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#041** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11 15:09–15:10 CEST (CPU, `run_cpu_chain.sh`)
- **Command executed:** `nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/cpu_probes.py robustness --workers 8`
- **Config / scenes:** identical to #032, with the swap JSON now complete (n = 40). Output `cpu_robustness.json`
  (overwrites pass 1; pass-1 numbers are preserved in #032).
- **Results:** every non-swap row identical to #032. Swap rows as reported under #039.
- **PASS/FAIL:** Target: **the interim n = 29 swap conclusion survives at n = 40.** Points null (perm p 0.694),
  camera effect robust (worst leave-one-out p 1.1e-05; without the bowl +0.9181, p 1e-05). **PASS.**
- **Interpretation:** the interim swap numbers in #032 are superseded by #039; the conclusion is unchanged.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/cpu_probes.py`
- `code/run_cpu_chain.sh`

**results/**

- `results/cpu_robustness.json`

**logs/**

- `logs/cpu_robustness_final.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

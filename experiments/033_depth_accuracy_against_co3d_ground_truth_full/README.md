# 033 — Depth accuracy against CO3D ground truth, full vs W4A4, all 1330 view-groups

This is entry **#033** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11 13:57 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/cpu_probes.py`
- **Command executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py depthgt --workers 8
  ```
- **Config:** no training. GT depth = CO3D depth map resampled onto the 518×518 pad grid with
  `prep_depth.gt_depth_518` (the loader the depth-regularised oracle used), valid where CO3D's depth mask AND the
  object mask hold — i.e. **the object only**. Each predicted depth is aligned to GT by one per-image median scale
  (VGGT depth is up to scale). Metrics: median absolute relative error, fraction of pixels with
  max(pred/gt, gt/pred) < 1.25. Edge test: full-vs-w4a4 depth disagreement (after a median scale) in a 5-px band
  inside the object boundary, divided by the interior's. Sanity check before the run: full-precision absrel on one
  frame 0.0067, δ<1.25 = 1.0.
- **Scenes used:** all 1330 view-groups (drawn from the 40 frozen scenes); 1330 usable, 0 failed.

**Results:**

| subset | metric | full | w4a4 | w4a4 − full (median) | Wilcoxon p | w4a4 larger |
|---|---|---:|---:|---:|---:|---|
| all, n = 1330 | absrel | 0.00542 | 0.00645 | +0.00103 (+0.00084) | 2.01e-191 | 1214/1330 |
| all, n = 1330 | δ<1.25 | 0.99851 | 0.99871 | +0.00019 (0.00000) | 9.1e-17 | 527/1330 |
| no bowl, n = 1296 | absrel | 0.00539 | 0.00644 | +0.00105 (+0.00086) | 5.9e-191 | 1194/1296 |
| no bowl, n = 1296 | δ<1.25 | 0.99847 | 0.99868 | +0.00021 (0.00000) | 2.27e-24 | 527/1296 |

Edge band over interior disagreement: median ratio **1.594**, > 1 in **99.1 %** of 1330 groups, Wilcoxon on log
ratio p = 2.34e-218.

**PASS/FAIL:**
- Target: **is W4A4 object depth measurably worse than full against GT?** Yes, in 1214/1330 groups, p = 2.01e-191 —
  but by +0.00103 absolute relative error on a base of 0.00542. **PASS (real, tiny).**
- δ<1.25 is saturated (≥ 0.9985 for both) and its median difference is 0: **uninformative, not a finding.**
- Target: **does quantization disagreement concentrate at object edges?** 1.594× in 99.1 % of groups. **PASS.**

**Interpretation.** On the object, per-view depth survives W4A4 almost untouched — about 0.1 % of depth extra error,
consistent in sign but negligible in size — and what disagreement there is sits disproportionately on object
boundaries. The edge band is defined with the GT mask, which a deployed predictor would not have; whether a
mask-free proxy (depth gradient) recovers it is tested in the predictability run.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/cpu_probes.py`

**results/**

- `results/cpu_depthgt.json`

**logs/**

- `logs/cpu_depthgt.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

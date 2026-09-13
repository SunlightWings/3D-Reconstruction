# 035 — W4A4 point error splits into thirds: depth shape, per-view scale drift, cameras (supersedes #034's inference)

This is entry **#035** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11 14:15–14:25 CEST (CPU; two re-runs of the #034 command with added measurements)
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py decompose` extended)
- **Command executed (final):**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py decompose --workers 6
  ```
- **Config:** as #034, plus (a) the same medians restricted to the object mask and to background content pixels,
  and (b) *depth-shape*: W4A4 depth with **each view rescaled by its own median factor**, unprojected through
  full's cameras — removing per-view scale drift as well as camera error. Per-view drift = std of log per-view
  scale factors across the six views.
- **Scenes used:** all 1330 view-groups.

**Results (median over groups of per-group medians, scene radii):**

| region | total | depth, one global scale | share | depth shape, per-view scale | share |
|---|---:|---:|---:|---:|---:|
| object | 0.02163 | 0.01397 | 0.637 | 0.00646 | 0.316 |
| background | 0.03278 | 0.02175 | 0.627 | 0.01089 | 0.337 |

Per-view depth-scale drift of W4A4 relative to full: median 0.01308 (≈ 1.3 %), p90 0.02292.

**PASS/FAIL:**
- #034's inference — *"W4A4's depth disagreement lives mainly in the background"* — **FAIL, refuted**: the depth
  share is 0.637 on the object and 0.627 in the background.
- Target: **reconcile #033 (object depth almost unchanged per view) with #034 (depth explains ~70 %).** With per-view
  scaling, depth shape explains only ~0.32–0.34 of the error; the other ~0.3 is per-view scale drift of ~1.3 %.
  **Reconciled.**

**Interpretation.** W4A4's world-point disagreement with full precision is roughly one third each: per-view depth
**shape**, per-view depth **scale drift** (the six views disagreeing about scale by ~1.3 %), and **cameras**. Each
view's depth is individually almost unchanged — which is why #033, scaling every image separately, saw only +0.1 %.
The damage is mostly in how views fit *together*: scale drift and camera error are both inter-view inconsistencies,
which is precisely what a cross-view self-consistency check can observe without full precision. Whether it does
is the predictability run. **#034's background inference is withdrawn.**

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/cpu_probes.py`

**results/**

- `results/cpu_decompose.json`

**logs/**

- `logs/cpu_decompose.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

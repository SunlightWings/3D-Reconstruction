# 045 — Input-order ensembling gives a deployable confidence signal for camera pose (ρ = 0.88)

This is entry **#045** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-12, w4a4 09:20–09:26, full 09:26–09:30, w4a4_rtn from 09:30 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/pose_uncertainty.py`, `run_pose_uncertainty.sh`
- **Commands executed:**
  ```
  ./code/quantization/run_pose_uncertainty.sh
  #   -> pose_uncertainty.py --variant w4a4    --perms 4 --scenes all
  #   -> pose_uncertainty.py --variant full     --perms 4 --scenes all
  #   -> pose_uncertainty.py --variant w4a4_rtn --perms 4 --scenes all
  ```
- **Config:** inference only, no 3DGS. Per scene, the same six 518×518 pad-mode images are run through the model in
  4 orderings (identity + 3 seeded shuffles); outputs are un-permuted, then every ordering is carried into ordering
  0's frame by Sim(3) on the six camera centres (a different reference frame is not an error). *Spread* = mean over
  the six cameras of the mean pairwise rotation angle across orderings. *Error* = ordering 0's pose error against
  full precision's saved cameras, and against CO3D ground truth, by the same construction. Frozen manifest SHA
  verified at start. Cost: 4× inference, ~10 s/scene.
- **Scenes used:** all 40.

**Determinism check:** full precision, canonical ordering, vs the saved `full.npz` cameras — max 2.37e-02°,
median 8.30e-03°. The pipeline is reproducible, so the spread below is genuinely order-induced.

**Results (n = 40):**

| variant | spread median (range) | spread vs error-vs-full | vs GT error |
|---|---|---|---|
| w4a4 | 1.1389° (0.4452–11.1944) | **ρ +0.880, p 6.97e-14** (no bowl +0.871, p 5.59e-13) | ρ +0.764, p 9.73e-09 |
| full (control) | 0.6084° (0.2715–6.4497) | ρ −0.062, p 0.703 (this target is ~0 by construction) | ρ +0.709, p 3.02e-07 |

Quantization increases order-sensitivity: w4a4's spread exceeds full's in **39/40** scenes, Wilcoxon p = 4.32e-09.
Excess spread (w4a4 − full) vs w4a4's pose error: ρ = +0.751, p = 2.37e-08. Full's spread alone vs w4a4's pose
error: ρ = +0.391, p = 0.0125.

**PASS/FAIL:**
- Target: **a signal computable from the quantized model alone predicts its camera-pose error.** ρ = 0.880,
  p = 7e-14, robust to removing the bowl scene. **PASS** — and it is the first confidence signal in this project
  that works, after per-point true error (#030) and cross-view self-consistency (#036) both failed.
- Target: **is it specific to quantization, or just a scene-ambiguity detector?** Partly the latter: full
  precision's own spread predicts its GT error (ρ +0.709) and predicts w4a4's error (ρ +0.391). But the
  **excess** spread — the part quantization adds — predicts quantization's pose error at ρ = 0.751.
  **PASS with the caveat stated.**

**Interpretation.** VGGT's output depends on input order, and the disagreement across orderings is a usable
uncertainty estimate: it needs no ground truth, no full-precision reference, and no training, and it targets camera
pose — the quantity #039/#043 show carries the rendering damage. Two honest limits. It costs 4× inference, which
must be weighed against the proposal's efficiency clause (a cheaper 2-ordering variant is untested). And it is not
purely a quantization signal: unstable scenes are unstable for full precision too, so part of the correlation is
"this scene is ambiguous" rather than "quantization broke this scene" — the excess-spread result is the way to
state the quantization-specific part. **What it does not yet show is that acting on the signal helps**: at W4A4 the
pose error it predicts, 1.28° on average, is below the 2–4° rendering-damage threshold of #042, so there is nothing
to gain downstream at this bit-width. Its value would be at W3A3 and below.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/pose_uncertainty.py`
- `code/run_pose_uncertainty.sh`

**results/**

- `results/pose_uncertainty_full_p4.json`
- `results/pose_uncertainty_w4a4_p4.json`

**logs/**

- `logs/pose_unc_chain.log`
- `logs/pose_uncertainty_full.log`
- `logs/pose_uncertainty_w4a4.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

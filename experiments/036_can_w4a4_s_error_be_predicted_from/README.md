# 036 — Can W4A4's error be predicted from W4A4's own output? Pixel level: partly; camera level: no

This is entry **#036** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11; absolute-target run 13:57–14:01:11, relative-target run finished 14:07:58 CEST (CPU)
- **Timestamp correction for #034 and #035:** those entries give 14:10 and 14:15–14:25. Those times were written
  without reading the clock and are **wrong**. The measured file times are: #033 `cpu_depthgt.json` 13:57:57;
  #034 and #035 were successive runs of `cpu_probes.py decompose` whose final output `cpu_decompose.json` is
  stamped 14:04:02. The results in those entries are unaffected; only their stated times are.
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py predict`)
- **Commands executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py predict --workers 8
  PREDICT_TARGET=relative nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py predict --workers 8
  ```
- **Config:** no training. All 1330 view-groups; 300 random content pixels per view (seeded per group), 2,394,000
  pixels. **Label:** pixel in the top decile of W4A4-vs-full world-point error *within its own view-group* (the
  "where in this image" question a confidence map must answer). **Target** either absolute error (scene radii) or
  **relative** error (divided by the point's distance from full's camera). **Features — W4A4 output and RGB only,
  nothing from full precision:** log depth; log depth / view median; relative depth gradient; relative local depth
  std (5×5); image gradient; image texture (7×7 std); image intensity; radial image position; cross-view
  self-inconsistency (each pixel's W4A4 point reprojected into the other five views with W4A4's cameras, median
  relative disagreement with W4A4's depth there). **Models:** L2 logistic regression on standardised features
  (`linear`) and on 10 quantile bins per feature (`binned`). **Validation:** 5 folds holding out whole scenes, and
  5 folds holding out whole categories (20 categories). Foreground mask used only to subset evaluation, never as a
  feature.
- **Scenes used:** all 1330 view-groups of the 40-scene / 20-category set; 0 failed.

**Pixel level (AUROC; 0.5 = chance):**

| target | region | best single feature | scene-held-out linear / binned | category-held-out linear / binned |
|---|---|---|---|---|
| absolute | all content | log_depth_rel_median 0.821 | 0.8882 / 0.8979 | 0.8873 / 0.8970 |
| absolute | object only (1,600 positives) | depth_local_std_rel 0.828 | 0.8806 / 0.8678 | 0.8968 / 0.8808 |
| **relative** | all content | depth_local_std_rel 0.721 | **0.8165 / 0.8310** | **0.8171 / 0.8328** |
| **relative** | **object only** (6,169 positives) | depth_local_std_rel 0.641 | **0.6432 / 0.5888** | **0.6677 / 0.6314** |

Image-appearance features alone are at or below chance in every setting (img_grad 0.436–0.471, img_texture
0.432–0.451, img_intensity 0.489–0.563). Self-inconsistency alone: 0.770 / 0.691 absolute (all / object), 0.728 /
0.594 relative. Fold ranges are in `cpu_predict*.json`; object-only category folds span 0.606–0.787 (relative).

**Camera level (per view-group, Spearman, n = 1330; no-bowl n = 1296):**

| camera error | ~ self-inconsistency | ~ depth spread |
|---|---|---|
| w4a4 vs full rotation | ρ −0.032, p 0.248 (no bowl −0.024, p 0.395) | ρ −0.361, p 3.19e-42 |
| w4a4 vs GT rotation | ρ −0.090, p 0.00106 | ρ −0.352, p 4.22e-40 |
| full vs GT rotation | ρ −0.048, p 0.0808 | ρ −0.274, p 2.36e-24 |

**PASS/FAIL:**
- Target: **W4A4 pixel error is predictable from W4A4's own output, and the predictor transfers to unseen
  categories.** All content, relative target: AUROC 0.817–0.833, category-held-out equal to scene-held-out.
  **PASS.** On the object only: 0.589–0.668. **FAIL — weak.**
- The absolute-target numbers (≈ 0.89) **overstate predictability**: absolute point error grows with distance, and
  depth alone scores 0.811. The relative target is the one to cite.
- Target: **a cross-view self-consistency signal flags W4A4's camera error.** ρ = −0.032, p = 0.248. **FAIL.**

**Interpretation.** Where W4A4's geometry is wrong is predictable from its own output mainly at depth
discontinuities and image periphery — geometric cues, not appearance ones, so the 40-object appearance-diversity
concern raised in discussion does not bite: RGB features carry no signal at all. But on the object, where
reconstruction quality is decided, predictability falls to AUROC ~0.6–0.67. And at the camera level, where #030 and
the swap probe place the rendering damage, W4A4's outputs are **internally consistent even when their cameras are
wrong** — self-consistency does not see the error. Depth spread predicts camera error for full precision as well as
W4A4 (flatter view-groups have worse poses), so it is a property of VGGT pose estimation, not a quantization
signal. **Net for the proposal:** a confidence signal is available where it is least needed (points, background,
edges — which 3DGS absorbs anyway) and absent where it would matter (cameras).

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/cpu_probes.py`

**results/**

- `results/cpu_predict.json`
- `results/cpu_predict_relative.json`

**logs/**

- `logs/cpu_predict.log`
- `logs/cpu_predict_relative.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

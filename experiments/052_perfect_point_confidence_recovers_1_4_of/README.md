# 052 — Perfect point confidence recovers −1.4 % of W3A3's gap: the proposal's mechanism fails on a second arm

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#052** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-12 21:44–22:15 CEST (GPU), 31 min
- **Git commit:** `f0c8e32` + working tree (`confidence_oracle.py` gains `--scene-set`)
- **Command executed:**
  ```
  python code/quantization/confidence_oracle.py --arm w3a3 --scene-set subset \
         --iterations 7000 --keep 0.5 --deadline 00:15
  ```
- **Config:** confidence = the **true** per-point error against full precision, i.e. a cheating predictor
  that no deployed system could have. Keep the best 50 % of initial points, retrain, re-render from the
  arm's own held-out source. The matched control drops an equally sized *random* half with a fixed seed,
  because halving the point count changes 3DGS behaviour on its own — **only (confidence − random) is
  attributable to confidence.**
- **Scenes used:** the same 8 (selection bias as stated in #050).

**The oracle's separation was real on every scene** (median true point error, kept / dropped): apple
0.1090/0.2140, ball 0.0256/0.1147, bowl 0.1604/0.3008, broccoli 0.0559/0.1253, hydrant 0.0488/0.1345,
remote 0.0706/0.1477, teddybear 0.0501/0.0998, toaster 0.0449/0.1386 — a 2–4× ratio throughout. Every
scene: 101,400 points → 50,700.

| condition | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| A — full precision | 9.9708 | 0.2403 | 0.6553 |
| B — w3a3, all points | 9.1880 | 0.2280 | 0.6787 |
| C — w3a3, confidence-kept 50 % | 9.1770 | 0.2163 | 0.6763 |
| R — w3a3, random-kept 50 % (control) | 9.2675 | 0.2204 | 0.6803 |

| paired delta (foreground) | PSNR | p | LPIPS | p |
|---|---:|---:|---:|---:|
| C − B  (confidence vs uniform) | −0.0109 | 0.930 | −0.0024 | 0.692 |
| **C − R  (confidence vs random — THE test)** | **−0.0904** | **0.635** | **−0.0040** | **0.507** |
| R − B  (point-count effect) | +0.0795 | 0.480 | +0.0016 | 0.174 |
| A − B  (the gap to close) | +0.7828 | 0.00629 | −0.0234 | 0.0266 |

**PASS/FAIL:**
- Target: **does perfect point confidence beat its matched random control?** −0.0904 dB, p 0.635.
  **FAIL** — the point estimate is on the wrong side and indistinguishable from random pruning.
- Target: **does it recover any of the +0.7828 dB gap?** C − B = −0.0109 dB, i.e. **−1.4 %**. **FAIL.**
- Target: **was the oracle strong enough for this to be informative?** Kept/dropped error separated 2–4×
  on all 8 scenes. **PASS — the null is not an artefact of a weak signal.**

**Interpretation.** This closes the proposal's mechanism by measurement on a **second, independent** arm:
`w4a4_rtn` recovered −12 % (#030), `w3a3` recovers −1.4 %. Because the oracle uses ground truth it is the
**ceiling** for any confidence predictor, so no learned head at any accuracy can help here — including the
AUROC 0.888 / 0.887 (scene- / category-held-out) pixel predictor already demonstrated in #036. The reason
is given by #050 and #040 together: W3A3's point error (0.101 scene radii) is an order of magnitude inside
the harmless regime, its damage is camera-borne, and 3DGS's appearance optimisation absorbs misplaced points
while camera poses are held fixed throughout training and never corrected.

**Consequence for the proposal.** *"Can a lightweight confidence predictor reduce novel-view rendering
degradation caused by QuantVGGT geometry errors?"* — answered **no** for point-level confidence, now on two
arms, with the ceiling measured rather than argued. The detector half (#036, #045) stands; the
intervention half does not. The only untested reading of the title is **soft weighting** rather than hard
pruning, which would require modifying the GraphDECO trainer to accept per-point weights; given that the
oracle shows no separation from random at all, it is not expected to change the conclusion and has not
been built.

**Threat to validity.** n = 8 with p 0.635 is a *null*, not evidence of harm; the confidence interval
includes zero in both directions. What makes it decisive is that the oracle bounds every predictor from
above, not the precision of this particular estimate.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/confidence_oracle.py`

**results/**

- `results/confidence_oracle_w3a3_it7000.json`

**logs/**

- `logs/confidence_oracle_w3a3.log.gz`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

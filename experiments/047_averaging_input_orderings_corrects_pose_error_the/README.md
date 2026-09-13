# 047 — Averaging input orderings corrects pose error: the corrector #046 was missing

This is entry **#047** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-12; w4a4_rtn 09:57–10:01, w4a4 10:01–10:08, full 10:08–10:11 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/pose_ensemble.py`, `run_pose_ensemble.sh`
- **Command executed:**
  ```
  ./code/quantization/run_pose_ensemble.sh
  #   -> pose_ensemble.py --variant {w4a4_rtn,w4a4,full} --perms 4 --scenes all --save-extrinsics
  ```
- **Config:** inference only. Four orderings per scene (identity + 3 seeded shuffles) of the same six
  518×518 pad-mode images; outputs un-permuted; every ordering carried into ordering 0's frame by Sim(3) on the
  six camera centres. **Ensemble pose** = chordal mean rotation (arithmetic mean projected back to SO(3) by SVD,
  determinant forced positive) and mean camera centre, per camera. Intrinsics are **not** ensembled. Error =
  mean per-camera rotation angle against full precision's saved cameras and against CO3D ground truth. Frozen
  manifest SHA checked at start. Cost 4× inference, ~6.2 s/scene.
- **Scenes used:** all 40.

**Results (median over scenes; "reduction" is the median of the per-scene relative reduction):**

| variant | target | single | ensemble | improved | median reduction | Wilcoxon p |
|---|---|---:|---:|---|---:|---:|
| w4a4_rtn | vs full | 6.0741° | **4.8713°** | 31/40 | +15.8 % | 5.28e-07 |
| w4a4_rtn | vs full, no bowl | 6.0306° | 4.8254° | 30/39 | +16.7 % | 1.05e-06 |
| w4a4_rtn | vs GT | 6.2565° | 4.7472° | 32/40 | +12.3 % | 7.14e-07 |
| w4a4 | vs full | 1.1162° | **0.8557°** | 34/40 | +23.8 % | 7.71e-05 |
| w4a4 | vs GT | 1.7123° | 1.3210° | 32/40 | +14.4 % | 5.08e-05 |
| full (control) | vs full | 0.0083° | 0.4546° | 0/40 | −4687 % | 1.82e-12 |
| full (control) | vs GT | 1.1121° | 1.1193° | 27/40 | +6.4 % | 0.0278 |

**PASS/FAIL:**
- Target: **a corrector that reduces the pose error #045 detects, using only the quantized model.** Pose error
  falls for both quantized arms, on both references, at p ≤ 8e-05. **PASS.**
- Target: **the control behaves sensibly.** For `full` the "vs full" row is an artefact of the definition — its
  reference *is* ordering 0's own output, so any averaging moves away from it; the number is reported rather than
  hidden but carries no meaning. The meaningful control is `full` vs GT: +6.4 % median reduction, p = 0.0278,
  27/40 — far smaller than the quantized arms' 12–24 %. **PASS: the correction is mostly cancelling
  quantization noise, not VGGT's systematic error.**

**Correction to a number quoted in conversation.** The 8-scene smoke test gave a 40.9 % median reduction for
`w4a4_rtn`; at n = 40 it is **15.8 %**. The smoke-test figure was not representative and should not be cited.

**Interpretation.** Quantization noise in VGGT's pose head is substantially **zero-mean with respect to input
ordering**, so averaging four orderings removes part of it — a corrector that needs no ground truth, no
full-precision model, and no training, and that reuses exactly the 4× inference already spent on the #045
detector. Two honest limits. The median `w4a4_rtn` pose error lands at 4.87°, still above the ~2–4° tolerance of
#042, so full rendering recovery should **not** be expected. And intrinsics are not corrected, so the arm's
18.652 % focal error (#031), worth 0.4876 dB on its own (#043), survives untouched. Whether the pose correction
buys anything downstream is being measured now (`ensemble_downstream.py`), and must not be assumed either way.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/pose_ensemble.py`
- `code/run_pose_ensemble.sh`

**results/**

- `results/pose_ensemble_full_p4.json`
- `results/pose_ensemble_w4a4_p4.json`
- `results/pose_ensemble_w4a4_rtn_p4.json`

**logs/**

- `logs/pose_ens_chain.log`
- `logs/pose_ensemble_full.log`
- `logs/pose_ensemble_w4a4.log`
- `logs/pose_ensemble_w4a4_rtn.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

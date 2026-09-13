# 048 — The corrected pose does NOT recover rendering; focal error is a bias, not order-noise

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#048** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-12; pose-only downstream 10:11–11:46, followups 11:46–13:31 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/ensemble_downstream.py`,
  `run_ensemble_downstream.sh`, `run_ensemble_followups.sh`
- **Commands executed:**
  ```
  ./code/quantization/run_ensemble_downstream.sh
  #   -> ensemble_downstream.py --arm w4a4_rtn --perms 4 --iterations 7000 --deadline 00:15
  ./code/quantization/run_ensemble_followups.sh
  #   -> pose_ensemble.py --variant w4a4_rtn --perms 4 --scenes all --save-extrinsics   (adds intrinsics)
  #   -> pose_ensemble.py --variant {w4a4_rtn,w4a4} --perms 2 --scenes all
  #   -> ensemble_downstream.py --arm w4a4_rtn --perms 4 --iterations 7000 --use-intrinsics
  ```
- **Config:** conditions built in full precision's frame, rendered from full's held-out source, 7000 iterations,
  6 input / 9 held-out views, foreground mask, GraphDECO defaults, no depth regularisation — identical to #039/#043
  so the numbers are directly comparable. A = `full`; K = `swapK_w4a4_rtn` (arm pose + arm intrinsics + full
  points); ensK = ensemble pose + arm intrinsics + full points; ensKF = ensemble pose + **ensemble intrinsics** +
  full points. Points are full precision everywhere, so only the camera channel varies.
- **Scenes used:** all 40.

**Downstream (n = 40, foreground):**

| condition | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| A full | 8.9642 | 0.2318 | 0.6604 |
| K uncorrected | 7.9922 | 0.2124 | 0.7269 |
| ensK pose-corrected | 8.0991 | 0.2135 | 0.7195 |
| ensKF pose+focal-corrected | 8.0339 | 0.2120 | 0.7183 |

| contrast | PSNR mean | p | LPIPS mean | p |
|---|---:|---:|---:|---:|
| ensK − K (what pose correction buys) | +0.1070 | 0.3807 | +0.0074 | 0.1458 |
| ensKF − K (pose + focal) | +0.0417 | 0.6726 | +0.0086 | 0.1338 |
| ensKF − ensK (focal on top) | −0.0653 | 0.5498 | +0.0012 | 0.7726 |
| A − ensK (what remains) | +0.8651 | 1.599e-08 | +0.0591 | 2.408e-11 |
| A − K (original gap) | +0.9721 | 5.068e-06 | +0.0665 | 5.783e-09 |

**Focal ensembling (n = 40):** focal error vs full, single median 0.17445 → ensemble 0.17700, improved in 18/40,
Wilcoxon p = 0.485 (w4a4: 0.02220 → 0.02153, 19/40, p = 0.493). **Averaging does not reduce focal error at all.**

**Ensemble size, pose error vs full (median, w4a4_rtn / w4a4):** 2 orderings 5.5751° (+7.3 %, p 0.0016) /
0.8373° (+13.1 %, p 0.00558); 4 orderings 5.1609° (+17.3 %, p 1.67e-05) / (#047: 0.8557°, +23.8 %).

**PASS/FAIL:**
- Target: **the pose correction of #047 recovers a measurable share of the 0.9721 dB rendering gap.** It recovers
  +0.1070 dB, 11 %, p = 0.3807. **FAIL — not significant.** With focal correction added, +0.0417 dB, p = 0.6726.
  **FAIL.**
- Target: **ensembling corrects intrinsics as it corrects pose.** p = 0.485, 18/40. **FAIL.**
- Target: **2 orderings retain most of the 4-ordering gain** (the efficiency question). They retain roughly half
  (7.3 % vs 17.3 % on rtn; 13.1 % vs 23.8 % on w4a4). **PARTIAL.**

**Reproducibility bug found, and its effect.** The ensemble seeded permutations with `abs(hash(name))`. Python
salts string hashing per interpreter, so **every process drew a different permutation set** and the ensemble was
not reproducible: the same command gave a 4.8713° median in #047 and 5.1609° here. Both runs show a significant
improvement and the conclusion is unchanged, but the exact reduction has run-to-run variance of a few tenths of a
degree and #047's 15.8 % should be read alongside this run's 17.3 %. Seeding is now `zlib.crc32(name)`, stable
across processes; all later runs are reproducible. Numbers produced before this fix are marked by the commands
above.

**Interpretation.** The detector works (#045) and the corrector reduces pose error (#047), but the correction is
not enough to matter downstream: it moves `w4a4_rtn` from ~6.07° to ~4.9–5.2°, and #042 puts the rendering-damage
threshold at ~2–4°, so the corrected prior is still inside the damaged regime. The focal result explains itself —
order-averaging can only cancel *order-dependent* noise, and quantization's focal error is a systematic bias that
every ordering shares, so averaging leaves it untouched. This is the honest state of the method: **a working
detector, a partial corrector, and no downstream gain at this damage level.** Whether more orderings cross the
threshold is being measured now (2/4/8/16, seeded); if the gain saturates above 4°, the ceiling is real.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/ensemble_downstream.py`
- `code/pose_ensemble.py`
- `code/run_ensemble_downstream.sh`
- `code/run_ensemble_followups.sh`

**results/**

- `results/ensemble_downstream_posefocal_w4a4_rtn_it7000.json`
- `results/ensemble_downstream_w4a4_rtn_it7000.json`

**logs/**

- `logs/ens_dn_chain.log`
- `logs/ens_followups.log`
- `logs/ensemble_downstream.log.gz`
- `logs/ensemble_downstream_posefocal.log.gz`
- `logs/pose_ensemble_w4a4_rtn_p4b.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

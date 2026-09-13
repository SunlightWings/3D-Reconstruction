# 049 — Ensemble size 2/4/8/16: the correction saturates above the damage threshold

This is entry **#049** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-12 13:39–14:58 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree (`pose_ensemble.py` with crc32 seeding, `run_ensemble_scaling.sh`)
- **Command executed:**
  ```
  ./code/quantization/run_ensemble_scaling.sh
  #   -> pose_ensemble.py --variant {w4a4_rtn,w4a4} --perms {2,4,8,16} --scenes all --save-extrinsics
  ```
- **Config:** inference only, as #047, with permutations seeded by `zlib.crc32(scene)` so every run is
  reproducible (the `hash()`-based seeding of #045/#047/#048 was not — see #048). Pose error is the mean
  per-camera rotation angle against full precision's saved cameras after Sim(3) on the six camera centres.
- **Scenes used:** all 40, every configuration.

**Pose error vs full precision (median over 40 scenes):**

| arm | orderings | single | ensemble | reduction | improved | Wilcoxon p | scenes < 4° |
|---|---:|---:|---:|---:|---|---:|---|
| w4a4_rtn | 2 | 6.0741° | 5.4213° | 14.1 % | 30/40 | 0.0061 | 9/40 |
| w4a4_rtn | 4 | 6.0741° | 5.1255° | 26.2 % | 31/40 | 0.000538 | 16/40 |
| w4a4_rtn | 8 | 6.0741° | 4.6820° | 31.4 % | 35/40 | 2.27e-05 | 14/40 |
| w4a4_rtn | 16 | 6.0741° | **4.3743°** | 29.5 % | 35/40 | 1.16e-06 | 17/40 |
| w4a4 | 2 | 1.1162° | 0.8977° | 10.9 % | 25/40 | 0.0381 | 39/40 |
| w4a4 | 4 | 1.1162° | 0.6921° | 17.6 % | 30/40 | 0.00105 | 39/40 |
| w4a4 | 8 | 1.1162° | 0.7961° | 23.1 % | 31/40 | 8.25e-05 | 39/40 |
| w4a4 | 16 | 1.1162° | 0.7283° | 28.4 % | 31/40 | 0.00017 | 39/40 |

Marginal gain per doubling on `w4a4_rtn`: 5.4213 → 5.1255 → 4.6820 → 4.3743°, i.e. +0.296, +0.444, +0.308°.
The median relative reduction does not increase monotonically (26.2 → 31.4 → 29.5 %), and `w4a4`'s does not
either (17.6 → 23.1 → 28.4 % with a dip in the median error at 8), which bounds how precisely these can be read.

**PASS/FAIL:**
- Target: **does a larger ensemble cross the ~2–4° rendering-damage threshold of #042?** At 16 orderings
  `w4a4_rtn` reaches 4.3743°, still above 4°, with 17/40 scenes below it. **FAIL — the threshold is not
  crossed even at 4× the cost tested downstream in #048.**
- Target: **does the gain keep scaling?** From 4 to 16 orderings — a 4× cost increase — the median falls by
  0.75° (5.1255 → 4.3743). **Saturating; no ensemble size in reach fixes this arm.**
- Target (efficiency): **is 2 enough?** 2 orderings give 14.1 % of the 29.5 % available at 16 on rtn, and
  10.9 % vs 28.4 % on w4a4. **No — roughly half the gain at a quarter the extra cost.**

**Interpretation.** Order-ensembling reduces quantization pose error reliably and reproducibly, on both arms,
at every size tested — but it saturates around a 30 % reduction, which leaves `w4a4_rtn` at 4.37°, inside the
regime where rendering is damaged. That closes the loop opened in #045: the signal is real, the correction is
real, and neither is large enough to recover rendering at this damage level, which is exactly what #048 measured
downstream. The mechanism is consistent throughout: only the order-*dependent* part of quantization error can be
averaged away, and for this arm the order-independent remainder is itself above the threshold. For W4A4 the
correction takes 1.12° to 0.69–0.80°, both already inside tolerance, so there is nothing downstream to win there
either. **No further downstream run is justified for `w4a4_rtn`;** the case where this machinery could matter is
an arm whose single-run error sits just *above* the threshold and whose order-dependent share is large — W3A3 is
the candidate, and it is not yet available.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/pose_ensemble.py`
- `code/run_ensemble_scaling.sh`

**results/**

- `results/pose_ensemble_w4a4_p16.json`
- `results/pose_ensemble_w4a4_p2.json`
- `results/pose_ensemble_w4a4_p8.json`
- `results/pose_ensemble_w4a4_rtn_p16.json`
- `results/pose_ensemble_w4a4_rtn_p2.json`
- `results/pose_ensemble_w4a4_rtn_p8.json`

**logs/**

- `logs/ens_scaling.log`
- `logs/pose_ensemble_w4a4_p16_seeded.log`
- `logs/pose_ensemble_w4a4_p2.log`
- `logs/pose_ensemble_w4a4_p2_seeded.log`
- `logs/pose_ensemble_w4a4_p4_seeded.log`
- `logs/pose_ensemble_w4a4_p8_seeded.log`
- `logs/pose_ensemble_w4a4_rtn_p16_seeded.log`
- `logs/pose_ensemble_w4a4_rtn_p2.log`
- `logs/pose_ensemble_w4a4_rtn_p2_seeded.log`
- `logs/pose_ensemble_w4a4_rtn_p4_seeded.log`
- `logs/pose_ensemble_w4a4_rtn_p8_seeded.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

# 023 — Sim(3) with rotation constraints, and the similarity-model floor

This is entry **#023** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-08 10:50 CEST
- **Git commit:** `231d1ea`
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sim3_rotation.py --all
  ```
  plus an inline `python3 -c` probe measuring the in-sample residual of the best similarity fit.

**Config:** No training, no rendering. Frozen prediction NPZs and GT annotations.

| Field | Value |
|---|---|
| Init source | `full` and `w4a4` predicted cameras and points |
| Iterations | n/a |
| Densify | n/a |
| Input views | 6 input cameras (fit 5, predict the 6th for LOO); point stride 8 for on-silhouette |
| Mask type | `foreground` |
| Background handling | n/a |
| Depth regularisation | n/a |

**Scenes used:** all 40 scenes.

**Estimators:** `centres` = the current `dv.umeyama` on 6 camera centres; `rotation` = A from chordal rotation
averaging of the 6 per-camera estimates `A_i = R_pred_i^T R_gt_i`, with s and b then in closed form from the centres;
`joint` = SO(3) re-projection of an equal blend of the rotation-averaged and centres-only A.

**Results (median LOO error in scene radii; on-silhouette is the mean over scenes):**

| variant | estimator | median LOO | vs centres | on-silhouette |
|---|---|---:|---:|---:|
| full | centres | 0.0813 |  | 8.70% |
| full | rotation | 0.0776 | -4.7% | 8.58% |
| full | joint | 0.0772 | -5.1% | 8.62% |
| w4a4 | centres | 0.1380 |  | 8.80% |
| w4a4 | rotation | 0.1294 | -6.2% | 8.65% |
| w4a4 | joint | 0.1243 | -10.0% | 8.70% |

The `centres` rows reproduce entry #018 exactly (0.0813 full, 0.1380 w4a4), confirming only the estimator changed.

**The reason the gain is small — the similarity model itself is the floor.** Fitting a similarity on all 6 cameras and
scoring the residual on those same 6 (best case, no extrapolation at all) still leaves:

| quantity | value |
|---|---:|
| in-sample residual of the best similarity fit, median over 40 scenes | **0.0485** |
| in-sample residual, mean | 0.0666 |
| leave-one-out error, `centres` (#018) | 0.0813 |
| A5 target | 0.0200 |

**PASS/FAIL:** Target: **adding rotation constraints should materially reduce the 0.081 / 0.138 residual.** Measured:
full 0.0813 -> **0.0772** (joint), w4a4 0.1380 -> **0.1243** (joint) — improvements of 5.1% and 10.0%.
On-silhouette is unchanged to slightly worse (8.70% -> 8.62%). **FAIL — the reduction is real but not material,
and the residual remains 4-6x the target.**

**Interpretation:** Rules out a better estimator of a similarity as the fix. The in-sample floor of 0.0485 scene radii
is already 2.4x the A5 target, and it is reached with the same six cameras used for fitting — so it is not an
extrapolation or a weak-constraint problem at all: **the GT-to-predicted map is not a similarity.** VGGT's predicted
geometry is non-rigidly distorted relative to GT, and no Sim(3), however well estimated, can absorb that. Rotation
averaging recovers only the extrapolation slice (0.0328 of the 0.0813), which is why it buys ~5%.

**What this implies for the harness:** closing the remaining alignment error requires either a non-rigid alignment
(per-camera or deformable), which the pipeline currently forbids by design (`no_per_camera_correction: True` in every
metrics.json), or accepting the alignment error as a floor and reporting downstream numbers with it stated. The
`joint` estimator is a free ~5-10% improvement and is worth adopting, but it does not change any gate verdict.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/sim3_rotation.py`

**results/**

- `results/sim3_rotation.json`

**logs/**

- `logs/sim3_rot40.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

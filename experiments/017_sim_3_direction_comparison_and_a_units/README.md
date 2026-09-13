# 017 — Sim(3) direction comparison, and a units bug in A5 (analysis, no training)

This is entry **#017** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-08 09:20 CEST
- **Git commit:** `1088b5a`
- **Command executed:**
  ```
  python3 code/downstream_3dgs/oracle_sweep/sim3_direction.py
  ```
  plus two inline `python3 -c` probes reproducing A5's normalisation (both reproduced by
  `loo_error()` in that module and by the table below)

**Config:** No training, no rendering. Operates on the frozen prediction NPZs and GT annotations.

| Field | Value |
|---|---|
| Init source | `full` and `w4a4` predicted geometry (frozen NPZs) |
| Iterations | n/a |
| Densify | n/a |
| Input views | 6 input frames; point stride 8 for the on-silhouette test |
| Mask type | `foreground` (object silhouette) |
| Background handling | n/a |
| Depth regularisation | n/a |

**Scenes used:** the 8-scene GPU subset.

**Result 1 — Sim(3) fitting direction makes no difference to A4.**

Comparing the current `umeyama(C_gt, C_pred)`-then-invert form against the direct
`umeyama(C_pred, C_gt)`-applied-forward form:

| variant | on-silhouette (inverse) | on-silhouette (direct) |
|---|---:|---:|
| full | 9.50% | 9.51% |
| w4a4 | 9.39% | 9.40% |

Per-scene the two agree to within 0.1 percentage point on every scene.

**Result 2 — A5 normalises a predicted-frame residual by a GT-frame radius.**

`test_a5_loo_sim3` computes `||pred_k - C_pred[k]|| / radius`, where the residual is in the PREDICTED
world frame but `radius` is `scene_radius` from per_scene.csv, i.e. the 90th-percentile radius of the CO3D
**GT** point cloud. The two frames differ by the Sim(3) scale, measured here at **0.0716** on average
(the predicted world is ~14x smaller than the GT world), so the reported figure is scaled by that factor.

Recomputing the same leave-one-out quantity consistently -- mapping the predicted camera back into the GT
frame and normalising by a GT-frame radius:

| scene | A5 as coded | consistent (GT frame) | ratio |
|---|---:|---:|---:|
| apple/110_13051_23361 | 0.0107 | 0.1441 | 13.4x |
| ball/123_14363_28981 | 0.0065 | 0.0760 | 11.7x |
| bowl/70_5792_13401 | 0.0341 | 0.4549 | 13.3x |
| broccoli/412_56288_108844 | 0.0052 | 0.0826 | 15.9x |
| hydrant/167_18184_34441 | 0.0023 | 0.0449 | 19.2x |
| remote/350_36761_68623 | 0.0042 | 0.0646 | 15.4x |
| teddybear/187_20215_38541 | 0.0044 | 0.0588 | 13.3x |
| toaster/372_41229_82130 | 0.0150 | 0.1821 | 12.2x |
| **mean** | **0.0103** | **0.1385** | **~13.4x** |

The conclusion does not depend on which GT-frame radius is chosen:

| normalisation | mean LOO error | vs 0.02 target |
|---|---:|---|
| GT point-cloud radius (what per_scene.csv holds) | 0.1385 | **FAIL** |
| GT camera-spread radius | 0.0473 | **FAIL** |

**PASS/FAIL:**
- Target: **Sim(3) direction should be chosen by whichever gives better A4/A5.** On A4 the two directions are
  indistinguishable (9.50% vs 9.51%). On A5 they are not comparable as written, because the as-coded metric is
  frame-inconsistent. **Keep the existing `inverse` form** — there is no measured reason to change it, and changing
  it would churn the alignment used by every existing result for no gain.
- Target: **A5 median LOO error well under 0.02 scene radii.** As coded: 0.0103, **PASS**. Measured consistently:
  **0.1385** (or 0.0473 against camera spread), **FAIL** by 2.4x to 7x.

**Interpretation:** Rules out the Sim(3) fitting direction as an explanation for A4's low on-silhouette, and rules IN
a units bug in A5 that converts a real FAIL into a reported PASS. This resolves the A4-vs-A5 tension flagged in the
previous DECISION STATE: A4 (frame-consistent) said the camera alignment was borderline, A5 (frame-inconsistent) said
it was comfortable, and A5 was wrong. **The camera alignment between the predicted and GT frames is genuinely poor,
which is a live candidate for the held-out generalisation failure that entry #015 localised to viewpoint change.**

**Superseded:** the A5 rows in entries #001 and its DECISION STATE ("A5 PASS", 0.0066 / 0.0106) are frame-inconsistent
and should not be cited. This entry supersedes them.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/sim3_direction.py`

**results/**

- `results/sim3_direction.json`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

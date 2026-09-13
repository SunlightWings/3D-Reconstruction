# 016 — A4 root cause: is the GT camera path self-consistent? (analysis, no training)

This is entry **#016** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-08 09:05 CEST
- **Git commit:** `1088b5a`
- **Command executed:**
  ```
  python3 code/downstream_3dgs/oracle_sweep/check_gt_projection.py
  ```

**Config:** No training, no rendering, no VGGT. Pure projection arithmetic on GT data.

| Field | Value |
|---|---|
| Init source | GT depth unprojection only (`gate_a._oracle_decode_frame`) |
| Iterations | n/a |
| Densify | n/a |
| Input views | 6 input frames per scene |
| Mask type | `foreground` (the CO3D object silhouette) |
| Background handling | n/a |
| Depth regularisation | n/a |

**Scenes used:** the 8-scene GPU subset.

**Method:** points unprojected from frame f's own GT depth using frame f's own GT camera are projected straight back
through that same camera (`self`), and into the other input frames' GT cameras (`cross-view`). The self case is a round
trip with no alignment, no prediction and no other view involved, so its answer is known in advance: it must be ~100%.

**Results (fraction of projected points landing on the object silhouette):**

| scene | self | inside image | cross-view |
|---|---:|---:|---:|
| apple/110_13051_23361 | 100.0% | 100.0% | 92.7% |
| ball/123_14363_28981 | 100.0% | 100.0% | 92.7% |
| bowl/70_5792_13401 | 100.0% | 100.0% | 97.6% |
| broccoli/412_56288_108844 | 100.0% | 100.0% | 99.3% |
| hydrant/167_18184_34441 | 100.0% | 100.0% | 95.9% |
| remote/350_36761_68623 | 100.0% | 100.0% | 99.3% |
| teddybear/187_20215_38541 | 100.0% | 100.0% | 96.2% |
| toaster/372_41229_82130 | 100.0% | 100.0% | 97.2% |
| **mean** | **100.00%** | 100.00% | **96.36%** |

**PASS/FAIL:** Target: **>= 90% self on-silhouette** (projecting GT points through their own GT camera).
Measured **100.00%**, with 100.00% of points landing inside the image. **PASS.**

**Interpretation:** Rules out the GT camera path as the cause of A4's 9-29% on-silhouette -- the pad affine applied to
the intrinsics, the CAM_FLIP sign convention, and the NDC->pixel conversion in `co3d_to_opencv_camera` are all
self-consistent, and the GT extrinsics are mutually consistent across views at 96.4% cross-view. The
remaining candidates for A4 are therefore VGGT's predicted geometry and the Sim(3) alignment between the predicted and
GT frames, which is exactly what task item 5 tests. The 3.6% cross-view shortfall is expected from genuine
occlusion (points visible in one view are hidden in another), not from a coordinate bug.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/check_gt_projection.py`

**results/**

- `results/gt_projection_check.json`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

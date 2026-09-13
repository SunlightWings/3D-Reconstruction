# 015 — A6 oracle self-fit (new diagnostic, no training)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#015** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-08 08:47 CEST
- **Git commit:** `f8b1f7d` + working-tree changes adding A6, the foreground mask, and the A3 gate redefinition
- **Command executed:**
  ```
  python3 -c "import sys; sys.path.insert(0,'code/downstream_3dgs/diagnostics'); import gate_a; gate_a.test_a6_oracle_selffit()"
  ```
  (registered as test `A6` in `gate_a.TESTS`, so it also runs via `run_all_diagnostics.py`)

**Config:** No training. Renders the EXISTING oracle models at their own 6 training views.

| Field | Value |
|---|---|
| Init source | `oracle` (GT cameras + GT-unprojected foreground points) |
| Iterations | 30000 (models as trained by A3) |
| Densify | default (as trained by A3) |
| Input views | 6 input; rendered at those same 6 training cameras |
| Mask type | both `foreground` (primary) and `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset.

**Results (self-fit PSNR at the model's own training views):**

| scene | content | foreground |
|---|---:|---:|
| apple/110_13051_23361 | 50.17 | 50.34 |
| ball/123_14363_28981 | 47.05 | 48.74 |
| bowl/70_5792_13401 | 47.81 | 46.63 |
| broccoli/412_56288_108844 | 49.60 | 46.28 |
| hydrant/167_18184_34441 | 46.95 | 48.52 |
| remote/350_36761_68623 | 48.33 | 47.73 |
| teddybear/187_20215_38541 | 46.93 | 46.07 |
| toaster/372_41229_82130 | 45.58 | 48.57 |
| **mean** | **47.801** | **47.859** |
| min (foreground) | | 46.07 |

**PASS/FAIL:** Target: **>= 30 dB self-fit.** Measured **47.859 dB** mean on the foreground
mask (47.801 dB content), minimum across scenes 46.07 dB, scenes below target: 0/8. **PASS.**

**Interpretation:** Rules out a renderer/adapter defect and a capacity limit in the oracle arm -- the oracle reproduces
its own training views at ~48 dB, so the ~14.8 dB held-out foreground score is a genuine failure to generalise to new
viewpoints, not a broken pipeline.

**Important asymmetry, stated so this gate is not over-read:** A6 is circular with respect to camera correctness. The
model was *trained* on these exact cameras, so it will fit them well whether or not they are geometrically right; a
systematic camera error would simply be baked into the Gaussians. A6 FAILING would have indicted the GT camera path,
but A6 PASSING does not exonerate it. The camera question stays open and belongs to A4/A5 (task items 5 and 6).

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/common.py`
- `code/gate_a.py`
- `code/run_all_diagnostics.py`

**results/**

- `results/a6_oracle_selffit.json`

**logs/**

- `logs/rerun_gates.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

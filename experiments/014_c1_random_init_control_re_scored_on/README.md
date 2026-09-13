# 014 — C1 random-init control, re-scored on the foreground mask (analysis, no training)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#014** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 23:40 CEST
- **Git commit:** `f8b1f7d` (branch `disagreement-dataset-validation`, clean tree at time of run)
- **Command executed:**
  ```
  python3 code/downstream_3dgs/oracle_sweep/reeval_c1.py
  ```

**Config:** No training and no rendering. Re-scores the C1 renders that already exist on disk
(`DIAG_HEAVY_ROOT/<scene>/heldout/{c1_random,c1_w4a4,c1_full,oracle}/renders`). All training config is
whatever C1 and A3 originally used (30000 iterations, default densification, 6 input views, GT-frame cameras);
the only thing varied here is the evaluation mask.

| Field | Value |
|---|---|
| Init source | all four arms compared: `random`, `w4a4`, `full`, `oracle` |
| Iterations | 30000 (as originally trained) |
| Densify | default (as originally trained) |
| Input views | 6 input, 9 held-out |
| Mask type | **`content` vs `foreground`, compared** |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset.

**Results:**

| arm | content PSNR | content SSIM | foreground PSNR | foreground SSIM |
|---|---:|---:|---:|---:|
| random | 13.481 | 0.3563 | 13.459 | 0.3107 |
| w4a4 | 13.873 | 0.3803 | 14.814 | 0.3389 |
| full | 13.962 | 0.3764 | **14.916** | 0.3354 |
| oracle | **14.061** | 0.3483 | 14.828 | 0.3213 |

| quantity | content mask | foreground mask |
|---|---:|---:|
| ordering (best first) | oracle > full > w4a4 > random | full > oracle > w4a4 > random |
| full - random | +0.481 dB | **+1.457 dB** |
| scenes where random is within 0.5 dB of full | 4 / 8 | **1 / 8** |

**PASS/FAIL:** Target (C1's own, as written in DIAGNOSTIC_RESULTS.md): **ordering random < quant <= full, with a
visible margin**; the stated failure mode is "if random ~= full, initialization is inert in this harness."

- On the **content** mask: random 13.481 < w4a4 13.873 < full 13.962, margin +0.481 dB, init inert on 4/8 scenes. **FAIL** (this was the published verdict).
- On the **foreground** mask: random 13.459 < w4a4 14.814 <= full 14.916, margin **+1.457 dB**, init inert on **1/8** scenes. **PASS.**

**Interpretation:** Rules out "initialization is inert in this harness" — that conclusion was an artifact of scoring on a
mask that is ~85% background, where no initialization of any kind can help (cf. entries #002 and #011); on the region
where the init prior actually places geometry, the geometry prior matters and the C1 ordering holds with a 3x larger
margin.

**Flag for follow-up, not resolved here:** on the foreground mask the `oracle` arm (14.828 dB) sits *below* `full`
(14.916 dB), i.e. GT geometry is beaten by VGGT geometry. The two arms share GT-frame cameras and differ only in the
initial point cloud, and the oracle cloud is far sparser (2,566-6,356 points from GT depth valid only on the object)
than VGGT's dense prediction. That is a plausible explanation but it is **not** measured, and it should be, because a
GT-geometry control that loses to the method it is meant to bound is a gate worth distrusting.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/reeval_c1.py`

**results/**

- `results/c1_both_masks.json`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

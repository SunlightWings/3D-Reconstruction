# 021 — Pose vs geometry decomposition of the oracle-minus-VGGT gap

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#021** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-08 10:25 CEST
- **Git commit:** `231d1ea`
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/pose_vs_geometry.py
  ```

**Config:** Render-only, reusing the trained oracle models. Held-out camera centres displaced by a random-direction
translation of magnitude `delta * scene_radius`, orientation unchanged; seed 20260908.

| Field | Value |
|---|---|
| Init source | `oracle` (GT cameras + GT foreground points, no Sim(3) anywhere) |
| Iterations | 30000 (models as trained by A3) |
| Densify | default (as trained) |
| Input views | 6 input, 9 held-out |
| Mask type | `foreground` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset.

**Why the literally-requested arm was not built:** a Sim(3) applied to both the points and the cameras leaves every
rendered image unchanged, and the w4a4 geometry's only link to the GT frame *is* its Sim(3). "Render the w4a4 cloud
through GT cameras directly" is therefore pixel-identical to "render it through its own Sim(3)" and can separate
nothing. (The full-arm-Sim(3) swap is well-posed -- the two predicted frames agree to a median 0.0243 of scene radius,
scale ratio ~1.00, rotation < 1.5 deg on 7/8 scenes -- but it would only compare two nearly identical estimates.)
Instead the causation is run the other way: measured alignment error is injected into the one arm that has none.

**Results (foreground PSNR vs injected camera error):**

| injected error (scene radii) | provenance | mean fg PSNR | cost vs unperturbed |
|---|---|---:|---:|
| 0.0000 | oracle as it stands | 14.828 | +0.000 |
| 0.0200 | A5's pass threshold | 13.648 | +1.180 |
| 0.0813 | A5 measured, full arm (#018) | 12.613 | +2.215 |
| 0.1380 | A5 measured, w4a4 arm (#018) | 12.045 | +2.783 |

The delta=0 row reproduces the oracle's 14.828 dB from entry #004 exactly, confirming the render path is unchanged.

**Attribution:**

| quantity | value |
|---|---:|
| oracle, no alignment error | 14.828 dB |
| full arm, 40 scenes, foreground (#020) | 9.028 dB |
| total oracle-minus-full gap | 5.800 dB |
| cost of the measured full-arm alignment error | 2.215 dB |
| **share attributable to alignment** | **38.2%** |
| **share attributable to geometry and everything else** | **61.8%** (3.585 dB) |

**PASS/FAIL:** Target: **decide whether a per-pixel confidence predictor is the right instrument**, by attributing the
gap. Measured split: alignment **38.2%**, geometry-and-other **61.8%**. **Geometry is the majority share, so a
per-pixel geometry-confidence predictor addresses the larger part -- but it cannot touch the 38.2% that is pose, and
that share is not small.**

**Interpretation:** Rules out treating the gap as purely a geometry problem: roughly two fifths of it is camera
alignment, which no per-pixel geometry confidence can fix, so a confidence predictor should be paired with (or preceded
by) the alignment fix in task item 3. It also shows the exchange rate is steep and saturating -- the first 0.02 radii of
camera error costs 1.180 dB while going from 0.0813 to 0.1380 costs only a further 0.568 dB -- so even a partial
alignment improvement should pay off, and the w4a4 arm's larger misalignment explains little of the Full-vs-W4A4
difference on its own.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/pose_vs_geometry.py`

**results/**

- `results/pose_vs_geom/d0.0000/apple/110_13051_23361/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0000/apple/110_13051_23361/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0000/apple/110_13051_23361/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0000/apple/110_13051_23361/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0000/apple/110_13051_23361/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0000/ball/123_14363_28981/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0000/ball/123_14363_28981/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0000/ball/123_14363_28981/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0000/ball/123_14363_28981/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0000/ball/123_14363_28981/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0000/bowl/70_5792_13401/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0000/bowl/70_5792_13401/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0000/bowl/70_5792_13401/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0000/bowl/70_5792_13401/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0000/bowl/70_5792_13401/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0000/broccoli/412_56288_108844/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0000/broccoli/412_56288_108844/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0000/broccoli/412_56288_108844/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0000/broccoli/412_56288_108844/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0000/broccoli/412_56288_108844/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0000/hydrant/167_18184_34441/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0000/hydrant/167_18184_34441/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0000/hydrant/167_18184_34441/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0000/hydrant/167_18184_34441/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0000/hydrant/167_18184_34441/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0000/remote/350_36761_68623/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0000/remote/350_36761_68623/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0000/remote/350_36761_68623/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0000/remote/350_36761_68623/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0000/remote/350_36761_68623/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0000/teddybear/187_20215_38541/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0000/teddybear/187_20215_38541/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0000/teddybear/187_20215_38541/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0000/teddybear/187_20215_38541/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0000/teddybear/187_20215_38541/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0000/toaster/372_41229_82130/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0000/toaster/372_41229_82130/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0000/toaster/372_41229_82130/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0000/toaster/372_41229_82130/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0000/toaster/372_41229_82130/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0200/apple/110_13051_23361/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0200/apple/110_13051_23361/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0200/apple/110_13051_23361/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0200/apple/110_13051_23361/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0200/apple/110_13051_23361/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0200/ball/123_14363_28981/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0200/ball/123_14363_28981/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0200/ball/123_14363_28981/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0200/ball/123_14363_28981/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0200/ball/123_14363_28981/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0200/bowl/70_5792_13401/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0200/bowl/70_5792_13401/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0200/bowl/70_5792_13401/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0200/bowl/70_5792_13401/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0200/bowl/70_5792_13401/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0200/broccoli/412_56288_108844/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0200/broccoli/412_56288_108844/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0200/broccoli/412_56288_108844/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0200/broccoli/412_56288_108844/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0200/broccoli/412_56288_108844/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0200/hydrant/167_18184_34441/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0200/hydrant/167_18184_34441/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0200/hydrant/167_18184_34441/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0200/hydrant/167_18184_34441/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0200/hydrant/167_18184_34441/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0200/remote/350_36761_68623/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0200/remote/350_36761_68623/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0200/remote/350_36761_68623/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0200/remote/350_36761_68623/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0200/remote/350_36761_68623/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0200/teddybear/187_20215_38541/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0200/teddybear/187_20215_38541/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0200/teddybear/187_20215_38541/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0200/teddybear/187_20215_38541/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0200/teddybear/187_20215_38541/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0200/toaster/372_41229_82130/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0200/toaster/372_41229_82130/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0200/toaster/372_41229_82130/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0200/toaster/372_41229_82130/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0200/toaster/372_41229_82130/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0813/apple/110_13051_23361/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0813/apple/110_13051_23361/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0813/apple/110_13051_23361/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0813/apple/110_13051_23361/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0813/apple/110_13051_23361/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0813/ball/123_14363_28981/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0813/ball/123_14363_28981/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0813/ball/123_14363_28981/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0813/ball/123_14363_28981/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0813/ball/123_14363_28981/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0813/bowl/70_5792_13401/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0813/bowl/70_5792_13401/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0813/bowl/70_5792_13401/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0813/bowl/70_5792_13401/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0813/bowl/70_5792_13401/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0813/broccoli/412_56288_108844/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0813/broccoli/412_56288_108844/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0813/broccoli/412_56288_108844/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0813/broccoli/412_56288_108844/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0813/broccoli/412_56288_108844/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0813/hydrant/167_18184_34441/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0813/hydrant/167_18184_34441/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0813/hydrant/167_18184_34441/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0813/hydrant/167_18184_34441/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0813/hydrant/167_18184_34441/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0813/remote/350_36761_68623/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0813/remote/350_36761_68623/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0813/remote/350_36761_68623/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0813/remote/350_36761_68623/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0813/remote/350_36761_68623/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0813/teddybear/187_20215_38541/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0813/teddybear/187_20215_38541/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0813/teddybear/187_20215_38541/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0813/teddybear/187_20215_38541/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0813/teddybear/187_20215_38541/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.0813/toaster/372_41229_82130/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.0813/toaster/372_41229_82130/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.0813/toaster/372_41229_82130/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.0813/toaster/372_41229_82130/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.0813/toaster/372_41229_82130/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.1380/apple/110_13051_23361/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.1380/apple/110_13051_23361/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.1380/apple/110_13051_23361/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.1380/apple/110_13051_23361/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.1380/apple/110_13051_23361/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.1380/ball/123_14363_28981/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.1380/ball/123_14363_28981/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.1380/ball/123_14363_28981/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.1380/ball/123_14363_28981/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.1380/ball/123_14363_28981/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.1380/bowl/70_5792_13401/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.1380/bowl/70_5792_13401/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.1380/bowl/70_5792_13401/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.1380/bowl/70_5792_13401/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.1380/bowl/70_5792_13401/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.1380/broccoli/412_56288_108844/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.1380/broccoli/412_56288_108844/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.1380/broccoli/412_56288_108844/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.1380/broccoli/412_56288_108844/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.1380/broccoli/412_56288_108844/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.1380/hydrant/167_18184_34441/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.1380/hydrant/167_18184_34441/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.1380/hydrant/167_18184_34441/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.1380/hydrant/167_18184_34441/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.1380/hydrant/167_18184_34441/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.1380/remote/350_36761_68623/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.1380/remote/350_36761_68623/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.1380/remote/350_36761_68623/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.1380/remote/350_36761_68623/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.1380/remote/350_36761_68623/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.1380/teddybear/187_20215_38541/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.1380/teddybear/187_20215_38541/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.1380/teddybear/187_20215_38541/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.1380/teddybear/187_20215_38541/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.1380/teddybear/187_20215_38541/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geom/d0.1380/toaster/372_41229_82130/heldout/PREPARED.ok`
- `results/pose_vs_geom/d0.1380/toaster/372_41229_82130/heldout/sparse/0/cameras.txt`
- `results/pose_vs_geom/d0.1380/toaster/372_41229_82130/heldout/sparse/0/images.txt`
- `results/pose_vs_geom/d0.1380/toaster/372_41229_82130/heldout/sparse/0/points3D.ply`
- `results/pose_vs_geom/d0.1380/toaster/372_41229_82130/heldout/sparse/0/points3D.txt`
- `results/pose_vs_geometry.json`

**logs/**

- `logs/pose_vs_geom.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

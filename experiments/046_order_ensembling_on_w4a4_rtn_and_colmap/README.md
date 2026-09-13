# 046 — Order-ensembling on `w4a4_rtn`, and COLMAP refinement of a damaged prior: BLOCKED by SfM failure

This is entry **#046** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-12; `w4a4_rtn` ensembling 09:30–09:34; refinement final run finished before 09:38 CEST
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/pose_refine.py`
- **Commands executed:**
  ```
  /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/pose_uncertainty.py \
      --variant w4a4_rtn --perms 4 --scenes all
  nice -n 15 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/pose_refine.py \
      refine --arm w4a4_rtn --scenes 40
  ```

**Part A — order-ensembling extends to the heavily damaged arm (n = 40).** Spread median 6.6314°
(range 3.1576–130.6178), against w4a4's 1.1389° and full's 0.6084° (#045). Spread vs pose error against full:
ρ = +0.678, p = 1.56e-06 (no bowl +0.652, p 6.78e-06); vs GT error ρ = +0.637, p 9.99e-06. **PASS:** the signal
holds at 14.66° of damage, though less tightly than at W4A4 (ρ 0.880).

**Part B — can classical SfM refine the damaged cameras?**

- **Config:** COLMAP via pycolmap 4.2.0 on the six 518×518 input images per scene: SIFT extraction and exhaustive
  matching on **CPU** (`device=pycolmap.Device.cpu`, so as not to contend with GPU jobs), `PER_IMAGE` PINHOLE
  cameras with intrinsics **fixed to the quantized model's prediction**, then `incremental_mapping`. No ground
  truth and no full-precision model are used — only what is available at deployment. Pose error is scored on
  exactly the subset of cameras COLMAP registered, against the same subset of full/GT cameras, by the usual
  Sim(3)-on-centres construction. A scene counts as usable only if ≥ 4 of 6 images register.
- **Scenes used:** all 40.

| registered images | 0 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| scenes | **28** | 3 | 2 | 1 | 1 | 5 |

Usable: **7/40**. On those 7: pose error vs full, arm median 5.878° → refined **19.757°**, improved in **1/7**,
Wilcoxon p = 0.0781; vs GT, 4.842° → 19.332°, improved in 1/7, under 4° goes from 2 scenes to 0. Full precision's
own error vs GT on the same scenes: median 1.223°. Individual runs show both outcomes — one scene improved
14.87° → 3.30°, another degraded 5.38° → 128.66°.

**PASS/FAIL:** Target: **classical refinement pulls the arm's 14.66° pose error under the ~4° tolerance of #042.**
COLMAP fails to reconstruct at all on 28/40 scenes, is usable on 7, and on those it makes pose error worse in 6 of
7. **FAIL.** Stage 2 (`pose_refine.py build`, 3DGS on refined cameras) was **not run**: with 7 scenes and poses
mostly worse than the input, it could not produce a meaningful rendering comparison. Recording that as a decision,
not a result.

**Bugs found and fixed during this run, listed so the numbers above are traceable:** (1) `cam_from_world` is a
method in pycolmap 4.2, not a property — first attempt reported 0/40 successes for that reason alone; (2) requiring
all six images to register discarded the common 5/6 case, so scoring now uses the registered subset; (3) a
`use_gpu` attribute that does not exist in this version turned every scene into an exception; (4) the
"no reconstruction" return path returned two values where three were expected, which mislabelled 28 genuine
COLMAP failures as exceptions. The table above is from the run after all four fixes. COLMAP's mapper is not
deterministic — an earlier post-fix run gave 8/40 usable and a 16.487° refined median — so these counts should be
read as approximate, not exact.

**Interpretation.** On six wide-baseline views of a mostly textureless object, structure-from-motion usually
produces nothing, and when it does produce something it is typically worse than the quantized prior it was meant to
fix. This is the regime feed-forward models exist for, and it means "just refine the poses classically" is not
available as a correction step here. It also raises the bar for the confidence signal of #045: a detector is only
useful with a corrector, and the obvious corrector does not work. Untested alternatives that remain: photometric
pose refinement inside 3DGS training (needs a rasterizer with camera gradients, which GraphDECO's default does not
expose), and re-running the feed-forward model with more input views.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/pose_refine.py`
- `code/pose_uncertainty.py`

**results/**

- `results/pose_refine_w4a4_rtn.json`
- `results/pose_uncertainty_w4a4_rtn_p4.json`

**logs/**

- `logs/pose_refine_rtn.log`
- `logs/pose_uncertainty_w4a4_rtn.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

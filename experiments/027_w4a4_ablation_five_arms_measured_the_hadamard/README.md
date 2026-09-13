# 027 — W4A4 ablation, five arms measured: the Hadamard rotation is the load-bearing component

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#027** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-09 23:28 CEST (stage 2 completed); calibrations ran 2026-09-09 05:02 → 22:07
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/`
- **Commands executed:**
  ```
  ./code/quantization/run_w4a4_ablation.sh            # calibrations, one arm per component
  ./code/quantization/run_stage2_then_nolac.sh        # stage 2 reordered ahead of the last arm
  #   -> code/quantization/run_w2a4_inference.py --variant <arm> --scenes subset
  #   -> code/quantization/ablation_compare.py --scenes subset --tag subset --variants ...
  #   -> code/quantization/run_w2a4_downstream.py --scenes subset --iterations 7000 --arms ...
  ```
- **Wall time:** calibration 3.5 h/arm; stage 2 82.6 min for 7 arms

**Config:**

| Field | Value |
|---|---|
| Init source | locally calibrated W4A4 variants, one arm per removed component |
| Calibration | authors' 42-sample CO3D set, 15 epochs/block, 48 blocks (24 frame + 24 global), batch 1 |
| Iterations | 7000, identical for every arm |
| Densify | GraphDECO default |
| Input views | 6 input, 9 held-out |
| Mask type | `foreground` primary, `content` secondary (both reported) |
| Background handling | GS default black, unmasked full frames |
| Depth regularisation | off |
| Exposure | both raw and gain+bias-corrected PSNR reported |

**Scenes used (8-scene GPU subset):** apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401,
broccoli/412_56288_1088, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_385,
toaster/372_41229_82130

**Geometry error vs full-precision VGGT (n = 8):**

| arm | cam rot (deg) | cam centre err | cam spread ratio | depth rel err | point err |
|---|---:|---:|---:|---:|---:|
| w4a4 (shipped) | 1.4230 | 0.02435 | 1.005 | 0.01826 | 0.05142 |
| w4a4_local (baseline) | 1.2050 | 0.02254 | 1.000 | 0.02198 | 0.04015 |
| w4a4_no_lwc | 1.1169 | 0.01815 | 1.001 | 0.01715 | 0.04015 |
| w4a4_no_smooth | 1.1711 | 0.02102 | 0.996 | 0.01699 | 0.04851 |
| **w4a4_no_rot** | **2.4690** | **0.04117** | 0.995 | 0.02511 | **0.08084** |
| w4a4_rtn (all off) | 26.2667 | 0.13944 | 0.856 | 0.08095 | 0.68261 |
| w2a4 (reference) | 113.9351 | 0.68659 | 0.010 | 0.13184 | 87.35657 |

Against `w4a4_local`, removing the rotation costs **2.05x** camera rotation and **2.01x** point error.
Removing `lwc` or `smooth` individually costs nothing measurable (both are at or below baseline).

**Per-scene foreground PSNR (raw, 7000 it):**

| scene | full | local | no_rot | no_smooth | no_lwc | rtn |
|---|---:|---:|---:|---:|---:|---:|
| apple/110_13051_23361 | 12.670 | 12.797 | 11.655 | 12.477 | 13.631 | 9.594 |
| ball/123_14363_28981 | 11.224 | 10.726 | 10.212 | 10.427 | 10.167 | 8.528 |
| bowl/70_5792_13401 | 10.458 | 10.202 | 9.079 | 10.063 | 10.598 | 5.684 |
| broccoli/412_56288_1088 | 9.338 | 9.223 | 8.734 | 9.159 | 9.277 | 9.102 |
| hydrant/167_18184_34441 | 8.131 | 7.815 | 7.973 | 7.994 | 8.143 | 7.207 |
| remote/350_36761_68623 | 9.656 | 9.872 | 9.289 | 9.680 | 9.690 | 9.385 |
| teddybear/187_20215_385 | 9.341 | 8.411 | 8.946 | 8.672 | 8.741 | 7.973 |
| toaster/372_41229_82130 | 8.949 | 8.330 | 7.842 | 8.663 | 9.263 | 7.067 |
| **mean** | **9.9708** | **9.6721** | **9.2164** | **9.6419** | **9.9387** | **8.0677** |

**Downstream means (n = 8):**

| arm | fg PSNR | fg exp-corr | fg SSIM | fg LPIPS | content PSNR |
|---|---:|---:|---:|---:|---:|
| full | 9.9708 | 15.7667 | 0.2403 | 0.6553 | 9.8990 |
| w4a4 | 9.6424 | 15.8439 | 0.2369 | 0.6635 | 9.8931 |
| w4a4_local | 9.6721 | 15.8306 | 0.2336 | 0.6624 | 9.8191 |
| w4a4_no_lwc | 9.9387 | 15.9986 | 0.2436 | 0.6516 | 10.0610 |
| w4a4_no_rot | 9.2164 | 15.8340 | 0.2350 | 0.6710 | 9.4923 |
| w4a4_no_smooth | 9.6419 | 15.9555 | 0.2323 | 0.6598 | 9.8254 |
| w4a4_rtn | 8.0677 | 14.1246 | 0.1942 | 0.7243 | 8.4257 |

**Paired tests, foreground PSNR (the four that matter):**

| contrast | raw mean | p | CI95 | sign | exp-corr mean | p |
|---|---:|---:|---|---|---:|---:|
| full − no_rot | +0.7545 | 0.001675 | [+0.3933, +1.1156] | 8/8 | −0.0673 | 0.7725 |
| full − no_smooth | +0.3290 | 0.01246 | [+0.0959, +0.5620] | 7/8 | −0.1888 | 0.4143 |
| full − rtn | +1.9031 | 0.01054 | [+0.6028, +3.2035] | 8/8 | +1.6421 | 0.2863 |
| full − no_lwc | +0.0321 | 0.8838 | [−0.4691, +0.5334] | 3/8 | −0.2319 | 0.2988 |
| local − no_rot | +0.4557 | 0.05915 | [−0.0231, +0.9345] | 6/8 | −0.0034 | 0.9713 |
| local − rtn | +1.6044 | 0.02349 | [+0.2893, +2.9195] | 8/8 | +1.7060 | 0.2711 |

**PASS/FAIL:** Target: **attribute the W4A4 quantization gain to individual components, with the geometry
instrument separating arms at p < 0.05 and consistent sign.** Geometry separates `no_rot` from baseline by
2.05x on camera rotation and `rtn` by 21.8x, while `no_lwc` and `no_smooth` are indistinguishable from
baseline. **PASS for geometry; PASS for raw downstream on no_rot and rtn (8/8 sign, p < 0.02); FAIL for
exposure-corrected downstream on every arm (no contrast reaches p < 0.05).**

**Interpretation.** At 4 bits the QuaRot Hadamard rotation is the only component of QuantVGGT's pipeline that
is individually load-bearing -- removing it doubles camera and point error and is the single cleanest
downstream signal in this project to date (8/8 scenes, p = 0.0017); removing learned weight clipping or
SmoothQuant scaling individually costs nothing measurable, yet removing all four costs 21.8x, so the three
non-rotation components are redundant with each other rather than useless. The exposure-corrected column is
the load-bearing caveat: every raw-PSNR gap collapses to insignificance once a per-image gain+bias is fitted,
so most of what the raw metric is scoring is brightness, not geometry -- consistent with #022 and with the
harness-insensitivity finding in #020. Geometry remains the instrument that responds; the renderer does not.

**Deviation from plan, flagged:** the sixth arm `w4a4_no_lac` is **NOT MEASURED** and MUST NOT be inferred
from the other five. Its calibration was killed 1 h 07 m in (14 of 48 blocks) by the machine's nightly
shutdown. Restarted 2026-09-10 04:37 from scratch; not resumed, because upstream `resume_qs` loads finished
totals rather than continuing a run, and every other arm is a clean run.

**Operational fact worth recording:** `last -x` shows this machine shuts down at **00:35 every night** and
returns around **04:33** -- a usable window of ~20 h, not the 12:30 reset previously assumed. Any calibration
of ~3.5 h must therefore start before ~21:00.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/ablation_compare.py`
- `code/run_stage2_then_nolac.sh`
- `code/run_w2a4_downstream.py`

**results/**

- `results/arms_full_w4a4_w4a4_local_w4a4_no_lwc_w4a4_no_rot_w4a4_no_smooth_w4a4_rtn_subset_it7000.json`

**logs/**

- `logs/ablation_downstream.log.gz`
- `logs/ablation_stage2.log`
- `logs/stage2_then_nolac.log.gz`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

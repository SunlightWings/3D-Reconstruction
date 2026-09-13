# 026 — W4A4 ablation study: design, instrument, and the round-to-nearest arm (IN PROGRESS)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#026** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-09, started 05:02 CEST; calibration of the remaining arms still running
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/`
- **Commands executed:**
  ```
  python3 code/quantization/calibrate_w2a4.py --wbit 4 --abit 4 --exp-name a44_rtn \
      --no-smooth --no-rot --no-lwc --no-lac
  python3 -u code/quantization/run_w2a4_inference.py --variant w4a4_rtn --scenes subset
  python3 code/quantization/ablation_compare.py --variants w4a4 w4a4_rtn w2a4 --scenes subset
  ./code/quantization/run_w4a4_ablation.sh          # 5 calibrations, ~17.5 h
  ./code/quantization/run_ablation_full.sh subset   # stage 2, queued behind it
  ```

**What is being ablated.** QuantVGGT's W4A4 pipeline has four components, each of which changes the quantized
linear layer's parameter structure (`quarot_linear.py`):

| component | flag | what it adds |
|---|---|---|
| SmoothQuant per-channel scaling | `not_smooth` | `channel_wise_scale` |
| QuaRot Hadamard rotation | `not_rot` | `rotation_matrix` |
| learned weight clipping | `lwc` | `clip_factor_w_max/min` |
| learned activation clipping | `lac` | activation-quantizer clipping |

Arms: `a44_local` (all four on -- the baseline), then one arm per component removed, plus `a44_rtn`
(all four off). Each arm needs its OWN calibration because the learned parameters are configuration-specific;
`a44_rtn` is the exception -- with smooth/lwc/lac all off QuantVGGT performs no calibration at all, so it is
round-to-nearest weights and costs seconds.

**Why `a44_local` and not the shipped `w4a4`.** The shipped W4A4 parameters were downloaded from the authors'
HuggingFace release with unknown calibration settings. Comparing an ablation against it would confound
"component removed" with "different calibration", so every ablation is compared against a locally calibrated
full pipeline instead.

**Config:**

| Field | Value |
|---|---|
| Init source | `w4a4` variants (4-bit weights, 4-bit activations throughout) |
| Calibration | authors' 42-sample CO3D set, 15 epochs/block, 48 blocks, batch 1 |
| Iterations | 7000 for the downstream stage; all arms rendered at the SAME iteration |
| Densify | GraphDECO default |
| Input views | 6 input, 9 held-out |
| Mask type | `foreground` primary, `content` secondary (both reported) |
| Background handling | GS default black, unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset for the geometry table; the downstream stage runs the same subset first.

**Measurement design.** Two levels, deliberately:

1. *Geometry* -- each arm's VGGT output against that scene's full-precision output, after a Sim(3) fit on the six
   input camera centres (every variant predicts in its own arbitrary world frame). Metrics: mean camera-orientation
   error, camera-centre error in scene radii, camera-spread ratio (the collapse detector that caught W2A4),
   median relative depth error, median world-point error. Cost ~2 s/scene.
2. *Downstream 3DGS* -- train and render each arm, PSNR/SSIM/LPIPS under both masks, raw and exposure-corrected,
   with paired statistics per arm pair. Cost ~1 h/arm at 40 scenes.

Geometry is the sensitive instrument: at W4A4 the full-vs-quant *rendering* difference is not resolvable
(p = 0.174, #020), so an ablation scored only downstream would measure noise. But the ablation arms are far more
degraded than `w4a4`, so a downstream gap may well be resolvable there -- and the proposal's claim is a
downstream one. Geometry says which component matters; 3DGS says whether it matters to renders.

**Results so far (geometry vs full precision, 8 scenes):**

| arm | cam rot (deg) | cam centre err | cam spread ratio | depth rel err | point err |
|---|---:|---:|---:|---:|---:|
| w4a4 | 1.4230 | 0.02435 | 1.005 | 0.01826 | 0.05142 |
| w4a4_rtn | 26.2667 | 0.13944 | 0.856 | 0.08095 | 0.68261 |
| w2a4 | 113.9351 | 0.68659 | 0.010 | 0.13184 | 87.35657 |

Ratio, round-to-nearest over full pipeline: camera rotation **18.5x** worse,
camera centre **5.7x**, depth **4.4x**,
world points **13.3x**.

Downstream Sim(3) alignment residuals for the RTN arm, measured while building its 3DGS sources, range
0.0634-0.1067 scene radii across the 8 scenes, against `w4a4`'s 0.0243 (#017) -- consistent with the geometry table.

**PASS/FAIL:** Target: **the geometry instrument must separate arms that the rendering metric cannot.** The
rendering metric cannot distinguish full from w4a4 (p = 0.174, #020); the geometry metric separates
full-pipeline W4A4 from round-to-nearest W4A4 by 18x on camera rotation and from W2A4 by a further
4.3x. **PASS.** The per-component attribution is **PENDING** -- 5 calibrations still running.

**Interpretation:** Rules in QuantVGGT's smooth+rotation+clipping machinery as doing substantial work at 4 bits
(removing all of it costs 18x on camera rotation), and rules in geometry-level measurement as an instrument
that responds where the rendering metric is inert -- the first such instrument in this project. Which individual
component carries that gain is not yet measured and MUST NOT be guessed.

**Still to come in this entry's experiment (will be logged as its own entries when measured):** the four
single-component arms, and the downstream 3DGS numbers for all arms.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/ablation_compare.py`
- `code/calibrate_w2a4.py`
- `code/run_ablation_full.sh`
- `code/run_w2a4_inference.py`
- `code/run_w4a4_ablation.sh`

**results/**

- `results/w4a4_ablation_subset.json`

**logs/**

- `logs/ablation_geometry.log`
- `logs/ablation_queue.log`
- `logs/calib_a44_local.log`
- `logs/calib_a44_nolac.log`
- `logs/calib_a44_nolwc.log`
- `logs/calib_a44_norot.log`
- `logs/calib_a44_nosmooth.log`
- `logs/infer_w4a4_local.log`
- `logs/infer_w4a4_no_lac.log`
- `logs/infer_w4a4_no_lwc.log`
- `logs/infer_w4a4_no_rot.log`
- `logs/infer_w4a4_no_smooth.log`
- `logs/infer_w4a4_rtn.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

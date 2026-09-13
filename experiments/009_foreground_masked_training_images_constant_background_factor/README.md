# 009 — Foreground-masked training images, constant background (factor 5)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#009** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 CEST (this run took 4.8 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag maskedbg --iters 500 1000 3000 7000 --source maskedbg
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 6 input, 9 held-out |
| Mask type | `content` (also re-scored under `foreground`) |
| Background handling | training images have background replaced by constant black; held-out GT unchanged |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 6.84 | 6.83 | 6.82 | 6.84 |
| ball/123_14363_28981 | 5.76 | 5.77 | 5.74 | 5.75 |
| bowl/70_5792_13401 | 7.33 | 7.33 | 7.37 | 7.37 |
| broccoli/412_56288_108844 | 4.85 | 4.84 | 4.84 | 4.84 |
| hydrant/167_18184_34441 | 5.79 | 5.77 | 5.75 | 5.74 |
| remote/350_36761_68623 | 6.90 | 6.85 | 6.80 | 6.79 |
| teddybear/187_20215_38541 | 5.56 | 5.56 | 5.57 | 5.57 |
| toaster/372_41229_82130 | 6.11 | 6.10 | 6.09 | 6.11 |
| **mean PSNR** | **6.142** | **6.129** | **6.122** | **6.126** |
| **mean SSIM** | 0.0547 | 0.0509 | 0.0494 | 0.0511 |
| min PSNR | 4.85 | 4.84 | 4.84 | 4.84 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **6.142 dB at 500 iterations**. **FAIL**, short by 13.86 dB.

**Interpretation:** Rules out foreground-masked training: under the A3 content-mask metric it collapses to 6.126 dB, and even re-scored on the foreground alone (the only region it models) it reaches 14.342 dB, still below the plain oracle's 14.828 dB foreground score from entry #004 -- so removing background supervision does not improve the object either.


**Secondary metric for this config (foreground-only mask, since the model deliberately does not represent the background):**

| iterations | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| mean foreground PSNR | 13.544 | 13.737 | 14.115 | **14.342** |
| mean foreground SSIM | 0.2442 | 0.2292 | 0.2238 | 0.2315 |
| min foreground PSNR | 8.33 | 8.26 | 8.14 | 8.61 |

Command: `python3 code/downstream_3dgs/oracle_sweep/reeval.py maskedbg foreground 500 1000 3000 7000`.
Reference: plain oracle at 30000 scores 14.828 dB under this same foreground mask (entry #004). Target under this
secondary metric is still 20 dB: **FAIL**, short by 5.66 dB.

The content-mask collapse to 6.126 dB is itself a confirmation of entry #002's model, which predicted a 6.26 dB
ceiling for a perfectly-reconstructed foreground on a black background.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/reeval.py`
- `code/sweep.py`

**results/**

- `results/maskedbg.json`
- `results/maskedbg_it1000_foreground.json`
- `results/maskedbg_it3000_foreground.json`
- `results/maskedbg_it500_foreground.json`
- `results/maskedbg_it7000_foreground.json`

**logs/**

- `logs/maskedbg.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

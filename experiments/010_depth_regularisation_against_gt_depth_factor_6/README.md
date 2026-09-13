# 010 — Depth regularisation against GT depth (factor 6)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#010** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 CEST (this run took 8.7 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag depthreg --iters 500 1000 3000 7000 --source depthreg --extra -d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1 -d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` + GT inverse-depth maps |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 6 input, 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | ON: `-d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01`; GT depth covers 2.6-6.2% of pixels, foreground only |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 16.00 | 15.29 | 14.98 | 14.90 |
| ball/123_14363_28981 | 10.47 | 10.95 | 14.91 | 14.82 |
| bowl/70_5792_13401 | 11.96 | 12.40 | 14.80 | 15.35 |
| broccoli/412_56288_108844 | 9.41 | 10.67 | 12.24 | 13.03 |
| hydrant/167_18184_34441 | 10.71 | 11.69 | 12.50 | 13.78 |
| remote/350_36761_68623 | 11.12 | 13.05 | 13.89 | 14.69 |
| teddybear/187_20215_38541 | 17.77 | 15.28 | 17.36 | 17.76 |
| toaster/372_41229_82130 | 10.93 | 10.37 | 12.88 | 13.95 |
| **mean PSNR** | **12.297** | **12.461** | **14.195** | **14.786** |
| **mean SSIM** | 0.4489 | 0.4207 | 0.3499 | 0.3561 |
| min PSNR | 9.41 | 10.37 | 12.24 | 13.03 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **14.786 dB at 7000 iterations**. **FAIL**, short by 5.21 dB.

**Interpretation:** Rules IN depth regularisation as a real but insufficient help: 14.786 dB at 7000 vs the baseline 14.023 dB (+0.763 dB), achieved even though GT depth covers only 2.6-6.2% of pixels and lies almost entirely on the object. Second-best single factor after view count, still 5.21 dB short. NOTE: the first attempt at this config failed (rc=1) because render.py inherits the training cfg_args 'depths' setting and then demands depth_params.json in the heldout source; fixed by passing -d "" at render time (sweep.py), and re-run.


---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/prep_depth.py`
- `code/sweep.py`

**results/**

- `results/depthreg.json`

**logs/**

- `logs/depthreg.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

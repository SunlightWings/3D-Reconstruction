# 003 — Baseline oracle, iteration sweep (factor 2)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#003** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 21:00-21:09 CEST (8.9 min wall for the 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (dirty, as above)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag base_itersweep --iters 500 1000 3000 7000
  ```
  which per scene invokes, from `/home/utn/luli38se/cv/gaussian-splatting`:
  ```
  /opt/saltstack/salt/bin/python3.10 train.py -s <oracle>/train -m <model> --iterations 7000 \
      --data_device cpu --resolution 1 --quiet --disable_viewer \
      --save_iterations 500 1000 3000 7000 --test_iterations -1
  /opt/saltstack/salt/bin/python3.10 render.py -m <model> -s <oracle>/heldout --iteration <it> --skip_test --quiet
  ```

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` (GT cameras + GT-unprojected foreground points; reused the existing `PREPARED.ok` sources) |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in a single training run per scene) |
| Densify | default (`densify_from_iter` 500, `densify_until_iter` 15000, interval 100, grad threshold 0.0002) |
| Input views | 6 input, 9 held-out |
| Mask type | `content` (identical to A3) |
| Background handling | GS default black; training images are unmasked full frames |
| Depth regularisation | off (`depth_l1_weight` unused, no `-d` depths folder) |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR / SSIM):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.75 | 15.36 | 14.41 | 15.80 |
| ball/123_14363_28981 | 10.73 | 10.44 | 13.32 | 13.40 |
| bowl/70_5792_13401 | 12.05 | 13.08 | 14.79 | 15.26 |
| broccoli/412_56288_108844 | 9.29 | 10.87 | 11.67 | 12.46 |
| hydrant/167_18184_34441 | 11.28 | 10.46 | 10.93 | 11.75 |
| remote/350_36761_68623 | 10.97 | 13.81 | 13.62 | 14.60 |
| teddybear/187_20215_38541 | 16.25 | 14.29 | 15.83 | 16.48 |
| toaster/372_41229_82130 | 10.71 | 11.07 | 11.04 | 12.41 |
| **mean PSNR** | **12.129** | **12.423** | **13.202** | **14.023** |
| **mean SSIM** | 0.4502 | 0.4308 | 0.3759 | 0.3768 |
| min PSNR | 9.29 | 10.44 | 10.93 | 11.75 |

Reference point at 30000 iterations, same config, from A3: **14.061 dB**. LPIPS not computed for this sweep (the A3 metric path used for comparability computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR.** Best measured: **14.023 dB at 7000 iterations** (14.061 dB at 30000). **FAIL**, by ~6 dB.

**Interpretation:** Rules out early stopping as the fix — mean PSNR rises monotonically with iterations (12.13 -> 14.02 -> 14.06 at 30k) rather than peaking early and decaying, so the sparse-view failure here is not the kind of overfitting that fewer iterations repairs; the falling SSIM (0.450 -> 0.377) alongside rising PSNR does indicate densification is trading structural fidelity for photometric mean.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/emit_entry.py`
- `code/prep.py`
- `code/run_configs.sh`
- `code/sweep.py`

**results/**

- `results/base_itersweep.json`
- `results/smoke_500.json`

**logs/**

- `logs/queue.log`
- `logs/queue2.log`
- `logs/queue3.log`
- `logs/queue4.log`
- `logs/queue5.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

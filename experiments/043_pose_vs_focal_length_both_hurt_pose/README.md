# 043 — Pose vs focal length: both hurt, pose more, and pose alone accounts for the perceptual loss

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#043** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11 21:11 → 2026-09-12 00:11 CEST (GPU); robustness pass 2026-09-12 09:07
- **Git commit:** `f0c8e32` + working tree (`code/quantization/geometry_probes.py`, `run_camsplit_after.sh`,
  `run_camsplit_finish.sh`)
- **Commands executed:**
  ```
  /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/geometry_probes.py camsplit \
      --arm w4a4_rtn --scenes 40 --deadline 00:15          # via run_camsplit_after.sh
  nice -n 15 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/cpu_probes.py robustness
  ```
  A resume path (`--resume`, `run_camsplit_finish.sh`) was queued in case the 00:15 guard cut the run short. It
  was **not needed**: the first run finished all 40 scenes at 00:11 and the continuation exited with nothing to do.
- **Config:** exactly #039's, except the two conditions, both built in full precision's frame and rendered from
  full's held-out source:

| condition | cameras | intrinsics | points |
|---|---|---|---|
| R (pose only) | `w4a4_rtn` pose, Sim(3)-carried | **full** | full |
| F (focal only) | **full** | `w4a4_rtn` | full |

  Offline check before launch: R leaves intrinsics equal to full's and changes pose; F leaves extrinsics equal to
  full's and changes intrinsics. `w4a4_rtn`'s focal error is 18.652 % and its camera rotation 14.6565° (#031, #029).
- **Scenes used:** all 40 of the frozen manifest; 0 failed.

**Per-scene foreground results:**

| scene | full PSNR | R PSNR | F PSNR | full LPIPS | R LPIPS | F LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| apple/110_13051_23361 | 12.670 | 10.427 | 10.297 | 0.6001 | 0.6906 | 0.6784 |
| apple/189_20393_38136 | 6.539 | 6.072 | 6.432 | 0.7393 | 0.8028 | 0.7755 |
| ball/123_14363_28981 | 11.224 | 8.957 | 10.294 | 0.5655 | 0.6427 | 0.5953 |
| ball/375_42693_85518 | 10.229 | 10.357 | 10.017 | 0.6800 | 0.7014 | 0.7161 |
| bench/415_57112_110099 | 12.687 | 11.776 | 12.067 | 0.6278 | 0.6377 | 0.6300 |
| bench/415_57121_110109 | 11.121 | 9.925 | 11.022 | 0.6127 | 0.6952 | 0.6079 |
| book/119_13962_28926 | 8.465 | 8.310 | 8.397 | 0.6617 | 0.7351 | 0.7036 |
| book/247_26469_51778 | 10.197 | 9.277 | 9.811 | 0.6969 | 0.7808 | 0.6997 |
| bowl/69_5465_12831 | 10.959 | 9.766 | 10.045 | 0.6136 | 0.7195 | 0.6167 |
| bowl/70_5792_13401 | 10.458 | 7.380 | 10.443 | 0.6549 | 0.8652 | 0.6576 |
| broccoli/372_41112_81867 | 8.211 | 7.351 | 8.458 | 0.7040 | 0.7352 | 0.7044 |
| broccoli/412_56288_108844 | 9.338 | 9.477 | 9.608 | 0.7163 | 0.7618 | 0.6909 |
| cake/374_42274_84517 | 7.306 | 7.398 | 6.197 | 0.6285 | 0.6711 | 0.6661 |
| cake/403_53094_103680 | 7.532 | 7.156 | 7.377 | 0.7464 | 0.7636 | 0.7461 |
| donut/391_47032_93657 | 8.537 | 7.830 | 8.683 | 0.7418 | 0.8178 | 0.7653 |
| donut/403_52964_103416 | 8.792 | 8.508 | 10.683 | 0.6310 | 0.6171 | 0.5718 |
| hydrant/167_18184_34441 | 8.131 | 7.456 | 8.087 | 0.7404 | 0.7666 | 0.7694 |
| hydrant/411_56064_108483 | 8.130 | 8.595 | 7.867 | 0.7223 | 0.6992 | 0.7057 |
| mouse/107_12753_23606 | 6.719 | 7.463 | 7.052 | 0.7095 | 0.7105 | 0.6990 |
| mouse/377_43416_86289 | 7.819 | 6.431 | 7.187 | 0.6685 | 0.7644 | 0.6685 |
| orange/374_42196_84367 | 9.734 | 8.645 | 7.985 | 0.7331 | 0.8520 | 0.8414 |
| orange/385_45386_90752 | 8.716 | 9.471 | 8.062 | 0.6867 | 0.6692 | 0.7227 |
| plant/247_26441_50907 | 9.124 | 7.676 | 7.518 | 0.6091 | 0.6893 | 0.6767 |
| plant/374_42005_84358 | 8.735 | 9.089 | 7.284 | 0.6161 | 0.6395 | 0.6628 |
| remote/195_20989_41543 | 9.215 | 8.790 | 8.568 | 0.6238 | 0.6962 | 0.6648 |
| remote/350_36761_68623 | 9.656 | 9.613 | 9.650 | 0.6703 | 0.7105 | 0.6491 |
| skateboard/245_26182_52130 | 9.546 | 8.396 | 9.509 | 0.5752 | 0.6954 | 0.6149 |
| skateboard/366_39266_76077 | 9.450 | 10.275 | 9.671 | 0.6357 | 0.6764 | 0.6381 |
| suitcase/410_55734_107452 | 8.462 | 6.700 | 8.508 | 0.6277 | 0.7267 | 0.6486 |
| suitcase/50_2928_8645 | 9.317 | 8.090 | 8.494 | 0.6498 | 0.6983 | 0.6880 |
| teddybear/187_20215_38541 | 9.341 | 6.897 | 8.963 | 0.6020 | 0.6739 | 0.5955 |
| teddybear/34_1479_4753 | 7.200 | 7.185 | 6.928 | 0.7102 | 0.7460 | 0.7220 |
| toaster/372_41229_82130 | 8.949 | 7.113 | 8.678 | 0.6932 | 0.7327 | 0.6935 |
| toaster/416_57389_110765 | 8.631 | 8.670 | 8.066 | 0.6369 | 0.7157 | 0.6689 |
| toytrain/240_25394_51994 | 8.422 | 9.238 | 7.620 | 0.6894 | 0.9396 | 0.6983 |
| toytrain/399_51323_100753 | 4.455 | 4.134 | 4.989 | 0.8379 | 0.8557 | 0.8247 |
| toytruck/190_20494_39385 | 10.355 | 8.350 | 6.695 | 0.5123 | 0.5662 | 0.6214 |
| toytruck/346_36113_66551 | 7.048 | 6.914 | 6.842 | 0.6621 | 0.7066 | 0.6722 |
| vase/374_41862_83720 | 6.780 | 7.589 | 5.876 | 0.6182 | 0.6286 | 0.6520 |
| vase/380_44863_89631 | 10.372 | 8.051 | 9.135 | 0.5649 | 0.6689 | 0.6168 |

**Means and paired tests (n = 40; positive Δ = worse than full):**

| condition | PSNR | SSIM | LPIPS | Δ PSNR | p | Δ LPIPS | p | sign |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full | 8.9642 | 0.2318 | 0.6604 | | | | | |
| F focal only | 8.4767 | 0.2220 | 0.6810 | +0.4876 | 0.00125 | +0.0206 | 0.0004171 | 32/40 |
| R pose only | 8.2700 | 0.2199 | 0.7216 | +0.6943 | 0.0001121 | +0.0612 | 1.325e-08 | 29/40 |
| K both (from #039) | 7.9922 | 0.2124 | 0.7269 | +0.9721 | 5.068e-06 | +0.0665 | 5.783e-09 | 30/40 |

**Robustness (`cpu_probes.py robustness`, 2026-09-12 09:07):** R PSNR boot CI95 [+0.3875, +1.0094], perm p 0.00045,
worst leave-one-out p 0.000228, no-bowl +0.6331 / p 0.00019. F PSNR CI95 [+0.2265, +0.7641], perm p 0.00085,
worst LOO p 0.00239, no-bowl +0.4997 / p 0.0012. R LPIPS CI95 [+0.0454, +0.0784], perm p 5e-05 (floor).
F LPIPS CI95 [+0.0102, +0.0311], perm p 0.0003.

**PASS/FAIL:**
- Target: **split #039's camera damage into pose and focal length.** Both are real and robust. PSNR: pose +0.6943,
  focal +0.4876, jointly +0.9721 — **sub-additive** (sum 1.1819 vs 0.9721). LPIPS: pose +0.0612 vs joint +0.0665,
  focal only +0.0206 — **pose alone reproduces 92 % of the joint perceptual loss.** **PASS.**
- The caveat carried since #031 and #039 — *"cameras" is not yet split* — is now **resolved**.

**Interpretation.** Both parts of a wrong camera hurt, and neither is negligible: an 18.65 % focal error alone costs
0.49 dB even with perfect pose and perfect points. But pose is the larger term and dominates the perceptual metric.
Combined with #042 — where pure 16° rotation with correct centres, intrinsics and points cost +1.0016 dB, close to
the +0.9721 dB of `rtn`'s full camera swap — the picture is consistent: `rtn`'s rendering loss is mostly its camera
orientation. For the proposal this narrows the target further: if anything is worth predicting or correcting in a
quantized geometry prior, it is the **camera pose**, then intrinsics; points remain irrelevant (#039, #042).
Caveat: measured on one heavily damaged arm (`w4a4_rtn`); at W4A4's own 1.28° / 2.6 % the sweep (#042) says both
are below the detection threshold.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/cpu_probes.py`
- `code/geometry_probes.py`
- `code/run_camsplit_after.sh`
- `code/run_camsplit_finish.sh`

**results/**

- `results/geometry_probe_camsplit_w4a4_rtn_it7000.json`

**logs/**

- `logs/camsplit_chain.log`
- `logs/camsplit_finish.log`
- `logs/probe_camsplit_rtn.log.gz`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

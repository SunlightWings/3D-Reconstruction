# 039 — Camera/point swap, 40 scenes: the rendering damage is in the cameras, not the points

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#039** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11 10:34 → 15:08 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/geometry_probes.py`
- **Command executed:**
  ```
  /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/geometry_probes.py swap \
      --arm w4a4_rtn --scenes 40 --deadline 00:15
  ```
  (launched by `code/quantization/run_probes_day.sh`)
- **Config:**

| Field | Value |
|---|---|
| Frame | full precision's world frame; `w4a4_rtn` carried in by Sim(3) on the six input camera centres |
| Conditions | A full cams + full pts (existing `full`); P full cams + rtn pts; K rtn cams (pose **and intrinsics**) + full pts; B' rtn cams + rtn pts |
| Evaluation cameras | full's own held-out source for **all four** cells |
| Init | stride-4 VGGT point grid, 101,400 points |
| Iterations | 7000 |
| Densify | GraphDECO default |
| Input views | 6 input, 9 held-out |
| Mask type | foreground |
| Background handling | GS default black, unmasked full frames |
| Depth regularisation | off |

  Offline check before launch (apple/110): K changes only cameras (points identical to full), P only points
  (cameras identical); rtn camera error on that scene 14.875°, point error 0.5426 radii.
- **Scenes used:** all 40 of the frozen manifest; 0 failed.

**Per-scene foreground results:**

| scene | A PSNR | P PSNR | K PSNR | B' PSNR | A LPIPS | P LPIPS | K LPIPS | B' LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| apple/110_13051_23361 | 12.670 | 12.425 | 9.940 | 9.190 | 0.6001 | 0.6064 | 0.6958 | 0.7021 |
| apple/189_20393_38136 | 6.539 | 7.450 | 5.874 | 6.279 | 0.7393 | 0.7136 | 0.7832 | 0.7689 |
| ball/123_14363_28981 | 11.224 | 11.116 | 8.986 | 8.610 | 0.5655 | 0.5425 | 0.6489 | 0.6361 |
| ball/375_42693_85518 | 10.229 | 10.420 | 10.363 | 10.065 | 0.6800 | 0.6653 | 0.7044 | 0.7015 |
| bench/415_57112_110099 | 12.687 | 11.735 | 11.549 | 10.860 | 0.6278 | 0.6512 | 0.6499 | 0.6653 |
| bench/415_57121_110109 | 11.121 | 11.094 | 10.263 | 9.191 | 0.6127 | 0.6204 | 0.6818 | 0.7123 |
| book/119_13962_28926 | 8.465 | 8.051 | 8.601 | 7.525 | 0.6617 | 0.6323 | 0.7511 | 0.7194 |
| book/247_26469_51778 | 10.197 | 9.705 | 9.423 | 8.909 | 0.6969 | 0.7027 | 0.7622 | 0.7846 |
| bowl/69_5465_12831 | 10.959 | 11.671 | 8.865 | 8.872 | 0.6136 | 0.6109 | 0.7357 | 0.7315 |
| bowl/70_5792_13401 | 10.458 | 5.684 | 7.380 | 5.684 | 0.6549 | 0.9102 | 0.8652 | 0.9102 |
| broccoli/372_41112_81867 | 8.211 | 8.312 | 7.929 | 7.766 | 0.7040 | 0.6969 | 0.7345 | 0.7222 |
| broccoli/412_56288_108844 | 9.338 | 9.656 | 7.946 | 8.810 | 0.7163 | 0.6956 | 0.7737 | 0.7720 |
| cake/374_42274_84517 | 7.306 | 6.445 | 6.237 | 5.732 | 0.6285 | 0.6554 | 0.6972 | 0.7150 |
| cake/403_53094_103680 | 7.532 | 7.867 | 6.778 | 6.542 | 0.7464 | 0.7330 | 0.7726 | 0.7829 |
| donut/391_47032_93657 | 8.537 | 9.193 | 8.862 | 8.438 | 0.7418 | 0.7314 | 0.7868 | 0.7831 |
| donut/403_52964_103416 | 8.792 | 7.833 | 10.360 | 8.949 | 0.6310 | 0.6188 | 0.5779 | 0.5933 |
| hydrant/167_18184_34441 | 8.131 | 7.694 | 7.385 | 7.123 | 0.7404 | 0.7637 | 0.7828 | 0.7831 |
| hydrant/411_56064_108483 | 8.130 | 8.217 | 8.927 | 8.601 | 0.7223 | 0.7003 | 0.6835 | 0.6612 |
| mouse/107_12753_23606 | 6.719 | 7.835 | 6.827 | 7.745 | 0.7095 | 0.6623 | 0.7279 | 0.7212 |
| mouse/377_43416_86289 | 7.819 | 7.634 | 5.672 | 6.084 | 0.6685 | 0.6654 | 0.7758 | 0.7491 |
| orange/374_42196_84367 | 9.734 | 9.543 | 7.485 | 8.043 | 0.7331 | 0.7753 | 0.8629 | 0.8552 |
| orange/385_45386_90752 | 8.716 | 8.366 | 8.014 | 8.269 | 0.6867 | 0.7046 | 0.7170 | 0.7182 |
| plant/247_26441_50907 | 9.124 | 9.529 | 7.038 | 7.110 | 0.6091 | 0.6046 | 0.6972 | 0.6835 |
| plant/374_42005_84358 | 8.735 | 9.732 | 7.962 | 8.402 | 0.6161 | 0.5883 | 0.6667 | 0.6368 |
| remote/195_20989_41543 | 9.215 | 9.392 | 8.662 | 8.025 | 0.6238 | 0.6252 | 0.6786 | 0.6913 |
| remote/350_36761_68623 | 9.656 | 9.712 | 9.581 | 9.433 | 0.6703 | 0.6395 | 0.7047 | 0.6954 |
| skateboard/245_26182_52130 | 9.546 | 8.844 | 8.451 | 7.424 | 0.5752 | 0.6158 | 0.7035 | 0.7320 |
| skateboard/366_39266_76077 | 9.450 | 9.846 | 10.087 | 9.654 | 0.6357 | 0.6299 | 0.6878 | 0.6826 |
| suitcase/410_55734_107452 | 8.462 | 7.885 | 6.816 | 6.166 | 0.6277 | 0.6536 | 0.7180 | 0.7525 |
| suitcase/50_2928_8645 | 9.317 | 9.095 | 7.595 | 7.835 | 0.6498 | 0.6506 | 0.7191 | 0.6920 |
| teddybear/187_20215_38541 | 9.341 | 10.649 | 7.349 | 7.768 | 0.6020 | 0.5376 | 0.6700 | 0.6418 |
| teddybear/34_1479_4753 | 7.200 | 8.210 | 7.249 | 7.563 | 0.7102 | 0.7272 | 0.7433 | 0.7685 |
| toaster/372_41229_82130 | 8.949 | 8.183 | 7.141 | 6.954 | 0.6932 | 0.6656 | 0.7222 | 0.7344 |
| toaster/416_57389_110765 | 8.631 | 8.617 | 7.652 | 8.082 | 0.6369 | 0.6298 | 0.7075 | 0.7011 |
| toytrain/240_25394_51994 | 8.422 | 8.292 | 9.188 | 8.295 | 0.6894 | 0.9813 | 0.9453 | 0.9808 |
| toytrain/399_51323_100753 | 4.455 | 4.275 | 4.085 | 5.135 | 0.8379 | 0.8354 | 0.8447 | 0.8144 |
| toytruck/190_20494_39385 | 10.355 | 10.128 | 7.465 | 7.151 | 0.5123 | 0.5404 | 0.6021 | 0.6354 |
| toytruck/346_36113_66551 | 7.048 | 7.163 | 6.105 | 6.060 | 0.6621 | 0.6748 | 0.7204 | 0.7457 |
| vase/374_41862_83720 | 6.780 | 6.609 | 6.795 | 6.771 | 0.6182 | 0.6357 | 0.6581 | 0.6659 |
| vase/380_44863_89631 | 10.372 | 11.342 | 6.796 | 7.575 | 0.5649 | 0.5657 | 0.7163 | 0.6978 |

**Means and paired tests (n = 40; positive Δ = condition worse than A):**

| condition | PSNR | SSIM | LPIPS | Δ PSNR | p | Δ LPIPS | p | sign |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A full | 8.9642 | 0.2318 | 0.6604 | | | | | |
| P bad points | 8.8862 | 0.2277 | 0.6715 | +0.0780 | 0.61 | +0.0111 | 0.2893 | 22/40 |
| K bad cameras | 7.9922 | 0.2124 | 0.7269 | +0.9721 | 5.068e-06 | +0.0665 | 5.783e-09 | 30/40 |
| B' both | 7.8172 | 0.2090 | 0.7285 | +1.1470 | 7.852e-07 | +0.0681 | 9.53e-08 | 34/40 |

Additivity (PSNR): points +0.0780 + cameras +0.9721 vs both +1.1470; interaction +0.0969.

**Robustness (final pass of `cpu_probes.py robustness`, 15:09–15:10):** full − P: boot CI95 [−0.1781, +0.3999],
perm p 0.694, no-bowl −0.0424 / p 0.66. full − K: CI95 [+0.6206, +1.3219], perm p 5e-05 (floor), worst
leave-one-out p 1.1e-05, no-bowl +0.9181 / p 1e-05. full − B': CI95 [+0.7768, +1.5273], perm p 5e-05.

**PASS/FAIL:** Target: **attribute `w4a4_rtn`'s rendering loss to its cameras or its points, with both
effects measured under identical evaluation cameras.** Cameras alone: +0.9721 dB, p = 5.1e-06, robust to every
test. Points alone: +0.0780 dB, p = 0.61, CI spanning zero. Cameras account for 0.9721 / 1.1470 = **85 %** of the
joint loss. **PASS.**

**Interpretation.** The geometry prior's *points* are close to irrelevant to held-out rendering here; its
*cameras* carry the damage. This is why #030's perfect point-level confidence recovered nothing. Caveat carried
forward: K replaces pose and intrinsics together, and `rtn`'s focal is 18.652 % off (#031), so "cameras" is not
yet split into pose vs focal — `geometry_probes.py camsplit` is queued for that.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/geometry_probes.py`
- `code/run_probes_day.sh`

**results/**

- `results/geometry_probe_swap_w4a4_rtn_it7000.json`

**logs/**

- `logs/probe_swap_rtn.log.gz`
- `logs/probes_day.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

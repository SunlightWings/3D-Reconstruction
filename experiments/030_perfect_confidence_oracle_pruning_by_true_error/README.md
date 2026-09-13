# 030 — Perfect-confidence oracle: pruning by TRUE error recovers none of a real gap

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#030** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-10 21:54 → 2026-09-11 00:15 CEST (self-stopped at its deadline)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/confidence_oracle.py`
- **Command executed:**
  ```
  python -u code/quantization/confidence_oracle.py --arm w4a4_rtn --scenes 40 \
      --iterations 7000 --keep 0.5 --deadline 00:15
  ```
- **Wall time:** 2 h 21 m; stopped cleanly after 36 of 40 scenes. The last four were **not run**, not estimated.

**What it tests.** An upper bound on the proposal. Instead of a learned confidence head, the TRUE per-point error
against full-precision VGGT (Sim(3) on the six input camera centres, residual in scene radii, on the stride-4
init grid) is used as confidence, and the worst half of the initial points is removed before 3DGS.

**Conditions (per scene):** A full precision; B `w4a4_rtn`, all points; C `w4a4_rtn`, lowest-error 50 % kept;
R `w4a4_rtn`, random 50 % kept (seed 0) — the matched-point-count control. **C − R is the test**; C − B
confounds confidence with halving the Gaussian count.

**Config:**

| Field | Value |
|---|---|
| Init source | `w4a4_rtn` VGGT geometry, stride 4, 101,400 → 50,700 points |
| Confidence | true per-point error vs full precision (oracle — not available at deployment) |
| Iterations | 7000 |
| Densify | GraphDECO default |
| Input views | 6 input, 9 held-out |
| Mask type | foreground |
| Background handling | GS default black, unmasked full frames |
| Depth regularisation | off |
| Cameras | the arm's own predicted cameras, unchanged by pruning |

**Scenes used:** the first 36 of the frozen 40-scene manifest, in manifest order.

**Per-scene results (foreground):**

| scene | A PSNR | B PSNR | C PSNR | R PSNR | B LPIPS | C LPIPS | R LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| apple/110_13051_23361 | 12.670 | 9.594 | 9.728 | 9.490 | 0.7101 | 0.6992 | 0.7102 |
| apple/189_20393_38136 | 6.539 | 6.527 | 6.337 | 6.316 | 0.7506 | 0.7606 | 0.7664 |
| ball/123_14363_28981 | 11.224 | 8.528 | 8.574 | 8.538 | 0.6304 | 0.6388 | 0.6255 |
| ball/375_42693_85518 | 10.229 | 10.243 | 10.832 | 10.498 | 0.6918 | 0.6946 | 0.6977 |
| bench/415_57112_110099 | 12.687 | 10.799 | 12.031 | 11.178 | 0.6609 | 0.6300 | 0.6500 |
| bench/415_57121_110109 | 11.121 | 9.421 | 9.998 | 9.393 | 0.6981 | 0.6779 | 0.6904 |
| book/119_13962_28926 | 8.465 | 7.876 | 7.965 | 7.844 | 0.7016 | 0.6898 | 0.6953 |
| book/247_26469_51778 | 10.197 | 8.996 | 8.482 | 8.846 | 0.7786 | 0.7744 | 0.7815 |
| bowl/69_5465_12831 | 10.959 | 10.152 | 10.270 | 10.548 | 0.7098 | 0.6971 | 0.7090 |
| bowl/70_5792_13401 | 10.458 | 5.684 | 5.684 | 5.684 | 0.9102 | 0.9102 | 0.9102 |
| broccoli/372_41112_81867 | 8.211 | 7.772 | 7.589 | 7.679 | 0.7098 | 0.7078 | 0.7055 |
| broccoli/412_56288_108844 | 9.338 | 9.102 | 8.102 | 9.253 | 0.7506 | 0.7465 | 0.7492 |
| cake/374_42274_84517 | 7.306 | 5.668 | 6.022 | 5.474 | 0.7218 | 0.6829 | 0.7230 |
| cake/403_53094_103680 | 7.532 | 6.284 | 5.990 | 6.464 | 0.7698 | 0.7810 | 0.7664 |
| donut/391_47032_93657 | 8.537 | 8.321 | 7.830 | 8.133 | 0.7864 | 0.7965 | 0.7582 |
| donut/403_52964_103416 | 8.792 | 9.867 | 7.803 | 9.653 | 0.5679 | 0.6068 | 0.5999 |
| hydrant/167_18184_34441 | 8.131 | 7.207 | 7.989 | 7.295 | 0.7744 | 0.7689 | 0.7733 |
| hydrant/411_56064_108483 | 8.130 | 8.777 | 8.928 | 8.817 | 0.6498 | 0.6460 | 0.6622 |
| mouse/107_12753_23606 | 6.719 | 7.672 | 7.660 | 7.847 | 0.7223 | 0.7050 | 0.7255 |
| mouse/377_43416_86289 | 7.819 | 5.686 | 5.753 | 5.811 | 0.7491 | 0.7393 | 0.7432 |
| orange/374_42196_84367 | 9.734 | 8.158 | 7.595 | 7.792 | 0.8378 | 0.8587 | 0.8456 |
| orange/385_45386_90752 | 8.716 | 8.720 | 8.071 | 8.267 | 0.6668 | 0.6867 | 0.6843 |
| plant/247_26441_50907 | 9.124 | 7.870 | 7.758 | 7.934 | 0.6639 | 0.6486 | 0.6655 |
| plant/374_42005_84358 | 8.735 | 8.265 | 8.089 | 8.121 | 0.6382 | 0.6670 | 0.6398 |
| remote/195_20989_41543 | 9.215 | 7.901 | 8.249 | 8.068 | 0.6961 | 0.6846 | 0.6942 |
| remote/350_36761_68623 | 9.656 | 9.385 | 9.361 | 9.516 | 0.6664 | 0.6901 | 0.6627 |
| skateboard/245_26182_52130 | 9.546 | 8.047 | 8.323 | 8.521 | 0.7052 | 0.6813 | 0.6997 |
| skateboard/366_39266_76077 | 9.450 | 9.549 | 9.252 | 9.944 | 0.6813 | 0.6779 | 0.6721 |
| suitcase/410_55734_107452 | 8.462 | 6.361 | 6.537 | 6.243 | 0.7491 | 0.7244 | 0.7450 |
| suitcase/50_2928_8645 | 9.317 | 8.353 | 8.326 | 8.292 | 0.6917 | 0.7099 | 0.7005 |
| teddybear/187_20215_38541 | 9.341 | 7.973 | 7.942 | 7.648 | 0.6389 | 0.6427 | 0.6558 |
| teddybear/34_1479_4753 | 7.200 | 8.224 | 7.001 | 8.110 | 0.6926 | 0.7009 | 0.6985 |
| toaster/372_41229_82130 | 8.949 | 7.067 | 6.740 | 6.915 | 0.7133 | 0.7194 | 0.7185 |
| toaster/416_57389_110765 | 8.631 | 7.865 | 7.493 | 8.030 | 0.7076 | 0.7114 | 0.7025 |
| toytrain/240_25394_51994 | 8.422 | 8.232 | 8.292 | 8.362 | 0.9714 | 0.9812 | 0.9785 |
| toytrain/399_51323_100753 | 4.455 | 4.363 | 3.840 | 4.507 | 0.8222 | 0.8253 | 0.8352 |

**Means (n = 36):**

| condition | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| A full precision | 9.0004 | 0.2341 | 0.6683 |
| B rtn, all points | 8.0697 | 0.2128 | 0.7218 |
| C rtn, confidence-kept | 7.9566 | 0.2062 | 0.7212 |
| R rtn, random-kept | 8.0842 | 0.2137 | 0.7234 |

**Paired (n = 36):**

| contrast | PSNR | p | LPIPS | p |
|---|---:|---:|---:|---:|
| A − B (the gap to close) | +0.9307 | 4.56e-05 | −0.0536 | 6.632e-05 |
| C − B (confidence vs uniform) | −0.1131 | 0.246 | −0.0006 | 0.8223 |
| **C − R (confidence vs random)** | **−0.1276** | **0.1541** | **−0.0022** | **0.4084** |
| R − B (point-count effect) | +0.0145 | 0.6948 | +0.0015 | 0.3833 |

C beats R on PSNR in 14/36 scenes and beats B in 15/36. The single-scene smoke test (apple, C − R +0.239 dB)
was not representative.

**PASS/FAIL:** Target: **a perfect-knowledge confidence signal recovers a meaningful fraction of the A − B gap,
beyond the matched random control.** The gap is real (+0.9307 dB, p = 4.56e-05). Recovery (C − B)/(A − B) on
PSNR = −0.1131 / 0.9307 = **−12 %**, not significant; C − R is −0.1276 dB, p = 0.1541. **FAIL.**

**Interpretation.** Removing the least trustworthy half of the initial points — chosen with error values no
deployed predictor could ever see — recovers none of a real, highly significant 0.93 dB gap, and does no better
than removing a random half. This rules out init-time point pruning as the mechanism for confidence-weighted
3DGS on this arm. It does **not** rule out every use of confidence: GraphDECO densification regrows points
from 50,700 anyway, and the pruning leaves the arm's cameras untouched. The second point matters most: `rtn`
carries 14.66° of camera-rotation error, and no point-level confidence can correct a wrong camera. **Whether
`rtn`'s gap is camera error or point error is not measured** — the discriminating experiment is to render `rtn`
points with full-precision cameras, and it should be run before any conclusion about the proposal is drawn.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/confidence_oracle.py`

**results/**

- `results/confidence_oracle_w4a4_rtn_it7000.json`

**logs/**

- `logs/confidence_oracle_rtn.log.gz`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

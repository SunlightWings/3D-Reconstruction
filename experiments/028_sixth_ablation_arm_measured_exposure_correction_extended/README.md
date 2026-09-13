# 028 — Sixth ablation arm measured; exposure correction extended to SSIM and LPIPS

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#028** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-10; `a44_nolac` calibrated 04:37–08:07, stage 2 rerun 08:07–08:31
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/`
- **Commands executed:**
  ```
  ./code/quantization/run_nolac_then_stage2.sh
  #   -> calibrate_w2a4.py --wbit 4 --abit 4 --exp-name a44_nolac --no-lac
  #   -> run_ablation_full.sh subset   (auto-detected all six arms)
  ```
- **Wall time:** calibration 3 h 30 m (rc=0); stage 2 rerun 24 min

**Why a restart and not a resume.** The first `a44_nolac` attempt was killed at 14/48 blocks by the
nightly 00:35 shutdown. It was restarted from scratch: upstream `resume_qs` loads finished totals rather
than continuing a run, and block-wise calibration feeds each block's output into the next, so a hand-rolled
resume would have had to replay layers 0–6 anyway and could diverge from a clean run. Every other arm is a
clean run; an arm not comparable to the others has no value here. Partial files retained under
`killed_run_2026-09-09T2328/`.

**Config:** identical to #027 in every field (7000 iterations, 6 input / 9 held-out views, foreground primary,
GraphDECO defaults, no depth regularisation, 8-scene subset). Only the arm set changed.

**Geometry error vs full-precision VGGT (n = 8), all six arms:**

| arm | cam rot (deg) | cam centre err | cam spread ratio | depth rel err | point err | vs `w4a4_local` |
|---|---:|---:|---:|---:|---:|---:|
| w4a4_no_lwc | 1.1169 | 0.01815 | 1.001 | 0.01715 | 0.04015 | 0.93x |
| w4a4_no_smooth | 1.1711 | 0.02102 | 0.996 | 0.01699 | 0.04851 | 0.97x |
| w4a4_local (baseline) | 1.2050 | 0.02254 | 1.000 | 0.02198 | 0.04015 | 1.00x |
| **w4a4_no_lac** | **1.5476** | 0.02645 | 0.991 | 0.02298 | 0.04957 | **1.28x** |
| **w4a4_no_rot** | **2.4690** | 0.04117 | 0.995 | 0.02511 | 0.08084 | **2.05x** |
| w4a4_rtn | 26.2667 | 0.13944 | 0.856 | 0.08095 | 0.68261 | 21.80x |

**Downstream, foreground, raw and exposure-corrected (n = 8):**

| arm | PSNR | PSNR* | SSIM | SSIM* | LPIPS | LPIPS* |
|---|---:|---:|---:|---:|---:|---:|
| full | 9.9708 | 15.7667 | 0.2403 | 0.3674 | 0.6553 | 0.6898 |
| w4a4 | 9.6424 | 15.8439 | 0.2369 | 0.3655 | 0.6635 | 0.6882 |
| w4a4_local | 9.6721 | 15.8306 | 0.2336 | 0.3662 | 0.6624 | 0.6981 |
| w4a4_no_lac | 9.8432 | 15.9031 | 0.2365 | 0.3639 | 0.6667 | 0.6927 |
| w4a4_no_lwc | 9.9387 | 15.9986 | 0.2436 | 0.3680 | 0.6516 | 0.6818 |
| w4a4_no_rot | 9.2164 | 15.8340 | 0.2350 | 0.3698 | 0.6710 | 0.7007 |
| w4a4_no_smooth | 9.6419 | 15.9555 | 0.2323 | 0.3650 | 0.6598 | 0.6773 |
| w4a4_rtn | 8.0677 | 14.1246 | 0.1942 | 0.3263 | 0.7243 | 0.7463 |

`*` = one per-channel least-squares gain+bias fitted per image over masked pixels, applied, clipped to
[0,1], then scored. One fit per image reused across all three metrics.

**Code change:** `run_w2a4_downstream.py` previously exposure-corrected PSNR only. SSIM carries a luminance
term and LPIPS is only partly exposure-robust, so two of three metrics remained confounded. All six metrics
are now computed, aggregated, paired-tested and printed raw beside corrected.

**PASS/FAIL:**
- Target: **complete the per-component attribution begun in #027.** All six arms measured. **PASS.**
- Target: **does `no_lac` overturn the #027 conclusion that rotation is the load-bearing component?**
  It does not: `no_lac` costs 1.28x camera rotation against rotation's 2.05x. **PASS, conclusion stands.**
- Target: **every metric corrected by the same transform.** **PASS.**

**Interpretation.** Learned *activation* clipping is the second-ranked single component (1.28x) while learned
*weight* clipping is inert (0.93x, at or below baseline) — so of QuantVGGT's four components, exactly two are
individually detectable at 4 bits, and the rotation is 1.6x more important than the runner-up. The
raw-versus-corrected split from #027 survives the added arm and the added metrics: every raw-PSNR contrast
that reached p < 0.05 (`no_rot` 0.0017, `no_smooth` 0.0125, `rtn` 0.0105) fails under correction, and the
first two *change sign*, i.e. corrected, those arms render marginally better than full precision. That sign
flip is not a real effect — it is noise around zero — but it is direct evidence that raw PSNR was
manufacturing the ordering rather than measuring it.

**Unexplained observation, flagged not resolved.** Exposure correction moves PSNR and SSIM *up* (SSIM
0.2403 → 0.3674 for `full`, a larger relative shift than PSNR's) but moves LPIPS *down in quality*
(0.6553 → 0.6898, higher = worse). The plausible cause is the clip to [0,1] after gain+bias destroying
highlight detail that LPIPS is sensitive to. **This is a hypothesis, not a measurement.** Exposure-corrected
LPIPS must not be used as a headline metric until an unclipped variant is tested against it.

**Next:** 40-scene run of all eight arms launched 08:43 (`run_40scene_exposure.sh`), to separate the two
distinct n=8 outcomes — a tight null (`full − no_rot`, −0.0673, CI [−0.5966, +0.4620]) from a large but
imprecise effect (`full − rtn`, +1.6421, CI [−1.7220, +5.0063]).

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/run_40scene_exposure.sh`
- `code/run_nolac_then_stage2.sh`
- `code/run_w2a4_downstream.py`

**results/**

- `results/arms_full_w4a4_w4a4_local_w4a4_no_lac_w4a4_no_lwc_w4a4_no_rot_w4a4_no_smooth_w4a4_rtn_subset_it7000.json`

**logs/**

- `logs/nolac_then_stage2.log.gz`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

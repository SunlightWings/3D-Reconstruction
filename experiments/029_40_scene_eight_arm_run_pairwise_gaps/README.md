# 029 — 40-scene, eight-arm run: pairwise gaps, dose-response, and the LPIPS clip test

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#029** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-10; 3DGS 08:43–15:56 CEST; LPIPS clip test 16:40–17:10
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/`
- **Commands executed:**
  ```
  ./code/quantization/run_40scene_exposure.sh
  #   -> run_w2a4_inference.py --variant <arm> --scenes all        (7 arms)
  #   -> ablation_compare.py --scenes all --tag all40 --variants <arms> w2a4
  #   -> run_w2a4_downstream.py --scenes all --iterations 7000 --arms <8 arms>
  python -u code/quantization/lpips_clip_test.py --scenes all --iterations 7000 \
      --arms full w4a4 w4a4_local w4a4_no_rot w4a4_rtn
  ```
- **Wall time:** 7 h 13 m; 192 trainings, 112 resume-skips, 0 failures

**Config:** identical to #028 except **Scenes used: all 40 of the frozen manifest** (sha256 `1ce2f3f8…968e06`).

**Geometry vs full precision (n = 40):**

| arm | cam rot (deg) | cam centre | spread | depth rel | point err |
|---|---:|---:|---:|---:|---:|
| w4a4 | 1.2754 | 0.02063 | 1.004 | 0.02255 | 0.06144 |
| w4a4_local | 1.1014 | 0.01896 | 1.002 | 0.02450 | 0.04987 |
| w4a4_no_lwc | 1.0799 | 0.01662 | 1.005 | 0.02201 | 0.05326 |
| w4a4_no_smooth | 1.1390 | 0.02034 | 0.991 | 0.02509 | 0.05840 |
| w4a4_no_lac | 1.3160 | 0.02286 | 1.002 | 0.02933 | 0.06447 |
| w4a4_no_rot | 2.0323 | 0.03665 | 0.996 | 0.04404 | 0.10293 |
| w4a4_rtn | 14.6565 | 0.12093 | 0.889 | 0.11145 | 0.46422 |

**Downstream, foreground (n = 40):**

| arm | PSNR | PSNR* | SSIM | SSIM* | LPIPS | LPIPS* |
|---|---:|---:|---:|---:|---:|---:|
| full | 8.9642 | 15.9047 | 0.2318 | 0.3744 | 0.6604 | 0.6895 |
| w4a4 | 8.8283 | 15.9085 | 0.2305 | 0.3739 | 0.6712 | 0.6952 |
| w4a4_local | 8.8004 | 15.8643 | 0.2278 | 0.3729 | 0.6683 | 0.6993 |
| w4a4_no_rot | 8.5326 | 15.9265 | 0.2278 | 0.3744 | 0.6749 | 0.6992 |
| w4a4_no_smooth | 8.9461 | 15.9213 | 0.2309 | 0.3753 | 0.6634 | 0.6913 |
| w4a4_no_lwc | 8.9986 | 15.9774 | 0.2318 | 0.3745 | 0.6616 | 0.6893 |
| w4a4_rtn | 7.9794 | 15.4445 | 0.2109 | 0.3618 | 0.7167 | 0.7327 |
| w4a4_no_lac | 8.8642 | 15.9000 | 0.2272 | 0.3741 | 0.6688 | 0.6989 |

**Paired, full minus arm, foreground (n = 40):**

| arm | raw PSNR | p | sign | corrected PSNR | p |
|---|---:|---:|---|---:|---:|
| w4a4 | +0.1359 | 0.09876 | 23/40 | −0.0038 | 0.966 |
| w4a4_local | +0.1638 | 0.04296 | 25/40 | +0.0404 | 0.503 |
| w4a4_no_rot | +0.4316 | 0.0001689 | 30/40 | −0.0218 | 0.7989 |
| w4a4_no_smooth | +0.0181 | 0.7953 | 19/40 | −0.0165 | 0.8609 |
| w4a4_no_lwc | −0.0344 | 0.6865 | 17/40 | −0.0727 | 0.37 |
| w4a4_rtn | +0.9848 | 7.508e-06 | 32/40 | +0.4602 | 0.1838 |
| w4a4_no_lac | +0.1000 | 0.1888 | 23/40 | +0.0047 | 0.9586 |

Full minus w4a4 on every metric: PSNR +0.1359 (p 0.099), SSIM +0.0013 (p 0.575), **LPIPS −0.0108 (p 0.00797,
28/40 favouring full)**, PSNR* −0.0038 (p 0.966), SSIM* +0.0005 (p 0.793), LPIPS* −0.0057 (p 0.206).

**Dose-response, Spearman vs 40-scene camera-rotation error over the six locally calibrated arms:**
LPIPS ρ = +0.9429 (p 0.004805); SSIM ρ = −0.8857 (p 0.01885); PSNR ρ = −0.8286 (p 0.04156);
LPIPS* ρ = +0.6571 (p 0.1562); PSNR* ρ = −0.4857 (p 0.3287); SSIM* ρ = −0.4857 (p 0.3287).

**LPIPS clip test (n = 40):** clipping to [0,1] after gain+bias alters 0.06–0.17 % of masked pixels, and
unclipped LPIPS equals clipped LPIPS to four decimals on every arm (e.g. full 0.6895 vs 0.6895).

**PASS/FAIL:**
- Target: **resolve the n = 8 ambiguity of `full − rtn` exposure-corrected (+1.6421, CI [−1.7220, +5.0063]).**
  At n = 40: +0.4602, CI [−0.2277, +1.1481]. The n = 8 estimate was inflated ~3.5x. **Resolved: small and bounded.**
- Target: **does W4A4 degrade rendering vs full precision at n = 40?** PSNR and SSIM, raw and corrected: no.
  Raw LPIPS: yes, p = 0.00797; Bonferroni over six metrics ≈ 0.048. **Marginal PASS on one metric only.**
- Target (#028 hypothesis): **is the LPIPS worsening under correction caused by the clip?** **FAIL — refuted.**
- Target: **do rendering metrics track geometry error across arms?** Raw metrics: all three at p < 0.05.
  Corrected: none. **PASS for raw, FAIL for corrected.**

**Interpretation.** At the configuration QuantVGGT ships, W4A4, quantization does not measurably degrade
novel-view rendering on PSNR or SSIM and only marginally on LPIPS; measurable degradation needs damage well
beyond anything the shipped method produces (`no_rot`, `rtn`). Rendering quality does follow geometry error
monotonically across the ladder, so the pipeline is not inert — but only raw metrics show it, and exposure
correction removes the relationship, i.e. it discards genuine geometry-driven signal. **This retracts the
#027/#028 recommendation that exposure-corrected metrics are "the column to trust"** for measuring geometry
degradation. **It also retracts #028's clip hypothesis:** the affine correction itself, not the clip, is what
worsens LPIPS, so exposure-corrected LPIPS is dropped rather than repaired.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/ablation_compare.py`
- `code/lpips_clip_test.py`
- `code/run_40scene_exposure.sh`
- `code/run_lpips_test_after.sh`
- `code/run_w2a4_downstream.py`
- `code/run_w2a4_inference.py`

**results/**

- `results/arms_full_w4a4_w4a4_local_w4a4_no_rot_w4a4_no_smooth_w4a4_no_lwc_w4a4_rtn_w4a4_no_lac_all_it7000.json`
- `results/lpips_clip_test_all_it7000.json`
- `results/lpips_clip_test_subset_it7000.json`
- `results/w4a4_ablation_all40.json`

**logs/**

- `logs/ablation_downstream_40.log.gz`
- `logs/ablation_geometry_40.log`
- `logs/infer40_w4a4.log`
- `logs/infer40_w4a4_local.log`
- `logs/infer40_w4a4_no_lac.log`
- `logs/infer40_w4a4_no_lwc.log`
- `logs/infer40_w4a4_no_rot.log`
- `logs/infer40_w4a4_no_smooth.log`
- `logs/infer40_w4a4_rtn.log`
- `logs/lpips_clip_test.log`
- `logs/lpips_clip_test_40.log`
- `logs/lpips_test_chain.log`
- `logs/metrics_check.log`
- `logs/run40_exposure.log.gz`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

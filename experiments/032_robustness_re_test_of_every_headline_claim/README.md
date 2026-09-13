# 032 — Robustness re-test of every headline claim (pass 1); the dose-response is weaker than reported

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#032** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11 13:56–13:57 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/cpu_probes.py`
- **Command executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py robustness --workers 8
  ```
- **Config:** no training. Inputs: #029 40-scene per-scene JSON (foreground, raw, 7000 it), #030 confidence-oracle
  JSON (n = 36), swap-probe JSON **as it stood at 13:56 (n = 29 of 40, interim)**, #031 per-scene camera errors.
  Per contrast: bootstrap 95 % CI (10,000 resamples, seed 0), sign-flip permutation p (20,000 flips), t and
  Wilcoxon p, leave-one-scene-out mean range and worst p, and the contrast with `bowl/70_5792_13401` removed.
  Dose-response: Spearman over the six locally calibrated arms, **exact** permutation p over all 720 orderings.
- **Scenes used:** frozen 40-scene manifest (swap: its first 29; confidence oracle: its first 36).

**Contrasts (positive = second arm renders worse):**

| contrast | n | mean | boot CI95 | t p | perm p | worst LOO p | no-bowl mean / p |
|---|---:|---:|---|---:|---:|---:|---|
| PSNR full − w4a4 | 40 | +0.1359 | [−0.0164, +0.2895] | 0.0988 | 0.0972 | 0.17 | +0.1408 / 0.095 |
| LPIPS w4a4 − full | 40 | +0.0108 | [+0.0033, +0.0182] | 0.00797 | 0.0077 | 0.0151 | +0.0111 / 0.008 |
| SSIM full − w4a4 | 40 | +0.0013 | [−0.0030, +0.0059] | 0.575 | 0.58 | 0.939 | +0.0015 / 0.53 |
| PSNR full − w4a4_local | 40 | +0.1638 | [+0.0184, +0.3160] | 0.043 | 0.0417 | 0.0799 | +0.1614 / 0.051 |
| LPIPS w4a4_local − full | 40 | +0.0079 | [+0.0019, +0.0141] | 0.0171 | 0.0176 | 0.032 | +0.0084 / 0.013 |
| PSNR full − w4a4_no_rot | 40 | +0.4316 | [+0.2374, +0.6371] | 0.000169 | 0.00025 | 0.000341 | +0.4073 / 0.00034 |
| LPIPS w4a4_no_rot − full | 40 | +0.0145 | [+0.0057, +0.0234] | 0.00285 | 0.0026 | 0.00544 | +0.0147 / 0.0033 |
| PSNR full − w4a4_rtn | 40 | +0.9848 | [+0.6222, +1.3575] | 7.51e-06 | 5e-05 | 1.62e-05 | +0.8877 / 5.7e-06 |
| LPIPS w4a4_rtn − full | 40 | +0.0563 | [+0.0363, +0.0778] | 6.61e-06 | 5e-05 | 1.43e-05 | +0.0512 / 6.5e-06 |
| PSNR conf-oracle C − R | 36 | −0.1276 | [−0.3028, +0.0372] | 0.154 | 0.162 | 0.301 | −0.1312 / 0.15 |
| LPIPS conf-oracle C − R | 36 | +0.0022 | [−0.0029, +0.0072] | 0.408 | 0.405 | 0.66 | +0.0022 / 0.41 |
| PSNR swap full − P (interim) | 29 | +0.1660 | [−0.1640, +0.5797] | 0.402 | 0.492 | 0.99 | +0.0014 / 0.99 |
| LPIPS swap full − P (interim) | 29 | +0.0061 | [−0.0090, +0.0276] | 0.541 | 0.739 | 0.639 | −0.0028 / 0.51 |
| PSNR swap full − K (interim) | 29 | +0.8771 | [+0.4934, +1.2767] | 0.000206 | 0.00015 | 0.000442 | +0.7985 / 0.00038 |
| LPIPS swap full − K (interim) | 29 | +0.0617 | [+0.0439, +0.0805] | 5.53e-07 | 5e-05 | 1.44e-06 | +0.0563 / 2.4e-07 |
| PSNR swap full − B (interim) | 29 | +1.1793 | [+0.7634, +1.6416] | 2.05e-05 | 5e-05 | 4.71e-05 | +1.0509 / 1.4e-05 |

(perm p of 5e-05 is the floor for 20,000 flips: no flip matched the observed mean.)

**Dose-response, exact p over 720 orderings:**

| metric | ρ | exact p | leave-one-arm-out ρ | no bowl: ρ / exact p |
|---|---:|---:|---|---|
| LPIPS | +0.9429 | 0.0167 | +0.900 … +1.000 | +0.8286 / 0.0583 |
| SSIM | −0.8857 | 0.0333 | −0.900 … −0.800 | −0.7143 / 0.1361 |
| PSNR | −0.8286 | 0.0583 | −1.000 … −0.700 | −0.8286 / 0.0583 |
| PSNR exposure-corrected | −0.4857 | 0.3556 | −0.700 … −0.100 | −0.4857 / 0.3556 |

**PASS/FAIL:**
- Target: **every headline pairwise claim survives permutation, bootstrap, leave-one-out and bowl removal.**
  W4A4 LPIPS (perm p 0.0077, CI excludes 0, worst LOO 0.0151, no-bowl 0.008): **PASS.** no_rot and rtn: **PASS.**
  W4A4 PSNR/SSIM null: **PASS (null robust).** Confidence-oracle null: **PASS (null robust).** Swap
  camera-vs-points (interim): **PASS**; points effect with bowl removed is +0.0014 dB, p 0.99.
  `w4a4_local` PSNR is fragile (worst LOO p 0.0799, no-bowl 0.051): **FAIL — not robust.**
- Target: **the dose-response survives an exact test.** LPIPS and SSIM yes (0.0167, 0.0333); PSNR no (0.0583);
  none of the three at p < 0.05 once the bowl scene is removed. **PARTIAL.**

**Correction to #029 and to statements made in conversation.** #029 reported the dose-response with scipy's
asymptotic Spearman p (LPIPS 0.004805, SSIM 0.01885, PSNR 0.04156). With n = 6 that approximation is too
optimistic. The exact values are LPIPS 0.0167, SSIM 0.0333, **PSNR 0.0583 (not significant)**, and none survive
removal of the bowl scene at 0.05. The claim "all three raw metrics track geometry error significantly" is
**withdrawn**; the licensed statement is that rendering quality is *consistent with* a monotone dependence on
geometry error (LPIPS, SSIM), resting on six arms.

**Interpretation.** The pairwise findings are robust by every test applied — W4A4's small LPIPS cost, the large
no_rot and rtn gaps, the confidence-oracle null, and (interim) cameras-not-points — and none depends on the
unstable bowl scene. The arm-level dose-response is the weak link: six points cannot carry a strong claim, which
is exactly why the synthetic damage sweep (running) exists.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/cpu_probes.py`

**logs/**

- `logs/cpu_robustness_pass1.log`

_No result file: this entry's numbers are in the entry text / logs only).

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

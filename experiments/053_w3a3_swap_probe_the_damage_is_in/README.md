# 053 — W3A3 swap probe: the damage is in the cameras (robust on LPIPS, not on PSNR at n = 8); points are null

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#053** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-12 23:13 – 2026-09-13 00:08 CEST (GPU), 55 min; finished before the 00:20 deadline
  (`stopped_at_deadline: false`) and the 00:35 shutdown.
- **Git commit:** `f0c8e32` + working tree (`geometry_probes.py` gains `--scene-set`)
- **Command executed:**
  ```
  python code/quantization/geometry_probes.py swap --arm w3a3 --scene-set subset \
         --iterations 7000 --deadline 00:20
  ```
- **Config:** the #039 construction applied to W3A3. All conditions are built in full precision's frame and
  rendered from full's held-out source: `swapP` = W3A3 points + full cameras, `swapK` = W3A3 cameras
  (pose + intrinsics) + full points, `swapB` = both from W3A3. Compared against the existing `full` renders.
- **Scenes used:** the 8 `EXPENSIVE_SCENES` (selection bias as stated in #050).

**Validity check — the swap construction reproduces the real arm.** `swapB` (both geometry channels from W3A3,
carried into full's frame) gives a mean gap of **+0.7691 dB**; the real W3A3 arm of #051 gives **+0.7828 dB**.
The `full` values used are identical to #051 on every scene (e.g. apple 12.6695, toaster 8.9490), so the two
runs share their reference renders. Per scene the two gaps agree in sign on 7/8; bowl diverges (+0.518 real,
−0.309 swapB). The probe therefore measures the same damage #051 measured.

**Per scene, full-minus-condition PSNR (positive = condition worse):**

| scene | full | swapP | swapK | swapB | Δ points | Δ cameras | Δ both |
|---|---:|---:|---:|---:|---:|---:|---:|
| apple/110 | 12.670 | 12.932 | 11.556 | 11.796 | −0.262 | **+1.114** | +0.873 |
| ball/123 | 11.224 | 10.577 | 9.545 | 9.477 | +0.647 | **+1.679** | +1.747 |
| bowl/70 | 10.458 | 11.181 | 9.403 | 10.767 | −0.723 | **+1.055** | −0.309 |
| broccoli/412 | 9.338 | 9.222 | 8.366 | 8.541 | +0.116 | **+0.972** | +0.797 |
| hydrant/167 | 8.131 | 8.028 | 8.343 | 8.496 | +0.103 | −0.212 | −0.365 |
| remote/350 | 9.656 | 9.741 | 9.669 | 9.398 | −0.085 | −0.013 | +0.259 |
| teddybear/187 | 9.341 | 9.465 | 8.098 | 8.048 | −0.124 | **+1.244** | +1.293 |
| toaster/372 | 8.949 | 7.537 | 9.187 | 7.092 | **+1.412** | −0.238 | +1.857 |

**Summary (condition vs full; positive = worse):**

| condition | PSNR | SSIM | LPIPS | Δ PSNR | t-test p | Wilcoxon p | worse | Δ LPIPS | t-test p | Wilcoxon p | worse |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| full | 9.9708 | 0.2403 | 0.6553 | | | | | | | | |
| swapP (points) | 9.8353 | 0.2388 | 0.6518 | +0.1355 | 0.571 | 0.945 | 4/8 | −0.0036 | 0.629 | 0.844 | 4/8 |
| **swapK (cameras)** | 9.2707 | 0.2299 | 0.6884 | **+0.7001** | **0.0319** | 0.109 | 5/8 | **+0.0330** | **0.00611** | **0.0078** | **8/8** |
| swapB (both) | 9.2017 | 0.2283 | 0.6860 | +0.7691 | 0.0387 | 0.0781 | 6/8 | +0.0307 | 0.0217 | 0.0391 | 7/8 |

SSIM: points +0.0015 (p 0.715), cameras +0.0104 (p 0.158), both +0.0119 (p 0.108) — no condition separates.
Decomposition of the joint loss: **cameras 91.0 % of PSNR, points 17.6 %**, interaction −0.0665 dB; on LPIPS
cameras account for 107.7 % (points marginally *improve* LPIPS). `swapB` 95 % CI on PSNR [+0.0529, +1.4853];
minimum detectable effect at 80 % power, n = 8: 0.849 dB PSNR, 0.0292 LPIPS.

**PASS/FAIL:**
- Target: **do W3A3's points damage rendering?** +0.1355 dB, p 0.571 (Wilcoxon 0.945), 4/8; LPIPS −0.0036,
  p 0.629. **FAIL — null on every metric under both tests**, as #050 predicted from the 0.101-radius point error.
- Target: **do W3A3's cameras damage rendering?** LPIPS +0.0330, **worse on 8/8 scenes**, t-test p 0.00611,
  Wilcoxon p 0.0078. **PASS on LPIPS, robust.** PSNR +0.7001, t-test p 0.0319 but **Wilcoxon p 0.109, worse
  on only 5/8. PASS on PSNR under the t-test only — not robust at n = 8.** The PSNR effect is carried by
  large drops (≥ +0.97 dB) on 5 scenes while 3 scenes are flat or slightly better.
- Target: **does W3A3 reproduce the `rtn` decomposition of #039?** Cameras 91.0 % of the joint PSNR loss here
  vs 85 % for `rtn` at n = 40. **PASS — same account, second arm.**

**Scene-level exceptions, reported rather than averaged away.** On **toaster** the pattern reverses: points
alone cost +1.412 dB and cameras alone *improve* PSNR by 0.238 — the only scene where the points carry the
damage. On **bowl** W3A3's points improve PSNR by 0.723 dB, and the combined condition is better than full
(−0.309) despite a +1.055 dB camera loss. On **hydrant** and **remote** neither channel does measurable damage,
consistent with W3A3 being close to full on those scenes in #051 (−0.226 and +0.208). With n = 8 these cannot
be separated from 3DGS training noise, and no claim is made about them.

**Interpretation.** This converts #050's inference into a direct measurement on W3A3 itself: its points are
harmless and its cameras carry the damage, reproducing #039's finding on a second, realistic arm. The strength
of that claim differs by metric and must be stated that way — **decisive on LPIPS** (8/8, both tests), and on
PSNR **supported by the mean and the t-test but not by the rank test**. Together with #052 it closes the case
against point-level intervention on W3A3 from both sides: pruning points by their true error recovers nothing
(#052), and replacing W3A3's points with full precision's recovers nothing either (+0.1355 dB, p 0.571). The
lever, if any exists, is the camera — which is the part 3DGS never optimises and which order-averaging could
not fix on `rtn` (#048). Whether it can on W3A3 remains blocked on the QS checkpoints (#050).

**Threat to validity.** n = 8, selected subset; the PSNR camera effect is not robust to a rank test, and the
MDE for PSNR at this n (0.849 dB) is larger than the effect itself. The 40-scene W3A3 predictions would
resolve this directly.

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/geometry_probes.py`

**results/**

- `results/geometry_probe_swap_w3a3_it7000.json`

**logs/**

- `logs/swap_w3a3.log.gz`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

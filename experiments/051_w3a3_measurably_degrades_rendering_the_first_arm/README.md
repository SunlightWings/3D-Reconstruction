# 051 — W3A3 measurably degrades rendering: the first arm to separate from W4A4

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#051** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-12 20:01–20:20 CEST (GPU), 18.7 min
- **Git commit:** `f0c8e32` + working tree
- **Command executed:**
  ```
  python code/quantization/run_w2a4_downstream.py --scenes subset --iterations 7000 \
         --arms full w4a4 w3a3
  ```
- **Config:** their 8 `w3a3.npz` were installed into the group directories and the sources, 3DGS training
  and scoring were **all rebuilt locally** by `build_variant_sources()` — the same code path that produced
  `full` and `w4a4`. Their own renders and their own results JSON were deliberately **not** used. 7000
  iterations, `--resolution 1`, foreground mask primary.
- **Scenes used:** the 8 `EXPENSIVE_SCENES` (selection bias as stated in #050).

**Arm means (foreground mask):**

| arm | PSNR | SSIM | LPIPS | content PSNR |
|---|---:|---:|---:|---:|
| full | 9.9708 | 0.2403 | 0.6553 | 9.8990 |
| w4a4 | 9.6424 | 0.2369 | 0.6635 | 9.8931 |
| **w3a3** | **9.1880** | **0.2280** | **0.6787** | **9.2106** |

**Paired tests over scenes (raw, foreground):**

| pair | PSNR | p | 95 % CI | SSIM | p | LPIPS | p |
|---|---:|---:|---|---:|---:|---:|---:|
| full − w3a3 | **+0.7828** | **0.00629** | [+0.3021, +1.2636] | +0.0123 | 0.0630 | **−0.0234** | **0.0266** |
| w4a4 − w3a3 | **+0.4544** | **0.0263** | [+0.0715, +0.8373] | +0.0089 | 0.157 | −0.0152 | 0.142 |
| full − w4a4 | +0.3284 | 0.0935 | [−0.0719, +0.7287] | +0.0033 | 0.488 | −0.0082 | 0.0628 |

Per-scene full-minus-w3a3 PSNR: apple +1.011, ball +1.419, bowl +0.518, broccoli +0.858, hydrant **−0.226**,
remote +0.208, teddybear +1.132, toaster +1.342 — 7/8 positive.

**Exposure-corrected columns show nothing:** full − w3a3 corrected PSNR +0.0962, p 0.718. This is the
expected behaviour, not a contradiction: #037 measured that the correction removes real geometry signal.
Corrected LPIPS does separate (−0.0314, p 0.0196). **Raw is cited throughout.**

**PASS/FAIL:**
- Target: **does W3A3 degrade rendering vs full precision?** +0.7828 dB, p 0.00629, CI excludes zero, 7/8
  scenes. **PASS.** First arm in this project to do so at a bit-width nobody deliberately broke.
- Target: **does W3A3 separate from W4A4?** +0.4544 dB, p 0.0263. **PASS.**
- Target: **does the W4A4 null reproduce at n = 8?** +0.3284 dB, p 0.0935 — consistent with the n = 40
  result (+0.1359, p 0.099, #029), inflated as expected by the small biased subset. **PASS as a control.**

**Correcting the number their pipeline reported.** Their results JSON gave W3A3 foreground PSNR 8.8469 on
these scenes, which paired against our `full` implies a gap of **+1.1239 dB**. Rebuilt locally the gap is
**+0.7828 dB** — theirs is 0.3411 dB larger, i.e. **44 % above** the like-for-like value. Their per-scene
bowl figure (10.916) also *beat* our full precision (10.458), which does not reproduce here (9.940, gap
+0.518). The cross-machine comparison was therefore materially optimistic and **must not be cited**; the
discrepancy is unexplained and is a second reason the preprocessing verification in #050 matters. Their
JSON also carried only `psnr_exposure_corrected` and not the SSIM/LPIPS corrected variants, i.e. it came
from a pre-#037 version of `run_w2a4_downstream.py` — their code has drifted from ours.

**Interpretation.** The prediction of #050 holds: the arm whose *cameras* cross into the damaging regime
while its *points* stay far below it is the arm that loses rendering quality. Together with #039/#043 this
is now a consistent account across four arms — `w4a4` (cameras inside tolerance → no damage), `w3a3`
(cameras straddle it → damage), `no_rot` and `rtn` (cameras far outside → large damage).

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/run_w2a4_downstream.py`
- `code/run_w3a3_chain.sh`

**results/**

- `results/arms_full_w4a4_w3a3_subset_it7000.json`
- `results/three_arm_subset.json`

**logs/**

- `logs/downstream_w3a3_subset_ourpipeline.log.gz`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

# 002 — Evaluation-region vs available-geometry audit (analysis, no training)

This is entry **#002** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 20:52 CEST
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 code/downstream_3dgs/oracle_sweep/ceiling.py
  ```
  (plus an inline `python3 -c` depth-validity probe using the same `common`/`dis` helpers; the probe is reproduced by the `depth_valid_in_BG` computation in that script's sibling analysis)

**Config:** No training. Pure measurement over the existing prepared scene assets.

| Field | Value |
|---|---|
| Init source | n/a (no 3DGS run) |
| Iterations | n/a |
| Densify | n/a |
| Input views | 6 (`input_frames` from the frozen manifest), 9 held-out |
| Mask type | `content` (the A3 metric region) vs `foreground`, compared |
| Background handling | n/a |
| Depth regularisation | n/a |

**Scenes used:** all 8 of the GPU subset for the ceiling computation; the depth-validity probe was run on the first 4 (apple, ball, bowl, broccoli).

**Results:**

Foreground fraction of the evaluated (content-mask) region, and the PSNR ceiling if the foreground were reconstructed *pixel-perfectly* while the background is filled with the best possible constant colour:

| scene | fg fraction of content mask | ceiling, bg = optimal constant | ceiling, bg = black |
|---|---:|---:|---:|
| apple/110_13051_23361 | 0.186 | 17.69 | 6.88 |
| ball/123_14363_28981 | 0.163 | 15.36 | 5.91 |
| bowl/70_5792_13401 | 0.232 | 17.34 | 7.50 |
| broccoli/412_56288_108844 | 0.080 | 12.44 | 4.84 |
| hydrant/167_18184_34441 | 0.131 | 13.20 | 5.82 |
| remote/350_36761_68623 | 0.106 | 20.74 | 7.22 |
| teddybear/187_20215_38541 | 0.192 | 22.84 | 5.61 |
| toaster/372_41229_82130 | 0.136 | 14.87 | 6.27 |
| **mean** | **0.153** | **16.81** | **6.26** |

GT depth validity, as a fraction of pixels (input frame 1 of each scene):

| scene | depth valid within content mask | depth valid within background |
|---|---:|---:|
| apple/110_13051_23361 | 0.039 | 0.001 |
| ball/123_14363_28981 | 0.107 | 0.001 |
| bowl/70_5792_13401 | 0.063 | 0.001 |
| broccoli/412_56288_108844 | 0.033 | 0.000 |

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Measured ceiling with a *pixel-perfect* foreground and an optimally chosen flat background: **16.81 dB mean**, with only 2/8 scenes (remote 20.74, teddybear 22.84) individually able to clear 20 dB. **FAIL — the target is unreachable under this metric by any 3DGS configuration that does not reconstruct the background.**

**Interpretation:** Rules out every configuration change as a route to 20 dB under the A3 content-mask metric, because 84.7% of the evaluated pixels are background for which CO3D supplies essentially no ground-truth depth (0.1% valid), so the oracle cannot be initialised there at all.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/ceiling.py`

**results/**

- `results/metric_ceiling.json`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

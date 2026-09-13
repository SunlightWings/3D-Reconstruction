# 001 — Backfill of pre-existing results (no new run)

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#001** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-07 (backfill; underlying runs completed 2026-08-23 13:48:28 +02:00 for the 40-scene main run, 2026-09-07 18:07:07 for the diagnostics)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; commit dated 2026-08-30, i.e. this is the tree state at backfill time, not necessarily the tree the runs executed on)
- **Command executed:** **UNRECORDED.** The runs predate this log and no command line was persisted; `results/downstream_3dgs/diagnostics/state.json` stores results only. The entry points were `code/downstream_3dgs/run_full_dataset_downstream.py` and `code/downstream_3dgs/diagnostics/run_all_diagnostics.py`, but their exact invocations and flags are not recoverable and are **not** reconstructed here.

**Config (as documented, not as observed at run time):**

| Field | Value |
|---|---|
| Init source | `full` and `w4a4` (main run); `oracle` and `random` additionally in A3/C1 |
| Iterations | 30000 default; C3 swept 500 / 1000 / 3000 / 7000 / 15000 / 30000 |
| Densify | On by default; C2 re-trained with `--densify_until_iter 0` |
| Input views | `--stride 4`, `--heldout-views 9` (defaults) |
| Mask type | `content` (rectangular) for the main aggregate; `foreground` (B1), `boundary_band` (B2), object/background split (B3) |
| Background handling | **UNRECORDED** — `gs_render_depth.py` selects white vs black via `dataset.white_background`; the value used for these runs was not persisted |
| Depth regularisation | **UNRECORDED** — no depth-regularisation flag is exposed by the runners; E2 renders depth for evaluation only |

**Scenes used:** 40 CO3D scenes for the main run and A1/A2/A4/A5/B1-B3/D1/D2/E1-E3. The 8-scene GPU subset for A3/C1/C2/C3: apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130.

**Results (40-scene main run, from `aggregate_metrics.json`):**

| Metric | Full | W4A4 | Full - W4A4 |
|---|---:|---:|---:|
| Mean PSNR (dB) | 9.9156 | 9.7117 | +0.2038 |
| Mean SSIM | 0.31485 | 0.31198 | +0.00287 |
| Mean LPIPS | 0.66245 | 0.67396 | -0.01151 (Full better) |

Median Full-minus-Quant PSNR +0.2322 dB; scenes favouring Full: 31/40 PSNR, 20/40 SSIM, 28/40 LPIPS. Status COMPLETE, 40/40 scenes.

Geometry metrics (40 scenes): Chamfer F-score @ tau 0.039 (Full) / 0.034 (W4A4). Floater mass 0.9768 / 0.9802. Mean Gaussian count 393,108 / 390,908.

**Per-scene PSNR/SSIM/LPIPS:** not reproduced here — they exist in `results/downstream_3dgs/aggregate_metrics.csv` and `results/per_scene.csv`. Per-scene oracle/random PSNR for the 8-scene subset is tabulated in `DIAGNOSTIC_RESULTS.md` (A3, C1). Copying them into this entry was avoided rather than risk transcription error; future entries must include them inline as the rule requires.

**PASS/FAIL:**

- Target: **A3 oracle >= 20 dB.** Measured **14.06 dB** mean (min 12.04 dB). **FAIL.**
- Target: **absolute PSNR above the A2 noise floor of 12.94 dB.** Measured **9.9156 dB** (Full). **FAIL.**
- Target: **A1 self-fit >= 15 dB.** Measured **50.92 dB** (Full) / 50.77 dB (W4A4), 0/40 below threshold. **PASS.**
- Target: **A5 LOO camera error well under 0.02 scene radii.** Measured **0.0066** (Full) / **0.0106** (W4A4). **PASS.**
- Target: **A4 camera-centre residual under the same 0.02.** Measured mean **0.0253**, max **0.1096**. **FAIL.**
- Target: **C1 ordering random < quant <= full with a visible margin.** Measured 13.48 / 13.87 / 13.96 dB, 4/8 scenes with random within 0.5 dB of full. **FAIL.**
- Target: **D2 non-zero disagreement-to-damage correlation.** Measured |rho| <= 0.229. **FAIL.**
- **D3: BLOCKED** — no W3A3/W2A4 calibration artifact exists; see open blockers.

**Interpretation:** Rules out the metric-masking and background-dilution explanations (Gate B) and rules in a harness-level failure — GT geometry itself cannot clear the noise floor (A3), so the +0.204 dB Full-vs-W4A4 delta cannot yet be read as a quality difference.

<!--
ENTRY TEMPLATE — copy for each new run.

### #NNN — <short title>

- **Date/time:** YYYY-MM-DD HH:MM TZ
- **Git commit:** <full hash> (branch <name>; dirty? yes/no)
- **Command executed:**
  ```
  <exact command line, copy-pasteable>
  ```

**Config:**

| Field | Value |
|---|---|
| Init source | full / w4a4 / oracle / random |
| Iterations | |
| Densify | |
| Input views | stride N, held-out M |
| Mask type | content / foreground / boundary_band / background |
| Background handling | |
| Depth regularisation | on / off |

**Scenes used:** <list, or named set + count>

**Results:** mean and per-scene PSNR / SSIM / LPIPS; plus Chamfer F-score, floater mass, Gaussian count where relevant.

| scene | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| | | | |
| **mean** | | | |

**PASS/FAIL:** Target: <target written out>. Measured: <value>. PASS/FAIL.

**Interpretation:** <one line: what this rules in or rules out>

Then rewrite DECISION STATE above.
-->

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/common.py`
- `code/evaluate_7k_convergence.py`
- `code/gate_a.py`
- `code/gate_b.py`
- `code/gate_c.py`
- `code/gate_d.py`
- `code/gate_e.py`
- `code/gs_render_depth.py`
- `code/render_report.py`
- `code/run_all_diagnostics.py`
- `code/run_downstream_validation.py`
- `code/run_full_dataset_downstream.py`
- `code/run_one_scene_timing.sh`

**results/**

- `results/DIAGNOSTIC_RESULTS.md`
- `results/aggregate_metrics.csv`
- `results/aggregate_metrics.json`
- `results/convergence_7k_vs_30k.csv`
- `results/convergence_7k_vs_30k.json`
- `results/figures/A4_reprojection_apple_110_13051_23361.png`
- `results/figures/A4_reprojection_ball_123_14363_28981.png`
- `results/figures/A4_reprojection_bowl_70_5792_13401.png`
- `results/figures/A4_reprojection_broccoli_412_56288_108844.png`
- `results/figures/A4_reprojection_hydrant_167_18184_34441.png`
- `results/per_scene.csv`
- `results/state.json`

**logs/**

- `logs/orchestrator.log.gz`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

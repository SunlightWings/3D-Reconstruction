# 025 — W2A4 calibrated locally; the pose head collapses (unblocks D3 at 2 bits)

This is entry **#025** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-08 15:08-18:36 CEST (calibration 209.6 min), evaluated 18:40-19:00
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/`
- **Commands executed:**
  ```
  python3 -u code/quantization/calibrate_w2a4.py
  python3 code/quantization/quant_loader.py w2a4
  python3 -u code/quantization/run_w2a4_inference.py --scenes subset
  ```

**Config:**

| Field | Value |
|---|---|
| Init source | `w2a4` — VGGT-1B quantized to 2-bit weights / 4-bit activations |
| Quantization | QuantVGGT quarot smooth+rotation, learned weight AND activation clipping (lwc+lac), symmetric |
| Calibration | authors' filtered CO3D set, 42 samples, 15 epochs/block, batch 1, 48 blocks (24 frame + 24 global) |
| Calibration provenance | **locally calibrated** (the shipped W4A4 is the authors' HuggingFace release) |
| Iterations / densify / views / mask / background / depth reg | n/a — no 3DGS was run; this entry is VGGT-level only |

**Scenes used:** the 8-scene GPU subset, one six-view group each (the group the downstream pipeline selects).

**Results — calibration itself converged.** 48/48 blocks, 209.6 min, artifacts written:
`qs_frame_parameters_total.pth` and `qs_global_parameters_total.pth`, 1.78 GiB each. Per-block reconstruction
error fell within every block (block 0: 0.00267 -> 0.00078). But it degrades sharply with depth:
first blocks ~0.0008, last blocks ~0.34, a ~440x rise across the network. For contrast the W4A4 calibration
running under identical settings goes 0.0000086 -> 0.0005, a ~50x rise.

**Results — the model is broken. Per-scene camera statistics:**

| scene | cam spread full | cam spread w2a4 | ratio | max pair-rot w2a4 (deg) | LOO cam err full | LOO cam err w2a4 |
|---|---:|---:|---:|---:|---:|---:|
| apple/110_13051_23361 | 0.5204 | 0.0113 | 0.022 | 2.6 | 0.1441 | 2.7158 |
| ball/123_14363_28981 | 0.9419 | 0.0082 | 0.009 | 1.5 | 0.0760 | 6.0907 |
| bowl/70_5792_13401 | 0.4640 | 0.0038 | 0.008 | 1.0 | 0.4549 | 5.5326 |
| broccoli/412_56288_108844 | 0.7118 | 0.0045 | 0.006 | 1.7 | 0.0826 | 10.2075 |
| hydrant/167_18184_34441 | 0.4375 | 0.0046 | 0.010 | 2.0 | 0.0449 | 3.2489 |
| remote/350_36761_68623 | 0.6093 | 0.0070 | 0.011 | 1.2 | 0.0646 | 4.8738 |
| teddybear/187_20215_38541 | 0.5535 | 0.0032 | 0.006 | 2.6 | 0.0588 | 5.0599 |
| toaster/372_41229_82130 | 0.8439 | 0.0077 | 0.009 | 1.9 | 0.1821 | 9.5981 |
| **mean** | 0.6353 | 0.0063 | **0.010** | 1.8 | 0.0793 | 5.2963 |

Full precision places the six input cameras across a wide orbit (max pairwise rotation 73.7 deg on apple);
W2A4 places them all within a few degrees of each other. Depth degrades far more gracefully -- median depth
0.85 vs full's 0.77-1.03, world-point spread compressed to ~60% -- so the failure is specific to the POSE head,
not global.

Inference cost, measured: 2.25 s/scene mean, 15,384 MiB peak (identical to W4A4, since both are simulated
quantization over full-precision matmuls).

**PASS/FAIL:** Target: **W2A4 should produce geometry degraded enough to measure but still usable**, so that a
confidence predictor has something to repair. Measured: camera-centre spread collapses to **1.0%** of full
precision and leave-one-out camera error rises to **5.296** scene radii against full's
**0.079**. **FAIL — the arm is collapsed, not degraded.**

**Interpretation:** Rules out W2A4 as an experimental setting for the confidence-predictor idea, and rules out
"just quantize harder" as the fix for the unresolvable W4A4 result (#020): 2-bit does not produce a larger
measurable gap, it produces a model that has stopped doing pose estimation. No 3DGS was run on this arm on
purpose -- the resulting gap would measure model collapse, not a quantization effect a per-pixel confidence
could repair.

**Also unblocks D3 partially:** DIAGNOSTIC_RESULTS.md marks D3 (harsher quantization) BLOCKED for want of a
calibrated artifact. That artifact now exists for W2A4. D3 remains BLOCKED for W3A3.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/calibrate_w2a4.py`
- `code/merge_qs_layers.py`
- `code/quant_loader.py`
- `code/run_w2a4_downstream.py`
- `code/run_w2a4_inference.py`

**logs/**

- `logs/calibrate.log`

_No result file: this entry's numbers are in the entry text / logs only).

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

# 022 — Exposure check: how much of the oracle's 14.83 dB is photometric, not geometric?

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#022** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-08 10:35 CEST
- **Git commit:** `231d1ea`
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/exposure_check.py
  ```

**Config:** No training, no rendering. Two probes over the existing oracle held-out renders.

| Field | Value |
|---|---|
| Init source | `oracle` |
| Iterations | 30000 (as trained) |
| Densify | default (as trained) |
| Input views | 6 input, 9 held-out |
| Mask type | `foreground` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset.

**Probes:** (a) score each held-out GT frame against the training GT frame whose camera centre is nearest — what a
method scores by copying the closest real photograph and modelling nothing; (b) fit one per-channel gain+bias per image
by least squares between the oracle render and the GT over the masked pixels, then re-score — whatever this recovers was
exposure/white-balance mismatch, not geometry. CO3D is captured with auto-exposure, so this is a real effect.

**Results (foreground PSNR):**

| scene | oracle | oracle + gain/bias | photometric headroom | nearest-pose GT frame |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 17.332 | 19.727 | +2.395 | 17.824 |
| ball/123_14363_28981 | 14.907 | 17.916 | +3.009 | 15.809 |
| bowl/70_5792_13401 | 17.449 | 19.187 | +1.738 | 16.981 |
| broccoli/412_56288_108844 | 15.087 | 20.341 | +5.254 | 14.674 |
| hydrant/167_18184_34441 | 12.900 | 18.331 | +5.431 | 16.590 |
| remote/350_36761_68623 | 11.795 | 14.251 | +2.456 | 12.109 |
| teddybear/187_20215_38541 | 15.680 | 20.770 | +5.090 | 15.501 |
| toaster/372_41229_82130 | 13.474 | 16.435 | +2.962 | 13.823 |
| **mean** | **14.828** | **18.370** | **+3.542** | **15.414** |

**PASS/FAIL:**
- Target: **quantify the photometric share of the oracle's score.** A single global gain+bias per image recovers
  **+3.542 dB** (14.828 -> 18.370), so that much of the oracle's deficit was never a geometry error.
- Target: **the oracle should beat the trivial nearest-photograph baseline.** It does not: nearest-pose GT scores
  **15.414 dB** against the oracle's **14.828 dB**, i.e. the oracle is **0.586 dB WORSE** than copying the
  nearest training photograph, and it wins on only 3/8 scenes. **FAIL.**

**Interpretation:** Rules out reading the oracle's 14.83 dB as a measure of reconstruction quality at all. Two
independent reasons: 3.54 dB of it is a per-image exposure offset that any global colour correction removes, and
the whole arm is beaten by a baseline that does no reconstruction whatsoever — copying the nearest training frame.
Once exposure is normalised the oracle reaches 18.370 dB, which is finally above the nearest-photo baseline and
close to the 19.430 dB A3 target from entry #002's derivation.

**This changes the diagnosis of A3.** The oracle's apparent failure is substantially photometric, and the harness does
not currently control for auto-exposure. Either the metric should fit a per-image gain+bias before scoring (standard
practice for NVS on auto-exposure capture, and what GraphDECO's own `--train_test_exp` option exists for), or the
comparison should be restricted to exposure-stable scenes. Note the existing pipeline writes an `exposure.json` per
model but the evaluation path does not apply any exposure correction.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/exposure_check.py`

**results/**

- `results/exposure_check.json`

**logs/**

- `logs/exposure.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

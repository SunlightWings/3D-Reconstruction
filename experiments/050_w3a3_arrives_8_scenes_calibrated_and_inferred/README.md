# 050 — W3A3 arrives (8 scenes, calibrated and inferred externally): pose 2.69°, focal 0.102, points 0.101

This is entry **#050** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-12 19:58–20:00 CEST (transfer), 20:05–20:12 CEST (verification + geometry, CPU)
- **Git commit:** `f0c8e32` + working tree
- **Provenance:** W3A3 was calibrated **and** inferred by Sharma Poudel / Prabin (UTN) on their machine.
  What arrived is `w3a3_8scenes_export.zip`, 1,065,794,236 bytes,
  sha256 `1f2c21c60f8f9504f80bd627f8eec6f97e91671d5dab55612fac76735e25b34a`, containing per scene one
  `w3a3.npz` (25,760,956 B) and one trained `point_cloud.ply`. **The QS checkpoints were NOT sent**, so
  nothing here can be re-inferred locally — see "What this does not unblock" below.
- **Command executed:**
  ```
  # contract check + geometry, both local, no GPU
  python - <<'…'   # shapes, frame numbers, finiteness vs our full.npz
  python - <<'…'   # pose_err / focal_err / point_err vs full and vs CO3D GT
  ```
- **Config:** pose error = mean per-camera rotation angle against full precision's saved cameras after the
  Sim(3) fitted on the six input camera centres — identical construction to #039/#043/#045. Focal error =
  median |K_arm/K_full − 1| over both axes and all six cameras. Point error = median ‖p_arm→full − p_full‖
  normalised by the scene radius, same Sim(3).
- **Scenes used:** the 8 `EXPENSIVE_SCENES`. These are **not** a random sample — they were selected in
  #002 to span categories *and both signs* of the full-minus-quant PSNR delta, which biases any effect
  size estimated on them. Stated here because every W3A3 number in #050–#052 inherits it.

**Contract check — all 8 pass.** Array shapes match the prediction contract
(`extrinsic` (6,3,4), `intrinsic` (6,3,3), `depth` (6,518,518,1), `world_points_from_depth` (6,518,518,3)),
all finite, and the frame numbers are **identical to our `full` arm** on every scene
(`[1,35,69,103,137,170]`, and `[1,35,69,102,135,168]` for hydrant).

**Geometry vs full precision:**

| scene | pose w4a4 | pose rtn | **pose w3a3** | focal w4a4 | focal rtn | **focal w3a3** | pts w4a4 | pts rtn | **pts w3a3** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| apple/110 | 0.718° | 14.875° | **2.865°** | 0.0449 | 0.3641 | **0.2909** | 0.0386 | 0.5421 | **0.1560** |
| ball/123 | 0.617° | 6.118° | **1.515°** | 0.0126 | 0.0536 | **0.0547** | 0.0218 | 0.2304 | **0.0388** |
| bowl/70 | 4.779° | 165.510° | **7.402°** | 0.1632 | 0.0445 | **0.0953** | 0.1879 | 3.4859 | **0.2271** |
| broccoli/412 | 0.681° | 4.473° | **1.934°** | 0.0173 | 0.1349 | **0.1128** | 0.0214 | 0.1782 | **0.0806** |
| hydrant/167 | 0.772° | 2.203° | **1.860°** | 0.0284 | 0.0695 | **0.0251** | 0.0345 | 0.2173 | **0.0680** |
| remote/350 | 1.335° | 4.198° | **2.011°** | 0.0089 | 0.1883 | **0.1444** | 0.0350 | 0.2591 | **0.1045** |
| teddybear/187 | 1.314° | 8.097° | **2.094°** | 0.0017 | 0.0989 | **0.0172** | 0.0434 | 0.3351 | **0.0714** |
| toaster/372 | 1.167° | 4.661° | **1.830°** | 0.0173 | 0.1478 | **0.0715** | 0.0286 | 0.2135 | **0.0592** |
| **mean** | 1.4230° | 26.2667° | **2.6887°** | 0.0368 | 0.1377 | **0.1015** | 0.0514 | 0.6827 | **0.1007** |
| **median** | 0.9699° | 5.3893° | **1.9725°** | 0.0173 | 0.1169 | **0.0834** | 0.0347 | 0.2447 | **0.0760** |

W3A3 is worse than W4A4 on **8/8** scenes for pose (Wilcoxon p 0.0078) and **8/8** for points (p 0.0078);
focal is worse on 6/8 (p 0.1094, not significant at n = 8). Full precision's own pose error against CO3D
ground truth on these scenes is 1.933° — i.e. **W3A3's 2.69° error relative to full is larger than full's
entire error relative to ground truth.**

**PASS/FAIL:**
- Target: **is W3A3 outside the ~2–4° rotation tolerance of #042?** Mean 2.69°, median 1.97°, max 7.40°;
  5/8 scenes above 2°. **PASS at the mean, marginal at the median — it straddles the threshold**, unlike
  W4A4 (1.42° mean) which sits below it.
- Target: **are W3A3's points in the damaging regime?** 0.101 scene radii, against #042's finding of no
  PSNR loss up to 1.0. **FAIL — an order of magnitude below the level at which points matter.**
- Target: **is the focal error in the damaging regime?** 0.1015, versus `rtn`'s 0.1377 which cost
  +0.4876 dB on its own (#043). **PASS — 74 % of a damage level already measured.**

**Interpretation.** W3A3 is the arm #049 said was needed: single-run error above the threshold where
rendering starts to care. The decomposition predicts, before any rendering was measured, that its damage
must be **camera-borne** — the points are far inside the harmless regime while pose straddles the tolerance
line and focal sits at three quarters of a known-damaging level. #051 tests that prediction downstream.

**What this does not unblock.** The QS checkpoints (`qs_frame_parameters_total.pth`,
`qs_global_parameters_total.pth`, `ablation_config.json`) were not transferred, so W3A3 cannot be run
locally. That blocks every experiment requiring *new* inference — specifically extending the
order-ensembling detector (#045) and corrector (#047) to W3A3, which #049 named as the case where the
machinery could plausibly pay off. It remains blocked.

**Threat to validity — not resolved.** `input_semantic_sha256` in the `w3a3_meta.json` files written here
is **copied from our own `full_meta.json`, not independently verified**: their meta files were not sent, so
we cannot confirm their preprocessing produced byte-identical 518×518 pad-mode inputs. Frame numbers match,
which is necessary but not sufficient. `w3a3_compare_contract.json` (40 scenes, sha256 of every train and
held-out GT PNG) was written for them to check against and has not been returned.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/run_w3a3_chain.sh`

**results/**

- `results/w3a3_compare_contract.json`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

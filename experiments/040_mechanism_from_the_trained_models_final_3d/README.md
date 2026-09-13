# 040 — Mechanism from the trained models: final 3D accuracy and render quality are decoupled

> **⚠ Rendering numbers in this entry are INVALID.** Every 3DGS render before 2026-09-13 used cameras rotated by Rᵀ instead of R (quaternion-conjugate bug in `rotmat2qvec`, see [054](../054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md)). PSNR / SSIM / LPIPS below are kept as the historical record only. Geometry-only numbers (depth, point error, pose and focal error) never went through the renderer and are unaffected.

This is entry **#040** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11; opacity > 0.5 run 15:09 (in `run_cpu_chain.sh`), opacity > 0.1 re-run finished 21:09:03 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py gaussians`)
- **Commands executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/cpu_probes.py gaussians --workers 8
  GS_OPACITY_MIN=0.1 nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py gaussians --workers 8
  ```
- **Config:** no training. Reads every trained 7000-iteration PLY for full, w4a4, w4a4_rtn, conf50/rand50, and the
  three swap cells, plus their initial `points3D.txt`. CO3D's GT `pointcloud.ply` (≤ 60,000 points) carried into
  full's frame by Sim(3) from GT to full input camera centres (sanity: median GT → full-init distance 0.011 radii).
  Arm-frame models carried into full's frame by their own Sim(3). "Surface" Gaussians = sigmoid(opacity) above the
  threshold (≈ 2–3 % of Gaussians at 0.5). **Accuracy** = median distance surface → GT inside the GT bounding box
  (+10 %); **completeness** = median distance GT → surface; scene radii. Two scenes (`bowl/70_5792_13401`,
  `toytrain/240_25394_51994`) have no surface Gaussians inside the GT box for some models and are excluded from
  the accuracy pairs (n = 38). The first run's printed paired means were NaN for this reason; the tests below were
  recomputed NaN-aware from `cpu_gaussians*.json`.
- **Scenes used:** all 40 (swap cells 40; conf/rand 36).

**Paired tests (median difference, Wilcoxon p, positive = first worse):**

| comparison | opacity > 0.5 | opacity > 0.1 |
|---|---|---|
| swapP final accuracy − full final accuracy | +0.0320, p 1.51e-09, 36/38 | +0.0292, p 1.46e-11, 37/38 |
| swapK final accuracy − full final accuracy | +0.0022, p 0.335, 23/38 | +0.0028, p 0.0844, 24/38 |
| swapP final completeness − full | +0.0264, p 7.28e-12, 38/38 | +0.0161, p 7.28e-12, 38/38 |
| swapK final completeness − full | +0.0043, p 3.25e-05, 30/39 | +0.0020, p 4.82e-06, 31/39 |
| swapP final accuracy − swapP initial accuracy | −0.0008, p 0.688, 18/38 | −0.0047, p 0.22, 16/38 |

Initial-vs-final *completeness* is **not** compared: the final surface set is far sparser than the 101,400-point
initialisation, so that difference would measure density, not geometry.

**PASS/FAIL:**
- Hypothesis stated in conversation and in this file's `gaussians` docstring — *"3DGS moves bad points to the right
  place, which is why bad points do not hurt rendering"* — **FAIL, refuted at both thresholds**: training does not
  improve bad-point accuracy (p 0.688 / 0.22), and the bad-point model ends with clearly worse 3D geometry than full.
- Target: **does final 3D accuracy explain the rendering results of #039?** No. swapP: worse geometry, renders like
  full. swapK: geometry like full, renders ~1 dB worse. **Dissociation, robust to the threshold.**

**Interpretation.** In this 6-view CO3D setting, held-out rendering quality is **decoupled from the 3D accuracy
of the reconstruction**: it is governed by whether the training cameras are right, not by whether the Gaussians
are in the right place. That explains three earlier results at once — the oracle losing to a nearest-photograph
baseline (#022), point-level confidence recovering nothing (#030), and bad points costing nothing (#039) — and
it means rendering PSNR/SSIM/LPIPS on this benchmark cannot, by construction, reward better geometry priors. It
is the strongest single finding for the experimental report.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/cpu_probes.py`
- `code/run_cpu_chain.sh`

**results/**

- `results/cpu_gaussians.json`
- `results/cpu_gaussians_op0.1.json`

**logs/**

- `logs/cpu_chain.log`
- `logs/cpu_gaussians.log`
- `logs/cpu_gaussians_op0.1.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

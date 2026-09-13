# 057 — Old 40 scenes re-trained with corrected cameras (full and W4A4, 80 runs)

_No `ongoing_logs.md` entry exists for this work yet. Every number below is read directly from the files in `results/`._

**What.** The frozen 40-scene manifest, full-precision VGGT and W4A4, each re-trained with GraphDECO 3DGS on the corrected cameras (054). Same configuration as before the fix: 6 input views, 9 held-out views, stride-4 point init, 518 pad preprocessing, 7000 iterations, `--data_device cpu --resolution 1`.

**Command:** `python -u code/run_qfix_all.py --old40 --arms full w4a4 --source-mode qfix --tag old40`

**Outputs:** 40 scenes, 80 scene-arms. Each has the VGGT prediction npz, its meta json, the 7000-iteration `point_cloud.ply`, and 9 held-out renders; every path is listed in `results/outputs_index_old40.json`. All were checked to exist, and each run has exactly 9 renders.

**Not in git (size).** PLYs and npz are in OneDrive `CV/full & w4a4 outputs.zip` (layout `full & w4a4 outputs/<category>/<sequence>/<arm>/{<arm>.npz, <arm>_meta.json, point_cloud.ply}`).

**Not done yet:** these 80 renders have not been scored. There is no PSNR / SSIM / LPIPS for them in this folder.

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/build_qfix_sources.py`
- `code/make_upload_archives.sh`
- `code/run_qfix_all.py`

**results/**

- `results/outputs_index_old40.json`

**logs/**

- `logs/all88.log.gz`
- `logs/archives.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

# 056 — Focal-length corrector and gsplat pose refinement: ABANDONED

_No `ongoing_logs.md` entry exists for this work yet. Every number below is read directly from the files in `results/`._

**Idea.** W3A3 / W4A4 damage lives mainly in the cameras (#039, #043, #053). Plan B was to correct the focal length from the prior itself, then refine pose inside a differentiable rasteriser (gsplat).

**Focal corrector** — mean focal error before and after, per method (from `results/focal_corrector_*.json`):

| file | n | before | after: oracle | after: const | after: iqr |
|---|---:|---:|---:|---:|---:|
| focal_corrector_w3a3_subset.json | 8 | 0.1015 | 0.0000 | 0.0690 | 0.0565 |
| focal_corrector_w4a4_rtn_all.json | 40 | 0.1865 | 0.0000 | 0.0753 | 0.0756 |

**gsplat pose refinement: ABANDONED.** Prebuilt wheels exist only for Python 3.10; the CUDA extension build from source failed (see `logs/gsplat_install*.log`). `code/gsplat_posefix.py` was written but never produced a result. No downstream number exists for this route.

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/focal_corrector.py`
- `code/gsplat_posefix.py`
- `code/run_focalfix_gate.sh`

**results/**

- `results/focal_corrector_w3a3_subset.json`
- `results/focal_corrector_w4a4_rtn_all.json`

**logs/**

- `logs/downstream_w3a3_focalfix.log`
- `logs/downstream_w3a3_focalfix2.log`
- `logs/focal_corrector_w3a3.log`
- `logs/focal_corrector_w4a4_rtn.log`
- `logs/focalfix_gate.log`
- `logs/gsplat_install.log`
- `logs/gsplat_install2.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

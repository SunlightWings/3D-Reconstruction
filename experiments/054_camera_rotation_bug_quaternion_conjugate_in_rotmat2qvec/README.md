# 054 — Camera rotation bug: quaternion conjugate in rotmat2qvec, fixed; corrected baselines

_No `ongoing_logs.md` entry exists for this work yet. Every number below is read directly from the files in `results/`._

**What happened.** `rotmat2qvec` in `code/run_downstream_validation.py` built the quaternion with the antisymmetric row of the K matrix negated. It therefore wrote the **conjugate** quaternion, so GraphDECO 3DGS trained and rendered every non-identity camera with rotation Rᵀ instead of R. Every rendering result produced before this fix (entries #001–#053 that report PSNR / SSIM / LPIPS) is affected. Geometry-only results are not.

**Fix.** One row of K (see `code/run_downstream_validation.py`, comment "sign fixed 2026-09-13"). `code/build_qfix_sources.py` rewrites existing COLMAP sources (`sources/<arm>` → `sources/<arm>_qfix`) by negating qx, qy, qz, and checks the result against the prediction npz. `code/qfix_downstream.py` retrains and re-renders; `code/score_arms.py` scores.

**Result — same 8 scenes, 7000 iterations, foreground mask, before vs after the fix** (from `results/qfix_baseline_full_w3a3_subset_it7000.json`, n = 8):

| arm | PSNR before fix | PSNR after fix | SSIM after | LPIPS after |
|---|---:|---:|---:|---:|
| full | 9.971 | 14.574 | 0.3195 | 0.3464 |
| w3a3 | 9.188 | 12.983 | 0.2485 | 0.4390 |

**Per scene (foreground PSNR):**

| scene | full before | full after | w3a3 before | w3a3 after |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 12.670 | 18.121 | 11.658 | 14.551 |
| ball/123_14363_28981 | 11.224 | 13.970 | 9.805 | 13.235 |
| bowl/70_5792_13401 | 10.458 | 15.166 | 9.940 | 12.678 |
| broccoli/412_56288_108844 | 9.338 | 15.411 | 8.480 | 14.265 |
| hydrant/167_18184_34441 | 8.131 | 13.115 | 8.356 | 12.299 |
| remote/350_36761_68623 | 9.656 | 14.370 | 9.448 | 11.276 |
| teddybear/187_20215_38541 | 9.341 | 17.583 | 8.209 | 16.358 |
| toaster/372_41229_82130 | 8.949 | 8.852 | 7.607 | 9.204 |

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/build_qfix_sources.py`
- `code/qfix_downstream.py`
- `code/run_downstream_validation.py`
- `code/run_downstream_validation.uncommitted.diff`
- `code/score_arms.py`

**results/**

- `results/qfix_baseline_full_w3a3_subset_it7000.json`

**logs/**

- `logs/qfix_baseline.log.gz`
- `logs/qfix_downstream_subset.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

# 024 — LLFF unblocking attempt: tooling resolved, benchmark data still BLOCKED

This is entry **#024** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-08 11:15 CEST
- **Git commit:** `231d1ea`
- **Commands executed:**
  ```
  python3 -m pip install pycolmap          # -> pycolmap 4.2.0 installed
  python3 -m pip install gdown
  python3 code/downstream_3dgs/llff/run_colmap_sfm.py llff_fern
  python3 code/downstream_3dgs/llff/run_colmap_sfm.py llff_flower
  curl -sSI -L https://people.eecs.berkeley.edu/~bmild/nerf/nerf_llff_data.zip
  python3 -c "import gdown; gdown.download('https://drive.google.com/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g', ...)"
  curl -s https://huggingface.co/api/datasets?search=llff
  ```

**Status: PARTIALLY UNBLOCKED. The LLFF downstream arm remains BLOCKED on data.**

**Resolved — SfM tooling.** `pycolmap` 4.2.0 installs from PyPI and works: `extract_features`,
`match_exhaustive` and `incremental_mapping` are all available (CPU only, `has_cuda=False`). Verified end to end on
`llff_fern`: **20/20 images registered, 2,959 points, 2 seconds**. A COLMAP reference arm is therefore constructible
in this environment, and `code/downstream_3dgs/llff/run_colmap_sfm.py` does it.

**Not resolved — the 8-scene benchmark.** Every canonical source failed:

| source | result |
|---|---|
| `people.eecs.berkeley.edu/~bmild/nerf/nerf_llff_data.zip` | HTTP 404 (redirects, then not found) |
| Google Drive `16VnMcF1KJYxN9QId6TClMsZRahHNMW5g` | 303; `gdown` refused — Drive blocks programmatic access to this file |
| Hugging Face `nielsr` / `dsrivastavv` / `pcuenq` mirrors | HTTP 401 |
| Hugging Face dataset search for "llff" | only `srikanthrangan/llffilters` and `poptree/LLFF`; the latter is `gated: manual` (401 without an accepted licence) and is actually mipnerf360 data — it contains no fern/fortress/horns/leaves/orchids/trex |

**Additional finding on the demo scenes:** of the two VGGT demo scenes, only `llff_fern` reconstructs.
`llff_flower` (25 images) fails SfM outright — every initialisation attempt is discarded with
"no initial pair", so it yields no reconstruction at all.

**PASS/FAIL:** Target: **obtain the real 8-scene LLFF benchmark with `poses_bounds.npy`, and install COLMAP or
pycolmap.** Tooling: **PASS**. Data: **BLOCKED** — 1 usable scene (fern) versus the 8 required, and it is a demo
asset rather than the benchmark, so the "published sparse-view range" gate still cannot be evaluated.

No pose source was substituted, and VGGT poses were again not used as the reference arm.

**Next step:** the benchmark needs a human with a browser (Drive's confirm page) or an accepted Hugging Face licence;
alternatively any mirror serving `nerf_llff_data.zip` over plain HTTP. Once the images and `poses_bounds.npy` are on
disk, `run_colmap_sfm.py` already covers the reconstruction half, and only the source builder (3/6/9 views, every 8th
frame held out, whole-image metrics) remains.

**Interpretation:** Rules out tooling as the LLFF blocker and localises it entirely to dataset access — worth
recording because the previous entry (#019) attributed the block partly to missing COLMAP, which is no longer true.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/README.md`
- `code/run_colmap_sfm.py`

_No result file: this entry's numbers are in the entry text / logs only (BLOCKED run).

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

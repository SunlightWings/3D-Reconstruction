# 019 — LLFF downstream pipeline: BLOCKED

This is entry **#019** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-08 09:45 CEST
- **Git commit:** `b838e3a`
- **Command executed:** environment probes only —
  ```
  which colmap; ls /usr/bin/colmap /usr/local/bin/colmap /opt/*/bin/colmap
  python3 -c "import pycolmap"
  find /var/tmp/luli38se /home/utn/luli38se -iname "*llff*"
  find /var/tmp/luli38se/vggt_official -name "sparse" -o -name "cameras.txt" -o -name "*.npy"
  ```

**Status: BLOCKED.**

**Config:** n/a — nothing was run, and no unverifiable code was written.

**Scenes available:** only `llff_fern` (20 images) and `llff_flower` (25 images) at 706x529, under
`/var/tmp/luli38se/vggt_official/examples/`. These are VGGT demo inputs, not the LLFF benchmark (8 scenes).

**Results:** none. The requested design gates running `full`/`w4a4` on a COLMAP reference arm landing in published
sparse-view range, and that reference arm cannot be constructed here:

1. No COLMAP binary on PATH, in `/usr/bin`, `/usr/local/bin` or `/opt/*/bin`; `pycolmap` not installed. The only
   COLMAP-related file present is GraphDECO's `scene/colmap_loader.py`, which reads a reconstruction and cannot
   produce one.
2. The LLFF images present carry no poses: no `poses_bounds.npy`, no `sparse/`, no `cameras.txt`/`images.txt`/
   `points3D.txt`.
3. Two demo scenes is not the benchmark, so even with poses the "published sparse-view range" comparison would not
   be meaningful.

**PASS/FAIL:** Target: **the COLMAP reference arm must land in published sparse-view PSNR range before running
full/w4a4.** The arm cannot be built, so the gate cannot be evaluated. **BLOCKED — not FAIL, and explicitly not
reported as a number.**

No pose source was substituted. Seeding the reference arm from VGGT's own predicted cameras was considered and
rejected: it would make the control depend on the method under test, producing a figure that looks like a reference
but measures nothing — the same failure mode D3 refused.

**Next step:** obtain the LLFF benchmark release with `poses_bounds.npy` (8 scenes), or install COLMAP/`pycolmap` and
run sparse reconstruction per scene; then build the source builder (3/6/9 views, every 8th frame held out, whole-image
metrics) and gate on the COLMAP arm. Recorded in `code/downstream_3dgs/llff/README.md`.

**Interpretation:** Rules nothing in or out about the method — this is an environment limitation, recorded so the
absence of an LLFF result is not later mistaken for an LLFF null result.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/README.md`

_No result file: this entry's numbers are in the entry text / logs only (BLOCKED run).

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

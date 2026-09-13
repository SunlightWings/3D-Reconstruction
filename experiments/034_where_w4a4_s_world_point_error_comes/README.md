# 034 — Where W4A4's world-point error comes from: depth maps vs cameras

This is entry **#034** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-11 14:10 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py decompose` added)
- **Command executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py decompose --workers 6
  ```
- **Config:** no training. Per view-group: *total* = W4A4 world points vs full's after the Sim(3) on camera centres;
  *depth-only* = W4A4 depth, rescaled by one global median factor, unprojected through **full's** cameras, vs full's
  points — camera error removed by construction. Median over every 7th pixel of all six views (**all content
  pixels, object and background**), in scene radii. Convention check: unprojecting full's depth through full's
  cameras reproduces full's points to 3.85e-07.
- **Scenes used:** all 1330 view-groups; 1330 usable.

**Results:**

| subset | n | total (median) | depth-only (median) | depth-only / total (median per group) | total > depth-only | Wilcoxon p |
|---|---:|---:|---:|---:|---|---:|
| all | 1330 | 0.03624 | 0.02510 | 0.705 | 1169/1330 | 4.02e-181 |
| no bowl | 1296 | 0.03559 | 0.02469 | 0.712 | 1135/1296 | 1.1e-174 |

**PASS/FAIL:** Target: **attribute W4A4's world-point disagreement to depth vs cameras.** With cameras held exact,
the error is still 70.5 % of its full size. **Measured: most of W4A4's point disagreement is in its depth maps, not
its cameras.**

**Interpretation, and a correction to an expectation stated in conversation.** I expected the point error to be
mostly camera-driven, because #033 shows object depth nearly unchanged. It is not: depth alone reproduces ~70 % of
the point error. The two results reconcile only if W4A4's depth disagreement lives mainly in the **background**,
which #033 cannot see (GT depth exists only on the object) — **that is an inference, not yet measured.** It does not
contradict the swap probe: the swap measures *rendering* loss, and it shows point damage barely affects rendering
at all, so where the point error comes from matters less than that 3DGS absorbs it. The swap was on `w4a4_rtn`;
this decomposition is on `w4a4`, and they must not be merged into one sentence.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/cpu_probes.py`

**results/**

- `results/cpu_decompose.json`

**logs/**

- `logs/cpu_decompose.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

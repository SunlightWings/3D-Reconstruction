# 018 — A5 re-run on all 40 scenes after the frame-consistency fix

This is entry **#018** of `ongoing_logs.md`, copied verbatim below (the full log is in [../ongoing_logs.md](../ongoing_logs.md)).


- **Date/time:** 2026-09-08 09:30 CEST
- **Git commit:** `1088b5a` + working-tree fix to `test_a5_loo_sim3`
- **Command executed:**
  ```
  python3 -c "import sys; sys.path.insert(0,'code/downstream_3dgs/diagnostics'); import gate_a; gate_a.test_a5_loo_sim3()"
  ```

**Config:** No training, no rendering. Frozen prediction NPZs plus GT annotations.

| Field | Value |
|---|---|
| Init source | `full` and `w4a4` predicted cameras |
| Iterations | n/a |
| Densify | n/a |
| Input views | 6 input cameras per scene (fit 5, predict the 6th) |
| Mask type | n/a (camera-geometry test) |
| Background handling | n/a |
| Depth regularisation | n/a |

**Scenes used:** all 40 scenes (A5 is a CPU test and runs on the full set).

**Results (median over scenes of each scene's median LOO error, in scene radii):**

| measure | full | w4a4 |
|---|---:|---:|
| **frame-consistent (fixed)** | **0.0813** | **0.1380** |
| legacy, frame-inconsistent (retained in output for traceability) | 0.0066 | — |

The legacy figure reproduces the value published in DIAGNOSTIC_RESULTS.md (0.0066 for Full) **exactly**, which
confirms the fix changes only the frame handling and nothing else about the computation.

**PASS/FAIL:** Target: **median LOO camera-centre error well under 0.02 scene radii.** Measured **0.0813** (Full)
and **0.1380** (W4A4). **FAIL**, by 4x and 7x respectively. The previously published **PASS (0.0066 / 0.0106) was an
artifact of the units bug** described in entry #017.

**Interpretation:** Rules in poor predicted-to-GT camera alignment as a real defect of this harness, on the full
40-scene set rather than the 8-scene subset — the Sim(3) that carries GT held-out cameras into each prediction's world
frame does not extrapolate to a held-out camera, so the held-out views every downstream metric is computed from are
rendered from materially wrong viewpoints. This is now the leading candidate cause of the held-out generalisation
failure that entry #015 isolated, and it applies to the `full` and `w4a4` arms but NOT to the `oracle` arm, which uses
GT cameras directly with no Sim(3).

**Supersedes:** the A5 rows in entry #001 and in every DECISION STATE written before this one.

---

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/common.py`
- `code/gate_a.py`

**results/**

- `results/a5_fixed.json`

**logs/**

- `logs/rerun_gates.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

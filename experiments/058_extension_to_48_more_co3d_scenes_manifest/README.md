# 058 — Extension to 48 more CO3D scenes: manifest built, run CANCELLED

_No `ongoing_logs.md` entry exists for this work yet. Every number below is read directly from the files in `results/`._

**What.** CO3D has 88 usable scenes in the local copy; 40 are the frozen manifest. `code/extend_new48.py` rebuilt the frame-selection rule (the original code was never committed) and validated it by reproducing the frozen manifest, then built a manifest for the other 48.

**Rule validation against the frozen 40-scene manifest** (verbatim from `logs/new48_validate.log`):

```
fg_min 0.0: usable sets exact 28/40, total differing frames 69 (of 7889), g0000 (used downstream) reproduced 29/40
fg_min 0.005: usable sets exact 31/40, total differing frames 51 (of 7889), g0000 (used downstream) reproduced 33/40
fg_min 0.01: usable sets exact 32/40, total differing frames 50 (of 7889), g0000 (used downstream) reproduced 33/40
fg_min 0.02: usable sets exact 32/40, total differing frames 50 (of 7889), g0000 (used downstream) reproduced 33/40

residual differences at fg_min 0.01: 0
```

`results/new48_rule_validation.json` records the minimum-valid-pixel sweep [1, 1000, 5000, 20000], the chosen value `1`, and 50 per-frame residual rows.

The 48 extra sequences are from the CO3D **test** split: GT depth is withheld and masks exist only on test-known frames, so their manifest uses the `no_gt_depth_fg_mask` rule instead of the GT-depth rule.

**Status: CANCELLED by decision on 2026-09-13** ("only do the old 40 scenes"). No inference or 3DGS was run on these scenes.

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/extend_new48.py`
- `code/run_all88_chain.sh`

**results/**

- `results/manifest_new48.json`
- `results/new48.txt`
- `results/new48_rule_validation.json`

**logs/**

- `logs/new48_manifest.log`
- `logs/new48_validate.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.

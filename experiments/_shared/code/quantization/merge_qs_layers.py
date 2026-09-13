"""Merge per-layer QS parameter files into the two totals QuantVGGT expects.

Reproduces exactly the merge at the end of QuantVGGT's `cali_qs_quant` (train_utils.py
~line 268): each `qs_{frame,global}_parameters_layer_<i>.pth` becomes entry `i` of a dict
saved as `qs_{frame,global}_parameters_total.pth`.

Exists because that merge is the LAST step of a ~3.5 h calibration and is not resumable: a
filesystem error there (here, an exhausted home quota) discards the whole run even though
every per-layer file is already on disk. This recovers such a run instead of recalibrating.

Refuses to write unless all `block_num` layers are present and non-empty, so a partial
calibration can never be silently promoted into a complete-looking artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def merge(exp_dir: Path, block_num: int = 24, keep_layers: bool = True) -> None:
    # A size check is NOT sufficient: a write killed by a quota can leave a small but
    # non-zero file that torch.load fails on. Every layer is actually opened.
    missing, empty = [], []
    for kind in ("frame", "global"):
        for i in range(block_num):
            p = exp_dir / f"qs_{kind}_parameters_layer_{i}.pth"
            if not p.is_file():
                missing.append(p.name)
                continue
            try:
                torch.load(p, map_location="cpu")
            except Exception as e:
                empty.append(f"{p.name} ({p.stat().st_size}B, {type(e).__name__})")
    if missing or empty:
        print(f"REFUSING to merge {exp_dir.name}")
        if missing:
            print(f"  missing {len(missing)} layer files, e.g. {missing[:3]}")
        if empty:
            print(f"  unreadable {len(empty)} layer files, e.g. {empty[:3]}")
        print("  -> calibration is incomplete; re-run it rather than merging a partial set")
        raise SystemExit(1)

    for kind in ("frame", "global"):
        out = exp_dir / f"qs_{kind}_parameters_total.pth"
        params = {i: torch.load(exp_dir / f"qs_{kind}_parameters_layer_{i}.pth",
                                map_location="cpu") for i in range(block_num)}
        tmp = out.with_suffix(".pth.tmp")
        torch.save(params, tmp)
        tmp.replace(out)  # atomic: a failed write never leaves a half-file at the real path
        print(f"  wrote {out.name}  ({out.stat().st_size / 2**30:.2f} GiB)")
        del params

    if not keep_layers:
        for kind in ("frame", "global"):
            for i in range(block_num):
                (exp_dir / f"qs_{kind}_parameters_layer_{i}.pth").unlink(missing_ok=True)
        print("  removed per-layer files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=Path)
    ap.add_argument("--block-num", type=int, default=24)
    ap.add_argument("--remove-layers", action="store_true",
                    help="delete the per-layer files after a successful merge")
    a = ap.parse_args()
    if not a.exp_dir.is_dir():
        raise SystemExit(f"not a directory: {a.exp_dir}")
    print(f"merging {a.exp_dir}")
    merge(a.exp_dir, a.block_num, keep_layers=not a.remove_layers)
    print("done")

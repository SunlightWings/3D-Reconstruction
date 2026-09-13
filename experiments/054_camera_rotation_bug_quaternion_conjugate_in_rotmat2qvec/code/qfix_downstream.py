"""Baseline 3DGS on camera-corrected sources (build_qfix_sources.py), scored scene by scene with score_scene."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
import common  # noqa: E402

dv = common.dv
RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["full", "w3a3"])
    ap.add_argument("--iterations", type=int, default=7000)
    a = ap.parse_args()
    subprocess.run([sys.executable, "-u", str(REPO / "code/quantization/build_qfix_sources.py"), "--arms", *a.arms], check=True)
    spec = importlib.util.spec_from_file_location("rwd", REPO / "code/quantization/run_w2a4_downstream.py")
    rwd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rwd)
    out = RES / f"qfix_baseline_{'_'.join(a.arms)}_subset_it{a.iterations}.json"
    rows = []
    for si, n in enumerate(common.EXPENSIVE_SCENES, 1):
        c, s = n.split("/", 1)
        root = common.scene_root(c, s)
        n_held = len(common.scene_manifest(c, s)["heldout_frames"])
        present = {}
        for arm in a.arms:
            q = f"{arm}_qfix"
            dv.train_3dgs(root / "sources" / q / "train", root / "models" / q, a.iterations, 20, 1)
            present[q] = dv.render_heldout(root / "sources" / q / "heldout", root / "models" / q,
                                           root / "heldout" / f"{q}_it{a.iterations}", a.iterations, n_held, 20, 1)
            old = root / "heldout" / f"{arm}_it{a.iterations}" / "renders"
            if old.is_dir() and len(list(old.glob("*.png"))) == n_held:
                present[arm] = old
        sc = rwd.score_scene(c, s, present)
        row = {"scene": n, **{f"{k}_{reg}_{m}": regs[reg][m] for k, regs in sc.items()
                              for reg in ("foreground", "content") for m in ("psnr", "ssim", "lpips")}}
        rows.append(row)
        print(f"[{si}/8] {n}  fg PSNR  " + "  ".join(f"{k}={row[f'{k}_foreground_psnr']:.3f}" for k in present), flush=True)
        out.write_text(json.dumps({"arms": a.arms, "per_scene": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

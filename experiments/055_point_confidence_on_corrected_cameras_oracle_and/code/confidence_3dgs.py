"""W3A3 output + a per-point confidence signal -> 3DGS. Does the reconstruction improve over plain W3A3?

Everything starts from the camera-corrected W3A3 source (sources/w3a3_qfix, see build_qfix_sources.py), so cameras,
colours, held-out views and scoring are identical to the baseline; only how the points enter 3DGS changes:

  prune   keep the most-confident `keep` fraction of the initial points (file order preserved)
  weight  keep every point, but start each with opacity 0.1 * (0.1 + 0.9 * confidence rank): the least trusted
          point starts at 0.01, the most trusted at GraphDECO's default 0.1 (GS_INIT_OPACITY_NPY, opt-in patch)
  random  prune an equally sized RANDOM subset (seed 0) -- the control, because removing points changes 3DGS by itself

Signals: `oracle` = negative true per-point error against full precision (the best any predictor could do);
`predictor` = a learned confidence loaded from conf_predictor/<cat>_<seq>_<arm>.npy (higher = more trusted).
Confidence arrays are aligned to points3D.txt order (the stride-4 grid, verified). Stale points3D.ply/.bin caches are
deleted from every rewritten source so GraphDECO cannot silently load the unpruned cloud.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
sys.path.insert(0, str(REPO / "code" / "quantization"))
import common  # noqa: E402
from confidence_oracle import per_point_error  # noqa: E402

dv = common.dv
RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")


def build(root: Path, base: Path, variant: str, conf: np.ndarray, mode: str, keep: float):
    dst = root / "sources" / variant
    if dst.exists():
        shutil.rmtree(dst)
    (dst).mkdir(parents=True)
    shutil.copytree(base / "train", dst / "train", symlinks=True)
    os.symlink(base / "heldout", dst / "heldout", target_is_directory=True)
    sp = dst / "train" / "sparse" / "0"
    for f in list(sp.iterdir()):
        if f.suffix in (".ply", ".bin"):
            f.unlink()
    lines = [l for l in (sp / "points3D.txt").read_text().splitlines()]
    header = [l for l in lines if l.startswith("#")]
    data = [l for l in lines if not l.startswith("#") and l.strip()]
    if len(data) != len(conf):
        raise SystemExit(f"{variant}: {len(data)} points but {len(conf)} confidences")
    n_keep = int(len(data) * keep)
    opac_file = None
    if mode == "prune":
        sel = np.sort(np.argsort(-conf, kind="stable")[:n_keep])
    elif mode == "random":
        sel = np.sort(np.random.default_rng(0).choice(len(data), size=n_keep, replace=False))
    elif mode == "weight":
        sel = np.arange(len(data))
        rank = np.empty(len(conf))
        rank[np.argsort(conf, kind="stable")] = np.linspace(0.0, 1.0, len(conf))
        opac_file = sp / "init_opacity.npy"
        np.save(opac_file, (0.1 * (0.1 + 0.9 * rank)).astype(np.float32))
    else:
        raise SystemExit(mode)
    if mode != "weight":
        (sp / "points3D.txt").write_text("\n".join(header + [data[i] for i in sel]) + "\n")
    (dst / "train" / "PREPARED.ok").write_text(f"PASS {mode}\n")
    return dst, len(sel), opac_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="w3a3")
    ap.add_argument("--signal", choices=["oracle", "predictor"], required=True)
    ap.add_argument("--modes", nargs="+", default=["prune", "weight", "random"])
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--iterations", type=int, default=7000)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("rwd", REPO / "code/quantization/run_w2a4_downstream.py")
    rwd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rwd)
    out = RES / f"confidence_3dgs_{a.arm}_{a.signal}_it{a.iterations}.json"
    rows = []
    for si, n in enumerate(common.EXPENSIVE_SCENES, 1):
        c, s = n.split("/", 1)
        root = common.scene_root(c, s)
        n_held = len(common.scene_manifest(c, s)["heldout_frames"])
        base = root / "sources" / f"{a.arm}_qfix"
        if a.signal == "oracle":
            conf = -per_point_error(c, s, a.arm)
        else:
            conf = np.load(RES / "conf_predictor" / f"{c}_{s}_{a.arm}.npy").astype(np.float64)
        present = {}
        for ref in (f"{a.arm}_qfix", "full_qfix"):
            d = root / "heldout" / f"{ref}_it{a.iterations}" / "renders"
            if d.is_dir() and len(list(d.glob("*.png"))) == n_held:
                present[ref] = d
        info = {}
        for mode in a.modes:
            v = f"{a.arm}_qfix_{'rand' if mode == 'random' else a.signal}_{mode if mode != 'random' else 'prune'}"
            if mode == "random" and a.signal == "predictor":
                continue   # the random control does not depend on the signal; oracle run owns it
            rdir = root / "heldout" / f"{v}_it{a.iterations}" / "renders"
            if not (rdir.is_dir() and len(list(rdir.glob("*.png"))) == n_held):
                src, npts, opac = build(root, base, v, conf, mode, a.keep)
                if opac is not None:
                    os.environ["GS_INIT_OPACITY_NPY"] = str(opac)
                try:
                    dv.train_3dgs(src / "train", root / "models" / v, a.iterations, 20, 1)
                finally:
                    os.environ.pop("GS_INIT_OPACITY_NPY", None)
                rdir = dv.render_heldout(src / "heldout", root / "models" / v,
                                         root / "heldout" / f"{v}_it{a.iterations}", a.iterations, n_held, 20, 1)
                info[v] = npts
            present[v] = rdir
        sc = rwd.score_scene(c, s, present)
        row = {"scene": n, "n_points": info,
               **{f"{k}_{reg}_{m}": regs[reg][m] for k, regs in sc.items()
                  for reg in ("foreground", "content") for m in ("psnr", "ssim", "lpips")}}
        rows.append(row)
        print(f"[{si}/8] {n}  fg PSNR  " + "  ".join(f"{k}={row[f'{k}_foreground_psnr']:.3f}" for k in present), flush=True)
        RES.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"arm": a.arm, "signal": a.signal, "keep": a.keep, "per_scene": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

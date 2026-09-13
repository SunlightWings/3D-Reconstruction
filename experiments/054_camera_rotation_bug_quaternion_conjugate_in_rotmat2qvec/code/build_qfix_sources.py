"""Write quaternion-CORRECTED copies of existing 3DGS sources, under new arm names, without touching the originals.

Bug being corrected: run_downstream_validation.rotmat2qvec builds the eigen-matrix with the antisymmetric row negated
relative to COLMAP's reference, so it writes the CONJUGATE quaternion, i.e. the quaternion of R^T. GraphDECO reads
world->view rotation as qvec2rotmat(q), so every source written by that function hands GraphDECO R^T instead of R for
every non-identity camera. Negating (qx, qy, qz) turns conj(q) back into q exactly -- nothing else in the source changes.

sources/<arm>/{train,heldout}  ->  sources/<arm>_qfix/{train,heldout}   (images symlink, cameras, points copied as-is)

Self-check (train split): re-reading the rewritten images.txt with the standard COLMAP qvec2rotmat must reproduce the
arm's .npz world->camera rotations; a mismatch aborts. The originals are never modified, so every earlier result stays
reproducible.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
import common  # noqa: E402

dv = common.dv


def q2r(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--scenes", nargs="+", default=None, help="category/sequence list; default = 8 subset scenes")
    ap.add_argument("--all", action="store_true", help="all 40 scenes")
    a = ap.parse_args()
    names = common.all_scenes() if a.all else (a.scenes or common.EXPENSIVE_SCENES)
    for n in names:
        c, s = n.split("/", 1)
        root = common.scene_root(c, s)
        g, _m, _w = dv.find_group(c, s)
        for arm in a.arms:
            for split in ("train", "heldout"):
                src, dst = root / "sources" / arm / split, root / "sources" / f"{arm}_qfix" / split
                if not (src / "sparse" / "0" / "images.txt").is_file():
                    raise SystemExit(f"{n}: missing source {src}")
                if (dst / "PREPARED.ok").is_file():
                    continue
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, symlinks=True)
                p = dst / "sparse" / "0" / "images.txt"
                out = []
                for ln in p.read_text().splitlines():
                    t = ln.split()
                    if not ln.startswith("#") and len(t) >= 10:
                        t[2], t[3], t[4] = (repr(-float(v)) for v in t[2:5])
                        ln = " ".join(t)
                    out.append(ln)
                p.write_text("\n".join(out) + "\n")
                if split == "train":
                    Q = dv.load_npz(g / f"{arm}.npz")["extrinsic"].astype(np.float64)
                    worst = 0.0
                    for ln in out:
                        t = ln.split()
                        if ln.startswith("#") or len(t) < 10:
                            continue
                        i = int(t[9].split("_")[1].split(".")[0]) - 1
                        worst = max(worst, float(np.abs(q2r(np.array(list(map(float, t[1:5])))) - Q[i, :, :3]).max()))
                    if worst > 1e-5:
                        shutil.rmtree(dst)
                        raise SystemExit(f"{n} {arm}: corrected rotations do not match npz (max {worst:.2e})")
                (dst / "PREPARED.ok").write_text("PASS qfix\n")
            print(f"  {n} {arm}_qfix ready", flush=True)


if __name__ == "__main__":
    main()

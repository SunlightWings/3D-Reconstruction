"""Augment the oracle init with random background points.

The pure oracle initialises ONLY foreground points, because CO3D GT depth is
valid on ~0.1% of background pixels. The evaluated content mask is ~85%
background, so the background has no geometry to start from. This variant keeps
every GT foreground point and adds a random shell of points around the scene so
adaptive density control has something to refine in the background region.

NOTE: this is no longer a pure GT-geometry oracle -- the added points are not
measured, they are a random prior. Reported as such.
"""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/utn/luli38se/cv/3D-Reconstruction/code/downstream_3dgs/diagnostics")
import common  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep  # noqa: E402

dv = common.dv
SWEEP = Path("/var/tmp/luli38se/quantsplat/oracle_sweep")


def read_points3D(path: Path):
    xyz, rgb = [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([int(p[4]), int(p[5]), int(p[6])])
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.uint8)


def build(tag: str, n_bg: int, r_lo: float, r_hi: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    for name in common.EXPENSIVE_SCENES:
        cat, seq = common.split_scene(name)
        src_root = common.DIAG_HEAVY_ROOT / cat / seq / "oracle"
        dst_root = SWEEP / tag / cat / seq
        ok = dst_root / "PREPARED.ok"
        if ok.is_file():
            print(f"  skip {tag} {name}")
            continue
        if dst_root.exists():
            shutil.rmtree(dst_root)
        # copy the whole oracle source (train + heldout, cameras and images)
        shutil.copytree(src_root, dst_root, symlinks=True)
        for stale in dst_root.glob("PREPARED.ok"):
            stale.unlink()
        prep.strip_model_artifacts(dst_root)

        p3d = dst_root / "train" / "sparse" / "0" / "points3D.txt"
        xyz, rgb = read_points3D(p3d)
        centre = xyz.mean(axis=0)
        radius = float(np.linalg.norm(xyz - centre, axis=1).max())

        # uniform directions, radii in [r_lo, r_hi] * scene radius
        d = rng.normal(size=(n_bg, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        r = (rng.uniform(r_lo, r_hi, size=(n_bg, 1)) ** (1.0 / 3.0)) * radius
        bg = centre + d * r
        bg_rgb = np.full((n_bg, 3), 128, dtype=np.uint8)

        all_xyz = np.concatenate([xyz, bg.astype(np.float32)])
        all_rgb = np.concatenate([rgb, bg_rgb])
        lines = ["# 3D point list", ""]
        for i, (p, c) in enumerate(zip(all_xyz, all_rgb), start=1):
            lines.append(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0")
        p3d.write_text("\n".join(lines) + "\n")
        shutil.copy2(p3d, dst_root / "heldout" / "sparse" / "0" / "points3D.txt")
        ok.write_text(f"PASS fg={len(xyz)} bg={n_bg} radius={radius:.4f}\n")
        print(f"  built {tag} {name}: {len(xyz)} GT fg + {n_bg} random bg, scene radius {radius:.3f}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "bgshell", n_bg=100_000, r_lo=1.2, r_hi=8.0)

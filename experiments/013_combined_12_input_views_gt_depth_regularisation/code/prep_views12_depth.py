"""Combine the only two factors that helped: 12 input views + GT depth regularisation.

Builds on the already-prepared `views12` source by adding the inverse-depth maps
and depth_params.json for that source's own 12 frames.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SP = "/home/utn/luli38se/cv/3D-Reconstruction/code/downstream_3dgs/oracle_sweep"
sys.path.insert(0, SP)
sys.path.insert(0, "/home/utn/luli38se/cv/3D-Reconstruction/code/downstream_3dgs/diagnostics")
import common  # noqa: E402
import prep  # noqa: E402
from prep_depth import gt_depth_518  # noqa: E402

dv = common.dv
SWEEP = Path("/var/tmp/luli38se/quantsplat/oracle_sweep")


def build(tag="views12_depth", n_views=12):
    for name in common.EXPENSIVE_SCENES:
        cat, seq = common.split_scene(name)
        src_root = SWEEP / "views12" / cat / seq
        dst_root = SWEEP / tag / cat / seq
        ok = dst_root / "PREPARED.ok"
        if ok.is_file():
            print(f"  skip {tag} {name}")
            continue
        if dst_root.exists():
            shutil.rmtree(dst_root)
        shutil.copytree(src_root, dst_root, symlinks=True)
        for stale in dst_root.glob("PREPARED.ok"):
            stale.unlink()
        prep.strip_model_artifacts(dst_root)

        recs = dv.load_annotations(cat, seq)
        held = common.scene_manifest(cat, seq)["heldout_frames"]
        frames = prep.pick_frames(recs, held, n_views)  # same selection as views12

        depths_dir = dst_root / "train" / "depths"
        depths_dir.mkdir(parents=True, exist_ok=True)
        inv_maps, covered = [], []
        for f in frames:
            d, v = gt_depth_518(recs[f])
            inv = np.zeros_like(d, dtype=np.float32)
            inv[v] = 1.0 / d[v]
            inv_maps.append(inv)
            covered.append(float(v.mean()))
        K = 65535.0 / max(float(np.max([m.max() for m in inv_maps])), 1e-6)
        params = {}
        for i, inv in enumerate(inv_maps, start=1):
            Image.fromarray(np.clip(inv * K, 0, 65535).astype(np.uint16)).save(depths_dir / f"image_{i}.png")
            params[f"image_{i}"] = {"scale": 65536.0 / K, "offset": 0.0}
        (dst_root / "train" / "sparse" / "0" / "depth_params.json").write_text(json.dumps(params, indent=1))
        ok.write_text(f"PASS views={len(frames)} K={K:.3f} coverage={np.mean(covered):.4f}\n")
        print(f"  built {tag} {name}: {len(frames)} views, depth coverage {np.mean(covered):.3f}")


if __name__ == "__main__":
    build()

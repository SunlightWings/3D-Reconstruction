"""Add GT inverse-depth maps to the oracle source so 3DGS depth regularisation
(`--depth_l1_weight_init`) can be switched on.

GraphDECO loads `<depths>/<image_name_without_ext>.png` as uint16, divides by
2**16, then applies per-image `scale`/`offset` from sparse/0/depth_params.json.
We store invdepth * K and set scale = 65536 / K so the loaded value is the true
inverse depth in world units. Pixels without valid GT depth are written as 0,
which GraphDECO treats as "no depth constraint here" (depth_mask = invdepth > 0).

Because CO3D GT depth is valid almost exclusively on the object, this
regulariser constrains the foreground only.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/utn/luli38se/cv/3D-Reconstruction/code/downstream_3dgs/diagnostics")
import common  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep  # noqa: E402

dv, dis = common.dv, common.dis
SWEEP = Path("/var/tmp/luli38se/quantsplat/oracle_sweep")


def gt_depth_518(rec):
    """GT metric depth resampled onto the same 518x518 grid as the training images."""
    h, w = [int(v) for v in rec["image"]["size"]]
    nw, nh, left, top, _ = dv.preprocess_geometry(w, h)
    tr = {"resized_width": nw, "resized_height": nh, "padding": {"left": left, "top": top}}
    dm = rec["depth"]
    with Image.open(dv.CO3D_ROOT / dm["path"]) as img:
        bits = np.array(img, dtype=np.uint16)
        raw = np.frombuffer(bits.tobytes(order="C"), dtype=np.float16).astype(np.float32).reshape(bits.shape)
    with Image.open(dv.CO3D_ROOT / dm["mask_path"]) as img:
        dmask = np.asarray(img)
    with Image.open(dv.CO3D_ROOT / rec["mask"]["path"]) as img:
        fg = np.asarray(img.convert("L"))
    depth = raw * float(dm.get("scale_adjustment", 1.0))
    d518, valid, fg518 = dis.resize_depth_and_masks(depth, dmask != 0, fg, tr)
    return d518, valid & (fg518 >= 0.5) & (d518 > 1e-6)


def build(tag="depthreg"):
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
        shutil.copytree(src_root, dst_root, symlinks=True)
        for stale in dst_root.glob("PREPARED.ok"):
            stale.unlink()
        prep.strip_model_artifacts(dst_root)

        recs = dv.load_annotations(cat, seq)
        frames = common.scene_manifest(cat, seq)["input_frames"]
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
            png = np.clip(inv * K, 0, 65535).astype(np.uint16)
            Image.fromarray(png).save(depths_dir / f"image_{i}.png")
            params[f"image_{i}"] = {"scale": 65536.0 / K, "offset": 0.0}
        (dst_root / "train" / "sparse" / "0" / "depth_params.json").write_text(json.dumps(params, indent=1))
        ok.write_text(f"PASS K={K:.3f} mean_depth_coverage={np.mean(covered):.4f}\n")
        print(f"  built {tag} {name}: K={K:.1f}, mean GT-depth pixel coverage {np.mean(covered):.3f}")


if __name__ == "__main__":
    build()

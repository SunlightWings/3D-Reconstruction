"""Measure the PSNR ceilings of the downstream-3DGS held-out metric.

Two different ceilings, because the two masks are limited by different things.

CONTENT MASK
    The content mask is the whole non-letterbox rectangle, of which the object is a
    mean 15.3%. CO3D supplies GT depth for ~0.1% of background pixels, so no arm can
    initialise geometry there. The ceiling is therefore what you score with a
    PIXEL-PERFECT object and the best possible constant background:
        ceiling_content = PSNR(gt with background replaced by its own mean, gt)
    Measured at 16.81 dB, which is why a 20 dB gate on this mask is unreachable.

FOREGROUND MASK
    A pixel-perfect object would score infinity here, so the content construction does
    not transfer. The binding limit instead is the GT geometry itself: the oracle is
    built from GT depth that is valid on only a few percent of pixels, so even Gaussians
    that reproduce the GT points exactly leave gaps. The ceiling is therefore the
    GT-POINT REPROJECTION ceiling: take the oracle's own world points and their GT
    colours, project them through the GT held-out cameras with a z-buffer, and score
    the result. This is what a reconstruction scores if it recovers the GT point cloud
    perfectly and adds nothing -- an upper bound on the oracle arm that is made of
    measurements only.

Both are written to results/metric_ceiling.json. run_all_diagnostics.py reads that file
to set the A3 target, so the gate threshold is derived rather than hand-picked.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP.parent / "diagnostics"))
import common  # noqa: E402

dv = common.dv
OUT = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
SPLAT_RADIUS = 1  # pixels; 1 -> a 3x3 footprint per point


def read_points3D(path: Path):
    xyz, rgb = [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([int(p[4]), int(p[5]), int(p[6])])
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.float32) / 255.0


def reproject(xyz, rgb, E, K, hw, radius=SPLAT_RADIUS):
    """Z-buffered point splat into one camera. Returns (image, covered_mask)."""
    H, W = hw
    cam = (E[:3, :3] @ xyz.T + E[:3, 3:4]).T
    z = cam[:, 2]
    front = z > 1e-6
    cam, z, col = cam[front], z[front], rgb[front]
    uv = (K @ cam.T).T
    u = np.round(uv[:, 0] / uv[:, 2]).astype(np.int64)
    v = np.round(uv[:, 1] / uv[:, 2]).astype(np.int64)

    img = np.zeros((H, W, 3), dtype=np.float32)
    depth = np.full((H, W), np.inf, dtype=np.float64)
    for du in range(-radius, radius + 1):
        for dv_ in range(-radius, radius + 1):
            uu, vv = u + du, v + dv_
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            if not ok.any():
                continue
            uu, vv, zz, cc = uu[ok], vv[ok], z[ok], col[ok]
            order = np.argsort(-zz)  # far first, so nearest wins the final write
            uu, vv, zz, cc = uu[order], vv[order], zz[order], cc[order]
            closer = zz < depth[vv, uu]
            uu, vv, zz, cc = uu[closer], vv[closer], zz[closer], cc[closer]
            depth[vv, uu] = zz
            img[vv, uu] = cc
    return img, np.isfinite(depth)


def noise_floors():
    """A2's construction (median pair-PSNR between distinct held-out GT frames), both masks.

    This is the uninformative-render floor: what you score by outputting some other real
    view of the same scene.
    """
    import itertools

    out = {}
    for masktype in ("content", "foreground"):
        meds = []
        for name in common.EXPENSIVE_SCENES:
            cat, seq = common.split_scene(name)
            recs = dv.load_annotations(cat, seq)
            held = common.scene_manifest(cat, seq)["heldout_frames"]
            gt_dir = common.scene_root(cat, seq) / "common" / "heldout_images"
            masks = {f: (dv.foreground_mask_from_record(recs[f]) if masktype == "foreground"
                         else dv.content_mask_from_record(recs[f])) for f in held}
            imgs = {f: np.asarray(Image.open(gt_dir / f"frame{f:06d}.png").convert("RGB"),
                                  dtype=np.float32) / 255.0 for f in held}
            pp = []
            for a, b in itertools.combinations(held, 2):
                m = masks[a] & masks[b]
                if m.sum() < 100:
                    continue
                v = dv.psnr(imgs[a], imgs[b], m)
                if np.isfinite(v):
                    pp.append(v)
            meds.append(float(np.median(pp)))
        out[masktype] = {"mean_of_scene_medians": float(np.mean(meds)), "per_scene_medians": meds}
    return out


def derive_a3_target(out):
    """Derive the A3 pass threshold instead of hand-picking it.

    The natural move would be `foreground_ceiling - 2 dB`, but the GT-point reprojection
    ceiling measured above is NOT an upper bound on a trained reconstruction: Gaussians
    interpolate across the gaps where CO3D depth is invalid (GT points cover only 49-84%
    of the object), so the trained oracle already scores ~3.7 dB ABOVE it. Subtracting from
    it would set a target below current performance, which would make the gate vacuous.

    Unlike the content mask -- whose 16.81 dB ceiling makes a 20 dB gate unreachable by
    construction -- the foreground mask has no low structural ceiling. So the threshold is
    instead derived by carrying over the ORIGINAL gate's own margin: A3 asked for 20 dB
    against A2's content-mask noise floor, and the same margin is applied to the
    foreground-mask noise floor.
    """
    nf = out["noise_floor"]
    content_floor = nf["content"]["mean_of_scene_medians"]
    fg_floor = nf["foreground"]["mean_of_scene_medians"]
    original_target = 20.0
    margin = original_target - content_floor
    return {
        "mask": "foreground",
        "target_db": float(fg_floor + margin),
        "derivation": (
            f"original A3 target {original_target:.1f} dB minus the content-mask noise floor "
            f"{content_floor:.3f} dB gives a margin of {margin:.3f} dB above 'uninformative'; "
            f"the same margin above the foreground-mask noise floor {fg_floor:.3f} dB gives "
            f"{fg_floor + margin:.3f} dB."
        ),
        "content_noise_floor_db": float(content_floor),
        "foreground_noise_floor_db": float(fg_floor),
        "margin_db": float(margin),
        "why_not_ceiling_minus_2db": (
            "The GT-point reprojection 'ceiling' is not an upper bound -- the trained oracle "
            "exceeds it because Gaussians fill the gaps left by invalid CO3D depth. The "
            "foreground mask has no binding structural ceiling, unlike the content mask."
        ),
    }


def main():
    content_rows, fg_rows = [], []
    for name in common.EXPENSIVE_SCENES:
        cat, seq = common.split_scene(name)
        recs = dv.load_annotations(cat, seq)
        held = common.scene_manifest(cat, seq)["heldout_frames"]
        gt_dir = common.scene_root(cat, seq) / "common" / "heldout_images"
        oracle_pts = common.DIAG_HEAVY_ROOT / cat / seq / "oracle" / "train" / "sparse" / "0" / "points3D.txt"
        xyz, rgb = read_points3D(oracle_pts)

        ff, c_const, c_black, fg_ceil, fg_cover = [], [], [], [], []
        for f in held:
            r = recs[f]
            cm = dv.content_mask_from_record(r)
            fg = dv.foreground_mask_from_record(r)
            bg = cm & ~fg
            gt = np.asarray(Image.open(gt_dir / f"frame{f:06d}.png").convert("RGB"), dtype=np.float32) / 255.0
            ff.append(fg.sum() / max(cm.sum(), 1))

            # --- content-mask ceiling: perfect object, best flat background
            if bg.sum():
                pred = gt.copy()
                pred[bg] = gt[bg].mean(axis=0)
                c_const.append(dv.psnr(pred, gt, cm))
            pred_black = gt.copy()
            pred_black[bg] = 0.0
            c_black.append(dv.psnr(pred_black, gt, cm))

            # --- foreground ceiling: reproject the GT point cloud through the GT camera
            E, K0 = dv.co3d_to_opencv_camera(r)
            K = dv.original_to_518_affine(r) @ K0
            img, covered = reproject(xyz, rgb, E, K, gt.shape[:2])
            if fg.sum():
                fg_ceil.append(dv.psnr(img, gt, fg))
                fg_cover.append(float((covered & fg).sum()) / float(fg.sum()))

        content_rows.append({"scene": name, "fg_fraction": float(np.mean(ff)),
                             "ceiling_bg_optimal_constant": float(np.mean(c_const)),
                             "ceiling_bg_black": float(np.mean(c_black))})
        fg_rows.append({"scene": name, "ceiling_gt_reprojection": float(np.mean(fg_ceil)),
                        "gt_point_coverage_of_foreground": float(np.mean(fg_cover))})
        print(f"{name:35s} fg_frac={content_rows[-1]['fg_fraction']:.3f}  "
              f"content_ceil={content_rows[-1]['ceiling_bg_optimal_constant']:6.2f} dB  "
              f"fg_ceil={fg_rows[-1]['ceiling_gt_reprojection']:6.2f} dB  "
              f"(GT points cover {fg_rows[-1]['gt_point_coverage_of_foreground']*100:.1f}% of object)")

    out = {
        "description": "PSNR ceilings for the downstream-3DGS held-out metric; see module docstring for derivation.",
        "scenes": common.EXPENSIVE_SCENES,
        "content_mask": {
            "construction": "pixel-perfect foreground, background replaced by its own mean colour",
            "mean_fg_fraction": float(np.mean([r["fg_fraction"] for r in content_rows])),
            "mean_ceiling": float(np.mean([r["ceiling_bg_optimal_constant"] for r in content_rows])),
            "mean_ceiling_bg_black": float(np.mean([r["ceiling_bg_black"] for r in content_rows])),
            "scenes_ceiling_ge_20db": int(sum(r["ceiling_bg_optimal_constant"] >= 20 for r in content_rows)),
            "per_scene": content_rows,
        },
        "foreground_mask": {
            "construction": f"z-buffered reprojection of the oracle GT point cloud through GT cameras, splat radius {SPLAT_RADIUS}",
            "mean_ceiling": float(np.mean([r["ceiling_gt_reprojection"] for r in fg_rows])),
            "min_ceiling": float(np.min([r["ceiling_gt_reprojection"] for r in fg_rows])),
            "mean_gt_point_coverage_of_foreground": float(np.mean([r["gt_point_coverage_of_foreground"] for r in fg_rows])),
            "per_scene": fg_rows,
        },
    }
    out["noise_floor"] = noise_floors()
    out["a3_target"] = derive_a3_target(out)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metric_ceiling.json").write_text(json.dumps(out, indent=2))
    print(f"\ncontent-mask ceiling    : {out['content_mask']['mean_ceiling']:.3f} dB "
          f"({out['content_mask']['scenes_ceiling_ge_20db']}/8 scenes could reach 20 dB)")
    print(f"foreground-mask ceiling : {out['foreground_mask']['mean_ceiling']:.3f} dB")
    print(f"saved {OUT/'metric_ceiling.json'}")


if __name__ == "__main__":
    main()

"""Compare the two Sim(3) fitting directions on A4 (on-silhouette) and A5 (LOO error).

Umeyama is not symmetric: fitting GT -> pred and inverting is not the same estimator as
fitting pred -> GT and applying it forward, because the least-squares residual is measured
in a different space and the scale estimate differs accordingly.

The codebase currently uses the INVERSE form everywhere:
    run_downstream_validation.py:   s,A,b = umeyama(C_gt, C_pred)   # then applied forward
                                    to carry GT held-out cameras into the predicted frame
    analyze_full_dataset_disagreement.py:
                                    s,A,b = umeyama(C_gt, C_pred)   # then pred_to_gt() inverts

This measures both forms on the two diagnostics that are sensitive to alignment:
    A5  leave-one-out camera-centre extrapolation error (fit 5 cameras, predict the 6th)
    A4  fraction of predicted points landing on the GT object silhouette

Entry #016 already established that the GT camera path itself is exact (100% self
on-silhouette), so any A4 shortfall lives in the prediction or in this alignment.
"""
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP.parent / "diagnostics"))
import common  # noqa: E402

dv = common.dv
OUT = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")


def map_pred_to_gt(P, C_gt, C_pred, direction):
    """Carry points from the predicted world frame into the GT world frame."""
    if direction == "inverse":            # current behaviour
        s, A, b = dv.umeyama(C_gt, C_pred)          # fits C_pred ~= s A C_gt + b
        return (P - b) @ A / s                      # invert it
    elif direction == "direct":           # proposed alternative
        s, A, b = dv.umeyama(C_pred, C_gt)          # fits C_gt ~= s A C_pred + b
        return (s * (A @ P.reshape(-1, 3).T)).T.reshape(P.shape) + b
    raise ValueError(direction)


def loo_error(C_gt, C_pred, radius, direction):
    """Leave-one-out camera-centre extrapolation error, in scene radii."""
    errs = []
    n = len(C_gt)
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        try:
            if direction == "inverse":
                s, A, b = dv.umeyama(C_gt[idx], C_pred[idx])
                pred_k = s * (A @ C_gt[k]) + b
                errs.append(float(np.linalg.norm(pred_k - C_pred[k]) / radius))
            else:
                s, A, b = dv.umeyama(C_pred[idx], C_gt[idx])
                # same quantity, measured in the GT frame then expressed in scene radii
                gt_k = s * (A @ C_pred[k]) + b
                errs.append(float(np.linalg.norm(gt_k - C_gt[k]) / radius))
        except Exception:
            continue
    return float(np.median(errs)) if errs else float("nan")


def on_silhouette(xyz, E, K, fg):
    cam = (E[:3, :3] @ xyz.T + E[:3, 3:4]).T
    z = cam[:, 2]
    ok = z > 1e-6
    if not ok.any():
        return 0.0
    uv = (K @ cam[ok].T).T
    u = np.round(uv[:, 0] / uv[:, 2]).astype(np.int64)
    v = np.round(uv[:, 1] / uv[:, 2]).astype(np.int64)
    H, W = fg.shape
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inside.any():
        return 0.0
    return float(fg[v[inside], u[inside]].mean())


def main(stride=8):
    per_scene_csv = common.load_per_scene_csv()
    rows = []
    for name in common.EXPENSIVE_SCENES:
        cat, seq = common.split_scene(name)
        recs = dv.load_annotations(cat, seq)
        frames = common.scene_manifest(cat, seq)["input_frames"]
        E_gt = np.stack([dv.co3d_to_opencv_camera(recs[f])[0] for f in frames])
        C_gt = dv.camera_centers(E_gt)
        radius = per_scene_csv[(cat, seq)]["scene_radius"]
        group_dir, _fm, _wm = dv.find_group(cat, seq)

        row = {"scene": name}
        for variant, npz in (("full", "full.npz"), ("w4a4", "w4a4.npz")):
            arrays = dv.load_npz(group_dir / npz)
            C_pred = dv.camera_centers(arrays["extrinsic"].astype(np.float64))
            P = arrays["world_points_from_depth"].astype(np.float64)
            for direction in ("inverse", "direct"):
                row[f"{variant}_{direction}_loo"] = loo_error(C_gt, C_pred, radius, direction)
                sil = []
                for i, f in enumerate(frames):
                    pts = P[i][::stride, ::stride, :].reshape(-1, 3)
                    pts_gt = map_pred_to_gt(pts, C_gt, C_pred, direction)
                    E, K0 = dv.co3d_to_opencv_camera(recs[f])
                    K = dv.original_to_518_affine(recs[f]) @ K0
                    sil.append(on_silhouette(pts_gt, E, K, dv.foreground_mask_from_record(recs[f])))
                row[f"{variant}_{direction}_on_silhouette"] = float(np.mean(sil))
        rows.append(row)
        print(f"{name:33s} full: LOO inv={row['full_inverse_loo']:.4f} dir={row['full_direct_loo']:.4f} | "
              f"sil inv={row['full_inverse_on_silhouette']*100:.1f}% dir={row['full_direct_on_silhouette']*100:.1f}%")

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    summary = {}
    for variant in ("full", "w4a4"):
        for direction in ("inverse", "direct"):
            summary[f"{variant}_{direction}_loo_mean"] = m(f"{variant}_{direction}_loo")
            summary[f"{variant}_{direction}_on_silhouette_mean"] = m(f"{variant}_{direction}_on_silhouette")

    better_loo = "direct" if summary["full_direct_loo_mean"] < summary["full_inverse_loo_mean"] else "inverse"
    better_sil = ("direct" if summary["full_direct_on_silhouette_mean"] > summary["full_inverse_on_silhouette_mean"]
                  else "inverse")
    out = {
        "description": "Sim(3) fitting direction comparison on A5 (LOO) and A4 (on-silhouette).",
        "point_stride": stride,
        "summary": summary,
        "better_on_loo": better_loo,
        "better_on_silhouette": better_sil,
        "agree": better_loo == better_sil,
        "per_scene": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sim3_direction.json").write_text(json.dumps(out, indent=2))

    print("\n--- means over 8 scenes ---")
    for variant in ("full", "w4a4"):
        print(f"{variant}:  LOO inverse={summary[f'{variant}_inverse_loo_mean']:.5f}  "
              f"direct={summary[f'{variant}_direct_loo_mean']:.5f}   |   "
              f"on-silhouette inverse={summary[f'{variant}_inverse_on_silhouette_mean']*100:.2f}%  "
              f"direct={summary[f'{variant}_direct_on_silhouette_mean']*100:.2f}%")
    print(f"\nbetter on LOO: {better_loo};  better on-silhouette: {better_sil};  agree: {out['agree']}")
    print(f"saved {OUT/'sim3_direction.json'}")


if __name__ == "__main__":
    main()

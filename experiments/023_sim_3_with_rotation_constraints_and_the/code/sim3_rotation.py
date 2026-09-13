"""Fit the GT->predicted Sim(3) using camera ORIENTATIONS as well as centres.

The current estimator (`dv.umeyama`) sees only the 6 camera centres. With 6 points in
3D the rotation is weakly constrained, which is why the leave-one-out extrapolation error
is 0.0813 / 0.1380 scene radii (entry #018) against a 0.02 target.

Every input camera carries an orientation too, and under a similarity that maps
GT world -> predicted world (X_pred = s A X_gt + b) the camera rotations satisfy

    R_pred = R_gt A^T          =>      A = R_pred^T R_gt

so each of the 6 camera pairs gives an INDEPENDENT estimate of A that does not depend on
the centres at all. Rotation-averaging those (the chordal L2 mean: SVD-project their sum
back onto SO(3)) gives an A estimated from 6 orientations rather than inferred from 6
points, after which the scale and translation follow in closed form from the centres.

Three estimators are compared:
    centres     dv.umeyama, the current behaviour
    rotation    A from rotation averaging; s, b from centres given that A
    joint       A from rotation averaging refined against centre residuals, w in [0,1]
                blending the two rotation estimates before re-projection onto SO(3)
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


def project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    return U @ D @ Vt


def scale_translation_given_A(C_gt, C_pred, A):
    """Closed-form s and b minimising ||C_pred - (s A C_gt + b)||^2 for fixed A."""
    mg, mp = C_gt.mean(axis=0), C_pred.mean(axis=0)
    X = (A @ (C_gt - mg).T).T
    Y = C_pred - mp
    denom = float(np.sum(X * X))
    s = float(np.sum(X * Y) / denom) if denom > 1e-18 else 1.0
    b = mp - s * (A @ mg)
    return s, b


def fit(C_gt, C_pred, R_gt, R_pred, method, w=0.5):
    if method == "centres":
        return dv.umeyama(C_gt, C_pred)
    A_rot = project_so3(sum(rp.T @ rg for rg, rp in zip(R_gt, R_pred)))
    if method == "rotation":
        s, b = scale_translation_given_A(C_gt, C_pred, A_rot)
        return s, A_rot, b
    if method == "joint":
        _, A_cen, _ = dv.umeyama(C_gt, C_pred)
        A = project_so3(w * A_rot + (1.0 - w) * A_cen)
        s, b = scale_translation_given_A(C_gt, C_pred, A)
        return s, A, b
    raise ValueError(method)


def loo(C_gt, C_pred, R_gt, R_pred, radius, method):
    """Frame-consistent leave-one-out error (entry #018 convention)."""
    errs = []
    n = len(C_gt)
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        try:
            s, A, b = fit(C_gt[idx], C_pred[idx], [R_gt[i] for i in idx],
                          [R_pred[i] for i in idx], method)
        except Exception:
            continue
        back = (C_pred[k] - b) @ A / s
        errs.append(float(np.linalg.norm(back - C_gt[k]) / radius))
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


def main(stride=8, scenes=None):
    per_scene_csv = common.load_per_scene_csv()
    names = scenes or common.EXPENSIVE_SCENES
    methods = ("centres", "rotation", "joint")
    rows = []
    for name in names:
        cat, seq = common.split_scene(name)
        recs = dv.load_annotations(cat, seq)
        frames = common.scene_manifest(cat, seq)["input_frames"]
        E_gt = [dv.co3d_to_opencv_camera(recs[f])[0] for f in frames]
        C_gt = dv.camera_centers(np.stack(E_gt))
        R_gt = [e[:, :3] for e in E_gt]
        radius = per_scene_csv[(cat, seq)]["scene_radius"]
        gd, _, _ = dv.find_group(cat, seq)

        row = {"scene": name}
        for variant, npz in (("full", "full.npz"), ("w4a4", "w4a4.npz")):
            arr = dv.load_npz(gd / npz)
            E_pred = arr["extrinsic"].astype(np.float64)
            C_pred = dv.camera_centers(E_pred)
            R_pred = [e[:, :3] for e in E_pred]
            P = arr["world_points_from_depth"].astype(np.float64)
            for m in methods:
                row[f"{variant}_{m}_loo"] = loo(C_gt, C_pred, R_gt, R_pred, radius, m)
                s, A, b = fit(C_gt, C_pred, R_gt, R_pred, m)
                sil = []
                for i, f in enumerate(frames):
                    pts = P[i][::stride, ::stride, :].reshape(-1, 3)
                    pts_gt = (pts - b) @ A / s
                    E, K0 = dv.co3d_to_opencv_camera(recs[f])
                    K = dv.original_to_518_affine(recs[f]) @ K0
                    sil.append(on_silhouette(pts_gt, E, K, dv.foreground_mask_from_record(recs[f])))
                row[f"{variant}_{m}_on_silhouette"] = float(np.mean(sil))
        rows.append(row)
        print(f"  {name:33s} LOO full: " + "  ".join(
            f"{m}={row[f'full_{m}_loo']:.4f}" for m in methods), flush=True)

    summary = {}
    for variant in ("full", "w4a4"):
        for m in methods:
            summary[f"{variant}_{m}_loo"] = float(np.mean([r[f"{variant}_{m}_loo"] for r in rows]))
            summary[f"{variant}_{m}_on_silhouette"] = float(
                np.mean([r[f"{variant}_{m}_on_silhouette"] for r in rows]))
    best = min(methods, key=lambda m: summary[f"full_{m}_loo"])
    out = {"description": "Sim(3) estimators: centres-only vs rotation-averaged vs joint.",
           "n_scenes": len(rows), "point_stride": stride,
           "summary": summary, "best_by_loo_full": best, "per_scene": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sim3_rotation.json").write_text(json.dumps(out, indent=2))

    print("\n=== means ===")
    for variant in ("full", "w4a4"):
        for m in methods:
            print(f"{variant:5s} {m:9s} LOO={summary[f'{variant}_{m}_loo']:.4f}   "
                  f"on-silhouette={summary[f'{variant}_{m}_on_silhouette']*100:.2f}%")
    print(f"\nbest by LOO (full): {best}")
    print(f"saved {OUT/'sim3_rotation.json'}")


if __name__ == "__main__":
    main(scenes=common.all_scenes() if "--all" in sys.argv else None)

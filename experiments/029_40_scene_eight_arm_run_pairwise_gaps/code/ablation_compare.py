"""Score W4A4 ablation arms by GEOMETRY error against full-precision VGGT.

WHY NOT SCORE THESE DOWNSTREAM
------------------------------
At W4A4 the downstream rendering difference from full precision is not statistically
resolvable: +0.1115 dB, p = 0.174, 95% CI [-0.051, +0.274] over 40 scenes
(ongoing_logs.md #020). An ablation measured through 3DGS would therefore be measuring
noise, at a cost of hours of training per arm. The quantities that DO respond are the
VGGT outputs themselves, and they cost ~2 s per scene.

WHAT IS MEASURED, per arm, against that scene's `full.npz`
    cam_rot_deg        mean camera-orientation error (degrees)
    cam_centre_err     camera-centre error after a Sim(3) fit, in scene radii
    cam_spread_ratio   predicted camera-centre spread / full's. ~1 is healthy; the W2A4
                       collapse showed up here as 0.01 (all six cameras at one pose), so
                       this is the guard against an arm that is broken rather than merely
                       degraded
    depth_rel_err      median |d_q - d_f| / d_f over valid pixels
    point_err          median world-point distance to full's, in scene radii, after Sim(3)

Sim(3) alignment is required because every variant predicts in its own arbitrary world
frame; comparing raw coordinates across arms is meaningless.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
import common  # noqa: E402

dv = common.dv
OUT = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")


def rot_angle_deg(A, B):
    c = np.clip((np.trace(A @ B.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def compare(cat, seq, variant):
    """Geometry error of `variant` against full precision for one scene."""
    group_dir, _fm, _wm = dv.find_group(cat, seq)
    f_npz, q_npz = group_dir / "full.npz", group_dir / f"{variant}.npz"
    if not q_npz.is_file():
        return None
    F, Q = dv.load_npz(f_npz), dv.load_npz(q_npz)

    Ef, Eq = F["extrinsic"].astype(np.float64), Q["extrinsic"].astype(np.float64)
    Cf, Cq = dv.camera_centers(Ef), dv.camera_centers(Eq)

    # Sim(3) from the quantized frame into full's frame, fitted on the six input cameras
    s, A, b = dv.umeyama(Cq, Cf)
    Cq_in_f = (s * (A @ Cq.T)).T + b

    radius = float(np.median(np.linalg.norm(Cf - np.median(Cf, axis=0), axis=1)))
    radius = max(radius, 1e-12)

    cam_rot = float(np.mean([rot_angle_deg(Ef[i][:, :3], Eq[i][:, :3] @ A.T) for i in range(len(Ef))]))
    cam_centre = float(np.median(np.linalg.norm(Cq_in_f - Cf, axis=1)) / radius)
    spread_f = float(np.median(np.linalg.norm(Cf - np.median(Cf, axis=0), axis=1)))
    spread_q = float(np.median(np.linalg.norm(Cq - np.median(Cq, axis=0), axis=1)))

    df, dq = F["depth"].astype(np.float64), Q["depth"].astype(np.float64)
    valid = (df > 1e-6) & np.isfinite(df) & np.isfinite(dq)
    depth_rel = float(np.median(np.abs(dq[valid] - df[valid]) / df[valid])) if valid.any() else float("nan")

    # world points, subsampled for speed, carried into full's frame by the same Sim(3)
    Pf = F["world_points_from_depth"].astype(np.float64).reshape(-1, 3)[::97]
    Pq = Q["world_points_from_depth"].astype(np.float64).reshape(-1, 3)[::97]
    Pq_in_f = (s * (A @ Pq.T)).T + b
    point_err = float(np.median(np.linalg.norm(Pq_in_f - Pf, axis=1)) / radius)

    return {"scene": f"{cat}/{seq}", "cam_rot_deg": cam_rot, "cam_centre_err": cam_centre,
            "cam_spread_ratio": spread_q / max(spread_f, 1e-12),
            "depth_rel_err": depth_rel, "point_err": point_err, "sim3_scale": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--scenes", choices=["subset", "all"], default="subset")
    ap.add_argument("--tag", default=None, help="output filename tag")
    a = ap.parse_args()

    scenes = common.EXPENSIVE_SCENES if a.scenes == "subset" else common.all_scenes()
    results = {}
    for v in a.variants:
        rows = [r for r in (compare(*common.split_scene(n), v) for n in scenes) if r]
        if not rows:
            print(f"{v}: no predictions found -- run run_w2a4_inference.py --variant {v}")
            continue
        agg = {k: float(np.mean([r[k] for r in rows]))
               for k in ("cam_rot_deg", "cam_centre_err", "cam_spread_ratio",
                         "depth_rel_err", "point_err")}
        agg["n_scenes"] = len(rows)
        results[v] = {"mean": agg, "per_scene": rows}

    OUT.mkdir(parents=True, exist_ok=True)
    name = f"w4a4_ablation_{a.tag or a.scenes}.json"
    (OUT / name).write_text(json.dumps(results, indent=2))

    print(f"\nGeometry error vs full-precision VGGT  ({a.scenes}, n={len(scenes)} scenes)")
    print(f"{'arm':16s} {'cam rot deg':>12s} {'cam centre':>11s} {'spread':>8s} "
          f"{'depth rel':>10s} {'point err':>10s}")
    print("-" * 72)
    for v, r in results.items():
        m = r["mean"]
        print(f"{v:16s} {m['cam_rot_deg']:12.4f} {m['cam_centre_err']:11.5f} "
              f"{m['cam_spread_ratio']:8.3f} {m['depth_rel_err']:10.5f} {m['point_err']:10.5f}")
    print(f"\nsaved {OUT/name}")
    print("cam_spread_ratio far below 1 means the pose head collapsed (cf. W2A4 at ~0.01).")


if __name__ == "__main__":
    main()

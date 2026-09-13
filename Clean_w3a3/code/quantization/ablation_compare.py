"""Score quantized VGGT variants by geometry error against full-precision VGGT.

WHY NOT SCORE THESE DOWNSTREAM
------------------------------
At W4A4, the downstream rendering difference from full precision is not
statistically resolvable: +0.1115 dB, p = 0.174,
95% CI [-0.051, +0.274] over 40 scenes.

An ablation measured through 3DGS would therefore be measuring noise,
at a cost of hours of training per arm.

The quantities that do respond are the VGGT outputs themselves.

WHAT IS MEASURED
----------------
Per arm, against that scene's full.npz:

    cam_rot_deg
        Mean camera-orientation error in degrees.

    cam_centre_err
        Camera-centre error after a Sim(3) fit, normalized by scene radius.

    cam_spread_ratio
        Predicted camera-centre spread divided by full-precision spread.
        ~1 is healthy.
        Very small values indicate pose collapse.
        W2A4 collapsed to about 0.01.

    depth_rel_err
        Median |d_q - d_f| / d_f over valid pixels.

    point_err
        Median world-point distance to full precision after Sim(3),
        normalized by scene radius.

Sim(3) alignment is required because every variant predicts in its own
arbitrary world coordinate frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
REPO = SCRIPT_PATH.parents[2]

CODE_DIR = REPO / "code"
DIAGNOSTICS_DIR = CODE_DIR / "downstream_3dgs" / "diagnostics"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(DIAGNOSTICS_DIR))

import common  # noqa: E402

dv = common.dv

OUT = Path("/var/tmp/poli22wo/quantsplat/oracle_sweep/results")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def rot_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    """Rotation difference between two 3x3 rotation matrices, in degrees."""
    c = np.clip(
        (np.trace(A @ B.T) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return float(np.degrees(np.arccos(c)))


def compare(cat: str, seq: str, variant: str):
    """Geometry error of `variant` against full precision for one scene."""

    group_dir, _full_meta, _w4_meta = dv.find_group(cat, seq)

    full_npz = group_dir / "full.npz"
    quant_npz = group_dir / f"{variant}.npz"

    if not full_npz.is_file():
        print(f"WARNING: missing full reference: {full_npz}")
        return None

    if not quant_npz.is_file():
        print(f"WARNING: missing {variant}: {quant_npz}")
        return None

    F = dv.load_npz(full_npz)
    Q = dv.load_npz(quant_npz)

    # -------------------------------------------------------------
    # Camera geometry
    # -------------------------------------------------------------

    Ef = F["extrinsic"].astype(np.float64)
    Eq = Q["extrinsic"].astype(np.float64)

    Cf = dv.camera_centers(Ef)
    Cq = dv.camera_centers(Eq)

    # Sim(3) from quantized frame into full-precision frame.
    # Fitted using the six input camera centres.
    s, A, b = dv.umeyama(Cq, Cf)

    Cq_in_f = (s * (A @ Cq.T)).T + b

    # Scene radius based on full-precision camera configuration.
    centre_f = np.median(Cf, axis=0)

    radius = float(
        np.median(
            np.linalg.norm(
                Cf - centre_f,
                axis=1,
            )
        )
    )

    radius = max(radius, 1e-12)

    # Camera rotation error after applying Sim(3) rotation.
    cam_rot = float(
        np.mean(
            [
                rot_angle_deg(
                    Ef[i][:, :3],
                    Eq[i][:, :3] @ A.T,
                )
                for i in range(len(Ef))
            ]
        )
    )

    # Camera-centre residual after Sim(3).
    cam_centre = float(
        np.median(
            np.linalg.norm(
                Cq_in_f - Cf,
                axis=1,
            )
        )
        / radius
    )

    # Camera spread ratio.
    spread_f = float(
        np.median(
            np.linalg.norm(
                Cf - np.median(Cf, axis=0),
                axis=1,
            )
        )
    )

    spread_q = float(
        np.median(
            np.linalg.norm(
                Cq - np.median(Cq, axis=0),
                axis=1,
            )
        )
    )

    cam_spread_ratio = spread_q / max(spread_f, 1e-12)

    # -------------------------------------------------------------
    # Depth error
    # -------------------------------------------------------------

    df = F["depth"].astype(np.float64)
    dq = Q["depth"].astype(np.float64)

    valid = (
        (df > 1e-6)
        & np.isfinite(df)
        & np.isfinite(dq)
    )

    if valid.any():
        depth_rel = float(
            np.median(
                np.abs(dq[valid] - df[valid]) / df[valid]
            )
        )
    else:
        depth_rel = float("nan")

    # -------------------------------------------------------------
    # World-point error
    # -------------------------------------------------------------

    # Subsample for speed.
    Pf = (
        F["world_points_from_depth"]
        .astype(np.float64)
        .reshape(-1, 3)[::97]
    )

    Pq = (
        Q["world_points_from_depth"]
        .astype(np.float64)
        .reshape(-1, 3)[::97]
    )

    # Transform quantized points into full-precision world frame.
    Pq_in_f = (s * (A @ Pq.T)).T + b

    point_err = float(
        np.median(
            np.linalg.norm(
                Pq_in_f - Pf,
                axis=1,
            )
        )
        / radius
    )

    return {
        "scene": f"{cat}/{seq}",
        "cam_rot_deg": cam_rot,
        "cam_centre_err": cam_centre,
        "cam_spread_ratio": cam_spread_ratio,
        "depth_rel_err": depth_rel,
        "point_err": point_err,
        "sim3_scale": float(s),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variants",
        nargs="+",
        required=True,
        help="Variants to compare, e.g. w3a3 w4a4",
    )

    parser.add_argument(
        "--scenes",
        choices=["subset", "all"],
        default="subset",
    )

    parser.add_argument(
        "--tag",
        default=None,
        help="Optional output filename tag",
    )

    args = parser.parse_args()

    FROZEN = REPO / "datasets" / "frozen_dataset_manifest.json"

    if args.scenes == "subset":
        scenes = common.EXPENSIVE_SCENES
    else:
        manifest = json.loads(FROZEN.read_text())
        scenes = [
            f"{s['category']}/{s['sequence']}"
            for s in manifest["scenes"]
        ]

    results = {}

    for variant in args.variants:
        rows = []

        for scene_name in scenes:
            cat, seq = common.split_scene(scene_name)

            row = compare(
                cat,
                seq,
                variant,
            )

            if row is not None:
                rows.append(row)

        if not rows:
            print(
                f"{variant}: no predictions found. "
                f"Run run_w2a4_inference.py --variant {variant}"
            )
            continue

        metric_names = (
            "cam_rot_deg",
            "cam_centre_err",
            "cam_spread_ratio",
            "depth_rel_err",
            "point_err",
        )

        aggregate = {
            key: float(
                np.mean(
                    [row[key] for row in rows]
                )
            )
            for key in metric_names
        }

        aggregate["n_scenes"] = len(rows)

        results[variant] = {
            "mean": aggregate,
            "per_scene": rows,
        }

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = args.tag or args.scenes
    output_name = f"geometry_ablation_{tag}.json"
    output_path = OUT / output_name

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n"
    )

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------

    print()
    print(
        f"Geometry error vs full-precision VGGT "
        f"({args.scenes}, requested n={len(scenes)} scenes)"
    )

    print(
        f"{'arm':16s}"
        f"{'cam rot deg':>14s}"
        f"{'cam centre':>14s}"
        f"{'spread':>10s}"
        f"{'depth rel':>12s}"
        f"{'point err':>12s}"
    )

    print("-" * 78)

    for variant, result in results.items():
        m = result["mean"]

        print(
            f"{variant:16s}"
            f"{m['cam_rot_deg']:14.4f}"
            f"{m['cam_centre_err']:14.5f}"
            f"{m['cam_spread_ratio']:10.3f}"
            f"{m['depth_rel_err']:12.5f}"
            f"{m['point_err']:12.5f}"
        )

    print()
    print(f"Saved: {output_path}")
    print(
        "cam_spread_ratio far below 1 indicates pose-head collapse "
        "(W2A4 was approximately 0.01)."
    )


if __name__ == "__main__":
    main()
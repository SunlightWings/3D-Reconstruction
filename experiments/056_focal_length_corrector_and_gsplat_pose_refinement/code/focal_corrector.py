"""Correct the systematic focal-length bias of a quantized VGGT prior BEFORE 3DGS, and write it as a new arm.

Why this and not point confidence: #052/#053 show W3A3's rendering damage is camera-borne and point-level
intervention recovers nothing. Of the camera error, focal length is the part that is (a) systematic -- W3A3
overestimates fx on 46/48 cameras, mean ratio 1.101, while full precision matches GT to 0.998 -- and therefore
learnable, and (b) the part order-averaging could NOT fix (#048, p 0.485). Pose cannot be refined inside the
installed GraphDECO rasterizer (no camera gradients), so it is out of scope here.

Methods (each writes <arm>_K<method>.npz next to the arm's npz, consumed unchanged by run_w2a4_downstream.py):
  oracle  intrinsics replaced by full precision's, per camera. The CEILING of any focal corrector; uses the
          full-precision model, so it is not deployable.
  const   divide fx, fy by the median per-scene bias of the OTHER scenes (leave-one-scene-out). No feature
          selection, no test-scene information: the clean deployable predictor.
  iqr     leave-one-scene-out linear fit of log(bias) on the arm's own depth IQR. The feature was chosen after
          looking at the same 8 scenes (rho -0.857), so on W3A3 this is EXPLORATORY and must be reported as such.

After changing K, world_points_from_depth is recomputed from the arm's own depth and extrinsics
(X_w = R^T (K^-1 [u v 1]^T d - t)); that reprojection was verified to reproduce the saved array to 2e-8.
Extrinsics and depth are untouched, so the only thing that differs from the uncorrected arm is focal length.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
import common  # noqa: E402

dv = common.dv
RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")


def unproject(depth, K, E):
    n, H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    out = np.empty((n, H, W, 3), np.float64)
    for i in range(n):
        d = depth[i]
        Xc = np.stack([(u - K[i, 0, 2]) / K[i, 0, 0] * d, (v - K[i, 1, 2]) / K[i, 1, 1] * d, d], -1)
        R, t = E[i, :, :3], E[i, :, 3]
        out[i] = ((Xc.reshape(-1, 3) - t) @ R).reshape(H, W, 3)
    return out


def focal_err(K, Kf):
    r = np.concatenate([K[:, 0, 0] / Kf[:, 0, 0], K[:, 1, 1] / Kf[:, 1, 1]])
    return float(np.median(np.abs(r - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="w3a3")
    ap.add_argument("--scenes", choices=["subset", "all"], default="subset")
    ap.add_argument("--methods", nargs="+", default=["oracle", "const", "iqr"])
    a = ap.parse_args()

    names = common.EXPENSIVE_SCENES if a.scenes == "subset" else common.all_scenes()
    data = []
    for n in names:
        c, s = n.split("/", 1)
        g, meta, _w = dv.find_group(c, s)
        if not (g / f"{a.arm}.npz").is_file():
            continue
        F, Q = dv.load_npz(g / "full.npz"), dv.load_npz(g / f"{a.arm}.npz")
        d = Q["depth"][..., 0].astype(np.float64)
        data.append({"scene": n, "g": g, "meta": meta, "F": F, "Q": Q,
                     "bias": float(np.mean(np.concatenate([Q["intrinsic"][:, 0, 0] / F["intrinsic"][:, 0, 0],
                                                            Q["intrinsic"][:, 1, 1] / F["intrinsic"][:, 1, 1]]))),
                     "iqr": float(np.subtract(*np.percentile(d, [75, 25])))})
    print(f"{a.arm}: {len(data)} scenes with a prediction", flush=True)

    rows = []
    for i, r in enumerate(data):
        others = [x for j, x in enumerate(data) if j != i]
        Kq = r["Q"]["intrinsic"].astype(np.float64)
        Kf = r["F"]["intrinsic"].astype(np.float64)
        # self-check: the unprojection must reproduce the arm's saved points with its own K before we trust it
        E0 = r["Q"]["extrinsic"].astype(np.float64)
        rep = unproject(r["Q"]["depth"][..., 0].astype(np.float64), Kq, E0)
        rep_err = float(np.abs(rep - r["Q"]["world_points_from_depth"].astype(np.float64)).max())
        if rep_err > 1e-3:
            raise SystemExit(f"{r['scene']}: unprojection does not reproduce saved points (max err {rep_err:.3e})")
        row = {"scene": r["scene"], "true_bias": r["bias"], "depth_iqr": r["iqr"], "focal_err_before": focal_err(Kq, Kf),
               "unproject_selfcheck_max_err": rep_err}
        for m in a.methods:
            if m == "oracle":
                Knew = Kq.copy(); Knew[:, 0, 0] = Kf[:, 0, 0]; Knew[:, 1, 1] = Kf[:, 1, 1]
                k = None
            elif m == "const":
                k = float(np.median([x["bias"] for x in others]))
            elif m == "iqr":
                b, c0 = np.polyfit([x["iqr"] for x in others], np.log([x["bias"] for x in others]), 1)
                k = float(np.exp(c0 + b * r["iqr"]))
            else:
                raise SystemExit(f"unknown method {m}")
            if k is not None:
                Knew = Kq.copy(); Knew[:, 0, 0] /= k; Knew[:, 1, 1] /= k
            E = r["Q"]["extrinsic"].astype(np.float64)
            pts = unproject(r["Q"]["depth"][..., 0].astype(np.float64), Knew, E)
            tag = f"{a.arm}_K{m}"
            out = r["g"] / f"{tag}.npz"
            np.savez(out, frame_numbers=r["Q"]["frame_numbers"], depth=r["Q"]["depth"],
                     intrinsic=Knew.astype(np.float32), extrinsic=r["Q"]["extrinsic"],
                     world_points_from_depth=pts.astype(np.float32))
            (r["g"] / f"{tag}_meta.json").write_text(json.dumps({
                **{k2: r["meta"].get(k2) for k2 in ("category", "sequence", "group_id", "frame_numbers",
                                                     "input_semantic_sha256", "frozen_manifest_sha256")},
                "variant": tag, "derived_from": f"{a.arm}.npz", "method": m,
                "predicted_bias": k, "leave_one_scene_out": m != "oracle",
                "note": "focal-corrected prior; extrinsics and depth identical to the source arm"}, indent=2) + "\n")
            row[f"{m}_predicted_bias"] = k
            row[f"{m}_focal_err_after"] = focal_err(Knew, Kf)
        rows.append(row)
        print(f"  {r['scene'][:28]:28s} bias {r['bias']:.4f}  err before {row['focal_err_before']:.4f}  " +
              "  ".join(f"{m}: {row.get(f'{m}_predicted_bias') or 0:.4f}->{row[f'{m}_focal_err_after']:.4f}"
                        for m in a.methods), flush=True)

    print("\nmedian-over-cameras focal error, mean over scenes:")
    print(f"  before  {np.mean([x['focal_err_before'] for x in rows]):.4f}")
    for m in a.methods:
        print(f"  {m:6s}  {np.mean([x[f'{m}_focal_err_after'] for x in rows]):.4f}")
    RES.mkdir(parents=True, exist_ok=True)
    p = RES / f"focal_corrector_{a.arm}_{a.scenes}.json"
    p.write_text(json.dumps({"arm": a.arm, "scenes": a.scenes, "methods": a.methods, "per_scene": rows}, indent=2))
    print(f"saved {p}")


if __name__ == "__main__":
    main()

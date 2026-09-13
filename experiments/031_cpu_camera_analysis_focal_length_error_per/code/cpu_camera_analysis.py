"""CPU-only analyses of predicted cameras, run while the GPU is busy. No training, no rendering.

Motivated by the swap probe (geometry_probes.py): with 25/40 scenes in, broken CAMERAS account for
essentially the whole rendering loss and broken POINTS for none of it. Three follow-ups:

  1. focal   The swap's "camera" condition replaced pose AND intrinsics together, so "cameras
             matter" could mean "focal length matters". Measure each arm's focal-length error
             beside its pose error, so the swap result can be read correctly.

  2. scenes  Per scene, per arm (40 x 7 = 280 pairs): which geometry error -- rotation, centre,
             focal, depth, points -- predicts that scene's rendering loss? Arm-level
             correlations (#029) have n = 6; this has n = 280, and a within-w4a4 slice (n = 40)
             that asks whether a per-scene signal exists at the configuration people ship.

  3. context How big is quantization's camera error next to VGGT's OWN camera error against
             CO3D ground truth? Measured on all 1330 view-groups of the disagreement dataset,
             paired full vs w4a4, so "w4a4 adds X deg" has a denominator.

All pose errors use the same construction as ablation_compare.py: Sim(3) fitted on the six input
camera centres, then mean per-camera rotation angle and median centre residual in scene radii.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
import common  # noqa: E402

dv = common.dv
RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
PRED = Path("/var/tmp/luli38se/quantsplat/disagreement_dataset_v1/run/predictions")
ARMS = ["w4a4", "w4a4_local", "w4a4_no_lwc", "w4a4_no_smooth", "w4a4_no_lac", "w4a4_no_rot", "w4a4_rtn"]


def angle(R1, R2):
    return float(np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1))))


def pose_err(E_ref, E_q):
    """Rotation (deg, mean) and centre (scene radii, median) of E_q against E_ref after Sim(3)."""
    Cr, Cq = dv.camera_centers(E_ref), dv.camera_centers(E_q)
    s, A, b = dv.umeyama(Cq, Cr)
    radius = max(float(np.median(np.linalg.norm(Cr - np.median(Cr, axis=0), axis=1))), 1e-12)
    rot = float(np.mean([angle(E_ref[i][:, :3], E_q[i][:, :3] @ A.T) for i in range(len(E_ref))]))
    cen = float(np.median(np.linalg.norm((s * (A @ Cq.T)).T + b - Cr, axis=1)) / radius)
    return rot, cen


def focal_err(K_ref, K_q):
    """Median |f_q / f_ref - 1| over cameras and both axes."""
    r = np.concatenate([K_q[:, 0, 0] / K_ref[:, 0, 0], K_q[:, 1, 1] / K_ref[:, 1, 1]])
    return float(np.median(np.abs(r - 1)))


def load(npz, keys=("extrinsic", "intrinsic")):
    z = np.load(npz)
    return {k: z[k].astype(np.float64) for k in keys}


def gt_cameras(cat, seq, frames):
    recs = dv.load_annotations(cat, seq)
    E, K = [], []
    for f in frames:
        e, k0 = dv.co3d_to_opencv_camera(recs[int(f)])
        E.append(e)
        K.append(dv.original_to_518_affine(recs[int(f)]) @ k0)
    return np.stack(E), np.stack(K)


# --------------------------------------------------------------------------- 1 + 2

def per_scene():
    downstream = json.load(open(sorted(RES.glob("arms_full_*_all_it7000.json"))[-1]))
    render = {r["scene"]: r for r in downstream["per_scene"]}
    rows = []
    for name in common.all_scenes():
        cat, seq = name.split("/", 1)
        gdir, _fm, _wm = dv.find_group(cat, seq)
        F = load(gdir / "full.npz", ("extrinsic", "intrinsic", "depth", "world_points_from_depth"))
        rF = render[name]
        for arm in ARMS:
            Q = load(gdir / f"{arm}.npz", ("extrinsic", "intrinsic", "depth", "world_points_from_depth"))
            rot, cen = pose_err(F["extrinsic"], Q["extrinsic"])
            fe = focal_err(F["intrinsic"], Q["intrinsic"])
            valid = (F["depth"] > 1e-6) & np.isfinite(Q["depth"])
            dep = float(np.median(np.abs(Q["depth"][valid] - F["depth"][valid]) / F["depth"][valid]))
            Cf, Cq = dv.camera_centers(F["extrinsic"]), dv.camera_centers(Q["extrinsic"])
            s, A, b = dv.umeyama(Cq, Cf)
            radius = max(float(np.median(np.linalg.norm(Cf - np.median(Cf, axis=0), axis=1))), 1e-12)
            Pf = F["world_points_from_depth"].reshape(-1, 3)[::97]
            Pq = Q["world_points_from_depth"].reshape(-1, 3)[::97]
            pe = float(np.median(np.linalg.norm((s * (A @ Pq.T)).T + b - Pf, axis=1)) / radius)
            rows.append({
                "scene": name, "arm": arm, "cam_rot": rot, "cam_centre": cen, "focal": fe,
                "depth_rel": dep, "point_err": pe,
                "d_psnr": rF["full_foreground_psnr"] - rF[f"{arm}_foreground_psnr"],
                "d_lpips": rF[f"{arm}_foreground_lpips"] - rF["full_foreground_lpips"],
            })
    return rows


def report_focal(rows):
    print("=" * 90)
    print("1. FOCAL LENGTH vs POSE error, each arm against full precision (n = 40 scenes)")
    print(f"{'arm':16s} {'cam rot deg':>12s} {'centre radii':>13s} {'focal |ratio-1|':>16s}")
    out = {}
    for arm in ARMS:
        v = [r for r in rows if r["arm"] == arm]
        m = {k: float(np.mean([r[k] for r in v])) for k in ("cam_rot", "cam_centre", "focal")}
        out[arm] = m
        print(f"{arm:16s} {m['cam_rot']:12.4f} {m['cam_centre']:13.5f} {m['focal']*100:15.3f}%")
    return out


def report_scenes(rows):
    feats = ("cam_rot", "cam_centre", "focal", "depth_rel", "point_err")
    out = {}
    print("\n" + "=" * 90)
    print("2. WHICH geometry error predicts a scene's rendering loss? Spearman rho")
    for label, sub in (("all arms (n = 280)", rows),
                       ("w4a4 only (n = 40)", [r for r in rows if r["arm"] == "w4a4"]),
                       ("mild arms, rtn and no_rot excluded (n = 200)",
                        [r for r in rows if r["arm"] not in ("w4a4_rtn", "w4a4_no_rot")])):
        print(f"\n  {label}")
        out[label] = {}
        for tgt in ("d_psnr", "d_lpips"):
            line = f"    {tgt:8s}"
            for f in feats:
                rho, p = stats.spearmanr([r[f] for r in sub], [r[tgt] for r in sub])
                out[label][f"{tgt}~{f}"] = {"rho": float(rho), "p": float(p)}
                star = "*" if p < 0.05 else " "
                line += f"  {f}={rho:+.3f}{star}(p={p:.2g})"
            print(line)
    print("\n  * p < 0.05.  d_psnr, d_lpips: positive = arm renders WORSE than full on that scene")
    return out


# --------------------------------------------------------------------------- 3

def context():
    groups = sorted(glob.glob(str(PRED / "*" / "*" / "*" / "full.npz")))
    rows = []
    for g in groups:
        g = Path(g)
        cat, seq = g.parts[-4], g.parts[-3]
        try:
            F, W = load(g), load(g.parent / "w4a4.npz")
            frames = np.load(g)["frame_numbers"]
            Eg, Kg = gt_cameras(cat, seq, frames)
        except Exception as e:  # noqa: BLE001 -- one unreadable group must not sink 1329 others
            rows.append({"group": str(g.parent), "error": f"{type(e).__name__}: {e}"})
            continue
        fr, fc = pose_err(Eg, F["extrinsic"])
        wr, wc = pose_err(Eg, W["extrinsic"])
        qr, qc = pose_err(F["extrinsic"], W["extrinsic"])
        rows.append({"group": str(g.parent), "full_vs_gt_rot": fr, "w4a4_vs_gt_rot": wr,
                     "w4a4_vs_full_rot": qr, "full_vs_gt_centre": fc, "w4a4_vs_gt_centre": wc,
                     "full_vs_gt_focal": focal_err(Kg, F["intrinsic"]),
                     "w4a4_vs_gt_focal": focal_err(Kg, W["intrinsic"])})
    ok = [r for r in rows if "error" not in r]
    print("\n" + "=" * 90)
    print(f"3. CONTEXT: quantization error next to VGGT's own error vs CO3D ground truth "
          f"({len(ok)} view-groups, {len(rows) - len(ok)} unreadable)")

    def q(k):
        a = np.array([r[k] for r in ok])
        return f"mean {a.mean():8.4f}  median {np.median(a):8.4f}  p90 {np.percentile(a, 90):8.4f}  " \
               f"p99 {np.percentile(a, 99):8.4f}  max {a.max():8.4f}"
    for k in ("full_vs_gt_rot", "w4a4_vs_gt_rot", "w4a4_vs_full_rot",
              "full_vs_gt_centre", "w4a4_vs_gt_centre", "full_vs_gt_focal", "w4a4_vs_gt_focal"):
        print(f"  {k:20s} {q(k)}")
    d = np.array([r["w4a4_vs_gt_rot"] - r["full_vs_gt_rot"] for r in ok])
    t, p = stats.ttest_1samp(d, 0)
    w = stats.wilcoxon(d)
    print(f"\n  paired: w4a4 minus full, rotation error vs GT: mean {d.mean():+.4f} deg, "
          f"median {np.median(d):+.4f}, t-test p={p:.3g}, Wilcoxon p={w.pvalue:.3g}, "
          f"w4a4 worse in {int((d > 0).sum())}/{len(d)}")
    a = np.array([r["w4a4_vs_full_rot"] for r in ok])
    for thr in (2, 5, 10):
        print(f"  groups where w4a4 departs from full by > {thr:2d} deg: {int((a > thr).sum())}/{len(a)}")
    return rows


def main():
    rows = per_scene()
    out = {"focal": report_focal(rows), "scenes": report_scenes(rows), "per_scene_rows": rows}
    out["context_rows"] = context()
    p = RES / "cpu_camera_analysis.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()

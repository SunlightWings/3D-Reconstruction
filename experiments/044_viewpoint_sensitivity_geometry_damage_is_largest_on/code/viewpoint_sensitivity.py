"""Does the benchmark only test interpolation? Per-held-out-view sensitivity vs distance from the training views.

#039-#043 measured rendering damage averaged over the nine held-out views of each scene. Those views
were chosen from the same CO3D orbit as the six training views, so every one of them is an
interpolation between cameras the model already saw. If geometry errors only show up when a render
has to EXTRAPOLATE, then the insensitivity measured so far is a property of the protocol, not of 3DGS
-- and the fix is to evaluate on distant views, not to give up on the metric.

This re-scores existing renders per view (no training, no rendering) and asks whether the damage from
a condition grows with that view's distance from the nearest training camera. Two distances, both in
degrees:

  axis_angle   angle between the held-out camera's optical axis and the nearest training camera's
  orbit_angle  angle subtended at the scene centroid between the held-out camera centre and the
               nearest training camera centre -- the natural one for CO3D's turntable capture

Conditions compared against `full`: the swap cells (bad points / bad cameras / both), the pose-only and
focal-only cells, and the measured arms w4a4 and w4a4_rtn.

The test is a per-scene Spearman between view distance and the per-view damage, then a Wilcoxon over
scenes -- never a pooled correlation over scene x view pairs, which would treat correlated views as
independent (the error made and corrected in #042).
"""
from __future__ import annotations

import argparse
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
CONDS = ["w4a4", "w4a4_rtn", "swapP_w4a4_rtn", "swapK_w4a4_rtn", "swapB_w4a4_rtn",
         "swapR_w4a4_rtn", "swapF_w4a4_rtn"]


def view_distances(cat, seq):
    """For each held-out view: degrees to the nearest training view, by optical axis and by orbit angle."""
    man = common.scene_manifest(cat, seq)
    recs = dv.load_annotations(cat, seq)
    E_tr = np.stack([dv.co3d_to_opencv_camera(recs[f])[0] for f in man["input_frames"]])
    E_ho = np.stack([dv.co3d_to_opencv_camera(recs[f])[0] for f in man["heldout_frames"]])
    C_tr, C_ho = dv.camera_centers(E_tr), dv.camera_centers(E_ho)
    centroid = C_tr.mean(axis=0)
    ax_tr = E_tr[:, 2, :3]                      # third row of R is the optical axis in world coords
    ax_ho = E_ho[:, 2, :3]
    out = []
    for i in range(len(E_ho)):
        ang = np.degrees(np.arccos(np.clip(ax_tr @ ax_ho[i], -1, 1)))
        v_ho = C_ho[i] - centroid
        orb = [np.degrees(np.arccos(np.clip(np.dot(v_ho, c - centroid) /
                                            (np.linalg.norm(v_ho) * np.linalg.norm(c - centroid)), -1, 1)))
               for c in C_tr]
        out.append({"axis_angle": float(ang.min()), "orbit_angle": float(min(orb))})
    return man, recs, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=7000)
    a = ap.parse_args()

    from PIL import Image
    import lpips as lpips_lib
    import torch
    net = lpips_lib.LPIPS(net="alex").cpu().eval()

    def lp(x, y, m):
        ys, xs = dv.bbox_from_mask(m)
        tx = torch.from_numpy(np.ascontiguousarray(x[ys, xs])).permute(2, 0, 1)[None].float() * 2 - 1
        ty = torch.from_numpy(np.ascontiguousarray(y[ys, xs])).permute(2, 0, 1)[None].float() * 2 - 1
        with torch.no_grad():
            return float(net(tx, ty).item())

    rows = []
    for si, name in enumerate(common.all_scenes(), 1):
        cat, seq = name.split("/", 1)
        root = common.scene_root(cat, seq)
        man, recs, dists = view_distances(cat, seq)
        held = man["heldout_frames"]
        gts = [root / "common" / "heldout_images" / f"frame{f:06d}.png" for f in held]
        dirs = {c: root / "heldout" / f"{c}_it{a.iterations}" / "renders" for c in ["full"] + CONDS}
        dirs = {c: d for c, d in dirs.items() if d.is_dir() and len(sorted(d.glob("*.png"))) == len(held)}
        if "full" not in dirs:
            continue
        per = {c: sorted(d.glob("*.png")) for c, d in dirs.items()}
        for i, f in enumerate(held):
            m = dv.foreground_mask_from_record(recs[f])
            if not m.any():
                continue
            g = np.asarray(Image.open(gts[i]).convert("RGB"), np.float32) / 255.0
            row = {"scene": name, "view": int(f), **dists[i]}
            for c, paths in per.items():
                r = np.asarray(Image.open(paths[i]).convert("RGB"), np.float32) / 255.0
                row[f"{c}_psnr"] = dv.psnr(r, g, m)
                row[f"{c}_lpips"] = lp(r, g, m)
            rows.append(row)
        print(f"[{si}/40] {name}  conditions: {len(per)}", flush=True)

    print("\n" + "=" * 100)
    print("VIEW DISTANCE vs DAMAGE. Per scene: Spearman(distance, per-view damage); then Wilcoxon over scenes.")
    print("Positive rho = the condition hurts MORE on views further from the training cameras.")
    d_all = np.array([r["orbit_angle"] for r in rows])
    print(f"held-out view distance to nearest training view: orbit angle median {np.median(d_all):.2f} deg, "
          f"range {d_all.min():.2f}-{d_all.max():.2f}; axis angle median "
          f"{np.median([r['axis_angle'] for r in rows]):.2f} deg")
    out = {"per_view": rows, "tests": {}}
    for dist in ("orbit_angle", "axis_angle"):
        print(f"\n  distance = {dist}")
        for cond in CONDS:
            for m, sgn in (("psnr", 1), ("lpips", -1)):
                rhos, scenes = [], sorted({r["scene"] for r in rows})
                for s in scenes:
                    sub = [r for r in rows if r["scene"] == s and f"{cond}_{m}" in r]
                    if len(sub) < 4:
                        continue
                    dmg = [sgn * (r[f"full_{m}"] - r[f"{cond}_{m}"]) for r in sub]
                    if np.std(dmg) == 0:
                        continue
                    rhos.append(stats.spearmanr([r[dist] for r in sub], dmg).statistic)
                rhos = [x for x in rhos if np.isfinite(x)]
                if len(rhos) < 5:
                    continue
                w = stats.wilcoxon(rhos)
                out["tests"][f"{dist}|{cond}|{m}"] = {
                    "n_scenes": len(rhos), "median_rho": float(np.median(rhos)),
                    "positive": int(sum(x > 0 for x in rhos)), "wilcoxon_p": float(w.pvalue)}
                star = "*" if w.pvalue < 0.05 else " "
                print(f"   {star}{cond:18s} {m:6s} median rho {np.median(rhos):+.3f}  "
                      f"positive {sum(x > 0 for x in rhos):2d}/{len(rhos)}  Wilcoxon p={w.pvalue:.3g}")
    RES.mkdir(parents=True, exist_ok=True)
    p = RES / "viewpoint_sensitivity.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()

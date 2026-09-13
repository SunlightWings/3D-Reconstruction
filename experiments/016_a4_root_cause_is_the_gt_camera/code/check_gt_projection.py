"""A4 root-cause probe: is the GT camera path self-consistent?

A4 reports only 9-29% of projected points landing on the object silhouette. That
number is computed for VGGT points through Sim(3)-aligned cameras, so it mixes three
possible faults: VGGT geometry, the Sim(3) alignment, and the GT camera/intrinsics path
itself.

This isolates the last one with a test that has a known answer. Take the points that
were unprojected FROM frame f's own GT depth using frame f's own GT camera, and project
them straight back through that same camera. Every point must land inside frame f's own
silhouette -- that is a round trip through unproject/project with no alignment, no
prediction, and no other view involved. Anything below ~90% is a bug in the GT path:
the pad affine on the intrinsics, the CAM_FLIP convention, or the NDC->pixel conversion.

It then repeats the projection into OTHER frames' cameras, which additionally exercises
the world-frame consistency of the GT extrinsics across views.
"""
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP.parent / "diagnostics"))
import common  # noqa: E402
import gate_a  # noqa: E402

dv = common.dv
OUT = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
SELF_TARGET = 0.90


def on_silhouette(xyz, E, K, fg):
    """Fraction of points that project inside the image and land on the object mask."""
    cam = (E[:3, :3] @ xyz.T + E[:3, 3:4]).T
    z = cam[:, 2]
    ok = z > 1e-6
    if not ok.any():
        return 0.0, 0.0
    uv = (K @ cam[ok].T).T
    u = np.round(uv[:, 0] / uv[:, 2]).astype(np.int64)
    v = np.round(uv[:, 1] / uv[:, 2]).astype(np.int64)
    H, W = fg.shape
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inside.any():
        return 0.0, 0.0
    hit = fg[v[inside], u[inside]]
    return float(hit.mean()), float(inside.mean())


def main():
    rows = []
    for name in common.EXPENSIVE_SCENES:
        cat, seq = common.split_scene(name)
        recs = dv.load_annotations(cat, seq)
        frames = common.scene_manifest(cat, seq)["input_frames"]

        self_fracs, cross_fracs, in_frame = [], [], []
        per_frame = {}
        for f in frames:
            d = gate_a._oracle_decode_frame(recs[f])
            pts = d["world"][d["valid"]]
            if len(pts) == 0:
                continue
            # (a) project back through the SAME frame's camera -- must be ~1.0
            fg_self = dv.foreground_mask_from_record(recs[f])
            s, ins = on_silhouette(pts, d["E"], d["K"], fg_self)
            self_fracs.append(s)
            in_frame.append(ins)
            per_frame[int(f)] = {"self_on_silhouette": s, "self_inside_image": ins}
            # (b) project into the OTHER input frames' cameras
            for g in frames:
                if g == f:
                    continue
                dg = gate_a._oracle_decode_frame(recs[g])
                sc, _ = on_silhouette(pts, dg["E"], dg["K"], dv.foreground_mask_from_record(recs[g]))
                cross_fracs.append(sc)

        rows.append({
            "scene": name,
            "self_on_silhouette": float(np.mean(self_fracs)),
            "self_inside_image": float(np.mean(in_frame)),
            "cross_on_silhouette": float(np.mean(cross_fracs)),
        })
        print(f"{name:35s} self={rows[-1]['self_on_silhouette']*100:5.1f}%  "
              f"(inside image {rows[-1]['self_inside_image']*100:5.1f}%)   "
              f"cross-view={rows[-1]['cross_on_silhouette']*100:5.1f}%")

    mean_self = float(np.mean([r["self_on_silhouette"] for r in rows]))
    mean_cross = float(np.mean([r["cross_on_silhouette"] for r in rows]))
    passed = mean_self >= SELF_TARGET
    out = {
        "description": "GT points projected back through their own GT camera (self) and through the "
                       "other input frames' GT cameras (cross-view).",
        "self_target": SELF_TARGET,
        "mean_self_on_silhouette": mean_self,
        "mean_cross_on_silhouette": mean_cross,
        "passed": passed,
        "verdict": ("GT camera path is self-consistent; A4's low on-silhouette is NOT explained by "
                    "the intrinsics/pad-affine/CAM_FLIP path"
                    if passed else
                    "GT camera path is BROKEN -- the bug is in the GT path, not in VGGT"),
        "per_scene": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gt_projection_check.json").write_text(json.dumps(out, indent=2))
    print(f"\nmean self on-silhouette : {mean_self*100:.2f}%  (target >= {SELF_TARGET*100:.0f}%)")
    print(f"mean cross-view         : {mean_cross*100:.2f}%")
    print(f"VERDICT: {out['verdict']}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Is exposure-corrected LPIPS worse because of the CLIP, or because of the correction itself?

#028 flagged an unexplained direction disagreement: fitting a per-image gain+bias moves PSNR
and SSIM up (better) but moves LPIPS DOWN in quality (0.6553 -> 0.6898 for `full`, higher is
worse). The stated hypothesis was that clipping to [0,1] after the affine destroys highlight
detail that LPIPS is sensitive to. That was a hypothesis, never a measurement, and #028 says
exposure-corrected LPIPS must not be a headline metric until it is tested.

This separates the two candidate causes by scoring three versions of every render:

  raw          no correction at all
  clipped      gain+bias, then clip to [0,1]        <- what the pipeline currently reports
  unclipped    gain+bias, no clip                   <- isolates the clip

If `unclipped` recovers to at or below `raw`, the clip is the cause and the fix is to stop
clipping. If `unclipped` is also worse than `raw`, the affine itself is what LPIPS dislikes,
and exposure-corrected LPIPS should be dropped rather than repaired.

`clipped_fraction` reports how many masked pixels the clip actually alters, which is the
direct evidence either way: a clip that touches almost nothing cannot explain a large gap.

Re-scores EXISTING renders. No training, no GPU.
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

OUT = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")


def fit_affine(render, gt, mask):
    """Per-channel least-squares gain+bias, render -> gt, over masked pixels. No clipping."""
    out = render.copy()
    for c in range(3):
        x, y = render[..., c][mask], gt[..., c][mask]
        if len(x) < 10 or float(np.var(x)) < 1e-12:
            continue
        sol, *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
        out[..., c] = render[..., c] * sol[0] + sol[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", choices=["subset", "all"], default="all")
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--arms", nargs="+", default=["full", "w4a4", "w4a4_local", "w4a4_rtn"])
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

    scenes = common.EXPENSIVE_SCENES if a.scenes == "subset" else common.all_scenes()
    rows = []
    for i, name in enumerate(scenes, 1):
        cat, seq = name.split("/", 1)
        root = common.scene_root(cat, seq)
        man = common.scene_manifest(cat, seq)
        recs = dv.load_annotations(cat, seq)
        held = man["heldout_frames"]
        gt_paths = [root / "common" / "heldout_images" / f"frame{f:06d}.png" for f in held]
        print(f"[{i}/{len(scenes)}] {name}", flush=True)

        row = {"scene": name}
        for arm in a.arms:
            rdir = root / "heldout" / f"{arm}_it{a.iterations}"
            paths = sorted(rdir.rglob("renders/*.png")) or sorted(rdir.glob("*.png"))
            if len(paths) != len(gt_paths):
                continue
            acc = {k: [] for k in ("raw", "clipped", "unclipped", "frac",
                                   "psnr_raw", "psnr_clipped", "psnr_unclipped")}
            for gtp, rp, f in zip(gt_paths, paths, held):
                m = dv.foreground_mask_from_record(recs[f])
                if not m.any():
                    continue
                g = np.asarray(Image.open(gtp).convert("RGB"), np.float32) / 255.0
                r = np.asarray(Image.open(rp).convert("RGB"), np.float32) / 255.0
                u = fit_affine(r, g, m)
                c = np.clip(u, 0.0, 1.0)
                # how much the clip actually changes, over masked pixels only
                acc["frac"].append(float(np.mean(np.any(np.abs(u - c) > 1e-6, axis=-1)[m])))
                acc["raw"].append(lp(r, g, m))
                acc["clipped"].append(lp(c, g, m))
                # LPIPS expects [-1,1]; feeding the unclipped array unmodified is the point
                acc["unclipped"].append(lp(u, g, m))
                acc["psnr_raw"].append(dv.psnr(r, g, m))
                acc["psnr_clipped"].append(dv.psnr(c, g, m))
                acc["psnr_unclipped"].append(dv.psnr(np.clip(u, 0, 1), g, m))
            if acc["raw"]:
                for k, v in acc.items():
                    row[f"{arm}_{k}"] = float(np.mean(v))
        rows.append(row)

    print("\n" + "=" * 92)
    print(f"{'arm':14s} {'LPIPS raw':>10s} {'clipped':>10s} {'unclipped':>10s} "
          f"{'clip-raw':>10s} {'unclip-raw':>11s} {'clipped px':>11s}")
    summary = {}
    for arm in a.arms:
        g = lambda k: [r[f"{arm}_{k}"] for r in rows if f"{arm}_{k}" in r]
        if not g("raw"):
            continue
        raw, cl, un, fr = map(lambda k: float(np.mean(g(k))), ("raw", "clipped", "unclipped", "frac"))
        summary[arm] = {"lpips_raw": raw, "lpips_clipped": cl, "lpips_unclipped": un,
                        "clipped_pixel_fraction": fr, "n": len(g("raw"))}
        print(f"{arm:14s} {raw:10.4f} {cl:10.4f} {un:10.4f} "
              f"{cl-raw:+10.4f} {un-raw:+11.4f} {fr*100:10.2f}%")
    print("\nhigher LPIPS = worse. If unclipped ~ raw, the clip is the cause.")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"lpips_clip_test_{a.scenes}_it{a.iterations}.json"
    p.write_text(json.dumps({"summary": summary, "per_scene": rows}, indent=2))
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()

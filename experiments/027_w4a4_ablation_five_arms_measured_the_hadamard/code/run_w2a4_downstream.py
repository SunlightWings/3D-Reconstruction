"""Downstream 3DGS for the W2A4 arm, and the three-way comparison the proposal needs.

The existing run_downstream_validation.py is structurally two-variant (full, w4a4) and it
produced the frozen 40-scene results, so it is not refactored here. This runner reuses its
helpers verbatim -- build_train_source, build_heldout_source, train_3dgs, render_heldout,
psnr/ssim, and the Sim(3) alignment -- to add a third arm without touching that file.

Arms
    full   full-precision VGGT      (already trained; reused, never retrained)
    w4a4   4-bit weights/activations (already trained; reused)
    w2a4   2-bit weights, 4-bit activations (built here)

Metrics, per the proposal's section VIII and this project's own corrections:
    PSNR / SSIM / LPIPS, each under BOTH the foreground mask (primary -- the object, where
      geometry exists) and the content mask (secondary -- kept for comparability with the
      published numbers). ongoing_logs.md #020 shows the two disagree in sign on 15/40
      scenes, so reporting only one is how the earlier result went wrong.
    Exposure-corrected PSNR: the same scores after a per-image per-channel gain+bias fit.
      #022 measured 3.542 dB of the oracle's deficit as pure auto-exposure mismatch, which
      is ~30x the effect being chased, so an uncorrected comparison is dominated by it.
    Paired statistics per arm pair (mean delta, t, p, 95% CI, minimum detectable effect),
      which is what decides whether any observed difference is real.
    Efficiency: VGGT inference seconds and peak GPU memory, read from the prediction meta
      files, plus Gaussian count per trained model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
import common  # noqa: E402

dv = common.dv
OUT = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
DEFAULT_ARMS = ("full", "w4a4", "w2a4")
# arms that already have sources+models from the original pipeline and must never be rebuilt
PREBUILT_ARMS = ("full", "w4a4")
ARMS = DEFAULT_ARMS  # overridden by --arms in main()


# --------------------------------------------------------------------------- sources

def build_variant_sources(cat: str, seq: str, variant: str, stride: int = 4):
    """Train + held-out COLMAP sources for any quantized arm, mirroring prepare_scene().

    Works for w2a4 and for every W4A4 ablation arm; `full` and `w4a4` already have sources
    built by the original pipeline and are never rebuilt here.
    """
    scene_root = common.scene_root(cat, seq)
    man = common.scene_manifest(cat, seq)
    frames, heldout = man["input_frames"], man["heldout_frames"]
    recs = dv.load_annotations(cat, seq)

    group_dir, _fm, _wm = dv.find_group(cat, seq)
    npz = group_dir / f"{variant}.npz"
    if not npz.is_file():
        raise SystemExit(f"{cat}/{seq}: missing {npz} -- "
                         f"run run_w2a4_inference.py --variant {variant} first")
    arrays = dv.load_npz(npz)
    dv.require_prediction_contract(arrays, variant.upper())

    common_train = scene_root / "common" / "train_images"
    common_heldout = scene_root / "common" / "heldout_images"
    train_names = [f"image_{i}.png" for i in range(1, len(frames) + 1)]
    train_tensor = dv.preprocess_images([dv.resolve_image_path(recs[f]) for f in frames])

    train_source = dv.build_train_source(scene_root, variant, arrays, train_tensor,
                                         train_names, common_train, stride)

    # Same Sim(3) construction the other arms use: fit GT->predicted on the six input
    # camera centres, then carry the GT held-out cameras into the predicted frame.
    E_gt = np.stack([dv.co3d_to_opencv_camera(recs[f])[0] for f in frames])
    C_gt = dv.camera_centers(E_gt)
    E_pred = arrays["extrinsic"].astype(np.float64)
    C_pred = dv.camera_centers(E_pred)
    s, A, b = dv.umeyama(C_gt, C_pred)
    C_fit = (s * (A @ C_gt.T)).T + b
    residual = np.linalg.norm(C_fit - C_pred, axis=1)
    radius = max(float(np.median(np.linalg.norm(C_pred - np.median(C_pred, axis=0), axis=1))), 1e-12)

    E_gt_held, K_held = [], []
    for f in heldout:
        Egt, K0 = dv.co3d_to_opencv_camera(recs[f])
        E_gt_held.append(Egt)
        K_held.append(dv.original_to_518_affine(recs[f]) @ K0)
    E_held = np.stack([dv.transform_camera_to_variant(e, s, A, b) for e in E_gt_held])

    heldout_source = dv.build_heldout_source(
        scene_root, variant, train_source, E_held, np.stack(K_held),
        [f"frame{f:06d}.png" for f in heldout], common_heldout,
    )
    alignment = {
        "scale": s,
        "camera_center_residual_median_scene_radius": float(np.median(residual) / radius),
    }
    return train_source, heldout_source, alignment


# --------------------------------------------------------------------------- metrics

# Every metric is reported raw and exposure-corrected. #022 measured 3.542 dB of the
# oracle's deficit as pure auto-exposure mismatch, and #027 found every raw-PSNR ablation
# gap collapsing once a per-image gain+bias is fitted -- so the corrected column is the
# one to trust, and it has to exist for the perceptual metrics too, not just PSNR.
METRICS = ("psnr", "ssim", "lpips",
           "psnr_exposure_corrected", "ssim_exposure_corrected", "lpips_exposure_corrected")


def gain_bias(render, gt, mask):
    out = render.copy()
    for c in range(3):
        x, y = render[..., c][mask], gt[..., c][mask]
        if len(x) < 10 or float(np.var(x)) < 1e-12:
            continue
        sol, *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
        out[..., c] = np.clip(render[..., c] * sol[0] + sol[1], 0.0, 1.0)
    return out


def paired_stats(deltas):
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=np.float64)
    n = len(d)
    if n < 2:
        return {"n": n}
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = mean / se if se > 0 else float("nan")
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
    except Exception:
        p, tcrit = float("nan"), 1.96
    return {"n": n, "mean_delta": mean, "sd": sd, "t": float(t), "p": p,
            "ci95": [mean - tcrit * se, mean + tcrit * se],
            "mde_80pct_power": float(2.802 * sd / np.sqrt(n)),
            "scenes_positive": int((d > 0).sum())}


def score_scene(cat, seq, arms_present):
    """PSNR/SSIM/LPIPS per arm, both masks, raw and exposure-corrected."""
    from PIL import Image
    import lpips as lpips_lib
    import torch

    global _LPIPS
    try:
        _LPIPS
    except NameError:
        _LPIPS = lpips_lib.LPIPS(net="alex").cpu().eval()

    man = common.scene_manifest(cat, seq)
    recs = dv.load_annotations(cat, seq)
    held = man["heldout_frames"]
    root = common.scene_root(cat, seq)
    gt_paths = [root / "common" / "heldout_images" / f"frame{f:06d}.png" for f in held]

    def rd(p):
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

    def lp(a, b, m):
        ys, xs = dv.bbox_from_mask(m)
        ta = torch.from_numpy(a[ys, xs]).permute(2, 0, 1)[None].float() * 2 - 1
        tb = torch.from_numpy(b[ys, xs]).permute(2, 0, 1)[None].float() * 2 - 1
        with torch.no_grad():
            return float(_LPIPS(ta, tb).item())

    out = {}
    for arm, rdir in arms_present.items():
        paths = sorted(Path(rdir).glob("*.png"))
        if len(paths) != len(gt_paths):
            continue
        acc = {}
        for region in ("foreground", "content"):
            masks = [(dv.foreground_mask_from_record(recs[f]) if region == "foreground"
                      else dv.content_mask_from_record(recs[f])) for f in held]
            ps, ss, ls, pe, se, le = [], [], [], [], [], []
            for gtp, rp, m in zip(gt_paths, paths, masks):
                if not m.any():
                    continue
                g, r = rd(gtp), rd(rp)
                # One gain+bias fit per image, reused for all three corrected metrics, so
                # PSNR/SSIM/LPIPS are corrected by the SAME transform and stay comparable.
                # SSIM carries a luminance term and LPIPS is only partly exposure-robust, so
                # correcting PSNR alone left two of the three metrics still confounded (#022).
                rc = gain_bias(r, g, m)
                ps.append(dv.psnr(r, g, m))
                ss.append(dv.ssim_value(r, g, m))
                ls.append(lp(r, g, m))
                pe.append(dv.psnr(rc, g, m))
                se.append(dv.ssim_value(rc, g, m))
                le.append(lp(rc, g, m))
            acc[region] = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
                           "lpips": float(np.mean(ls)), "psnr_exposure_corrected": float(np.mean(pe)),
                           "ssim_exposure_corrected": float(np.mean(se)),
                           "lpips_exposure_corrected": float(np.mean(le))}
        out[arm] = acc
    return out


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", choices=["subset", "all"], default="all")
    ap.add_argument("--iterations", type=int, default=7000,
                    help="all arms are rendered at this iteration. 7000 is the default because "
                         "7k->30k buys only +0.095 dB (full) / +0.065 dB (quant) and costs SSIM "
                         "-0.016 (results/downstream_3dgs/convergence_7k_vs_30k.json), and the "
                         "oracle sweep measured 14.023 -> 14.061 dB over the same range.")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--gpu-poll-seconds", type=int, default=10)
    ap.add_argument("--gpu-stable-checks", type=int, default=1)
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS),
                    help="arms to compare, e.g. full w4a4_local w4a4_rtn w4a4_no_rot")
    ap.add_argument("--prepare-only", action="store_true", help="build sources only, no GPU")
    ap.add_argument("--evaluate-only", action="store_true", help="score existing renders only")
    a = ap.parse_args()

    global ARMS
    ARMS = tuple(a.arms)
    scenes = common.EXPENSIVE_SCENES if a.scenes == "subset" else common.all_scenes()
    print(f"scenes={len(scenes)} ({a.scenes})  iterations={a.iterations}  arms={list(ARMS)}")

    rows, t_start = [], time.time()
    for i, name in enumerate(scenes, 1):
        cat, seq = common.split_scene(name)
        root = common.scene_root(cat, seq)
        print(f"\n[{i}/{len(scenes)}] {name}", flush=True)

        n_held = len(common.scene_manifest(cat, seq)["heldout_frames"])
        it = a.iterations

        if not a.evaluate_only:
            for arm in ARMS:
                if arm in PREBUILT_ARMS:
                    continue
                train_src, held_src, align = build_variant_sources(cat, seq, arm, a.stride)
                print(f"  {arm} sources ready; Sim(3) residual "
                      f"{align['camera_center_residual_median_scene_radius']:.4f} scene radii")
                if not a.prepare_only:
                    dv.train_3dgs(train_src, root / "models" / arm, it,
                                  a.gpu_poll_seconds, a.gpu_stable_checks)
        if a.prepare_only:
            continue

        # EVERY arm must be rendered at the SAME iteration, or bit-width is confounded
        # with training length. full/w4a4 are already trained and carry an iteration_7000
        # checkpoint (verified 40/40), so this renders rather than retrains them.
        arms_present = {}
        for arm in ARMS:
            model = root / "models" / arm
            if not (model / "point_cloud" / f"iteration_{it}" / "point_cloud.ply").is_file():
                print(f"    {arm}: no iteration_{it} checkpoint, skipping arm")
                continue
            held_src = (root / "sources" / arm / "heldout")
            if not held_src.is_dir():
                print(f"    {arm}: no heldout source at {held_src}, skipping arm")
                continue
            out_dir = root / "heldout" / f"{arm}_it{it}"
            rdir = dv.render_heldout(held_src, model, out_dir, it, n_held,
                                     a.gpu_poll_seconds, a.gpu_stable_checks)
            arms_present[arm] = rdir
        scores = score_scene(cat, seq, arms_present)
        rows.append({"scene": name, **{f"{arm}_{reg}_{k}": v
                                       for arm, regs in scores.items()
                                       for reg, kv in regs.items()
                                       for k, v in kv.items()}})
        for arm in ARMS:
            if arm in scores:
                f = scores[arm]["foreground"]
                print(f"    {arm:5s} fg PSNR={f['psnr']:6.3f}  (exp-corr {f['psnr_exposure_corrected']:6.3f})  "
                      f"SSIM={f['ssim']:.4f}  LPIPS={f['lpips']:.4f}", flush=True)

    if a.prepare_only:
        print("\nprepare-only: w2a4 sources built, no GPU work done")
        return

    # aggregate + paired statistics for every arm pair
    summary = {"n_scenes": len(rows), "minutes": (time.time() - t_start) / 60.0, "arms": {}, "pairs": {}}
    for arm in ARMS:
        for reg in ("foreground", "content"):
            keys = [f"{arm}_{reg}_{m}" for m in METRICS]
            if not any(k in (rows[0] if rows else {}) for k in keys):
                continue
            summary["arms"].setdefault(arm, {})[reg] = {
                m: float(np.mean([r[f"{arm}_{reg}_{m}"] for r in rows if f"{arm}_{reg}_{m}" in r]))
                for m in METRICS if any(f"{arm}_{reg}_{m}" in r for r in rows)
            }
    import itertools
    for x, y in itertools.combinations(ARMS, 2):
        for reg in ("foreground", "content"):
            for m in METRICS:
                kx, ky = f"{x}_{reg}_{m}", f"{y}_{reg}_{m}"
                d = [r[kx] - r[ky] for r in rows if kx in r and ky in r]
                if len(d) >= 2:
                    summary["pairs"][f"{x}_minus_{y}__{reg}__{m}"] = paired_stats(d)

    summary["per_scene"] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    summary["iterations"] = a.iterations
    tag = "_".join(ARMS)
    (OUT / f"arms_{tag}_{a.scenes}_it{a.iterations}.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 100)
    print("FOREGROUND, raw vs exposure-corrected (corrected is the column to trust, #027)")
    print(f"{'arm':16s} {'PSNR':>8s} {'PSNR*':>8s} {'SSIM':>8s} {'SSIM*':>8s} "
          f"{'LPIPS':>8s} {'LPIPS*':>8s} {'cont PSNR':>10s}")
    for arm in ARMS:
        if arm in summary["arms"]:
            f, c = summary["arms"][arm]["foreground"], summary["arms"][arm]["content"]
            g = lambda k: f.get(k, float("nan"))
            print(f"{arm:16s} {g('psnr'):8.4f} {g('psnr_exposure_corrected'):8.4f} "
                  f"{g('ssim'):8.4f} {g('ssim_exposure_corrected'):8.4f} "
                  f"{g('lpips'):8.4f} {g('lpips_exposure_corrected'):8.4f} {c['psnr']:10.4f}")
    print("  * = per-image gain+bias fitted before scoring")

    # Raw and corrected are printed side by side on purpose: the whole point of #027 is that
    # they disagree, and a table showing only one of them hides the finding.
    for label, suffix in (("raw PSNR", "__foreground__psnr"),
                          ("exposure-corrected PSNR", "__foreground__psnr_exposure_corrected"),
                          ("exposure-corrected LPIPS", "__foreground__lpips_exposure_corrected")):
        print(f"\npaired tests (foreground, {label}):")
        for k, v in summary["pairs"].items():
            if k.endswith(suffix) and v.get("n", 0) >= 2:
                star = "  *" if v.get("p", 1) < 0.05 else "   "
                print(f" {star}{k[:-len(suffix)]:34s} mean={v['mean_delta']:+.4f} p={v['p']:.4g} "
                      f"CI=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] {v['scenes_positive']}/{v['n']}")
    print(f"\nsaved {OUT/f'arms_{tag}_{a.scenes}_it{a.iterations}.json'}")


if __name__ == "__main__":
    main()

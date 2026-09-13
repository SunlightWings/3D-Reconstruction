"""Upper bound on the whole confidence-weighting proposal, using a CHEATING predictor.

The proposal is: a learned confidence head spots unreliable points in a quantized geometry
prior, 3DGS down-weights them, renders improve. Before building that head it is worth asking
whether the ceiling is above zero at all.

So this skips the learning entirely and uses the TRUE error as confidence: full-precision
VGGT geometry is available offline, so for every initial point we know exactly how wrong the
quantized arm is. Keep the best half, throw away the worst half, retrain, re-render.

  If a PERFECT confidence signal buys nothing, no learned predictor can, and the proposal is
  closed by measurement rather than left open.
  If it buys a lot, the idea is alive and only the predictor is missing.

Both outcomes are results. This is the cheapest experiment that can decide it.

THE CONTROL MATTERS AS MUCH AS THE TREATMENT. Keeping half the points also halves the
initial Gaussian count, and 3DGS is sensitive to that on its own -- so a `random` arm drops
an equally sized random half. Only (confidence - random) is attributable to confidence;
(confidence - uniform) confounds it with point count.

Arms produced per scene:
  <arm>_conf50   keep the half with lowest true error
  <arm>_rand50   keep a random half, same count, fixed seed
Compared against the already-trained <arm> (uniform, all points) and `full`.
"""
from __future__ import annotations

import argparse
import json
import shutil
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
STRIDE = 4


def per_point_error(cat, seq, arm):
    """True per-point error of `arm` against full precision, on the stride-4 init grid.

    Same Sim(3) construction ablation_compare.py uses: fit the quantized frame into full's
    frame on the six input camera centres, carry the points through it, measure residual in
    scene radii so scenes are comparable.
    """
    gdir, _fm, _wm = dv.find_group(cat, seq)
    F, Q = dv.load_npz(gdir / "full.npz"), dv.load_npz(gdir / f"{arm}.npz")

    Cf, Cq = dv.camera_centers(F["extrinsic"].astype(np.float64)), \
             dv.camera_centers(Q["extrinsic"].astype(np.float64))
    s, A, b = dv.umeyama(Cq, Cf)
    radius = max(float(np.median(np.linalg.norm(Cf - np.median(Cf, axis=0), axis=1))), 1e-12)

    def grid(x):
        return x["world_points_from_depth"].astype(np.float64)[:, ::STRIDE, ::STRIDE, :].reshape(-1, 3)

    Pf, Pq = grid(F), grid(Q)
    Pq_in_f = (s * (A @ Pq.T)).T + b
    return np.linalg.norm(Pq_in_f - Pf, axis=1) / radius


def build_pruned_source(cat, seq, arm, variant, keep_idx):
    """Train source for `variant`: the arm's own geometry, restricted to keep_idx."""
    scene_root = common.scene_root(cat, seq)
    man = common.scene_manifest(cat, seq)
    recs = dv.load_annotations(cat, seq)
    frames = man["input_frames"]
    gdir, _fm, _wm = dv.find_group(cat, seq)
    arrays = dv.load_npz(gdir / f"{arm}.npz")

    tensor = dv.preprocess_images([dv.resolve_image_path(recs[f]) for f in frames])
    pts, colors = dv.sampled_points_and_colors(arrays["world_points_from_depth"], tensor, STRIDE)
    pts, colors = pts[keep_idx], colors[keep_idx]

    root = scene_root / "sources" / variant / "train"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    import os
    os.symlink(scene_root / "common" / "train_images", root / "images", target_is_directory=True)
    names = [f"image_{i}.png" for i in range(1, len(frames) + 1)]
    dv.write_colmap_text_model(root, arrays["extrinsic"].astype(np.float64),
                               arrays["intrinsic"].astype(np.float64), names, pts, colors)
    (root / "PREPARED.ok").write_text("PASS\n")
    return root, len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="w4a4_rtn")
    ap.add_argument("--scenes", type=int, default=20, help="how many scenes, in manifest order")
    ap.add_argument("--scene-set", choices=["manifest", "subset"], default="manifest",
                    help="'subset' = the 8 EXPENSIVE_SCENES, the only ones with a w3a3 prediction")
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--deadline", default="00:15",
                    help="HH:MM local; stop launching new work after this and bank results")
    a = ap.parse_args()

    import datetime as dt
    now = dt.datetime.now()
    dl = now.replace(hour=int(a.deadline[:2]), minute=int(a.deadline[3:]), second=0)
    if dl <= now:
        dl += dt.timedelta(days=1)
    print(f"deadline {dl:%H:%M} ({(dl-now).total_seconds()/60:.0f} min from now)", flush=True)

    scenes = (common.EXPENSIVE_SCENES if a.scene_set == "subset"
              else common.all_scenes()[: a.scenes])
    rng = np.random.default_rng(0)
    rows = []
    for i, name in enumerate(scenes, 1):
        if dt.datetime.now() >= dl:
            print(f"\n!! deadline reached, stopping after {i-1} scenes -- results below are complete "
                  f"for those scenes and nothing is estimated", flush=True)
            break
        cat, seq = name.split("/", 1)
        print(f"\n[{i}/{len(scenes)}] {name}  ({dt.datetime.now():%H:%M})", flush=True)
        root = common.scene_root(cat, seq)

        err = per_point_error(cat, seq, a.arm)
        n_keep = int(len(err) * a.keep)
        conf_idx = np.argsort(err)[:n_keep]                       # lowest true error
        rand_idx = rng.choice(len(err), size=n_keep, replace=False)  # matched-size control
        print(f"  points {len(err)} -> {n_keep};  err kept/dropped median "
              f"{np.median(err[conf_idx]):.4f}/{np.median(err[np.argsort(err)[n_keep:]]):.4f}", flush=True)

        scores = {}
        for variant, idx in ((f"{a.arm}_conf{int(a.keep*100)}", conf_idx),
                             (f"{a.arm}_rand{int(a.keep*100)}", rand_idx)):
            src, npts = build_pruned_source(cat, seq, a.arm, variant, idx)
            # cameras are unchanged by pruning, so the arm's existing heldout source is reused
            held = root / "sources" / a.arm / "heldout"
            model = root / "models" / variant
            dv.train_3dgs(src, model, a.iterations, 20, 1)
            rdir = dv.render_heldout(held, model, root / "heldout" / f"{variant}_it{a.iterations}",
                                     a.iterations, len(common.scene_manifest(cat, seq)["heldout_frames"]), 20, 1)
            scores[variant] = rdir

        # score the two new arms plus the already-trained baselines, identically
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rwd", REPO / "code" / "quantization" / "run_w2a4_downstream.py")
        rwd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rwd)
        present = dict(scores)
        for base in ("full", a.arm):
            d = root / "heldout" / f"{base}_it{a.iterations}" / "renders"
            if d.is_dir():
                present[base] = d
        s = rwd.score_scene(cat, seq, present)
        row = {"scene": name, "n_points_kept": n_keep}
        for arm, regs in s.items():
            for m, v in regs["foreground"].items():
                row[f"{arm}_{m}"] = v
        rows.append(row)
        f = row.get(f"{a.arm}_lpips"), row.get(f"{a.arm}_conf{int(a.keep*100)}_lpips")
        print(f"  LPIPS uniform={f[0]} conf={f[1]}", flush=True)

        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"confidence_oracle_{a.arm}_it{a.iterations}.json").write_text(
            json.dumps({"arm": a.arm, "keep": a.keep, "per_scene": rows}, indent=2))

    # ---- summary
    if not rows:
        print("no scenes completed")
        return
    C, R, U = f"{a.arm}_conf{int(a.keep*100)}", f"{a.arm}_rand{int(a.keep*100)}", a.arm
    print("\n" + "=" * 88)
    print(f"n = {len(rows)} scenes,  arm = {a.arm},  keep = {a.keep:.0%}")
    print(f"{'condition':28s} {'PSNR':>9s} {'SSIM':>9s} {'LPIPS':>9s}")
    for lbl, k in (("A  full precision", "full"), ("B  arm, all points", U),
                   ("C  arm, confidence-kept", C), ("    arm, random-kept (control)", R)):
        v = [r for r in rows if f"{k}_psnr" in r]
        if v:
            print(f"{lbl:28s} {np.mean([r[f'{k}_psnr'] for r in v]):9.4f} "
                  f"{np.mean([r[f'{k}_ssim'] for r in v]):9.4f} "
                  f"{np.mean([r[f'{k}_lpips'] for r in v]):9.4f}")
    print("\npaired deltas (foreground):")
    for lbl, x, y in (("C - B  (confidence vs uniform)", C, U),
                      ("C - R  (confidence vs random, THE test)", C, R),
                      ("R - B  (random vs uniform, point-count effect)", R, U),
                      ("A - B  (the gap to close)", "full", U)):
        for m in ("psnr", "lpips"):
            d = [r[f"{x}_{m}"] - r[f"{y}_{m}"] for r in rows if f"{x}_{m}" in r and f"{y}_{m}" in r]
            if len(d) >= 2:
                st = rwd.paired_stats(d)
                print(f"  {lbl:44s} {m:6s} mean={st['mean_delta']:+.4f} "
                      f"p={st.get('p', float('nan')):.4g} n={st['n']}")
    print("\nlower LPIPS = better; higher PSNR = better")


if __name__ == "__main__":
    main()

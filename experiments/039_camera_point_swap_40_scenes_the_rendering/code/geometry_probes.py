"""Two probes that locate WHERE geometry damage hurts 3DGS, and HOW MUCH it takes.

`swap`  -- is a damaged arm's rendering gap caused by its CAMERAS or by its POINTS?
           #030 found that removing the worst half of `w4a4_rtn`'s points, chosen with the true
           error, recovers none of its 0.93 dB gap. `rtn` also carries 14.66 deg of camera error,
           which no point-level confidence can touch. This 2x2 separates the two.

           Everything is built in FULL precision's world frame and rendered from full's own
           held-out source, so all four cells share one set of evaluation cameras:

               A        full cameras  + full points      (already trained: `full`)
               P        full cameras  + ARM points       -> point damage alone
               K        ARM cameras   + full points      -> camera damage alone
               B'       ARM cameras   + ARM points       -> both, in the same evaluation

           The arm is carried into full's frame by the Sim(3) fitted on the six input camera
           centres, the same construction as ablation_compare.py. B' is NOT the same as the
           arm's original run: that one derives its held-out cameras by aligning GT to the arm's
           cameras, which absorbs part of the camera error. B' exists so the four cells are
           comparable with each other; do not compare it to the original `w4a4_rtn` number.

`sweep` -- how much damage does rendering tolerate before it measurably degrades?
           Full-precision geometry with controlled synthetic damage, one kind at a time:

               cameraR<deg>   each training camera rotated by exactly <deg> about its own
                              centre (random axis); centres, intrinsics, points untouched
               points<sigma>  isotropic Gaussian jitter on every initial point, scaled so the
                              MEDIAN displacement is <sigma> scene radii -- the same statistic
                              ablation_compare.py reports as point_err, so measured arms sit on
                              the same axis

           Rendered from full's held-out source. Measured reference points (#029, n=40):
           w4a4 cam rot 1.2754 deg / point err 0.06144; w4a4_rtn 14.6565 deg / 0.46422.

Both probes re-use full's existing trained model and renders as condition A. Results are written
after every scene; a --deadline stops new training cleanly before the nightly shutdown.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
import common  # noqa: E402

dv = common.dv
OUT = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
STRIDE = 4
MEDIAN_CHI3 = 1.5382  # median norm of a 3-D standard normal: sets per-axis std from a median target

_spec = importlib.util.spec_from_file_location("rwd", REPO / "code" / "quantization" / "run_w2a4_downstream.py")
rwd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rwd)

CAM_LEVELS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
PTS_LEVELS = (0.03, 0.06, 0.12, 0.25, 0.5, 1.0)


def tag(x: float) -> str:
    return f"{x:g}".replace(".", "p")


# --------------------------------------------------------------------------- geometry

def scene_inputs(cat, seq):
    man = common.scene_manifest(cat, seq)
    recs = dv.load_annotations(cat, seq)
    frames = man["input_frames"]
    gdir, _fm, _wm = dv.find_group(cat, seq)
    tensor = dv.preprocess_images([dv.resolve_image_path(recs[f]) for f in frames])
    return man, gdir, tensor, len(frames)


def grid_points(arrays, tensor):
    return dv.sampled_points_and_colors(arrays["world_points_from_depth"], tensor, STRIDE)


def arm_into_full(F, Q):
    """Sim(3) carrying the arm's frame into full's frame, fitted on input camera centres."""
    Cf = dv.camera_centers(F["extrinsic"].astype(np.float64))
    Cq = dv.camera_centers(Q["extrinsic"].astype(np.float64))
    s, A, b = dv.umeyama(Cq, Cf)
    radius = max(float(np.median(np.linalg.norm(Cf - np.median(Cf, axis=0), axis=1))), 1e-12)
    return s, A, b, radius


def rotation(axis, deg):
    axis = axis / np.linalg.norm(axis)
    t = np.deg2rad(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def rotate_cameras(E, deg, rng):
    """Rotate each world-to-camera pose by exactly `deg` about its own centre."""
    out = []
    for e in E:
        R, C = e[:, :3], dv.camera_center(e)
        R2 = rotation(rng.normal(size=3), deg) @ R
        out.append(np.concatenate([R2, (-R2 @ C)[:, None]], axis=1))
    return np.stack(out)


# --------------------------------------------------------------------------- 3DGS

def build_source(cat, seq, variant, E, K, pts, colors, n_views):
    root = common.scene_root(cat, seq) / "sources" / variant / "train"
    if (root / "PREPARED.ok").is_file():
        return root
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    os.symlink(common.scene_root(cat, seq) / "common" / "train_images", root / "images",
               target_is_directory=True)
    names = [f"image_{i}.png" for i in range(1, n_views + 1)]
    dv.write_colmap_text_model(root, np.asarray(E, np.float64), np.asarray(K, np.float64),
                               names, np.asarray(pts, np.float32), colors)
    (root / "PREPARED.ok").write_text("PASS\n")
    return root


def train_and_render(cat, seq, variant, src, iterations):
    root = common.scene_root(cat, seq)
    model = root / "models" / variant
    dv.train_3dgs(src, model, iterations, 20, 1)
    n_held = len(common.scene_manifest(cat, seq)["heldout_frames"])
    # full's held-out source: every condition is judged from the same evaluation cameras
    return dv.render_heldout(root / "sources" / "full" / "heldout", model,
                             root / "heldout" / f"{variant}_it{iterations}", iterations, n_held, 20, 1)


def score(cat, seq, present):
    s = rwd.score_scene(cat, seq, present)
    return {f"{arm}_{m}": v for arm, regs in s.items() for m, v in regs["foreground"].items()}


# --------------------------------------------------------------------------- probes

def conditions_swap(cat, seq, arm):
    man, gdir, tensor, n = scene_inputs(cat, seq)
    F, Q = dv.load_npz(gdir / "full.npz"), dv.load_npz(gdir / f"{arm}.npz")
    s, A, b, _r = arm_into_full(F, Q)
    pf, colors = grid_points(F, tensor)
    pq, _ = grid_points(Q, tensor)
    pq_f = (s * (A @ pq.T.astype(np.float64))).T + b
    Eq_f = np.stack([dv.transform_camera_to_variant(e, s, A, b) for e in Q["extrinsic"].astype(np.float64)])
    Ef, Kf, Kq = F["extrinsic"], F["intrinsic"], Q["intrinsic"]
    return {
        f"swapP_{arm}": (Ef, Kf, pq_f, colors),
        f"swapK_{arm}": (Eq_f, Kq, pf, colors),
        f"swapB_{arm}": (Eq_f, Kq, pq_f, colors),
    }, n


def conditions_camsplit(cat, seq, arm):
    """Split the swap's CAMERA damage into pose and focal length.

    The swap's K condition replaced the arm's pose AND intrinsics together, and the CPU
    analysis (cpu_camera_analysis.py) found `w4a4_rtn` focal lengths off by a median 18.65 %
    against full -- large enough to matter by itself. 3DGS optimises neither, so both are
    candidates. Points stay full precision in both cells.

        R  arm POSE  (carried into full's frame) + FULL intrinsics  -> pose damage alone
        F  FULL pose                             + ARM intrinsics   -> focal damage alone
    """
    man, gdir, tensor, n = scene_inputs(cat, seq)
    F, Q = dv.load_npz(gdir / "full.npz"), dv.load_npz(gdir / f"{arm}.npz")
    s, A, b, _r = arm_into_full(F, Q)
    pf, colors = grid_points(F, tensor)
    Eq_f = np.stack([dv.transform_camera_to_variant(e, s, A, b) for e in Q["extrinsic"].astype(np.float64)])
    return {
        f"swapR_{arm}": (Eq_f, F["intrinsic"], pf, colors),
        f"swapF_{arm}": (F["extrinsic"], Q["intrinsic"], pf, colors),
    }, n


def conditions_sweep(cat, seq):
    man, gdir, tensor, n = scene_inputs(cat, seq)
    F = dv.load_npz(gdir / "full.npz")
    pf, colors = grid_points(F, tensor)
    Ef, Kf = F["extrinsic"].astype(np.float64), F["intrinsic"]
    Cf = dv.camera_centers(Ef)
    radius = max(float(np.median(np.linalg.norm(Cf - np.median(Cf, axis=0), axis=1))), 1e-12)
    seed = zlib.crc32(f"{cat}/{seq}".encode())
    out = {}
    for d in CAM_LEVELS:
        rng = np.random.default_rng(seed + int(d * 1000))
        out[f"cameraR{tag(d)}"] = (rotate_cameras(Ef, d, rng), Kf, pf, colors)
    for sg in PTS_LEVELS:
        rng = np.random.default_rng(seed + 7 + int(sg * 1000))
        jitter = rng.normal(size=pf.shape) * (sg * radius / MEDIAN_CHI3)
        out[f"points{tag(sg)}"] = (Ef, Kf, pf + jitter, colors)
    return out, n


# --------------------------------------------------------------------------- driver

def summarise(kind, rows, arm):
    print("\n" + "=" * 92)
    print(f"{kind}: n = {len(rows)} scenes")
    conds = sorted({k[: -len('_psnr')] for r in rows for k in r if k.endswith("_psnr")})
    print(f"{'condition':22s} {'PSNR':>9s} {'SSIM':>9s} {'LPIPS':>9s}   {'Δ PSNR vs A':>12s} {'p':>9s}"
          f"   {'Δ LPIPS vs A':>13s} {'p':>9s}  sign")
    stats = {}
    for c in (["full"] + [x for x in conds if x != "full"]):
        v = [r for r in rows if f"{c}_psnr" in r]
        if not v:
            continue
        line = (f"{c:22s} {np.mean([r[f'{c}_psnr'] for r in v]):9.4f} "
                f"{np.mean([r[f'{c}_ssim'] for r in v]):9.4f} {np.mean([r[f'{c}_lpips'] for r in v]):9.4f}")
        if c != "full":
            dp = [r["full_psnr"] - r[f"{c}_psnr"] for r in v if "full_psnr" in r]
            dl = [r[f"{c}_lpips"] - r["full_lpips"] for r in v if "full_lpips" in r]
            sp, sl = rwd.paired_stats(dp), rwd.paired_stats(dl)
            stats[c] = {"psnr_drop": sp, "lpips_rise": sl}
            if sp.get("n", 0) >= 2:
                line += (f"   {sp['mean_delta']:+12.4f} {sp['p']:9.4g}   {sl['mean_delta']:+13.4f} "
                         f"{sl['p']:9.4g}  {sp['scenes_positive']}/{sp['n']}")
        print(line)
    print("Δ = A minus condition for PSNR, condition minus A for LPIPS: positive = condition is WORSE.")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["swap", "sweep", "camsplit"])
    ap.add_argument("--arm", default="w4a4_rtn", help="swap only")
    ap.add_argument("--scenes", type=int, default=40, help="first N scenes of the frozen manifest")
    ap.add_argument("--scene-set", choices=["manifest", "subset"], default="manifest",
                    help="'subset' = the 8 EXPENSIVE_SCENES, the only ones with a w3a3 prediction")
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--deadline", default="00:15", help="HH:MM; no new training starts after this")
    ap.add_argument("--resume", action="store_true",
                    help="keep scenes already saved in this probe's JSON and run only the missing ones")
    a = ap.parse_args()

    now = dt.datetime.now()
    dl = now.replace(hour=int(a.deadline[:2]), minute=int(a.deadline[3:]), second=0, microsecond=0)
    if dl <= now:
        dl += dt.timedelta(days=1)
    print(f"{a.probe}: deadline {dl:%Y-%m-%d %H:%M} ({(dl - now).total_seconds() / 60:.0f} min)", flush=True)

    label = "sweep" if a.probe == "sweep" else f"{a.probe}_{a.arm}"
    out_json = OUT / f"geometry_probe_{label}_it{a.iterations}.json"
    rows, stopped = [], False
    if a.resume and out_json.is_file():
        rows = json.loads(out_json.read_text())["per_scene"]
        print(f"resume: {len(rows)} scenes already complete in {out_json.name}", flush=True)
    done = {r["scene"] for r in rows}
    scene_list = (common.EXPENSIVE_SCENES if a.scene_set == "subset"
                  else common.all_scenes()[: a.scenes])
    for i, name in enumerate(scene_list, 1):
        if name in done:
            continue
        cat, seq = name.split("/", 1)
        print(f"\n[{i}/{len(scene_list)}] {name}  ({dt.datetime.now():%H:%M})", flush=True)
        builder = {"swap": lambda: conditions_swap(cat, seq, a.arm),
                   "camsplit": lambda: conditions_camsplit(cat, seq, a.arm),
                   "sweep": lambda: conditions_sweep(cat, seq)}[a.probe]
        conds, n_views = builder()
        present, complete = {}, True
        for variant, (E, K, pts, colors) in conds.items():
            if dt.datetime.now() >= dl:
                complete, stopped = False, True
                break
            src = build_source(cat, seq, variant, E, K, pts, colors, n_views)
            present[variant] = train_and_render(cat, seq, variant, src, a.iterations)
        if not complete:
            # a scene with only some conditions would bias every per-condition mean; drop it whole
            print(f"!! deadline reached mid-scene; {name} discarded, {len(rows)} complete scenes kept",
                  flush=True)
            break
        fr = common.scene_root(cat, seq) / "heldout" / f"full_it{a.iterations}" / "renders"
        present["full"] = fr
        row = {"scene": name, **score(cat, seq, present)}
        rows.append(row)
        print("  " + "  ".join(f"{v}={row.get(f'{v}_psnr', float('nan')):.3f}" for v in present), flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"probe": a.probe, "arm": a.arm, "per_scene": rows}, indent=2))
        if dt.datetime.now() >= dl:
            stopped = True
            print("!! deadline reached; stopping before the next scene", flush=True)
            break

    if not rows:
        print("no complete scenes")
        return
    stats = summarise(label, rows, a.arm)
    if a.probe == "swap":
        # additivity: does camera damage + point damage account for joint damage?
        P, K, B = (stats.get(f"swap{x}_{a.arm}", {}).get("psnr_drop", {}) for x in "PKB")
        if all(x.get("n", 0) >= 2 for x in (P, K, B)):
            inter = B["mean_delta"] - P["mean_delta"] - K["mean_delta"]
            print(f"\nPSNR drop: points alone {P['mean_delta']:+.4f}, cameras alone {K['mean_delta']:+.4f}, "
                  f"both {B['mean_delta']:+.4f}; interaction (both - points - cameras) {inter:+.4f}")
    elif a.probe == "sweep":
        print("\nreference (#029, n=40): w4a4 cam rot 1.2754 deg, point err 0.06144; "
              "w4a4_rtn 14.6565 deg, 0.46422")
    out_json.write_text(json.dumps({"probe": a.probe, "arm": a.arm, "per_scene": rows,
                                    "summary": stats, "stopped_at_deadline": stopped}, indent=2))
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()

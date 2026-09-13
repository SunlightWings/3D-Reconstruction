"""Decompose the oracle-to-VGGT gap into an alignment part and a geometry part.

WHY NOT THE LITERAL "w4a4 points through GT cameras" ARM
--------------------------------------------------------
A Sim(3) applied to BOTH the points and the cameras leaves every rendered image
unchanged. The w4a4 geometry's only link to the GT frame *is* its Sim(3), so
"render the w4a4 cloud through GT cameras" is pixel-identical to "render it through
its own Sim(3)". That arm cannot separate anything. (Swapping in the full arm's
Sim(3) is well-posed -- the two predicted frames agree to ~0.024 of scene radius --
but it only measures the difference between two nearly identical estimates.)

WHAT IS WELL-POSED
------------------
Run the causation the other way. The oracle arm has GT geometry AND GT cameras and
uses no Sim(3) at all, so it is the one arm with zero alignment error. Inject a
measured amount of alignment error into its held-out cameras and watch the PSNR fall.
That converts "camera error of size d" into "PSNR cost", which is exactly the
exchange rate needed to attribute the oracle-minus-full gap.

The perturbation sizes are the ones actually measured, not invented:
    0.000  no perturbation (the oracle as it stands)
    0.020  A5's pass threshold
    0.081  A5 frame-consistent, full arm   (entry #018)
    0.138  A5 frame-consistent, w4a4 arm   (entry #018)
expressed in GT point-cloud scene radii, applied as a random-direction translation of
the held-out camera centres with the orientation held fixed.

Render-only: reuses the existing trained oracle models.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP.parent / "diagnostics"))
import common  # noqa: E402
from reeval import fgmask  # noqa: E402

dv = common.dv
OUT = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
WORK = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/pose_vs_geom")
DELTAS = [0.0, 0.02, 0.0813, 0.1380]
SEED = 20260908


def perturbed_heldout_source(cat, seq, delta, rng):
    """A copy of the oracle held-out source with camera centres displaced by delta*radius."""
    src = common.DIAG_HEAVY_ROOT / cat / seq / "oracle" / "heldout"
    dst = WORK / f"d{delta:.4f}" / cat / seq / "heldout"
    if (dst / "PREPARED.ok").is_file():
        return dst
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=True)

    recs = dv.load_annotations(cat, seq)
    held = common.scene_manifest(cat, seq)["heldout_frames"]
    radius = common.load_per_scene_csv()[(cat, seq)]["scene_radius"]

    E_list, K_list, names = [], [], []
    for f in held:
        E, K0 = dv.co3d_to_opencv_camera(recs[f])
        K = dv.original_to_518_affine(recs[f]) @ K0
        R = E[:, :3]
        C = dv.camera_center(E)
        if delta > 0:
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            C = C + d * (delta * radius)
        t = -R @ C
        E_list.append(np.concatenate([R, t[:, None]], axis=1))
        K_list.append(K)
        names.append(f"frame{f:06d}.png")

    pts = np.zeros((10, 3), dtype=np.float32)
    cols = np.full((10, 3), 128, dtype=np.uint8)
    dv.write_colmap_text_model(dst, np.stack(E_list), np.stack(K_list), names, pts, cols)
    (dst / "PREPARED.ok").write_text(f"PASS delta={delta} radius={radius:.5f}\n")
    return dst


def main():
    rng = np.random.default_rng(SEED)
    results = {}
    for delta in DELTAS:
        rows = []
        for name in common.EXPENSIVE_SCENES:
            cat, seq = common.split_scene(name)
            model = common.DIAG_HEAVY_ROOT / cat / seq / "models" / "oracle"
            src = perturbed_heldout_source(cat, seq, delta, rng)
            cache = model / "train" / "ours_30000"
            if cache.exists():
                shutil.rmtree(cache)
            common.run([str(dv.GS_PY), "render.py", "-m", str(model), "-s", str(src),
                        "--iteration", "30000", "--skip_test", "-d", ""],
                       cwd=dv.GS_ROOT, env=dv.gs_env())
            rp = sorted((cache / "renders").glob("*.png"))
            recs = dv.load_annotations(cat, seq)
            held = common.scene_manifest(cat, seq)["heldout_frames"]
            gt = [common.scene_root(cat, seq) / "common" / "heldout_images" / f"frame{f:06d}.png"
                  for f in held]
            masks = [fgmask(recs[f]) for f in held]
            r = common.evaluate_renders(gt, rp, masks, with_ssim=False)
            rows.append({"scene": name, "psnr": r["mean_psnr"]})
            print(f"  delta={delta:.4f} {name:33s} fg PSNR={r['mean_psnr']:.3f}", flush=True)
        results[f"{delta:.4f}"] = {
            "delta_scene_radii": delta,
            "mean_foreground_psnr": float(np.mean([x["psnr"] for x in rows])),
            "per_scene": rows,
        }
        print(f"delta={delta:.4f}: MEAN fg PSNR = {results[f'{delta:.4f}']['mean_foreground_psnr']:.3f} dB\n",
              flush=True)

    base = results[f"{DELTAS[0]:.4f}"]["mean_foreground_psnr"]
    out = {
        "description": "Oracle arm with measured alignment error injected into its held-out cameras.",
        "seed": SEED,
        "baseline_oracle_foreground_psnr": base,
        "by_delta": results,
        "psnr_cost_of_alignment": {k: base - v["mean_foreground_psnr"] for k, v in results.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pose_vs_geometry.json").write_text(json.dumps(out, indent=2))
    print("=== PSNR cost of alignment error (oracle arm, foreground mask) ===")
    for k, v in results.items():
        print(f"  delta={v['delta_scene_radii']:.4f} radii -> {v['mean_foreground_psnr']:.3f} dB "
              f"(cost {base - v['mean_foreground_psnr']:+.3f} dB)")
    print(f"saved {OUT/'pose_vs_geometry.json'}")


if __name__ == "__main__":
    main()

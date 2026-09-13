"""Does the ensemble-corrected pose recover RENDERING quality? The payoff test for #045 + #047.

Chain so far: the rendering damage of a quantized prior is in its cameras (#039), mostly the pose
(#043); rendering tolerates pose error below ~2-4 degrees (#042); order-spread detects the pose error
(#045); COLMAP cannot correct it (#046); averaging the orderings does correct it, roughly 40 % on
`w4a4_rtn` (#047). What remains is whether that correction is worth anything downstream.

Conditions, all built in full precision's frame and rendered from full's own held-out source, so they
drop straight into the #039/#043 table:

    A       full cameras + full points                      (existing `full` arm)
    K       arm pose + arm intrinsics + full points         (existing `swapK_<arm>`, #039)
    ensK    ENSEMBLE pose + arm intrinsics + full points    (built here)

ensK differs from K only in the camera pose, so (K - ensK) is exactly what the correction buys, and
(A - ensK) is what still remains. Points are full precision in all three so the comparison is not
contaminated by the point channel, which #039 and #042 show is irrelevant anyway.

Intrinsics are NOT ensembled: #043 measured pose and focal separately and the correction under test is
a pose correction. That leaves ensK carrying the arm's focal error by construction, which bounds how
much of the gap it can close -- stated here rather than discovered later.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
sys.path.insert(0, str(REPO / "code" / "quantization"))
import common  # noqa: E402
import geometry_probes as gp  # noqa: E402

dv = common.dv
RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="w4a4_rtn")
    ap.add_argument("--perms", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--deadline", default="00:15")
    ap.add_argument("--use-intrinsics", action="store_true",
                    help="also swap in the ENSEMBLE intrinsics, not only the ensemble pose")
    a = ap.parse_args()

    src = RES / f"pose_ensemble_{a.arm}_p{a.perms}.json"
    ens = {r["scene"]: r for r in json.load(open(src))["per_scene"] if "ensemble_extrinsic" in r}
    print(f"{len(ens)} scenes with ensemble cameras from {src.name}")

    now = dt.datetime.now()
    dl = now.replace(hour=int(a.deadline[:2]), minute=int(a.deadline[3:]), second=0, microsecond=0)
    if dl <= now:
        dl += dt.timedelta(days=1)

    variant = f"ensKF_{a.arm}" if a.use_intrinsics else f"ensK_{a.arm}"
    rows = []
    out_json = RES / (f"ensemble_downstream_{'posefocal' if a.use_intrinsics else 'pose'}_"
                      f"{a.arm}_it{a.iterations}.json")
    for i, name in enumerate(common.all_scenes(), 1):
        if name not in ens or dt.datetime.now() >= dl:
            continue
        cat, seq = name.split("/", 1)
        man, gdir, tensor, n_views = gp.scene_inputs(cat, seq)
        F, Q = dv.load_npz(gdir / "full.npz"), dv.load_npz(gdir / f"{a.arm}.npz")
        pf, colors = gp.grid_points(F, tensor)

        # the ensemble lives in ordering 0's frame, i.e. the arm's frame: carry it into full's frame
        # with the SAME Sim(3) construction the swap cells use, so ensK and K differ only in pose
        s, A, b, _r = gp.arm_into_full(F, Q)
        E_ens = np.asarray(ens[name]["ensemble_extrinsic"], np.float64)
        E_in_full = np.stack([dv.transform_camera_to_variant(e, s, A, b) for e in E_ens])

        K_use = (np.asarray(ens[name]["ensemble_intrinsic"], np.float64) if a.use_intrinsics
                 else Q["intrinsic"])
        source = gp.build_source(cat, seq, variant, E_in_full, K_use, pf, colors, n_views)
        rdir = gp.train_and_render(cat, seq, variant, source, a.iterations)

        present = {variant: rdir}
        for base in ("full", f"swapK_{a.arm}", f"ensK_{a.arm}"):
            if base == variant:
                continue
            d = common.scene_root(cat, seq) / "heldout" / f"{base}_it{a.iterations}" / "renders"
            if d.is_dir():
                present[base] = d
        row = {"scene": name, **gp.score(cat, seq, present),
               "pose_err_single": ens[name]["single_vs_full"], "pose_err_ensemble": ens[name]["ensemble_vs_full"]}
        rows.append(row)
        print(f"[{i}/40] {name}  full {row.get('full_psnr', float('nan')):.3f}  "
              f"K {row.get(f'swapK_{a.arm}_psnr', float('nan')):.3f}  ensK {row[f'{variant}_psnr']:.3f}",
              flush=True)
        out_json.write_text(json.dumps({"arm": a.arm, "per_scene": rows}, indent=2))

    if not rows:
        print("no scenes completed")
        return
    K, E = f"swapK_{a.arm}", variant
    print("\n" + "=" * 96)
    print(f"ENSEMBLE-CORRECTED POSE, downstream, arm {a.arm}, n = {len(rows)} scenes")
    print(f"{'condition':28s} {'PSNR':>9s} {'SSIM':>9s} {'LPIPS':>9s}")
    for lbl, k in (("A  full", "full"), ("K  arm pose (uncorrected)", K), ("ensK  ensemble pose", E)):
        v = [r for r in rows if f"{k}_psnr" in r]
        if v:
            print(f"{lbl:28s} {np.mean([r[f'{k}_psnr'] for r in v]):9.4f} "
                  f"{np.mean([r[f'{k}_ssim'] for r in v]):9.4f} {np.mean([r[f'{k}_lpips'] for r in v]):9.4f}")
    tests = {}
    pairs = [(f"{E} - K   (what the correction buys)", E, K),
             (f"A - {E}   (what still remains)", "full", E),
             ("A - K      (the original gap)", "full", K)]
    if a.use_intrinsics:
        pairs.append((f"{E} - ensK  (focal correction on top)", E, f"ensK_{a.arm}"))
    for lbl, x, y in pairs:
        for m, sgn in (("psnr", 1), ("lpips", -1)):
            d = [sgn * (r[f"{x}_{m}"] - r[f"{y}_{m}"]) for r in rows if f"{x}_{m}" in r and f"{y}_{m}" in r]
            if len(d) >= 2:
                d = np.asarray(d)
                t = stats.ttest_1samp(d, 0)
                tests[f"{lbl}|{m}"] = {"n": len(d), "mean": float(d.mean()), "p": float(t.pvalue),
                                       "positive": int((d > 0).sum())}
                print(f"  {lbl:40s} {m:6s} mean {d.mean():+.4f}  p={t.pvalue:.4g}  "
                      f"positive {int((d > 0).sum())}/{len(d)}")
    print("  PSNR: positive = first is better. LPIPS: sign flipped so positive = first is better.")
    out_json.write_text(json.dumps({"arm": a.arm, "per_scene": rows, "tests": tests}, indent=2))
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()

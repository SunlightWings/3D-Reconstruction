"""Does AVERAGING the input orderings correct the pose error it detects? (detector -> corrector)

#045 found that running the same six images through a quantized VGGT in different orders gives camera
poses whose spread predicts the pose error (rho = 0.880). #046 then found the obvious corrector -- COLMAP
on the same six images -- fails outright (no reconstruction on 28/40 scenes).

If the order-induced perturbation is roughly zero-mean about the model's own answer, then averaging the
orderings should cancel part of it, and the same 4x inference already paid for the detector would also
buy a correction. That is the cheapest corrector available and it is tested here.

Per scene and variant:
  * run `--perms` orderings, un-permute, carry each into ordering 0's frame by Sim(3) on camera centres
  * ensemble pose  = chordal mean rotation (sum of rotations projected back to SO(3) by SVD) and mean
                     camera centre, per camera
  * compare pose error of ordering 0 (what a deployment would use today) against the ensemble, both
    versus full precision and versus CO3D ground truth, paired over scenes

`--save-extrinsics` writes the ensemble cameras so a downstream 3DGS arm can be built if -- and only if
-- the pose error actually drops.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from scipy import stats

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
sys.path.insert(0, str(REPO / "code" / "quantization"))
import common  # noqa: E402
from quant_loader import load_quantized, _load_base_model  # noqa: E402
from pose_uncertainty import align_into, angle, pose_error, sha_file, FROZEN, EXPECT_MANIFEST_SHA  # noqa: E402

dv = common.dv
RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
BOWL = "bowl/70_5792_13401"


def chordal_mean(Rs):
    """Rotation average: project the arithmetic mean back onto SO(3), det > 0 enforced."""
    U, _S, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    return U @ D @ Vt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="w4a4")
    ap.add_argument("--perms", type=int, default=4)
    ap.add_argument("--scenes", choices=["subset", "all"], default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-extrinsics", action="store_true")
    a = ap.parse_args()

    if sha_file(FROZEN) != EXPECT_MANIFEST_SHA:
        raise SystemExit("frozen manifest SHA mismatch -- refusing to run")

    from prepare_co3d_scene import import_local_preprocessor
    from vggt_inference_core import infer
    _P, _impl, _sig = import_local_preprocessor(Path("/home/utn/luli38se/cv/QuantVGGT"))

    def preprocess(paths):
        kw = {"mode": "pad"}
        if "target_size" in _sig.parameters:
            kw["target_size"] = 518
        x = _P([str(q) for q in paths], **kw).detach().cpu().float().contiguous()
        if tuple(x.shape) != (6, 3, 518, 518) or not torch.isfinite(x).all():
            raise SystemExit("bad frozen preprocessing output")
        return x

    scenes = common.EXPENSIVE_SCENES if a.scenes == "subset" else common.all_scenes()
    by_scene = {f"{s['category']}/{s['sequence']}": s for s in json.loads(FROZEN.read_text())["scenes"]}

    print(f"loading {a.variant} ...", flush=True)
    model = _load_base_model(a.device) if a.variant == "full" else load_quantized(a.variant, a.device)
    model.to(a.device)

    rows, t0 = [], time.time()
    for si, name in enumerate(scenes, 1):
        cat, seq = common.split_scene(name)
        gdir, full_meta, _w = dv.find_group(cat, seq)
        frames = list(map(int, full_meta["frame_numbers"]))
        lookup = {r["frame_number"]: r["image_path"] for r in by_scene[name]["usable_frames"]}
        x = preprocess([lookup[f] for f in frames])

        rng = np.random.default_rng(zlib.crc32(name.encode()))  # stable across processes:
        # Python's str hash is salted per interpreter (PYTHONHASHSEED), so abs(hash(name)) gave a
        # DIFFERENT permutation set on every run and made the ensemble irreproducible
        perms = [np.arange(len(frames))] + [rng.permutation(len(frames)) for _ in range(a.perms - 1)]
        Es, Ks = [], []
        for p in perms:
            arr, _rt = infer(model, x[p].contiguous(), [frames[i] for i in p], device=a.device)
            inv = np.argsort(p)
            Es.append(arr["extrinsic"].astype(np.float64)[inv])
            Ks.append(arr["intrinsic"].astype(np.float64)[inv])

        # every ordering into ordering 0's frame, then average rotations and centres per camera
        aligned_R, aligned_C = [], []
        for E in Es:
            R, (s, A, b) = align_into(Es[0], E)
            aligned_R.append(R)
            aligned_C.append((s * (A @ dv.camera_centers(E).T)).T + b)
        E_ens = []
        for c in range(len(frames)):
            Rm = chordal_mean([R[c] for R in aligned_R])
            Cm = np.mean([C[c] for C in aligned_C], axis=0)
            E_ens.append(np.concatenate([Rm, (-Rm @ Cm)[:, None]], axis=1))
        E_ens = np.stack(E_ens)
        # intrinsics are order-independent per camera once un-permuted, so a plain elementwise mean is
        # the matching ensemble; #043 showed focal error costs 0.4876 dB on its own, so it is worth correcting
        K_ens = np.mean(np.stack(Ks), axis=0)

        F = dv.load_npz(gdir / "full.npz")
        Ef = F["extrinsic"].astype(np.float64)
        recs = dv.load_annotations(cat, seq)
        Egt = np.stack([dv.co3d_to_opencv_camera(recs[f])[0] for f in frames])
        Kf = F["intrinsic"].astype(np.float64)
        def ferr(K):
            r = np.concatenate([K[:, 0, 0] / Kf[:, 0, 0], K[:, 1, 1] / Kf[:, 1, 1]])
            return float(np.median(np.abs(r - 1)))
        row = {"scene": name,
               "single_focal_err": ferr(Ks[0]), "ensemble_focal_err": ferr(K_ens),
               "single_vs_full": pose_error(Ef, Es[0]), "ensemble_vs_full": pose_error(Ef, E_ens),
               "single_vs_gt": pose_error(Egt, Es[0]), "ensemble_vs_gt": pose_error(Egt, E_ens),
               "spread_deg": float(np.mean([
                   np.mean([angle(aligned_R[i][c], aligned_R[j][c])
                            for i in range(len(Es)) for j in range(i + 1, len(Es))])
                   for c in range(len(frames))]))}
        if a.save_extrinsics:
            row["ensemble_extrinsic"] = E_ens.tolist()
            row["ensemble_intrinsic"] = K_ens.tolist()
        rows.append(row)
        print(f"[{si}/{len(scenes)}] {name}  single {row['single_vs_full']:.3f} -> ensemble "
              f"{row['ensemble_vs_full']:.3f} deg  ({(time.time() - t0) / si:.1f}s/scene)", flush=True)
        RES.mkdir(parents=True, exist_ok=True)
        (RES / f"pose_ensemble_{a.variant}_p{a.perms}.json").write_text(
            json.dumps({"variant": a.variant, "perms": a.perms, "per_scene": rows}, indent=2))

    print("\n" + "=" * 96)
    print(f"POSE ENSEMBLING, variant {a.variant}, {a.perms} orderings, n = {len(rows)} scenes")
    out = {"variant": a.variant, "perms": a.perms, "per_scene": rows, "tests": {}}
    s1 = np.array([r["single_focal_err"] for r in rows]); se = np.array([r["ensemble_focal_err"] for r in rows])
    wf = stats.wilcoxon(se - s1)
    out["focal"] = {"single_median": float(np.median(s1)), "ensemble_median": float(np.median(se)),
                    "improved": int((se < s1).sum()), "n": len(rows), "wilcoxon_p": float(wf.pvalue)}
    print(f"  focal error vs full: single median {np.median(s1):.5f} -> ensemble {np.median(se):.5f}  "
          f"improved {int((se < s1).sum())}/{len(rows)}  Wilcoxon p={wf.pvalue:.3g}")
    for tgt in ("vs_full", "vs_gt"):
        for excl in (False, True):
            sub = [r for r in rows if not (excl and r["scene"] == BOWL)]
            s1 = np.array([r[f"single_{tgt}"] for r in sub])
            se = np.array([r[f"ensemble_{tgt}"] for r in sub])
            w = stats.wilcoxon(se - s1)
            k = f"{tgt}{' (no bowl)' if excl else ''}"
            out["tests"][k] = {"n": len(sub), "single_median": float(np.median(s1)),
                               "ensemble_median": float(np.median(se)),
                               "single_mean": float(s1.mean()), "ensemble_mean": float(se.mean()),
                               "improved": int((se < s1).sum()), "wilcoxon_p": float(w.pvalue),
                               "median_reduction_pct": float(100 * np.median((s1 - se) / np.maximum(s1, 1e-9)))}
            t = out["tests"][k]
            print(f"  {k:16s} n={t['n']:3d}  single median {t['single_median']:.4f} -> ensemble "
                  f"{t['ensemble_median']:.4f} deg  improved {t['improved']}/{t['n']}  "
                  f"median reduction {t['median_reduction_pct']:+.1f}%  Wilcoxon p={t['wilcoxon_p']:.3g}")
    (RES / f"pose_ensemble_{a.variant}_p{a.perms}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {RES}/pose_ensemble_{a.variant}_p{a.perms}.json")


if __name__ == "__main__":
    main()

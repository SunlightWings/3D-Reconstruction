"""A confidence signal for CAMERA POSE, from input-order ensembling. Inference only, no training.

#036 tested cross-view self-consistency as a confidence signal and it failed at the camera level
(rho = -0.032, p = 0.248): a quantized model stays internally consistent even when its cameras are
wrong. But #039/#043 place the rendering damage squarely on the cameras, so a pose-level signal is the
one worth having.

VGGT is not permutation-invariant -- its global attention and its choice of reference frame both depend
on input order -- so running the SAME six images in different orders gives slightly different cameras.
The spread across orderings is an uncertainty estimate computable at deployment, with no ground truth
and no full-precision model. This measures whether that spread predicts the actual pose error.

For each scene and variant:
  * run the model on `--perms` orderings (the identity plus seeded shuffles), un-permuting the outputs
  * align every ordering into the first ordering's frame by Sim(3) on the six camera centres, because
    a different reference frame is not an error
  * spread = mean over the six cameras of the mean pairwise rotation angle across orderings
  * error  = that variant's pose error against full precision's canonical-order cameras

A positive rank correlation between spread and error would give the proposal a confidence signal aimed
at the quantity that actually matters. The cost is `perms` x inference, which must be reported honestly
against the proposal's efficiency claim.
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

dv = common.dv
RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
FROZEN = Path("/var/tmp/luli38se/quantsplat/disagreement_dataset_v1/frozen_dataset_manifest.json")
EXPECT_MANIFEST_SHA = "1ce2f3f8d43cc17f61d84545b82f132a3b64d49d5acf70ff0fe0ac6aa9968e06"


def sha_file(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def angle(R1, R2):
    return float(np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1))))


def align_into(E_ref, E):
    """Carry E into E_ref's frame by the Sim(3) on camera centres; returns rotations only."""
    s, A, b = dv.umeyama(dv.camera_centers(E), dv.camera_centers(E_ref))
    return np.stack([e[:, :3] @ A.T for e in E]), (s, A, b)


def pose_error(E_ref, E):
    R, _ = align_into(E_ref, E)
    return float(np.mean([angle(E_ref[i][:, :3], R[i]) for i in range(len(E_ref))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="w4a4")
    ap.add_argument("--perms", type=int, default=4, help="orderings per scene, including the identity")
    ap.add_argument("--scenes", choices=["subset", "all"], default="all")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    if sha_file(FROZEN) != EXPECT_MANIFEST_SHA:
        raise SystemExit("frozen manifest SHA mismatch -- refusing to run")

    from prepare_co3d_scene import import_local_preprocessor
    from vggt_inference_core import infer

    # identical wrapper to run_w2a4_inference.py: raw load_and_preprocess_images defaults to crop mode
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
    manifest = json.loads(FROZEN.read_text())
    by_scene = {f"{s['category']}/{s['sequence']}": s for s in manifest["scenes"]}

    print(f"loading {a.variant} ...", flush=True)
    # `full` is the control: if order-spread predicts pose error for full precision too, the signal
    # measures VGGT's own instability on ambiguous scenes, not anything about quantization.
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
        Es = []
        for p in perms:
            arr, _rt = infer(model, x[p].contiguous(), [frames[i] for i in p], device=a.device)
            inv = np.argsort(p)                       # put cameras back in canonical frame order
            Es.append(arr["extrinsic"].astype(np.float64)[inv])

        # spread: mean pairwise rotation disagreement per camera, after aligning to ordering 0
        Rs = [align_into(Es[0], E)[0] for E in Es]
        per_cam = []
        for c in range(len(frames)):
            pair = [angle(Rs[i][c], Rs[j][c]) for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
            per_cam.append(float(np.mean(pair)))
        spread = float(np.mean(per_cam))

        F = dv.load_npz(gdir / "full.npz")
        # for the `full` control this is canonical-order full vs saved full: a reproducibility check, ~0
        err_vs_full = pose_error(F["extrinsic"].astype(np.float64), Es[0])
        recs = dv.load_annotations(cat, seq)
        Egt = np.stack([dv.co3d_to_opencv_camera(recs[f])[0] for f in frames])
        rows.append({"scene": name, "spread_deg": spread, "per_camera_spread": per_cam,
                     "err_vs_full_deg": err_vs_full,
                     "err_vs_gt_deg": pose_error(Egt, Es[0]),
                     "full_err_vs_gt_deg": pose_error(Egt, F["extrinsic"].astype(np.float64))})
        print(f"[{si}/{len(scenes)}] {name}  spread {spread:.3f} deg  err-vs-full {err_vs_full:.3f}  "
              f"({(time.time() - t0) / si:.1f}s/scene)", flush=True)
        RES.mkdir(parents=True, exist_ok=True)
        (RES / f"pose_uncertainty_{a.variant}_p{a.perms}.json").write_text(
            json.dumps({"variant": a.variant, "perms": a.perms, "per_scene": rows}, indent=2))

    print("\n" + "=" * 96)
    print(f"POSE UNCERTAINTY from {a.perms} input orderings, variant {a.variant}, n = {len(rows)} scenes")
    sp = np.array([r["spread_deg"] for r in rows])
    print(f"  order-induced spread: median {np.median(sp):.4f} deg, range {sp.min():.4f}-{sp.max():.4f}")
    out = {"variant": a.variant, "perms": a.perms, "per_scene": rows, "tests": {}}
    for tgt in ("err_vs_full_deg", "err_vs_gt_deg"):
        for excl in (False, True):
            sub = [r for r in rows if not (excl and r["scene"] == "bowl/70_5792_13401")]
            rho, p = stats.spearmanr([r["spread_deg"] for r in sub], [r[tgt] for r in sub])
            k = f"{tgt}{' (no bowl)' if excl else ''}"
            out["tests"][k] = {"n": len(sub), "rho": float(rho), "p": float(p)}
            print(f"  spread vs {k:26s} n={len(sub)} rho={rho:+.3f} p={p:.3g}")
    (RES / f"pose_uncertainty_{a.variant}_p{a.perms}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {RES}/pose_uncertainty_{a.variant}_p{a.perms}.json")


if __name__ == "__main__":
    main()

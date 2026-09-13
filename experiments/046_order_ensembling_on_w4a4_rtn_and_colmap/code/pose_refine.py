"""Can the camera damage in a quantized geometry prior be CORRECTED? Classical SfM as the corrector.

#039 and #043 place `w4a4_rtn`'s rendering loss on its cameras (pose +0.6943 dB, focal +0.4876 dB), and
#042 says rendering stops caring below roughly 2-4 degrees of rotation error. So the actionable question
for the proposal is no longer "can we predict the error" but "can we remove it": if the six input images
alone can pull a 14.66-degree pose error under ~4 degrees, a quantized prior is safe to deploy with a
refinement step, and the confidence-weighting idea is replaced by something that demonstrably works.

Stage 1 (this file, `refine`): run COLMAP SIFT + exhaustive matching + incremental mapping on the six
518x518 input images of each scene, with the intrinsics the quantized model predicted held fixed. The
result is a pose estimate that uses NO ground truth and NO full-precision model -- exactly what is
available at deployment. Report, per scene: whether all six images registered, and the pose error of
the refined cameras against full precision and against CO3D ground truth, beside the quantized arm's
own error, all by the same Sim(3)-on-camera-centres construction used everywhere else.

A failure here is a result too: classical SfM is known to struggle on six wide-baseline views of a
textureless object, and that is precisely why feed-forward priors exist. Success rate is reported, never
hidden, and scenes that fail are excluded explicitly rather than silently.

Stage 2 (`build`): for scenes where refinement succeeded, write a 3DGS source with the refined cameras
and the arm's points, so the recovered rendering quality can be measured against `full` and against the
uncorrected arm with the same evaluation cameras as #039/#043.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
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
BOWL = "bowl/70_5792_13401"


def angle(R1, R2):
    return float(np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1))))


def pose_err(E_ref, E):
    """Mean per-camera rotation error after the Sim(3) that maps E's centres onto E_ref's."""
    s, A, b = dv.umeyama(dv.camera_centers(E), dv.camera_centers(E_ref))
    return float(np.mean([angle(E_ref[i][:, :3], E[i][:, :3] @ A.T) for i in range(len(E_ref))]))


def colmap_poses(image_dir: Path, K: np.ndarray, names, workdir: Path):
    """SIFT + exhaustive matching + incremental mapping on the six input images.

    Intrinsics are fixed to the quantized model's prediction (PINHOLE, per-image camera), so this
    refines POSE only -- matching #043's split, where pose and focal are separate conditions.
    Returns world-to-camera extrinsics in COLMAP's frame for the images that registered, or None.
    """
    import pycolmap
    db = workdir / "db.db"
    # CPU device: this runs beside GPU jobs and must not contend for the device
    pycolmap.extract_features(
        db, image_dir, camera_mode=pycolmap.CameraMode.PER_IMAGE, device=pycolmap.Device.cpu,
        reader_options=pycolmap.ImageReaderOptions(
            camera_model="PINHOLE",
            camera_params=",".join(str(v) for v in (K[0, 0], K[1, 1], K[0, 2], K[1, 2])),
        ),
    )
    pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    recs = pycolmap.incremental_mapping(db, image_dir, workdir / "sparse")
    if not recs:
        return None, None, 0      # COLMAP produced no reconstruction at all
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    by_name = {im.name: im for im in rec.images.values()}
    # COLMAP often registers a subset (5/6 is typical here). Score the subset it did register and
    # report the count: demanding all six would discard usable information and hide the true failure mode.
    E, idx = [], []
    for i, n in enumerate(names):
        if n not in by_name:
            continue
        cfw = by_name[n].cam_from_world
        cfw = cfw() if callable(cfw) else cfw     # pycolmap 4.2 exposes this as a method
        E.append(np.asarray(cfw.matrix(), np.float64)[:3, :4])
        idx.append(i)
    if len(E) < 4:                                 # a Sim(3) on fewer than four centres is not meaningful
        return None, None, len(E)
    return np.stack(E), np.asarray(idx), len(E)


def refine(a):
    rows = []
    for si, name in enumerate(common.all_scenes()[: a.scenes], 1):
        cat, seq = name.split("/", 1)
        man, gdir, tensor, n_views = gp.scene_inputs(cat, seq)
        Q = dv.load_npz(gdir / f"{a.arm}.npz")
        F = dv.load_npz(gdir / "full.npz")
        recs = dv.load_annotations(cat, seq)
        Egt = np.stack([dv.co3d_to_opencv_camera(recs[f])[0] for f in man["input_frames"]])
        img_dir = common.scene_root(cat, seq) / "common" / "train_images"
        names = sorted(p.name for p in img_dir.glob("*.png"))
        t0 = time.time()
        with tempfile.TemporaryDirectory(dir="/var/tmp/luli38se") as td:
            td = Path(td)
            (td / "sparse").mkdir()
            try:
                E_ref, idx, n_reg = colmap_poses(img_dir, Q["intrinsic"].astype(np.float64)[0], names, td)
            except Exception as e:  # noqa: BLE001 -- a COLMAP failure is data, not a crash
                E_ref, idx, n_reg = None, None, -1
                print(f"    {name}: pycolmap raised {type(e).__name__}: {e}", flush=True)
        row = {"scene": name, "n_registered": int(n_reg), "seconds": round(time.time() - t0, 1),
               "arm_err_vs_full": pose_err(F["extrinsic"].astype(np.float64), Q["extrinsic"].astype(np.float64)),
               "arm_err_vs_gt": pose_err(Egt, Q["extrinsic"].astype(np.float64)),
               "full_err_vs_gt": pose_err(Egt, F["extrinsic"].astype(np.float64))}
        if E_ref is not None:
            # compare on exactly the cameras COLMAP registered, so refined and arm errors are like-for-like
            Ef_s, Eq_s, Eg_s = (F["extrinsic"].astype(np.float64)[idx], Q["extrinsic"].astype(np.float64)[idx],
                                Egt[idx])
            row["refined_err_vs_full"] = pose_err(Ef_s, E_ref)
            row["refined_err_vs_gt"] = pose_err(Eg_s, E_ref)
            row["arm_err_vs_full_subset"] = pose_err(Ef_s, Eq_s)
            row["arm_err_vs_gt_subset"] = pose_err(Eg_s, Eq_s)
            row["registered_idx"] = [int(i) for i in idx]
            row["refined_extrinsic"] = E_ref.tolist()
        rows.append(row)
        ok = "refined_err_vs_full" in row
        print(f"[{si}/{a.scenes}] {name}  registered {n_reg}/6  "
              + (f"arm {row['arm_err_vs_full']:.2f} deg -> refined {row['refined_err_vs_full']:.2f} deg"
                 if ok else "FAILED") + f"  ({row['seconds']}s)", flush=True)
        RES.mkdir(parents=True, exist_ok=True)
        (RES / f"pose_refine_{a.arm}.json").write_text(json.dumps({"arm": a.arm, "per_scene": rows}, indent=2))

    ok = [r for r in rows if "refined_err_vs_full" in r]
    print("\n" + "=" * 96)
    print(f"POSE REFINEMENT by COLMAP on the six input images, arm {a.arm}")
    reg = [r["n_registered"] for r in rows]
    print(f"  usable (>= 4 of 6 images registered): {len(ok)}/{len(rows)} scenes; "
          f"registered-image counts: " + ", ".join(f"{k}:{reg.count(k)}" for k in sorted(set(reg))))
    out = {"arm": a.arm, "per_scene": rows, "n_success": len(ok), "n_total": len(rows), "tests": {}}
    if ok:
        for tgt in ("vs_full", "vs_gt"):
            arm = np.array([r[f"arm_err_{tgt}_subset"] for r in ok])
            ref = np.array([r[f"refined_err_{tgt}"] for r in ok])
            w = stats.wilcoxon(ref - arm)
            out["tests"][tgt] = {"n": len(ok), "arm_mean": float(arm.mean()), "refined_mean": float(ref.mean()),
                                 "arm_median": float(np.median(arm)), "refined_median": float(np.median(ref)),
                                 "wilcoxon_p": float(w.pvalue), "improved": int((ref < arm).sum()),
                                 "under_4deg": int((ref < 4).sum()), "arm_under_4deg": int((arm < 4).sum())}
            t = out["tests"][tgt]
            print(f"  {tgt}: arm median {t['arm_median']:.3f} deg -> refined {t['refined_median']:.3f} deg; "
                  f"improved in {t['improved']}/{len(ok)}, Wilcoxon p={t['wilcoxon_p']:.3g}; "
                  f"under 4 deg: arm {t['arm_under_4deg']} -> refined {t['under_4deg']}")
        fg = np.array([r["full_err_vs_gt"] for r in ok])
        print(f"  reference: full precision's own error vs GT, same scenes, median {np.median(fg):.3f} deg")
    (RES / f"pose_refine_{a.arm}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {RES}/pose_refine_{a.arm}.json")


def build(a):
    """3DGS sources with COLMAP-refined cameras + the arm's points, in full's frame."""
    d = json.load(open(RES / f"pose_refine_{a.arm}.json"))
    made = []
    for r in d["per_scene"]:
        if "refined_extrinsic" not in r:
            continue
        cat, seq = r["scene"].split("/", 1)
        man, gdir, tensor, n_views = gp.scene_inputs(cat, seq)
        F, Q = dv.load_npz(gdir / "full.npz"), dv.load_npz(gdir / f"{a.arm}.npz")
        E_ref = np.asarray(r["refined_extrinsic"], np.float64)
        # COLMAP works in its own frame: carry it into full's by the Sim(3) on camera centres
        s, A, b = dv.umeyama(dv.camera_centers(E_ref), dv.camera_centers(F["extrinsic"].astype(np.float64)))
        E_in_full = np.stack([dv.transform_camera_to_variant(e, s, A, b) for e in E_ref])
        sq, Aq, bq, _ = gp.arm_into_full(F, Q)
        pq, colors = gp.grid_points(Q, tensor)
        pq_f = (sq * (Aq @ pq.T.astype(np.float64))).T + bq
        gp.build_source(cat, seq, f"refined_{a.arm}", E_in_full, Q["intrinsic"], pq_f, colors, n_views)
        made.append(r["scene"])
    print(f"built {len(made)} refined sources for {a.arm}")
    (RES / f"pose_refine_{a.arm}_sources.json").write_text(json.dumps({"scenes": made}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["refine", "build"])
    ap.add_argument("--arm", default="w4a4_rtn")
    ap.add_argument("--scenes", type=int, default=40)
    a = ap.parse_args()
    (refine if a.stage == "refine" else build)(a)

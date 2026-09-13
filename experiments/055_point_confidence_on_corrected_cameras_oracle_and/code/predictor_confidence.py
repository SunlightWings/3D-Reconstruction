"""Real (learned) per-point confidence for W3A3, from W3A3's OWN output only -- no full precision at test time.

Same predictor as cpu_probes.py `predict` (#036): 9 features per pixel (depth, depth gradients/roughness, image
gradient/texture/intensity, radial position, cross-view self-inconsistency), label = pixel in the top decile of
point error within its own view-group, L2 logistic regression on standardised features.

Training: W4A4 vs full precision, one view-group per scene, on the 32 scenes that are NOT the 8 test scenes.
Test: W3A3 on the 8 EXPENSIVE_SCENES, every pixel of the stride-4 init grid (row-major per view, i.e. exactly the
order of points3D.txt). Confidence = -logit (higher = more trusted). Also reports how well it ranks W3A3's true error.
"""
from __future__ import annotations

import importlib.util
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
from confidence_oracle import per_point_error  # noqa: E402

spec = importlib.util.spec_from_file_location("cpu_probes", REPO / "code/quantization/cpu_probes.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)
dv = common.dv
RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
OUT = RES / "conf_predictor"


def main():
    test = set(common.EXPENSIVE_SCENES)
    Xs, ys = [], []
    for n in common.all_scenes():
        if n in test:
            continue
        c, s = n.split("/", 1)
        g, _m, _w = dv.find_group(c, s)
        r = cp.predict_group(str(g / "full.npz"))
        if "error" in r:
            print("  skip", n, r["error"]); continue
        Xs.append(r["X"].astype(np.float64))
        ys.append((r["err"] >= np.quantile(r["err"], 0.9)).astype(np.float64))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w = cp.fit_logistic((X - mu) / sd, y)
    print(f"trained on {len(Xs)} W4A4 scenes, {len(y)} pixels; weights {np.round(w, 3).tolist()}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    vv, uu = np.mgrid[0:518:4, 0:518:4]
    grid = (vv.ravel(), uu.ravel())
    for n in common.EXPENSIVE_SCENES:
        c, s = n.split("/", 1)
        g, _m, _w = dv.find_group(c, s)
        Q = dv.load_npz(g / "w3a3.npz")
        recs = dv.load_annotations(c, s)
        img = dv.preprocess_images([dv.resolve_image_path(recs[int(f)]) for f in Q["frame_numbers"]])
        Xq = cp.pixel_features(Q["depth"][..., 0].astype(np.float64), img, Q["extrinsic"].astype(np.float64),
                               Q["intrinsic"].astype(np.float64), Q["world_points_from_depth"].astype(np.float64),
                               [grid] * len(Q["frame_numbers"]))
        logit = np.c_[(Xq - mu) / sd, np.ones(len(Xq))] @ w
        conf = -logit
        np.save(OUT / f"{c}_{s}_w3a3.npy", conf.astype(np.float32))
        err = per_point_error(c, s, "w3a3")
        lab = err >= np.quantile(err, 0.9)
        au = cp.auroc(logit, lab)
        rho = stats.spearmanr(conf, -err).statistic
        report.append({"scene": n, "n_points": int(len(conf)), "auroc_top_decile": au, "spearman_conf_vs_neg_err": float(rho)})
        print(f"  {n}: {len(conf)} points  AUROC(top-decile W3A3 error) {au:.3f}  Spearman(conf, -err) {rho:+.3f}", flush=True)
    print(f"mean AUROC on W3A3 {np.mean([r['auroc_top_decile'] for r in report]):.3f}")
    (RES / "predictor_confidence_w3a3.json").write_text(json.dumps({"weights": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
                                                                    "features": cp.FEATS, "per_scene": report}, indent=2))


if __name__ == "__main__":
    main()

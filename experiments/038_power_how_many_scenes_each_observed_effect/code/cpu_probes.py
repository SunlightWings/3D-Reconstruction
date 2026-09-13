"""CPU-only experiments, run while the GPU is busy. Each subcommand saves raw JSON and prints a summary.

  robustness   Re-tests every headline claim with assumption-light statistics: bootstrap 95% CI,
               sign-flip permutation p, leave-one-scene-out range, the claim with the unstable
               `bowl/70_5792_13401` scene removed (#031), and exact permutation p for the n = 6
               dose-response. A claim that only survives a t-test is not a claim.

  gaussians    Mechanism, read off the TRAINED 3DGS models against CO3D ground-truth point clouds.
               The swap probe says bad cameras hurt and bad points do not. If that is because 3DGS
               moves bad points to the right place but cannot undo bad cameras, it is visible in
               the final Gaussians: swapP should start far from GT and end near it; swapK should
               start near GT and end away from it.

  predict      Can W4A4's error be predicted from W4A4's OWN output (the proposal's core)? Per
               pixel and per camera, using only quantized outputs and RGB. Validated by scene-held-out
               and category-held-out folds, because the 1330 view-groups are drawn from only 40
               scenes in 20 categories and a random split would leak.

  depthgt      Depth accuracy of full and W4A4 against CO3D ground-truth depth, all 1330 view-groups,
               plus whether their disagreement concentrates at the object boundary.

All geometry is compared after the ablation_compare.py construction: Sim(3) on the six input
camera centres, distances in scene radii (median camera distance from the camera centroid).
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize, stats
from scipy.spatial import cKDTree

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
import common  # noqa: E402

dv = common.dv
RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
PRED = Path("/var/tmp/luli38se/quantsplat/disagreement_dataset_v1/run/predictions")
DSV = Path("/var/tmp/luli38se/quantsplat/downstream_validation_v1")
BOWL = "bowl/70_5792_13401"
SEED = 0


def save(name, obj):
    p = RES / f"cpu_{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"\nsaved {p}")


def sim3(C_src, C_dst):
    s, A, b = dv.umeyama(C_src, C_dst)
    radius = max(float(np.median(np.linalg.norm(C_dst - np.median(C_dst, axis=0), axis=1))), 1e-12)
    return s, A, b, radius


def apply(s, A, b, X):
    return (s * (A @ np.asarray(X, np.float64).T)).T + b


def centers(E):
    return dv.camera_centers(np.asarray(E, np.float64))


# =========================================================================== robustness

def contrast(d, scenes, n_boot=10000, n_perm=20000):
    d = np.asarray(d, np.float64)
    rng = np.random.default_rng(SEED)
    boot = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = np.abs((flips * d).mean(axis=1))
    loo = np.array([np.delete(d, i).mean() for i in range(len(d))])
    loo_p = [stats.ttest_1samp(np.delete(d, i), 0).pvalue for i in range(len(d))]
    keep = np.array([s != BOWL for s in scenes])
    out = {
        "n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)),
        "t_p": float(stats.ttest_1samp(d, 0).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
        "perm_p": float((np.sum(perm >= abs(d.mean())) + 1) / (n_perm + 1)),
        "boot_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "positive": int((d > 0).sum()),
        "loo_mean_range": [float(loo.min()), float(loo.max())], "loo_max_t_p": float(max(loo_p)),
    }
    if keep.sum() < len(d):
        out["no_bowl"] = {"n": int(keep.sum()), "mean": float(d[keep].mean()),
                          "t_p": float(stats.ttest_1samp(d[keep], 0).pvalue)}
    cats = sorted({s.split("/")[0] for s in scenes})
    out["per_category_mean"] = {c: float(np.mean([x for x, s in zip(d, scenes) if s.startswith(c + "/")]))
                                for c in cats}
    return out


def exact_spearman_p(x, y):
    r = stats.spearmanr(x, y).statistic
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    perms = [stats.spearmanr(rx, np.array(p)).statistic for p in itertools.permutations(ry)]
    return float(r), float(np.mean(np.abs(perms) >= abs(r) - 1e-12))


def robustness():
    out = {}
    arms40 = json.load(open(sorted(RES.glob("arms_full_*_all_it7000.json"))[-1]))["per_scene"]
    sc = [r["scene"] for r in arms40]

    def g(r, arm, m):
        return r[f"{arm}_foreground_{m}"]

    print("=" * 100)
    print("ROBUSTNESS of headline contrasts. Positive = second arm renders WORSE than the first.")
    hdr = f"{'contrast':40s} {'n':>3s} {'mean':>9s} {'boot CI95':>21s} {'t p':>9s} {'perm p':>9s} " \
          f"{'LOO max p':>9s} {'no-bowl mean/p':>18s}"
    print(hdr)

    def show(label, res):
        nb = res.get("no_bowl")
        nbs = f"{nb['mean']:+.4f}/{nb['t_p']:.2g}" if nb else "-"
        print(f"{label:40s} {res['n']:3d} {res['mean']:+9.4f} [{res['boot_ci95'][0]:+.4f},{res['boot_ci95'][1]:+.4f}] "
              f"{res['t_p']:9.3g} {res['perm_p']:9.3g} {res['loo_max_t_p']:9.3g} {nbs:>18s}")
        out[label] = res

    for arm in ("w4a4", "w4a4_local", "w4a4_no_rot", "w4a4_rtn"):
        show(f"PSNR  full - {arm}", contrast([g(r, "full", "psnr") - g(r, arm, "psnr") for r in arms40], sc))
        show(f"LPIPS {arm} - full", contrast([g(r, arm, "lpips") - g(r, "full", "lpips") for r in arms40], sc))
    show("SSIM  full - w4a4", contrast([g(r, "full", "ssim") - g(r, "w4a4", "ssim") for r in arms40], sc))

    for fname, tag, pairs in (
        ("confidence_oracle_w4a4_rtn_it7000.json", "conf-oracle",
         (("C-R", "w4a4_rtn_conf50", "w4a4_rtn_rand50"), ("C-B", "w4a4_rtn_conf50", "w4a4_rtn"))),
        ("geometry_probe_camsplit_w4a4_rtn_it7000.json", "camsplit",
         (("full-R pose", "full", "swapR_w4a4_rtn"), ("full-F focal", "full", "swapF_w4a4_rtn"))),
        ("geometry_probe_swap_w4a4_rtn_it7000.json", "swap",
         (("full-P", "full", "swapP_w4a4_rtn"), ("full-K", "full", "swapK_w4a4_rtn"),
          ("full-B", "full", "swapB_w4a4_rtn"))),
    ):
        p = RES / fname
        if not p.is_file():
            continue
        rows = json.load(open(p))["per_scene"]
        s2 = [r["scene"] for r in rows]
        for lbl, x, y in pairs:
            # PSNR: x minus y; LPIPS: y minus x, so positive always means y is worse
            show(f"PSNR  {tag} {lbl}", contrast([r[f"{x}_psnr"] - r[f"{y}_psnr"] for r in rows], s2))
            show(f"LPIPS {tag} {lbl}", contrast([r[f"{y}_lpips"] - r[f"{x}_lpips"] for r in rows], s2))

    # dose-response: arm means vs 40-scene camera error, exact permutation p over 6! orderings
    cam = json.load(open(RES / "cpu_camera_analysis.json"))["per_scene_rows"]
    arms6 = ["w4a4_local", "w4a4_no_lwc", "w4a4_no_smooth", "w4a4_no_lac", "w4a4_no_rot", "w4a4_rtn"]
    print("\nDOSE-RESPONSE, six arms, exact permutation p (all 720 orderings)")
    out["dose"] = {}
    for excl in (False, True):
        keep = [s for s in sc if not (excl and s == BOWL)]
        x = [np.mean([c["cam_rot"] for c in cam if c["arm"] == a and c["scene"] in keep]) for a in arms6]
        for m in ("psnr", "ssim", "lpips", "psnr_exposure_corrected"):
            y = [np.mean([g(r, a, m) for r in arms40 if r["scene"] in keep]) for a in arms6]
            rho, p = exact_spearman_p(x, y)
            loo = [stats.spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(6)]
            key = f"{m}{' (no bowl)' if excl else ''}"
            out["dose"][key] = {"rho": rho, "exact_p": p, "leave_one_arm_out_rho": [float(v) for v in loo]}
            print(f"  {key:34s} rho={rho:+.4f}  exact p={p:.4f}  leave-one-arm-out rho "
                  f"{min(loo):+.3f}..{max(loo):+.3f}")
    save("robustness", out)


# =========================================================================== gaussians

def read_ply_xyz_opacity(path):
    """x, y, z, opacity from a binary little-endian GraphDECO PLY via memmap (files are ~120 MB)."""
    with open(path, "rb") as f:
        props, n, hdr = [], 0, 0
        while True:
            line = f.readline()
            hdr += len(line)
            t = line.decode().strip()
            if t.startswith("element vertex"):
                n = int(t.split()[-1])
            elif t.startswith("property"):
                props.append(t.split()[-1])
            elif t == "end_header":
                break
    dt = np.dtype([(p, "<f4") for p in props])
    mm = np.memmap(path, dtype=dt, mode="r", offset=hdr, shape=(n,))
    xyz = np.stack([mm["x"], mm["y"], mm["z"]], 1).astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-mm["opacity"].astype(np.float64)))
    return xyz, op


def gt_cloud(cat, seq):
    from plyfile import PlyData
    p = dv.CO3D_ROOT / cat / seq / "pointcloud.ply"
    v = PlyData.read(str(p))["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def cloud_vs_gt(X, G_tree, G, lo, hi, radius):
    """Accuracy: model -> GT, restricted to the GT bounding box (+10%); completeness: GT -> model."""
    if len(X) == 0:
        return {"acc": float("nan"), "comp": float("nan"), "n_in_box": 0}
    inb = np.all((X >= lo) & (X <= hi), axis=1)
    Xb = X[inb]
    acc = float(np.median(G_tree.query(Xb)[0]) / radius) if len(Xb) else float("nan")
    comp = float(np.median(cKDTree(X).query(G)[0]) / radius)
    return {"acc": acc, "comp": comp, "n_in_box": int(inb.sum())}


OPACITY_MIN = float(os.environ.get("GS_OPACITY_MIN", "0.5"))  # which Gaussians count as surface
GS_ARMS = ["full", "w4a4", "w4a4_rtn", "w4a4_rtn_conf50", "w4a4_rtn_rand50",
           "swapP_w4a4_rtn", "swapK_w4a4_rtn", "swapB_w4a4_rtn"]
FULL_FRAME = {"full", "swapP_w4a4_rtn", "swapK_w4a4_rtn", "swapB_w4a4_rtn"}


def gaussians_scene(name):
    cat, seq = name.split("/", 1)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    man = common.scene_manifest(cat, seq)
    recs = dv.load_annotations(cat, seq)
    gdir, _fm, _wm = dv.find_group(cat, seq)
    F = dv.load_npz(gdir / "full.npz")
    Cf = centers(F["extrinsic"])
    Egt = np.stack([dv.co3d_to_opencv_camera(recs[f])[0] for f in man["input_frames"]])
    s, A, b, radius = sim3(centers(Egt), Cf)           # GT world -> full's frame
    G = apply(s, A, b, gt_cloud(cat, seq))
    if len(G) > 60000:
        G = G[rng.choice(len(G), 60000, replace=False)]
    lo, hi = G.min(0), G.max(0)
    pad = 0.1 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    tree = cKDTree(G)
    row = {"scene": name, "gt_points": int(len(G))}
    for arm in GS_ARMS:
        ply = DSV / cat / seq / "models" / arm / "point_cloud" / "iteration_7000" / "point_cloud.ply"
        src = DSV / cat / seq / "sources" / arm / "train" / "sparse" / "0" / "points3D.txt"
        if not ply.is_file():
            continue
        xyz, op = read_ply_xyz_opacity(ply)
        init = common_read_points3d(src) if src.is_file() else None
        if arm not in FULL_FRAME:
            # arm-frame models: carry into full's frame with the arm->full Sim(3) on camera centres
            base = "w4a4_rtn" if arm.startswith("w4a4_rtn") else arm
            Q = dv.load_npz(gdir / f"{base}.npz")
            sa, Aa, ba, _ = sim3(centers(Q["extrinsic"]), Cf)
            xyz = apply(sa, Aa, ba, xyz)
            if init is not None:
                init = apply(sa, Aa, ba, init)
        solid = xyz[op > OPACITY_MIN]
        if len(solid) > 60000:
            solid = solid[rng.choice(len(solid), 60000, replace=False)]
        fin = cloud_vs_gt(solid, tree, G, lo, hi, radius)
        row[arm] = {"n_gaussians": int(len(xyz)), "frac_solid": float(np.mean(op > OPACITY_MIN)),
                    "final_acc": fin["acc"], "final_comp": fin["comp"], "final_n_in_box": fin["n_in_box"]}
        if init is not None:
            ini = cloud_vs_gt(init, tree, G, lo, hi, radius)
            row[arm].update({"init_acc": ini["acc"], "init_comp": ini["comp"], "n_init": int(len(init))})
    return row


def common_read_points3d(path):
    xyz = []
    for line in open(path):
        if line and not line.startswith("#"):
            p = line.split()
            if len(p) >= 4:
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(xyz, np.float64)


def gaussians(workers):
    scenes = common.all_scenes()
    with Pool(workers) as pool:
        rows = pool.map(gaussians_scene, scenes)
    print("=" * 100)
    print("TRAINED-MODEL GEOMETRY vs CO3D ground-truth point cloud (median distance, scene radii; "
          "lower = better)")
    print(f"{'arm':18s} {'n':>3s} {'#gauss':>9s} {'solid':>6s} {'init acc':>9s} {'final acc':>10s} "
          f"{'init comp':>10s} {'final comp':>11s}")
    summ = {}
    for arm in GS_ARMS:
        v = [r[arm] for r in rows if arm in r]
        if not v:
            continue
        m = lambda k: float(np.nanmean([x.get(k, np.nan) for x in v]))  # noqa: E731
        summ[arm] = {k: m(k) for k in ("n_gaussians", "frac_solid", "init_acc", "final_acc",
                                        "init_comp", "final_comp")} | {"n": len(v)}
        s = summ[arm]
        print(f"{arm:18s} {len(v):3d} {s['n_gaussians']:9.0f} {s['frac_solid']:6.3f} {s['init_acc']:9.4f} "
              f"{s['final_acc']:10.4f} {s['init_comp']:10.4f} {s['final_comp']:11.4f}")
    # paired mechanism tests on scenes that have every swap model
    both = [r for r in rows if all(a in r for a in ("full", "swapP_w4a4_rtn", "swapK_w4a4_rtn"))]
    print(f"\nMECHANISM, paired over the {len(both)} scenes with full, swapP and swapK models:")
    mech = {}
    for lbl, arr in (
        ("swapP init acc - full init acc (bad points start worse)",
         [r["swapP_w4a4_rtn"]["init_acc"] - r["full"]["init_acc"] for r in both]),
        ("swapP final acc - full final acc (does training close it?)",
         [r["swapP_w4a4_rtn"]["final_acc"] - r["full"]["final_acc"] for r in both]),
        ("swapK init acc - full init acc (same points: expect 0)",
         [r["swapK_w4a4_rtn"]["init_acc"] - r["full"]["init_acc"] for r in both]),
        ("swapK final acc - full final acc (do bad cameras corrupt it?)",
         [r["swapK_w4a4_rtn"]["final_acc"] - r["full"]["final_acc"] for r in both]),
    ):
        a = np.asarray(arr)
        if len(a) >= 2:
            p = stats.wilcoxon(a).pvalue if np.any(a != 0) else 1.0
            mech[lbl] = {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
                         "wilcoxon_p": float(p), "positive": int((a > 0).sum())}
            print(f"  {lbl:62s} mean {a.mean():+.4f}  median {np.median(a):+.4f}  "
                  f"Wilcoxon p={p:.3g}  positive {int((a > 0).sum())}/{len(a)}")
    save("gaussians" if OPACITY_MIN == 0.5 else f"gaussians_op{OPACITY_MIN:g}",
         {"per_scene": rows, "summary": summ, "mechanism": mech, "opacity_min": OPACITY_MIN})


# =========================================================================== predict

def box(x, k):
    return ndimage.uniform_filter(x, size=k, mode="nearest")


def pixel_features(depth, img, E, K, pts, sel):
    """Features for sampled pixels of every view, from ONE arm's output plus RGB. No full precision."""
    V, H, W = depth.shape
    feats = []
    yy, xx = np.mgrid[0:H, 0:W]
    rc = np.hypot(yy - H / 2, xx - W / 2) / (H / 2)
    for i in range(V):
        d = np.maximum(depth[i], 1e-6)
        gray = img[i].mean(0)
        gy, gx = np.gradient(d)
        iy, ix = np.gradient(gray)
        dstd = np.sqrt(np.maximum(box(d ** 2, 5) - box(d, 5) ** 2, 0)) / d
        tex = np.sqrt(np.maximum(box(gray ** 2, 7) - box(gray, 7) ** 2, 0))
        v, u = sel[i]
        # cross-view self-consistency: reproject this view's points into the other views using the
        # SAME arm's cameras and compare with that arm's depth there
        X = pts[i, v, u]
        rel = []
        for j in range(V):
            if j == i:
                continue
            Xc = (E[j][:, :3] @ X.T).T + E[j][:, 3]
            z = Xc[:, 2]
            p = (K[j] @ Xc.T).T
            with np.errstate(divide="ignore", invalid="ignore"):
                uu, vv = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
            ok = (z > 1e-6) & (uu >= 0) & (uu < W - 1) & (vv >= 0) & (vv < H - 1)
            r = np.full(len(X), np.nan)
            dj = depth[j][np.clip(np.round(vv).astype(int), 0, H - 1), np.clip(np.round(uu).astype(int), 0, W - 1)]
            r[ok] = np.abs(z[ok] - dj[ok]) / np.maximum(dj[ok], 1e-6)
            rel.append(r)
        rel = np.array(rel)
        with np.errstate(all="ignore"):
            cons = np.nanmedian(rel, axis=0)
        cons = np.where(np.isfinite(cons), cons, np.nanmax(cons[np.isfinite(cons)]) if np.isfinite(cons).any() else 1.0)
        med = np.median(d)
        feats.append(np.stack([
            np.log(d[v, u]), np.log(d[v, u] / med), (np.hypot(gx, gy) / d)[v, u], dstd[v, u],
            np.hypot(ix, iy)[v, u], tex[v, u], gray[v, u], rc[v, u], np.log(cons + 1e-4),
        ], 1))
    return np.concatenate(feats)


TARGET = os.environ.get("PREDICT_TARGET", "absolute")  # "relative": error / distance from full's camera
FEATS = ["log_depth", "log_depth_rel_median", "depth_grad_rel", "depth_local_std_rel", "img_grad",
         "img_texture", "img_intensity", "radial_pos", "log_self_inconsistency"]


def predict_group(npz_full):
    npz_full = Path(npz_full)
    cat, seq = npz_full.parts[-4], npz_full.parts[-3]
    name = f"{cat}/{seq}"
    rng = np.random.default_rng(zlib.crc32(str(npz_full).encode()))
    try:
        Fz, Wz = np.load(npz_full), np.load(npz_full.parent / "w4a4.npz")
        frames = Fz["frame_numbers"]
        recs = dv.load_annotations(cat, seq)
        img = dv.preprocess_images([dv.resolve_image_path(recs[int(f)]) for f in frames])
        Ef, Ew = Fz["extrinsic"].astype(np.float64), Wz["extrinsic"].astype(np.float64)
        Kw = Wz["intrinsic"].astype(np.float64)
        Pf, Pw = Fz["world_points_from_depth"].astype(np.float64), Wz["world_points_from_depth"].astype(np.float64)
        Dw = Wz["depth"][..., 0].astype(np.float64)
        s, A, b, radius = sim3(centers(Ew), centers(Ef))
        content = [dv.content_mask_from_record(recs[int(f)]) for f in frames]
        fg = [dv.foreground_mask_from_record(recs[int(f)]) for f in frames]
        sel, err, isfg = [], [], []
        for i in range(len(frames)):
            vv, uu = np.nonzero(content[i])
            k = rng.choice(len(vv), size=min(300, len(vv)), replace=False)
            v, u = vv[k], uu[k]
            sel.append((v, u))
            e_abs = np.linalg.norm(apply(s, A, b, Pw[i, v, u]) - Pf[i, v, u], axis=1)
            rng_f = np.linalg.norm(Pf[i, v, u] - dv.camera_center(Ef[i]), axis=1)
            err.append(e_abs / radius if TARGET == "absolute" else e_abs / np.maximum(rng_f, 1e-9))
            isfg.append(fg[i][v, u])
        X = pixel_features(Dw, img, Ew, Kw, Pw, sel)
        err = np.concatenate(err)
        # group-level self-consistency: median over all sampled pixels of the per-pixel feature
        return {"scene": name, "group": str(npz_full.parent), "X": X.astype(np.float32),
                "err": err.astype(np.float32), "fg": np.concatenate(isfg),
                "group_self_inconsistency": float(np.median(np.exp(X[:, -1]))),
                "group_depth_spread": float(np.std(np.log(np.maximum(Dw, 1e-6))))}
    except Exception as e:  # noqa: BLE001 -- one bad group must not sink the rest; it is recorded
        return {"scene": name, "group": str(npz_full.parent), "error": f"{type(e).__name__}: {e}"}


def auroc(score, y):
    y = np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def fit_logistic(X, y, l2=1e-3):
    Xb = np.c_[X, np.ones(len(X))]

    def f(w):
        z = Xb @ w
        ll = np.logaddexp(0, z) - y * z
        g = Xb.T @ (1 / (1 + np.exp(-z)) - y) / len(y)
        return ll.mean() + l2 * (w[:-1] ** 2).sum(), g + np.r_[2 * l2 * w[:-1], 0]
    return optimize.minimize(f, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B").x


def binned(Xtr, Xte, nb=10):
    cols_tr, cols_te = [], []
    for j in range(Xtr.shape[1]):
        edges = np.unique(np.quantile(Xtr[:, j], np.linspace(0, 1, nb + 1)[1:-1]))
        btr, bte = np.digitize(Xtr[:, j], edges), np.digitize(Xte[:, j], edges)
        cols_tr.append(np.eye(len(edges) + 1)[btr])
        cols_te.append(np.eye(len(edges) + 1)[bte])
    return np.hstack(cols_tr), np.hstack(cols_te)


def cv_auroc(X, y, groups, n_folds=5, model="linear"):
    ug = np.array(sorted(set(groups)))
    np.random.default_rng(SEED).shuffle(ug)
    folds = np.array_split(ug, n_folds)
    scores = np.full(len(y), np.nan)
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if model == "binned":
            Xtr, Xte = binned(Xtr, Xte)
        w = fit_logistic(Xtr, y[tr].astype(np.float64))
        scores[te] = np.c_[Xte, np.ones(te.sum())] @ w
    per_fold = [auroc(scores[np.isin(groups, f)], y[np.isin(groups, f)]) for f in folds]
    return auroc(scores, y), [float(v) for v in per_fold]


def predict(workers):
    groups = sorted(glob.glob(str(PRED / "*" / "*" / "*" / "full.npz")))
    with Pool(workers) as pool:
        rows = pool.map(predict_group, groups, chunksize=4)
    bad = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    print("=" * 100)
    print(f"PREDICTABILITY of W4A4 error from W4A4's own output: {len(ok)} view-groups ({len(bad)} failed), "
          f"target = {TARGET} point error")
    X = np.concatenate([r["X"] for r in ok]).astype(np.float64)
    err = np.concatenate([r["err"] for r in ok]).astype(np.float64)
    fg = np.concatenate([r["fg"] for r in ok])
    scene = np.concatenate([[r["scene"]] * len(r["err"]) for r in ok])
    gid = np.concatenate([[i] * len(r["err"]) for i, r in enumerate(ok)])
    cat = np.array([s.split("/")[0] for s in scene])
    # label: pixel in the top decile of error WITHIN ITS OWN view-group -> "where in this image"
    thr = np.zeros(len(err))
    for i in np.unique(gid):
        m = gid == i
        thr[m] = np.quantile(err[m], 0.9)
    y = err >= thr
    out = {"n_pixels": int(len(y)), "n_groups": len(ok), "failed_groups": bad, "features": FEATS}
    for region, m in (("all content pixels", np.ones(len(y), bool)), ("foreground pixels only", fg)):
        print(f"\n  {region}: n = {int(m.sum())} pixels, positives {int(y[m].sum())}")
        res = {"single_feature_auroc": {}}
        for j, fn in enumerate(FEATS):
            a = auroc(X[m, j], y[m])
            res["single_feature_auroc"][fn] = a
        print("    single-feature AUROC (descriptive, no fitting): " +
              ", ".join(f"{k}={v:.3f}" for k, v in res["single_feature_auroc"].items()))
        for split, grp in (("scene-held-out", scene[m]), ("category-held-out", cat[m])):
            for model in ("linear", "binned"):
                a, pf = cv_auroc(X[m], y[m], grp, model=model)
                res[f"{split} {model}"] = {"auroc": a, "per_fold": pf}
                print(f"    {split:18s} {model:6s} AUROC={a:.4f}  folds {min(pf):.3f}..{max(pf):.3f}")
        out[region] = res

    # camera level: does group self-inconsistency flag groups whose cameras are off?
    cam = {r["group"]: r for r in json.load(open(RES / "cpu_camera_analysis.json"))["context_rows"]
           if "error" not in r}
    print("\n  CAMERA LEVEL (per view-group): Spearman of self-inconsistency with camera error")
    out["camera_level"] = {}
    for excl in (False, True):
        sub = [r for r in ok if r["group"] in cam and not (excl and r["scene"] == BOWL)]
        for tgt in ("w4a4_vs_full_rot", "w4a4_vs_gt_rot", "full_vs_gt_rot"):
            for feat in ("group_self_inconsistency", "group_depth_spread"):
                rho, p = stats.spearmanr([r[feat] for r in sub], [cam[r["group"]][tgt] for r in sub])
                key = f"{tgt} ~ {feat}{' (no bowl)' if excl else ''}"
                out["camera_level"][key] = {"n": len(sub), "rho": float(rho), "p": float(p)}
                print(f"    {key:62s} n={len(sub)} rho={rho:+.3f} p={p:.3g}")
    out["target"] = TARGET
    save("predict" if TARGET == "absolute" else f"predict_{TARGET}", out)


# =========================================================================== depthgt

def depth_metrics(pred, gt, valid):
    if valid.sum() < 50:
        return None
    s = np.median(gt[valid] / pred[valid])            # per-image scale: VGGT depth is up to scale
    r = np.abs(s * pred[valid] - gt[valid]) / gt[valid]
    ratio = np.maximum(s * pred[valid] / gt[valid], gt[valid] / (s * pred[valid]))
    return float(np.median(r)), float(np.mean(ratio < 1.25))


def depthgt_group(npz_full):
    import prep_depth
    npz_full = Path(npz_full)
    cat, seq = npz_full.parts[-4], npz_full.parts[-3]
    try:
        Fz, Wz = np.load(npz_full), np.load(npz_full.parent / "w4a4.npz")
        recs = dv.load_annotations(cat, seq)
        out = {"scene": f"{cat}/{seq}", "group": str(npz_full.parent)}
        acc = {"full": [], "w4a4": []}
        band_ratio = []
        for i, f in enumerate(Fz["frame_numbers"]):
            rec = recs[int(f)]
            gt, valid = prep_depth.gt_depth_518(rec)
            for arm, z in (("full", Fz), ("w4a4", Wz)):
                m = depth_metrics(z["depth"][i, ..., 0].astype(np.float64), gt, valid)
                if m:
                    acc[arm].append(m)
            # where do full and w4a4 disagree? boundary band (<= 5 px from the object edge) vs interior
            fgm = dv.foreground_mask_from_record(rec)
            df, dw = Fz["depth"][i, ..., 0].astype(np.float64), Wz["depth"][i, ..., 0].astype(np.float64)
            sw = np.median(df[fgm] / np.maximum(dw[fgm], 1e-9)) if fgm.any() else 1.0
            dis = np.abs(sw * dw - df) / np.maximum(df, 1e-9)
            edge = fgm & ~ndimage.binary_erosion(fgm, iterations=5)
            inner = fgm & ~edge
            if edge.sum() > 20 and inner.sum() > 20:
                band_ratio.append(float(np.median(dis[edge]) / max(np.median(dis[inner]), 1e-12)))
        for arm in acc:
            if acc[arm]:
                a = np.array(acc[arm])
                out[f"{arm}_absrel"], out[f"{arm}_delta125"] = float(a[:, 0].mean()), float(a[:, 1].mean())
        out["edge_over_interior_disagreement"] = float(np.median(band_ratio)) if band_ratio else float("nan")
        return out
    except Exception as e:  # noqa: BLE001
        return {"scene": f"{cat}/{seq}", "group": str(npz_full.parent), "error": f"{type(e).__name__}: {e}"}


def depthgt(workers):
    groups = sorted(glob.glob(str(PRED / "*" / "*" / "*" / "full.npz")))
    with Pool(workers) as pool:
        rows = pool.map(depthgt_group, groups, chunksize=4)
    ok = [r for r in rows if "error" not in r and "full_absrel" in r and "w4a4_absrel" in r]
    print("=" * 100)
    print(f"DEPTH vs CO3D GT (per-image median scale), {len(ok)} of {len(rows)} view-groups usable")
    out = {"per_group": rows}
    for excl in (False, True):
        sub = [r for r in ok if not (excl and r["scene"] == BOWL)]
        tag = " (no bowl)" if excl else ""
        for m in ("absrel", "delta125"):
            f_, w_ = np.array([r[f"full_{m}"] for r in sub]), np.array([r[f"w4a4_{m}"] for r in sub])
            d = w_ - f_
            p = stats.wilcoxon(d).pvalue
            out[f"{m}{tag}"] = {"n": len(sub), "full_mean": float(f_.mean()), "w4a4_mean": float(w_.mean()),
                                "w4a4_minus_full_mean": float(d.mean()), "median": float(np.median(d)),
                                "wilcoxon_p": float(p), "w4a4_larger": int((d > 0).sum())}
            print(f"  {m:9s}{tag:10s} full {f_.mean():.5f}  w4a4 {w_.mean():.5f}  diff {d.mean():+.5f} "
                  f"(median {np.median(d):+.5f})  Wilcoxon p={p:.3g}  w4a4 larger in {int((d > 0).sum())}/{len(d)}")
    br = np.array([r["edge_over_interior_disagreement"] for r in ok])
    br = br[np.isfinite(br)]
    out["edge_ratio"] = {"n": len(br), "median": float(np.median(br)), "frac_above_1": float(np.mean(br > 1)),
                         "wilcoxon_log_p": float(stats.wilcoxon(np.log(br)).pvalue)}
    print(f"\n  full-vs-w4a4 depth disagreement, object-edge band over interior: median ratio "
          f"{np.median(br):.3f}, > 1 in {np.mean(br > 1):.1%} of {len(br)} groups, "
          f"Wilcoxon(log ratio) p={out['edge_ratio']['wilcoxon_log_p']:.3g}")
    save("depthgt", out)


# =========================================================================== decompose

def unproject(depth, E, K):
    """World points from one view's depth with the given camera (VGGT's world_points_from_depth)."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    rays = np.linalg.inv(K) @ np.stack([u.ravel() + 0.0, v.ravel() + 0.0, np.ones(H * W)])
    Xc = rays * depth.ravel()
    R, t = E[:, :3], E[:, 3]
    return (R.T @ (Xc - t[:, None])).T.reshape(H, W, 3)


def decompose_group(npz_full):
    """How much of W4A4's world-point error comes from its DEPTH, and how much from its CAMERAS?

    total      W4A4's own world points vs full's, after the usual Sim(3) on camera centres
    depth-only W4A4 depth unprojected through FULL's cameras (after one global depth scale, since
               VGGT depth is up to scale), vs full's points -- camera error removed by construction
    """
    npz_full = Path(npz_full)
    try:
        F, W = np.load(npz_full), np.load(npz_full.parent / "w4a4.npz")
        Ef, Ew = F["extrinsic"].astype(np.float64), W["extrinsic"].astype(np.float64)
        Kf = F["intrinsic"].astype(np.float64)
        Df, Dw = F["depth"][..., 0].astype(np.float64), W["depth"][..., 0].astype(np.float64)
        Pf, Pw = F["world_points_from_depth"].astype(np.float64), W["world_points_from_depth"].astype(np.float64)
        s, A, b, radius = sim3(centers(Ew), centers(Ef))
        idx = (slice(None), slice(None, None, 7), slice(None, None, 7))
        total = np.linalg.norm(apply(s, A, b, Pw[idx].reshape(-1, 3)) - Pf[idx].reshape(-1, 3), axis=1)
        k = np.median(Df / np.maximum(Dw, 1e-9))
        H = np.stack([unproject(k * Dw[i], Ef[i], Kf[i]) for i in range(len(Ef))])
        depth_only = np.linalg.norm(H[idx].reshape(-1, 3) - Pf[idx].reshape(-1, 3), axis=1)
        # same, but each view rescaled on its own: separates per-view scale drift from depth SHAPE error
        kv = [np.median(Df[i] / np.maximum(Dw[i], 1e-9)) for i in range(len(Ef))]
        Hv = np.stack([unproject(kv[i] * Dw[i], Ef[i], Kf[i]) for i in range(len(Ef))])
        depth_shape = np.linalg.norm(Hv[idx].reshape(-1, 3) - Pf[idx].reshape(-1, 3), axis=1)
        scale_spread = float(np.std(np.log(np.array(kv) / np.median(kv))))
        check = float(np.max(np.abs(unproject(Df[0], Ef[0], Kf[0]) - Pf[0])))  # convention check, ~0
        cat, seq = npz_full.parts[-4], npz_full.parts[-3]
        recs = dv.load_annotations(cat, seq)
        fg = np.stack([dv.foreground_mask_from_record(recs[int(f)]) for f in F["frame_numbers"]])
        ct = np.stack([dv.content_mask_from_record(recs[int(f)]) for f in F["frame_numbers"]])
        fgs, bgs = fg[idx].reshape(-1), (ct & ~fg)[idx].reshape(-1)
        out = {"group": str(npz_full.parent), "scene": f"{cat}/{seq}",
               "total_point_err": float(np.median(total) / radius),
               "depth_only_point_err": float(np.median(depth_only) / radius), "unproject_check": check,
               "depth_shape_point_err": float(np.median(depth_shape) / radius), "per_view_scale_logstd": scale_spread}
        for reg, m in (("object", fgs), ("background", bgs)):
            if m.sum() > 20:
                out[f"{reg}_total"] = float(np.median(total[m]) / radius)
                out[f"{reg}_depth_only"] = float(np.median(depth_only[m]) / radius)
                out[f"{reg}_depth_shape"] = float(np.median(depth_shape[m]) / radius)
        return out
    except Exception as e:  # noqa: BLE001
        return {"group": str(npz_full.parent), "error": f"{type(e).__name__}: {e}"}


def decompose(workers):
    groups = sorted(glob.glob(str(PRED / "*" / "*" / "*" / "full.npz")))
    with Pool(workers) as pool:
        rows = pool.map(decompose_group, groups, chunksize=8)
    ok = [r for r in rows if "error" not in r]
    print("=" * 100)
    print(f"W4A4 WORLD-POINT ERROR, depth vs camera share ({len(ok)} of {len(rows)} view-groups)")
    print(f"  unprojection convention check, max |unproject(full depth, full cam) - full points| = "
          f"{max(r['unproject_check'] for r in ok):.3g}")
    out = {"per_group": rows}
    for excl in (False, True):
        sub = [r for r in ok if not (excl and r["scene"] == BOWL)]
        t = np.array([r["total_point_err"] for r in sub])
        d = np.array([r["depth_only_point_err"] for r in sub])
        frac = d / np.maximum(t, 1e-12)
        tag = "no bowl" if excl else "all"
        out[tag] = {"n": len(sub), "total_mean": float(t.mean()), "total_median": float(np.median(t)),
                    "depth_only_mean": float(d.mean()), "depth_only_median": float(np.median(d)),
                    "depth_share_median": float(np.median(frac)),
                    "wilcoxon_p_total_gt_depth": float(stats.wilcoxon(t - d).pvalue),
                    "total_larger": int((t > d).sum())}
        print(f"  {tag:8s} n={len(sub)}  total median {np.median(t):.5f}  depth-only median {np.median(d):.5f}"
              f"  depth share (median of per-group ratio) {np.median(frac):.3f}  "
              f"total > depth-only in {int((t > d).sum())}/{len(sub)}  Wilcoxon p={out[tag]['wilcoxon_p_total_gt_depth']:.3g}")
    print("\n  by region (medians over groups of per-group medians):")
    for reg in ("object", "background"):
        sub = [r for r in ok if f"{reg}_total" in r]
        t = np.array([r[f"{reg}_total"] for r in sub]); d = np.array([r[f"{reg}_depth_only"] for r in sub])
        v = np.array([r[f"{reg}_depth_shape"] for r in sub])
        out[reg] = {"n": len(sub), "total_median": float(np.median(t)), "depth_only_median": float(np.median(d)),
                    "depth_share_median": float(np.median(d / np.maximum(t, 1e-12))),
                    "depth_shape_median": float(np.median(v)),
                    "shape_share_median": float(np.median(v / np.maximum(t, 1e-12)))}
        print(f"    {reg:10s} n={len(sub)} total {np.median(t):.5f}  depth-only (global scale) {np.median(d):.5f} "
              f"share {out[reg]['depth_share_median']:.3f}  |  depth-shape (per-view scale) {np.median(v):.5f} "
              f"share {out[reg]['shape_share_median']:.3f}")
    sp = np.array([r["per_view_scale_logstd"] for r in ok])
    out["per_view_scale_logstd"] = {"median": float(np.median(sp)), "p90": float(np.percentile(sp, 90))}
    print(f"    per-view depth-scale drift of w4a4 relative to full, std of log scale across the 6 views: "
          f"median {np.median(sp):.5f}, p90 {np.percentile(sp, 90):.5f}")
    save("decompose", out)


# =========================================================================== exposure + power

def exposure():
    """Does exposure correction remove MORE from arms whose geometry is worse?

    correction gain = PSNR(after per-image gain+bias) - PSNR(raw), per scene and arm, read from #029's
    40-scene JSON. If the gain grows with the arm's geometry error (relative to full's own gain on the
    same scene), the correction is absorbing geometry-driven error, not only auto-exposure.
    """
    arms40 = {r["scene"]: r for r in json.load(open(sorted(RES.glob("arms_full_*_all_it7000.json"))[-1]))["per_scene"]}
    cam = json.load(open(RES / "cpu_camera_analysis.json"))["per_scene_rows"]
    rows = []
    for c in cam:
        r = arms40[c["scene"]]
        g_arm = r[f"{c['arm']}_foreground_psnr_exposure_corrected"] - r[f"{c['arm']}_foreground_psnr"]
        g_full = r["full_foreground_psnr_exposure_corrected"] - r["full_foreground_psnr"]
        rows.append({"scene": c["scene"], "arm": c["arm"], "cam_rot": c["cam_rot"], "focal": c["focal"],
                     "point_err": c["point_err"], "extra_correction_gain": g_arm - g_full,
                     "raw_psnr_loss": r["full_foreground_psnr"] - r[f"{c['arm']}_foreground_psnr"]})
    print("=" * 100)
    print("EXPOSURE CORRECTION vs GEOMETRY ERROR (extra PSNR the correction recovers for the arm, beyond full's)")
    out = {"rows": rows}
    for label, sub in (("all arms, n = 280", rows),
                       ("mild arms (no rtn, no_rot), n = 200", [r for r in rows if r["arm"] not in ("w4a4_rtn", "w4a4_no_rot")]),
                       ("all arms, no bowl", [r for r in rows if r["scene"] != BOWL])):
        res = {}
        for f in ("cam_rot", "focal", "point_err", "raw_psnr_loss"):
            rho, p = stats.spearmanr([r[f] for r in sub], [r["extra_correction_gain"] for r in sub])
            res[f] = {"rho": float(rho), "p": float(p), "n": len(sub)}
        out[label] = res
        print(f"  {label:38s} " + "  ".join(f"{k}: rho={v['rho']:+.3f} (p={v['p']:.2g})" for k, v in res.items()))
    print("\n  per-arm mean extra correction gain (dB):")
    out["per_arm"] = {}
    for arm in ARMS_ALL:
        v = [r["extra_correction_gain"] for r in rows if r["arm"] == arm]
        if v:
            w = stats.wilcoxon(v).pvalue
            out["per_arm"][arm] = {"mean": float(np.mean(v)), "wilcoxon_p": float(w), "n": len(v)}
            print(f"    {arm:16s} {np.mean(v):+.4f}  Wilcoxon p={w:.3g}")
    save("exposure", out)


ARMS_ALL = ["w4a4", "w4a4_local", "w4a4_no_lwc", "w4a4_no_smooth", "w4a4_no_lac", "w4a4_no_rot", "w4a4_rtn"]


def n_for_power(delta, sd, alpha=0.05, power=0.8):
    """Scenes needed for a two-sided paired t-test to detect `delta` with the observed per-scene sd."""
    if delta == 0 or sd == 0:
        return float("inf")
    for n in range(3, 1_000_000):
        df = n - 1
        nc = abs(delta) / sd * np.sqrt(n)
        tcrit = stats.t.ppf(1 - alpha / 2, df)
        if 1 - stats.nct.cdf(tcrit, df, nc) + stats.nct.cdf(-tcrit, df, nc) >= power:
            return n
    return float("inf")


def power():
    arms40 = json.load(open(sorted(RES.glob("arms_full_*_all_it7000.json"))[-1]))["per_scene"]
    print("=" * 100)
    print("POWER: scenes needed to detect the OBSERVED effect at 80 % power, alpha 0.05, paired t (noncentral t)")
    print(f"{'contrast':44s} {'mean':>9s} {'sd':>8s} {'d':>7s} {'n needed':>9s}  (have 40)")
    out = {}
    for arm in ARMS_ALL:
        for m, sgn in (("psnr", 1), ("ssim", 1), ("lpips", -1), ("psnr_exposure_corrected", 1)):
            d = np.array([sgn * (r[f"full_foreground_{m}"] - r[f"{arm}_foreground_{m}"]) for r in arms40])
            mu, sd = float(d.mean()), float(d.std(ddof=1))
            n = n_for_power(mu, sd)
            key = f"{m} full vs {arm}"
            out[key] = {"mean": mu, "sd": sd, "cohen_d": mu / sd if sd else float("nan"), "n_needed": n}
            print(f"{key:44s} {mu:+9.4f} {sd:8.4f} {mu / sd:+7.3f} {n if n != float('inf') else 'inf':>9}")
    save("power", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["robustness", "gaussians", "predict", "depthgt", "decompose", "exposure", "power"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"robustness": robustness, "gaussians": lambda: gaussians(a.workers),
     "predict": lambda: predict(a.workers), "depthgt": lambda: depthgt(a.workers),
     "decompose": lambda: decompose(a.workers), "exposure": exposure, "power": power}[a.probe]()

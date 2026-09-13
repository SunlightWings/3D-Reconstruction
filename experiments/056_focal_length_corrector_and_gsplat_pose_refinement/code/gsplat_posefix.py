"""3DGS with optional photometric CAMERA POSE refinement (gsplat) -- plan B for camera-borne quantization damage.

Why: #053 puts 91 % of W3A3's rendering loss on its cameras, and #052/#053 show point-level intervention recovers
nothing. The installed GraphDECO rasterizer returns no camera gradients (backward returns None for raster_settings),
so pose cannot be refined there. gsplat differentiates through the view matrix.

One scene per invocation. Reads sources/<src-arm>/{train,heldout}/sparse/0/*.txt exactly as GraphDECO does and trains
with GraphDECO's defaults as our pipeline invokes them (--iterations 7000, so the position-lr schedule still spans
30000 steps; densify from 500 every 100 until 15000; opacity reset every 3000; lambda_dssim 0.2; one SH degree per
1000 iterations up to 3; black background; opacity init 0.1; scales from mean squared 3-NN distance). Renders the
held-out views, sorted by image name like GraphDECO's render.py, into heldout/<out-arm>_it<iters>/renders/ so
run_w2a4_downstream.score_scene scores them with unchanged metrics.

Held-out cameras: built upstream by carrying CO3D GT held-out poses through the Sim(3) fitted on the six INPUT camera
centres. If pose refinement moves the training cameras, the Sim(3) from initial to refined training centres is
composed onto every held-out camera (R' = R A^T, C' = s A C + b, the dv.transform_camera_to_variant convention), which
also removes the global gauge freedom of joint camera+scene motion. Without refinement it is the identity. Held-out
IMAGES are used only for the final comparison, never for training or pose.

Baselines must be re-run through this trainer (gs_full, gs_<arm>) so pose-on vs pose-off is compared inside one
implementation; GraphDECO and gsplat numbers are not interchangeable.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from scipy.spatial import cKDTree

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

BASE = Path("/var/tmp/luli38se/quantsplat/downstream_validation_v1")
SH_C0 = 0.28209479177387814


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_split(root: Path):
    cams = {}
    for ln in open(root / "sparse/0/cameras.txt"):
        if ln.startswith("#") or not ln.strip():
            continue
        a = ln.split()
        if a[1] != "PINHOLE":
            raise SystemExit(f"unsupported camera model {a[1]}")
        fx, fy, cx, cy = map(float, a[4:8])
        cams[int(a[0])] = (int(a[2]), int(a[3]), np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64))
    views = []
    for ln in open(root / "sparse/0/images.txt"):
        if ln.startswith("#"):
            continue
        a = ln.split()
        if len(a) < 10:
            continue
        E = np.concatenate([qvec2rotmat(np.array(list(map(float, a[1:5])))),
                            np.array(list(map(float, a[5:8])))[:, None]], 1)
        W, H, K = cams[int(a[8])]
        views.append({"name": a[9], "E": E, "K": K, "W": W, "H": H, "path": root / "images" / a[9]})
    return sorted(views, key=lambda v: v["name"])


def read_points(root: Path):
    xyz, rgb = [], []
    for ln in open(root / "sparse/0/points3D.txt"):
        if ln.startswith("#") or not ln.strip():
            continue
        a = ln.split()
        xyz.append((float(a[1]), float(a[2]), float(a[3])))
        rgb.append((int(a[4]), int(a[5]), int(a[6])))
    return np.asarray(xyz, np.float32), np.asarray(rgb, np.float32) / 255.0


def centers(E):
    return np.stack([-e[:, :3].T @ e[:, 3] for e in E])


def umeyama(src, dst):
    sm, dm = src.mean(0), dst.mean(0)
    X, Y = src - sm, dst - dm
    U, S, Vt = np.linalg.svd((Y.T @ X) / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1
    A = U @ D @ Vt
    s = float(np.sum(S * np.diag(D)) / (np.sum(X * X) / len(src)))
    return s, A, dm - s * (A @ sm)


def transform_camera(E, s, A, b):
    R = E[:, :3] @ A.T
    C = s * (A @ (-E[:, :3].T @ E[:, 3])) + b
    return np.concatenate([R, (-R @ C)[:, None]], 1)


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    th = th[..., None]
    return torch.eye(3, device=w.device, dtype=w.dtype)[None] + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def ssim(x, y, win):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = Fn.conv2d(x, win, padding=5, groups=3), Fn.conv2d(y, win, padding=5, groups=3)
    s1 = Fn.conv2d(x * x, win, padding=5, groups=3) - mu1 ** 2
    s2 = Fn.conv2d(y * y, win, padding=5, groups=3) - mu2 ** 2
    s12 = Fn.conv2d(x * y, win, padding=5, groups=3) - mu1 * mu2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean()


def load_img(p, dev):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="category/sequence")
    ap.add_argument("--src-arm", required=True, help="reads sources/<src-arm>/{train,heldout}")
    ap.add_argument("--out-arm", required=True, help="writes heldout/<out-arm>_it<iters>/renders")
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--pose-opt", action="store_true")
    ap.add_argument("--pose-lr", type=float, default=1e-4)
    ap.add_argument("--pose-reg", type=float, default=1e-4)
    ap.add_argument("--pose-start", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda"
    cat, seq = a.scene.split("/", 1)
    root = BASE / cat / seq
    out_dir = root / "heldout" / f"{a.out_arm}_it{a.iterations}"
    tr_root, ho_root = root / "sources" / a.src_arm / "train", root / "sources" / a.src_arm / "heldout"
    train, held = read_split(tr_root), read_split(ho_root)
    if (out_dir / "renders").is_dir() and len(list((out_dir / "renders").glob("*.png"))) == len(held):
        print(f"resume skip: {out_dir}/renders complete")
        return
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.time()

    E0 = np.stack([v["E"] for v in train])
    C_init = centers(E0)
    scene_scale = 1.1 * float(np.max(np.linalg.norm(C_init - C_init.mean(0), axis=1)))
    W, H = train[0]["W"], train[0]["H"]
    gts = [load_img(v["path"], dev) for v in train]
    V0 = torch.zeros(len(train), 4, 4, device=dev)
    V0[:, :3, :4] = torch.from_numpy(E0).float()
    V0[:, 3, 3] = 1
    Ks = torch.from_numpy(np.stack([v["K"] for v in train])).float().to(dev)

    xyz, rgb = read_points(tr_root)
    d, _ = cKDTree(xyz).query(xyz, k=4)
    d2 = np.maximum((d[:, 1:] ** 2).mean(1), 1e-7)
    N = len(xyz)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(xyz).to(dev)),
        "scales": torch.nn.Parameter(torch.from_numpy(np.log(np.sqrt(d2))[:, None].repeat(3, 1)).float().to(dev)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(((torch.from_numpy(rgb).to(dev) - 0.5) / SH_C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device=dev)),
    })
    lr_means0, lr_means1 = 1.6e-4 * scene_scale, 1.6e-6 * scene_scale
    lrs = {"means": lr_means0, "scales": 5e-3, "quats": 1e-3, "opacities": 2.5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    strategy = DefaultStrategy(prune_opa=0.005, grow_grad2d=0.0002, grow_scale3d=0.01, prune_scale3d=0.1,
                               refine_start_iter=500, refine_stop_iter=15000, reset_every=3000, refine_every=100,
                               absgrad=False, verbose=False)
    strategy.check_sanity(params, opt)
    state = strategy.initialize_state(scene_scale=scene_scale)

    pose = torch.nn.Parameter(torch.zeros(len(train), 6, device=dev))
    pose_opt = torch.optim.Adam([pose], lr=a.pose_lr)
    g = torch.exp(-((torch.arange(11, device=dev) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)

    def viewmats(idx):
        V = V0[idx]
        if not a.pose_opt:
            return V
        D = torch.zeros(len(idx), 4, 4, device=dev)
        D[:, :3, :3] = so3_exp(pose[idx, :3])
        D[:, :3, 3] = pose[idx, 3:]
        D[:, 3, 3] = 1
        return D @ V

    rng = np.random.default_rng(a.seed)
    stack = []
    last_l1 = float("nan")
    for step in range(a.iterations):
        t = min(step / 30000.0, 1.0)
        for grp in opt["means"].param_groups:
            grp["lr"] = math.exp(math.log(lr_means0) * (1 - t) + math.log(lr_means1) * t)
        if not stack:
            stack = list(rng.permutation(len(train)))
        i = int(stack.pop())
        idx = torch.tensor([i], device=dev)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        img, _alpha, info = rasterization(params["means"], params["quats"], torch.exp(params["scales"]),
                                          torch.sigmoid(params["opacities"]), colors, viewmats(idx), Ks[idx],
                                          W, H, sh_degree=min(step // 1000, 3), packed=False)
        strategy.step_pre_backward(params, opt, state, step, info)
        pred = img.permute(0, 3, 1, 2)
        l1 = (pred - gts[i]).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim(pred, gts[i], win))
        if a.pose_opt:
            loss = loss + a.pose_reg * (pose ** 2).sum()
        loss.backward()
        for o in opt.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if a.pose_opt and step >= a.pose_start:
            pose_opt.step()
        pose_opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, opt, state, step, info, packed=False)
        last_l1 = float(l1.detach())
        if step % 1000 == 0 or step == a.iterations - 1:
            msg = f"  step {step:5d}  L1 {last_l1:.4f}  N {params['means'].shape[0]}"
            if a.pose_opt:
                msg += f"  pose |w| max {float(pose[:, :3].norm(dim=1).max()) * 180 / math.pi:.3f} deg"
            print(msg, flush=True)

    with torch.no_grad():
        E_ref = viewmats(torch.arange(len(train), device=dev))[:, :3, :4].double().cpu().numpy()
        C_ref = centers(E_ref)
        s, A, b = umeyama(C_init, C_ref) if a.pose_opt else (1.0, np.eye(3), np.zeros(3))
        rdir = out_dir / "renders"
        rdir.mkdir(parents=True, exist_ok=True)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        for j, v in enumerate(held):
            Eh = transform_camera(v["E"], s, A, b)
            Vh = torch.eye(4, device=dev)[None]
            Vh[0, :3, :4] = torch.from_numpy(Eh).float()
            Kh = torch.from_numpy(v["K"]).float().to(dev)[None]
            im, _, _ = rasterization(params["means"], params["quats"], torch.exp(params["scales"]),
                                     torch.sigmoid(params["opacities"]), colors, Vh, Kh, v["W"], v["H"],
                                     sh_degree=3, packed=False)
            arr = (im[0].clamp(0, 1) * 255 + 0.5).clamp(0, 255).byte().cpu().numpy()
            Image.fromarray(arr).save(rdir / f"{j:05d}.png")
    pose_deg = [float(x) for x in (pose[:, :3].norm(dim=1) * 180 / math.pi).detach().cpu()] if a.pose_opt else None
    meta = {"scene": a.scene, "src_arm": a.src_arm, "out_arm": a.out_arm, "iterations": a.iterations,
            "pose_opt": a.pose_opt, "pose_lr": a.pose_lr, "pose_reg": a.pose_reg, "pose_start": a.pose_start,
            "seed": a.seed, "scene_scale": scene_scale, "n_init_points": int(N),
            "n_final_gaussians": int(params["means"].shape[0]), "final_train_l1": last_l1,
            "pose_delta_deg_per_camera": pose_deg,
            "pose_delta_translation_over_scene_scale": (
                [float(x) for x in (pose[:, 3:].norm(dim=1) / scene_scale).detach().cpu()] if a.pose_opt else None),
            "heldout_sim3": {"s": s, "rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(A) - 1) / 2, -1, 1)))),
                             "b_norm": float(np.linalg.norm(b))},
            "heldout_order": [v["name"] for v in held],
            "refined_train_extrinsics": E_ref.tolist(), "seconds": round(time.time() - t0, 1)}
    (out_dir / "gsplat_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done {a.scene} {a.out_arm}: {meta['n_final_gaussians']} gaussians, {meta['seconds']} s", flush=True)


if __name__ == "__main__":
    main()

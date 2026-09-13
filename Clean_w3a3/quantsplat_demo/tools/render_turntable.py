import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# -----------------------------------------------------------------------------
# Make gaussian-splatting importable
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
GS_ROOT = REPO_ROOT / "gaussian-splatting"
sys.path.insert(0, str(GS_ROOT))

from scene import GaussianModel  # noqa: E402
from scene.cameras import MiniCam  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from utils.graphics_utils import getProjectionMatrix  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Render turntable/orbit views from trained 3DGS models."
    )
    ap.add_argument(
        "--downstream",
        type=Path,
        required=True,
        help="Root like /var/tmp/.../quantsplat/downstream_validation_v1",
    )
    ap.add_argument(
        "--scenes",
        nargs="+",
        required=True,
        help="Scene IDs like apple/110_13051_23361",
    )
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["full", "w4a4", "w3a3"],
        help="Model arms to render",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=7000,
        help="3DGS iteration to load",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=36,
        help="Number of orbit frames",
    )
    ap.add_argument(
        "--size",
        type=int,
        default=518,
        help="Render width/height",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "quantsplat_demo" / "data" / "turntable",
        help="Output directory for rendered PNGs",
    )
    return ap.parse_args()


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_first_camera_intrinsics(cameras_txt: Path):
    lines = cameras_txt.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        # CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
        _, model, width, height, fx, fy, cx, cy = toks[:8]
        return {
            "model": model,
            "width": int(width),
            "height": int(height),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }
    raise RuntimeError(f"No camera entry found in {cameras_txt}")


def read_camera_centers(images_txt: Path) -> np.ndarray:
    centers = []
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        # image line has at least 10 tokens:
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if len(toks) < 10:
            continue
        try:
            int(toks[0])
        except ValueError:
            continue

        q = np.array(list(map(float, toks[1:5])), dtype=np.float64)
        t = np.array(list(map(float, toks[5:8])), dtype=np.float64)
        R = qvec2rotmat(q)
        C = -R.T @ t
        centers.append(C)

    if not centers:
        raise RuntimeError(f"No cameras parsed from {images_txt}")
    return np.stack(centers, axis=0)


def read_points3d_xyz(points_txt: Path) -> np.ndarray:
    pts = []
    lines = points_txt.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            xyz = np.array(list(map(float, toks[1:4])), dtype=np.float64)
        except ValueError:
            continue
        pts.append(xyz)

    if not pts:
        raise RuntimeError(f"No points parsed from {points_txt}")
    return np.stack(pts, axis=0)


def robust_scene_center(points_xyz: np.ndarray) -> np.ndarray:
    return np.median(points_xyz, axis=0)


def compute_orbit_basis(train_centers: np.ndarray, center: np.ndarray):
    X = train_centers - train_centers.mean(axis=0, keepdims=True)
    cov = X.T @ X / max(len(X), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Smallest eigenvector = approximate plane normal
    normal = eigvecs[:, 0]
    # Remaining two span the camera plane
    v = eigvecs[:, 1]
    u = eigvecs[:, 2]

    # Make a right-handed basis
    if np.linalg.det(np.stack([u, v, normal], axis=1)) < 0:
        v = -v

    rel = train_centers - center[None, :]
    height = np.median(rel @ normal)

    planar = rel - np.outer(rel @ normal, normal)
    planar_r = np.linalg.norm(planar, axis=1)
    radius = float(np.median(planar_r))
    radius = max(radius, 1e-3)

    return u, v, normal, radius, float(height)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def look_at_world_to_camera(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """
    Returns 4x4 world->camera matrix in OpenCV-style coordinates.
    """
    z_cam = normalize(target - eye)  # forward
    up_hint = normalize(up_hint)

    # avoid degeneracy
    if abs(np.dot(z_cam, up_hint)) > 0.98:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    x_cam = normalize(np.cross(z_cam, up_hint))   # right
    y_cam = normalize(np.cross(z_cam, x_cam))     # down-ish / orthonormal

    R = np.stack([x_cam, y_cam, z_cam], axis=0)
    t = -R @ eye

    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R.astype(np.float32)
    W2C[:3, 3] = t.astype(np.float32)
    return W2C


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(pixels / (2.0 * focal))


def parse_white_background_from_cfg(cfg_args_path: Path) -> bool:
    if not cfg_args_path.is_file():
        return False
    txt = cfg_args_path.read_text(encoding="utf-8")
    return "white_background=True" in txt


def make_camera(W2C: np.ndarray, width: int, height: int, fx: float, fy: float):
    fovx = focal_to_fov(fx, width)
    fovy = focal_to_fov(fy, height)

    world_view = torch.tensor(W2C, dtype=torch.float32, device="cuda").transpose(0, 1)
    proj = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=fovx,
        fovY=fovy,
    ).transpose(0, 1).to(device="cuda", dtype=torch.float32)

    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

    cam = MiniCam(
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=0.01,
        zfar=100.0,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
    )
    return cam


class Pipe:
    compute_cov3D_python = False
    convert_SHs_python = False
    debug = False
    antialiasing = False


def save_tensor_image(t: torch.Tensor, out_path: Path):
    t = t.detach().clamp(0.0, 1.0).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def render_one_arm(
    model_ply: Path,
    cfg_args_path: Path,
    out_dir: Path,
    width: int,
    height: int,
    fx: float,
    fy: float,
    eyes: list[np.ndarray],
    target: np.ndarray,
    up_hint: np.ndarray,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    gaussians = GaussianModel(3)
    gaussians.load_ply(str(model_ply))

    white_bg = parse_white_background_from_cfg(cfg_args_path)
    bg = torch.tensor(
        [0.85, 0.85, 0.85],
        dtype=torch.float32,
        device="cuda",
    )
    pipe = Pipe()

    with torch.no_grad():
        for idx, eye in enumerate(eyes):
            W2C = look_at_world_to_camera(eye, target, up_hint)
            cam = make_camera(W2C, width, height, fx, fy)
            result = render(cam, gaussians, pipe, bg)
            img = result["render"]
            save_tensor_image(img, out_dir / f"{idx:03d}.png")


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rendering turntables.")

    manifest = {
        "frames": args.frames,
        "size": args.size,
        "scenes": {},
    }

    for scene_id in args.scenes:
        category, sequence = scene_id.split("/")
        print("=" * 80)
        print(f"SCENE: {scene_id}")
        print("=" * 80)

        scene_root = args.downstream / category / sequence
        ref_train = scene_root / "sources" / "w3a3" / "train" / "sparse" / "0"
        if not ref_train.is_dir():
            raise RuntimeError(f"Reference train source missing: {ref_train}")

        cameras_txt = ref_train / "cameras.txt"
        images_txt = ref_train / "images.txt"
        points_txt = ref_train / "points3D.txt"

        intr = read_first_camera_intrinsics(cameras_txt)
        train_centers = read_camera_centers(images_txt)
        pts = read_points3d_xyz(points_txt)

        center = robust_scene_center(pts)
        u, v, normal, radius, height = compute_orbit_basis(train_centers, center)

        print(f"center  = {center}")
        print(f"radius  = {radius:.4f}")
        print(f"height  = {height:.4f}")
        print(f"normal  = {normal}")

        eyes = []
        for i in range(args.frames):
            theta = 2.0 * math.pi * (i / args.frames)
            eye = center + height * normal + radius * (math.cos(theta) * u + math.sin(theta) * v)
            eyes.append(eye.astype(np.float64))

        scene_manifest = {
            "center": center.tolist(),
            "radius": radius,
            "height": height,
            "normal": normal.tolist(),
            "arms": {},
        }

        for arm in args.arms:
            model_dir = scene_root / "models" / arm
            model_ply = model_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
            cfg_args = model_dir / "cfg_args"

            if not model_ply.is_file():
                print(f"[WARN] missing model ply for {arm}: {model_ply}")
                continue

            out_dir = args.out / category / sequence / arm
            print(f"Rendering {arm} -> {out_dir}")

            render_one_arm(
                model_ply=model_ply,
                cfg_args_path=cfg_args,
                out_dir=out_dir,
                width=args.size,
                height=args.size,
                fx=intr["fx"],
                fy=intr["fy"],
                eyes=eyes,
                target=center,
                up_hint=-normal,
            )

            scene_manifest["arms"][arm] = {
                "dir": str(out_dir),
                "frames": args.frames,
            }

        manifest["scenes"][scene_id] = scene_manifest

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"Saved turntable renders to: {args.out}")
    print(f"Saved manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
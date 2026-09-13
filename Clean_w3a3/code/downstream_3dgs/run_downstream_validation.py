#!/usr/bin/env python3
"""Controlled Full-VGGT vs W4A4-QuantVGGT downstream 3DGS validation.

This runner reuses the frozen six-view prediction NPZ files. It does not rerun VGGT.
For each frozen scene it:
  1. chooses one six-primary-view prediction group,
  2. reconstructs the exact 518x518 pad-mode input images with QuantVGGT's preprocessor,
  3. writes matched COLMAP/GraphDECO inputs for Full and W4A4,
  4. chooses nine held-out CO3D known-camera frames and maps their GT cameras into
     each variant's predicted world frame using one global proper Sim(3),
  5. waits for the shared GPU to have no compute process,
  6. trains official GraphDECO 3DGS for the same number of iterations,
  7. renders the same held-out views,
  8. computes content-region PSNR/SSIM/LPIPS and aggregate Full-minus-Quant deltas.

The design intentionally does not filter points by confidence, visibility, reprojection,
or GT. Point sampling is deterministic stride sampling, matching the earlier friend518
control (stride=4 -> 101400 points for six 518x518 views).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

TARGET = 518
PATCH = 14
EXPECTED_MANIFEST_SHA256 = "1ce2f3f8d43cc17f61d84545b82f132a3b64d49d5acf70ff0fe0ac6aa9968e06"
CAM_FLIP = np.diag([-1.0, -1.0, 1.0])

DEFAULT_SCENES = [
    "remote/350_36761_68623",      # strong disagreement-risk scene
    "broccoli/412_56288_108844",  # mid-range scene
    "toaster/372_41229_82130",    # weak-rho scene
]

SCRIPT_PATH = Path(__file__).resolve()
REPO = SCRIPT_PATH.parents[2]

QUANTVGGT = REPO / "QuantVGGT"
GS_ROOT = REPO / "gaussian-splatting"

# Use the Python environment that launched this script.
CV_PY = Path(sys.executable)

GS_PY = Path("/home/utn/poli22wo/miniconda3/envs/gs/bin/python")
GS_SITE = Path("/var/tmp/poli22wo/3dgs-site")

PRED_ROOT = Path("/var/tmp/poli22wo/quantsplat/disagreement_dataset_v1/run/predictions")

CO3D_ROOT = Path("/var/tmp/poli22wo/co3d_single_all")

OUT_ROOT = Path("/var/tmp/poli22wo/quantsplat/downstream_validation_v1")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    raise RuntimeError(msg)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: Sequence[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    log("+ " + " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def require_prediction_contract(arrays: Dict[str, np.ndarray], label: str) -> None:
    expected = {
        "frame_numbers": (6,),
        "intrinsic": (6, 3, 3),
        "extrinsic": (6, 3, 4),
        "world_points_from_depth": (6, TARGET, TARGET, 3),
    }
    for key, shape in expected.items():
        if key not in arrays:
            die(f"{label}: missing NPZ key {key}")
        if tuple(arrays[key].shape) != shape:
            die(f"{label}: {key} shape {arrays[key].shape} != {shape}")
        if key != "frame_numbers" and not np.isfinite(arrays[key]).all():
            die(f"{label}: non-finite values in {key}")
    det = np.linalg.det(arrays["extrinsic"][:, :, :3].astype(np.float64))
    if not np.allclose(det, 1.0, atol=2e-3):
        die(f"{label}: camera rotation determinant is not approximately +1: {det}")


def find_group(category: str, sequence: str) -> Tuple[Path, dict, dict]:
    """Pick the first group that has six primary frames and no context-only fillers."""
    metas = sorted(PRED_ROOT.rglob("full_meta.json"))
    candidates: List[Tuple[str, Path, dict]] = []
    for p in metas:
        try:
            m = read_json(p)
        except Exception:
            continue
        if m.get("category") != category or m.get("sequence") != sequence:
            continue
        frames = list(map(int, m.get("frame_numbers", [])))
        primary = list(map(int, m.get("primary_frame_numbers", frames)))
        context = list(map(int, m.get("context_only_frame_numbers", [])))
        if len(frames) == 6 and len(primary) == 6 and not context:
            candidates.append((str(m.get("group_id", p.parent.name)), p.parent, m))
    if not candidates:
        die(f"No six-primary-view group found for {category}/{sequence}")
    candidates.sort(key=lambda x: x[0])
    _, group_dir, full_meta = candidates[0]
    wmeta_path = group_dir / "w4a4_meta.json"
    if not wmeta_path.is_file():
        die(f"Missing {wmeta_path}")
    wmeta = read_json(wmeta_path)
    if full_meta.get("input_semantic_sha256") != wmeta.get("input_semantic_sha256"):
        die(f"Input semantic SHA mismatch in {group_dir}")
    if list(map(int, full_meta["frame_numbers"])) != list(map(int, wmeta["frame_numbers"])):
        die(f"Full/W4A4 frame order mismatch in {group_dir}")
    return group_dir, full_meta, wmeta


def load_annotations(category: str, sequence: str) -> Dict[int, dict]:
    ann = CO3D_ROOT / category / "frame_annotations.jgz"
    if not ann.is_file():
        die(f"Missing CO3D annotations: {ann}")
    with gzip.open(ann, "rt", encoding="utf-8") as f:
        data = json.load(f)
    records = {
        int(r["frame_number"]): r
        for r in data
        if r.get("sequence_name") == sequence
    }
    if not records:
        die(f"No annotations for {category}/{sequence} in {ann}")
    return records


def resolve_image_path(record: dict) -> Path:
    rel = record.get("image", {}).get("path")
    if not rel:
        die("CO3D record has no image.path")
    path = CO3D_ROOT / rel
    if not path.is_file():
        die(f"Missing RGB image: {path}")
    return path


def resolve_mask_path(record: dict) -> Path | None:
    rel = record.get("mask", {}).get("path")
    if not rel:
        return None
    path = CO3D_ROOT / rel
    return path if path.is_file() else None


def import_vggt_preprocessor():
    sys.path.insert(0, str(QUANTVGGT))
    try:
        from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore
    except Exception as e:
        die(f"Cannot import QuantVGGT pad preprocessor: {e}")
    return load_and_preprocess_images


def preprocess_images(paths: Sequence[Path]) -> np.ndarray:
    load_and_preprocess_images = import_vggt_preprocessor()
    tensor = load_and_preprocess_images([str(p) for p in paths], mode="pad")
    # Torch tensor expected, but keep this dependency local to the CV environment.
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    arr = arr.astype(np.float32, copy=False)
    if arr.shape != (len(paths), 3, TARGET, TARGET):
        die(f"Unexpected preprocessor output shape: {arr.shape}")
    if not np.isfinite(arr).all() or arr.min() < -1e-6 or arr.max() > 1.000001:
        die("Preprocessed image tensor is non-finite or outside [0,1]")
    return arr


def save_tensor_pngs(tensor: np.ndarray, image_dir: Path, names: Sequence[str]) -> List[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for i, name in enumerate(names):
        rgb = np.transpose(tensor[i], (1, 2, 0))
        u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        path = image_dir / name
        Image.fromarray(u8).save(path)
        out.append(path)
    return out


def preprocess_geometry(width: int, height: int) -> Tuple[int, int, int, int, np.ndarray]:
    """Recover the exact pad-mode content rectangle by preprocessing a synthetic black image.

    This intentionally delegates resize rounding/padding to the same QuantVGGT preprocessor
    instead of reimplementing its patch-grid rounding rules.
    """
    cache_dir = OUT_ROOT / ".geometry_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"geometry_{width}x{height}.json"
    if meta_path.is_file():
        m = read_json(meta_path)
        new_w, new_h, left, top = [int(m[k]) for k in ("new_width", "new_height", "left", "top")]
    else:
        black_path = cache_dir / f"black_{width}x{height}.png"
        if not black_path.is_file():
            Image.new("RGB", (width, height), (0, 0, 0)).save(black_path)
        arr = preprocess_images([black_path])[0]
        rgb = np.transpose(arr, (1, 2, 0))
        content = np.mean(rgb, axis=2) < 0.5
        ys, xs = np.nonzero(content)
        if len(xs) == 0:
            die(f"Could not recover pad geometry for {width}x{height}")
        left, right = int(xs.min()), int(xs.max())
        top, bottom = int(ys.min()), int(ys.max())
        new_w, new_h = right - left + 1, bottom - top + 1
        # The synthetic image is uniformly black, so the recovered support must be a rectangle.
        if int(content.sum()) != new_w * new_h:
            die(f"Non-rectangular synthetic pad support for {width}x{height}")
        write_json(meta_path, {
            "original_width": width, "original_height": height,
            "new_width": new_w, "new_height": new_h, "left": left, "top": top,
            "method": "QuantVGGT pad-mode synthetic-black support",
        })
    mask = np.zeros((TARGET, TARGET), dtype=bool)
    mask[top : top + new_h, left : left + new_w] = True
    return new_w, new_h, left, top, mask


def content_mask_from_record(record: dict) -> np.ndarray:
    h, w = [int(v) for v in record["image"]["size"]]
    return preprocess_geometry(w, h)[4]


_DISAGREEMENT_MODULE = None


def import_disagreement_module():
    """Load analyze_full_dataset_disagreement.py, which owns resize_depth_and_masks.

    Imported lazily and cached so that the plain metric path stays importable in
    environments where that module's own dependencies are unavailable.
    """
    global _DISAGREEMENT_MODULE
    if _DISAGREEMENT_MODULE is None:
        path = REPO / "code" / "analyze_full_dataset_disagreement.py"
        spec = importlib.util.spec_from_file_location("dv_disagreement", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["dv_disagreement"] = module
        spec.loader.exec_module(module)
        _DISAGREEMENT_MODULE = module
    return _DISAGREEMENT_MODULE


def foreground_mask_from_record(record: dict, threshold: float = 0.5) -> np.ndarray:
    """The CO3D object mask, warped onto the 518x518 grid and ANDed with the content mask.

    The content mask is only the pad rectangle, so it admits the whole background. On this
    dataset the object occupies a mean 15.3% of that rectangle and CO3D supplies GT depth for
    ~0.1% of background pixels, which makes background-inclusive PSNR mostly a measurement of
    a region no arm can reconstruct (see ongoing_logs.md entries #002 and #011). Metrics should
    therefore be reported under BOTH masks, with this one primary.

    Falls back to the content mask when a record carries no usable mask, so callers always get
    a valid mask; `foreground_mask_available()` distinguishes the two cases.
    """
    content = content_mask_from_record(record)
    mask_path = resolve_mask_path(record)
    if mask_path is None:
        return content
    h, w = [int(v) for v in record["image"]["size"]]
    new_w, new_h, left, top, _ = preprocess_geometry(w, h)
    transform = {"resized_width": new_w, "resized_height": new_h, "padding": {"left": left, "top": top}}
    with Image.open(mask_path) as img:
        raw = np.asarray(img.convert("L"))
    ones = np.ones(raw.shape, dtype=np.float32)
    _depth, _valid, fg = import_disagreement_module().resize_depth_and_masks(ones, ones > 0, raw, transform)
    return (fg >= threshold) & content


def foreground_mask_available(record: dict) -> bool:
    return resolve_mask_path(record) is not None


def original_to_518_affine(record: dict) -> np.ndarray:
    h, w = [int(v) for v in record["image"]["size"]]
    new_w, new_h, left, top, _ = preprocess_geometry(w, h)
    sx = new_w / float(w)
    sy = new_h / float(h)
    return np.array([[sx, 0.0, left], [0.0, sy, top], [0.0, 0.0, 1.0]], dtype=np.float64)

def camera_center(E: np.ndarray) -> np.ndarray:
    R = E[:3, :3]
    t = E[:3, 3]
    return -R.T @ t


def camera_centers(E: np.ndarray) -> np.ndarray:
    return np.stack([camera_center(x) for x in E], axis=0)


def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Fit dst ~= s * A @ src + b with a proper rotation A (no reflection)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    X = src - src_mean
    Y = dst - dst_mean
    cov = (Y.T @ X) / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    A = U @ D @ Vt
    var_src = np.sum(X * X) / len(src)
    if var_src <= 1e-12:
        die("Degenerate camera-center geometry for Sim(3)")
    s = float(np.sum(S * np.diag(D)) / var_src)
    if not np.isfinite(s) or s <= 0:
        die(f"Invalid Sim(3) scale {s}")
    b = dst_mean - s * (A @ src_mean)
    return s, A, b


def co3d_to_opencv_camera(record: dict) -> Tuple[np.ndarray, np.ndarray]:
    vp = record.get("viewpoint")
    if not isinstance(vp, dict):
        die("CO3D record has no viewpoint")
    R_p3d = np.asarray(vp["R"], dtype=np.float64)
    T_p3d = np.asarray(vp["T"], dtype=np.float64)
    R_cv = CAM_FLIP @ R_p3d.T
    t_cv = CAM_FLIP @ T_p3d
    if abs(np.linalg.det(R_cv) - 1.0) > 1e-4:
        die("Invalid CO3D->OpenCV rotation determinant")
    if np.linalg.norm(R_cv @ R_cv.T - np.eye(3)) > 1e-4:
        die("Non-orthogonal CO3D->OpenCV rotation")
    h, w = [int(v) for v in record["image"]["size"]]
    focal = np.asarray(vp["focal_length"], dtype=np.float64)
    principal = np.asarray(vp["principal_point"], dtype=np.float64)
    fmt = vp.get("intrinsics_format", "ndc_norm_image_bounds")
    if fmt == "ndc_norm_image_bounds":
        sx, sy = w / 2.0, h / 2.0
    elif fmt == "ndc_isotropic":
        common = min(h, w) / 2.0
        sx = sy = common
    else:
        die(f"Unsupported CO3D intrinsics format {fmt}")
    fx, fy = focal[0] * sx, focal[1] * sy
    cx = w / 2.0 - principal[0] * sx
    cy = h / 2.0 - principal[1] * sy
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    E = np.concatenate([R_cv, t_cv[:, None]], axis=1)
    return E, K


def transform_camera_to_variant(E_gt: np.ndarray, s: float, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    R_gt = E_gt[:, :3]
    C_gt = camera_center(E_gt)
    C_variant = s * (A @ C_gt) + b
    R_variant = R_gt @ A.T
    t_variant = -R_variant @ C_variant
    return np.concatenate([R_variant, t_variant[:, None]], axis=1)


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """COLMAP quaternion order: qw qx qy qz."""
    R = np.asarray(R, dtype=np.float64)
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
            [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if q[0] < 0:
        q = -q
    return q


def write_colmap_text_model(
    root: Path,
    E: np.ndarray,
    K: np.ndarray,
    image_names: Sequence[str],
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    with (sparse / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, k in enumerate(K, start=1):
            fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
            f.write(f"{i} PINHOLE {TARGET} {TARGET} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n")
    with (sparse / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i, (e, name) in enumerate(zip(E, image_names), start=1):
            q = rotmat2qvec(e[:, :3])
            t = e[:, 3]
            f.write(
                f"{i} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
                f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} {i} {name}\n\n"
            )
    with (sparse / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, (p, c) in enumerate(zip(points, colors), start=1):
            f.write(
                f"{i} {p[0]:.9g} {p[1]:.9g} {p[2]:.9g} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} 0\n"
            )


def sampled_points_and_colors(world_points: np.ndarray, tensor: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = world_points[:, ::stride, ::stride, :].reshape(-1, 3).astype(np.float32)
    rgb = np.transpose(tensor, (0, 2, 3, 1))[:, ::stride, ::stride, :].reshape(-1, 3)
    colors = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    if not np.isfinite(pts).all():
        die("Sampled world points contain non-finite values; refusing silent filtering")
    expected = 6 * len(range(0, TARGET, stride)) ** 2
    if len(pts) != expected:
        die(f"Unexpected point count {len(pts)} != {expected}")
    return pts, colors


def choose_heldout_frames(records: Dict[int, dict], input_frames: Sequence[int], n: int, fixed: Sequence[int] | None = None) -> List[int]:
    if fixed is not None:
        out = list(map(int, fixed))
        for frame in out:
            if frame in input_frames or frame not in records:
                die(f"Invalid frozen heldout frame {frame}")
        return out
    candidates = []
    input_set = set(map(int, input_frames))
    for frame, record in sorted(records.items()):
        if frame in input_set:
            continue
        if not isinstance(record.get("viewpoint"), dict):
            continue
        if not record.get("image", {}).get("path"):
            continue
        ft = str(record.get("meta", {}).get("frame_type", ""))
        if ft and "known" not in ft:
            continue
        candidates.append(frame)
    if len(candidates) < n:
        die(f"Only {len(candidates)} valid heldout candidates; need {n}")
    idx = np.linspace(0, len(candidates) - 1, n)
    picked = sorted({candidates[int(round(x))] for x in idx})
    # Very unlikely with long CO3D sequences; deterministic fallback fills any duplicate gaps.
    if len(picked) < n:
        for f in candidates:
            if f not in picked:
                picked.append(f)
                if len(picked) == n:
                    break
        picked.sort()
    return picked[:n]


def build_train_source(
    scene_root: Path,
    variant: str,
    arrays: Dict[str, np.ndarray],
    tensor: np.ndarray,
    image_names: Sequence[str],
    common_images: Path,
    stride: int,
) -> Path:
    root = scene_root / "sources" / variant / "train"
    ok = root / "PREPARED.ok"
    if ok.is_file():
        return root
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    os.symlink(common_images, root / "images", target_is_directory=True)
    pts, colors = sampled_points_and_colors(arrays["world_points_from_depth"], tensor, stride)
    E = arrays["extrinsic"].astype(np.float64)
    K = arrays["intrinsic"].astype(np.float64)
    write_colmap_text_model(root, E, K, image_names, pts, colors)
    write_json(
        root / "adapter_meta.json",
        {
            "variant": variant,
            "camera_model": "PINHOLE",
            "shared_camera": False,
            "sampling_stride": stride,
            "point_count": int(len(pts)),
            "source_point_array": "world_points_from_depth",
            "confidence_threshold": None,
            "visibility_threshold": None,
            "reprojection_filtering": False,
            "tracking": False,
            "bundle_adjustment": False,
            "random_sampling": False,
        },
    )
    ok.write_text("PASS\n", encoding="utf-8")
    return root


def build_heldout_source(
    scene_root: Path,
    variant: str,
    train_source: Path,
    E_heldout: np.ndarray,
    K_heldout: np.ndarray,
    heldout_names: Sequence[str],
    heldout_image_dir: Path,
) -> Path:
    root = scene_root / "sources" / variant / "heldout"
    ok = root / "PREPARED.ok"
    if ok.is_file():
        return root
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    os.symlink(heldout_image_dir, root / "images", target_is_directory=True)
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    # Cameras/images are heldout; point source is copied from matched training initialization
    # only because GraphDECO's COLMAP loader expects a point cloud source.
    with (sparse / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, k in enumerate(K_heldout, start=1):
            f.write(
                f"{i} PINHOLE {TARGET} {TARGET} {float(k[0,0]):.12g} {float(k[1,1]):.12g} "
                f"{float(k[0,2]):.12g} {float(k[1,2]):.12g}\n"
            )
    with (sparse / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i, (e, name) in enumerate(zip(E_heldout, heldout_names), start=1):
            q = rotmat2qvec(e[:, :3])
            t = e[:, 3]
            f.write(
                f"{i} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
                f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} {i} {name}\n\n"
            )
    shutil.copy2(train_source / "sparse" / "0" / "points3D.txt", sparse / "points3D.txt")
    ok.write_text("PASS\n", encoding="utf-8")
    return root


def gpu_compute_processes() -> List[Tuple[int, int]]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out: List[Tuple[int, int]] = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            out.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return out


def wait_for_gpu(poll_seconds: int, stable_checks: int) -> None:
    stable = 0
    while True:
        procs = gpu_compute_processes()
        if not procs:
            stable += 1
            if stable >= stable_checks:
                log(f"GPU gate: free for {stable_checks} consecutive checks -> proceed")
                return
            log(f"GPU gate: no compute process ({stable}/{stable_checks}); rechecking...")
        else:
            stable = 0
            desc = ", ".join(f"pid={pid} mem={mem}MiB" for pid, mem in procs)
            log(f"GPU gate: busy ({desc}); waiting {poll_seconds}s. No process will be killed.")
        time.sleep(poll_seconds)


def gs_env() -> dict:
    env = os.environ.copy()
    extra = f"{GS_SITE}:{GS_ROOT}"
    env["PYTHONPATH"] = extra + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def train_3dgs(source: Path, model: Path, iterations: int, poll: int, stable: int) -> None:
    final_ply = model / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if final_ply.is_file():
        log(f"3DGS train: resume skip, found {final_ply}")
        return
    wait_for_gpu(poll, stable)
    model.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(GS_PY),
            "train.py",
            "-s", str(source),
            "-m", str(model),
            "--iterations", str(iterations),
            "--data_device", "cpu",
            "--resolution", "1",
        ],
        cwd=GS_ROOT,
        env=gs_env(),
    )
    if not final_ply.is_file():
        die(f"3DGS training completed but final PLY missing: {final_ply}")


def render_heldout(source: Path, model: Path, out_dir: Path, iterations: int, n_views: int, poll: int, stable: int) -> Path:
    final_dir = out_dir / "renders"
    existing = sorted(final_dir.glob("*.png")) if final_dir.is_dir() else []
    if len(existing) == n_views:
        log(f"Render: resume skip, found {n_views} heldout renders in {final_dir}")
        return final_dir
    wait_for_gpu(poll, stable)
    # GraphDECO writes train/ours_<iter>/renders under model_path. Remove only that render cache,
    # never the trained point cloud.
    render_cache = model / "train" / f"ours_{iterations}"
    if render_cache.exists():
        shutil.rmtree(render_cache)
    run(
        [
            str(GS_PY),
            "render.py",
            "-m", str(model),
            "-s", str(source),
            "--iteration", str(iterations),
            "--skip_test",
        ],
        cwd=GS_ROOT,
        env=gs_env(),
    )
    generated = render_cache / "renders"
    imgs = sorted(generated.glob("*.png"))
    if len(imgs) != n_views:
        die(f"Expected {n_views} heldout renders, found {len(imgs)} in {generated}")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(generated, final_dir)
    return final_dir


def bbox_from_mask(mask: np.ndarray) -> Tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        die("Empty content mask")
    return slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    diff = (a - b) ** 2
    mse = float(diff[mask].mean())
    if mse <= 1e-15:
        return float("inf")
    return -10.0 * math.log10(mse)


def ssim_value(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception as e:
        die(f"scikit-image required for SSIM: {e}")
    ys, xs = bbox_from_mask(mask)
    aa = a[ys, xs]
    bb = b[ys, xs]
    return float(structural_similarity(aa, bb, channel_axis=2, data_range=1.0))


def evaluate_scene(
    scene_root: Path,
    heldout_frames: Sequence[int],
    content_masks: Sequence[np.ndarray],
    full_renders: Path,
    quant_renders: Path,
    heldout_image_dir: Path,
    foreground_masks: Sequence[np.ndarray] | None = None,
) -> dict:
    """Score held-out renders under BOTH masks.

    `foreground` is the primary region and `content` the secondary one. The content mask is
    the whole non-letterbox rectangle, of which the object is a mean 15.3% on this dataset,
    and CO3D supplies GT depth for ~0.1% of background pixels -- so a content-mask number is
    mostly a measurement of a region no arm can reconstruct. See ongoing_logs.md entries
    #002, #004 and #014; re-scoring C1 on the foreground flipped that gate's verdict.

    `foreground_masks` is optional only so old callers keep working; when omitted it is
    derived per frame from the CO3D object mask.
    """
    out_json = scene_root / "metrics.json"
    if out_json.is_file():
        log(f"Metrics: resume skip, found {out_json}")
        return read_json(out_json)

    # Load LPIPS lazily and keep it on CPU so evaluation does not contend for the shared GPU.
    try:
        import torch
        import lpips
    except Exception as e:
        die(f"LPIPS evaluation dependencies unavailable in CV environment: {e}")
    lpips_model = lpips.LPIPS(net="alex").cpu().eval()

    gt_paths = [heldout_image_dir / f"frame{f:06d}.png" for f in heldout_frames]
    full_paths = sorted(full_renders.glob("*.png"))
    quant_paths = sorted(quant_renders.glob("*.png"))
    n = len(heldout_frames)
    if len(full_paths) != n or len(quant_paths) != n:
        die("Heldout render count mismatch during evaluation")

    def read_rgb(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    def lpips_crop(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        ys, xs = bbox_from_mask(mask)
        aa = a[ys, xs]
        bb = b[ys, xs]
        ta = torch.from_numpy(aa).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
        tb = torch.from_numpy(bb).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
        with torch.no_grad():
            return float(lpips_model(ta, tb).item())

    if foreground_masks is not None:
        fg_masks = list(foreground_masks)
    else:
        records_for_masks = load_annotations(*scene_root.parts[-2:])
        fg_masks = [foreground_mask_from_record(records_for_masks[f]) for f in heldout_frames]

    rows = []
    for i, frame in enumerate(heldout_frames):
        gt = read_rgb(gt_paths[i])
        fr = read_rgb(full_paths[i])
        qr = read_rgb(quant_paths[i])
        row = {"frame": int(frame)}
        for region, mask in (("content", content_masks[i]), ("foreground", fg_masks[i])):
            if not mask.any():
                continue
            row[f"full_psnr_{region}"] = psnr(fr, gt, mask)
            row[f"quant_psnr_{region}"] = psnr(qr, gt, mask)
            row[f"full_ssim_{region}"] = ssim_value(fr, gt, mask)
            row[f"quant_ssim_{region}"] = ssim_value(qr, gt, mask)
            row[f"full_lpips_{region}"] = lpips_crop(fr, gt, mask)
            row[f"quant_lpips_{region}"] = lpips_crop(qr, gt, mask)
            row[f"full_minus_quant_psnr_{region}"] = row[f"full_psnr_{region}"] - row[f"quant_psnr_{region}"]
            row[f"full_minus_quant_ssim_{region}"] = row[f"full_ssim_{region}"] - row[f"quant_ssim_{region}"]
            row[f"quant_minus_full_lpips_{region}"] = row[f"quant_lpips_{region}"] - row[f"full_lpips_{region}"]
        # Back-compat aliases: unsuffixed deltas kept pointing at the content mask so existing
        # readers (aggregate_metrics.json, gate D) are not silently re-based onto a new region.
        row["full_minus_quant_psnr"] = row.get("full_minus_quant_psnr_content", float("nan"))
        row["full_minus_quant_ssim"] = row.get("full_minus_quant_ssim_content", float("nan"))
        row["quant_minus_full_lpips"] = row.get("quant_minus_full_lpips_content", float("nan"))
        rows.append(row)

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n_views": n,
        "primary_region": "CO3D object foreground (content region reported as secondary)",
        "full_mean_psnr_content": mean("full_psnr_content"),
        "quant_mean_psnr_content": mean("quant_psnr_content"),
        "full_minus_quant_psnr_content": mean("full_psnr_content") - mean("quant_psnr_content"),
        "full_mean_ssim_content": mean("full_ssim_content"),
        "quant_mean_ssim_content": mean("quant_ssim_content"),
        "full_minus_quant_ssim_content": mean("full_ssim_content") - mean("quant_ssim_content"),
        "full_mean_lpips_content": mean("full_lpips_content"),
        "quant_mean_lpips_content": mean("quant_lpips_content"),
        "quant_minus_full_lpips_content": mean("quant_lpips_content") - mean("full_lpips_content"),
        "full_mean_psnr_foreground": mean("full_psnr_foreground"),
        "quant_mean_psnr_foreground": mean("quant_psnr_foreground"),
        "full_minus_quant_psnr_foreground": mean("full_psnr_foreground") - mean("quant_psnr_foreground"),
        "full_mean_ssim_foreground": mean("full_ssim_foreground"),
        "quant_mean_ssim_foreground": mean("quant_ssim_foreground"),
        "full_minus_quant_ssim_foreground": mean("full_ssim_foreground") - mean("quant_ssim_foreground"),
        "full_mean_lpips_foreground": mean("full_lpips_foreground"),
        "quant_mean_lpips_foreground": mean("quant_lpips_foreground"),
        "quant_minus_full_lpips_foreground": mean("quant_lpips_foreground") - mean("full_lpips_foreground"),
        "gt_images_identical": True,
        "camera_alignment": "separate one-global-proper-Sim(3) GT->Full and GT->W4A4 using the same six input camera correspondences",
        "no_per_camera_correction": True,
    }
    result = {"summary": summary, "per_frame": rows}
    write_json(out_json, result)
    with (scene_root / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return result


def prepare_scene(category: str, sequence: str, stride: int, heldout_n: int) -> dict:
    scene_root = OUT_ROOT / category / sequence
    scene_root.mkdir(parents=True, exist_ok=True)
    manifest_path = scene_root / "scene_manifest.json"

    group_dir, full_meta, wmeta = find_group(category, sequence)
    full_npz = group_dir / "full.npz"
    quant_npz = group_dir / "w4a4.npz"
    if not full_npz.is_file() or not quant_npz.is_file():
        die(f"Missing Full/W4A4 NPZ pair in {group_dir}")
    full_sha = sha256_file(full_npz)
    quant_sha = sha256_file(quant_npz)
    if full_meta.get("prediction_sha256") and full_sha != full_meta["prediction_sha256"]:
        die(f"Full prediction SHA mismatch: {full_npz}")
    if wmeta.get("prediction_sha256") and quant_sha != wmeta["prediction_sha256"]:
        die(f"W4A4 prediction SHA mismatch: {quant_npz}")
    for label, meta in (("Full", full_meta), ("W4A4", wmeta)):
        msh = meta.get("frozen_manifest_sha256")
        if msh and msh != EXPECTED_MANIFEST_SHA256:
            die(f"{label} frozen manifest SHA mismatch: {msh}")
    full = load_npz(full_npz)
    quant = load_npz(quant_npz)
    require_prediction_contract(full, "Full")
    require_prediction_contract(quant, "W4A4")
    frames = list(map(int, full["frame_numbers"].tolist()))
    if frames != list(map(int, quant["frame_numbers"].tolist())):
        die("Full/W4A4 NPZ frame order mismatch")

    records = load_annotations(category, sequence)
    for f in frames:
        if f not in records:
            die(f"Training frame {f} missing from CO3D annotations")

    frozen = read_json(manifest_path) if manifest_path.is_file() else None
    if frozen is not None:
        if frozen["input_frames"] != frames:
            die(f"Frozen scene manifest input frames changed for {category}/{sequence}")
        heldout_frames = choose_heldout_frames(records, frames, heldout_n, frozen["heldout_frames"])
    else:
        heldout_frames = choose_heldout_frames(records, frames, heldout_n)

    train_paths = [resolve_image_path(records[f]) for f in frames]
    train_tensor = preprocess_images(train_paths)
    common_train = scene_root / "common" / "train_images"
    train_names = [f"image_{i}.png" for i in range(1, 7)]
    train_images_ok = scene_root / "common" / "train_images.ok"
    if not train_images_ok.is_file():
        if common_train.exists():
            shutil.rmtree(common_train)
        save_tensor_pngs(train_tensor, common_train, train_names)
        train_images_ok.write_text("PASS\n", encoding="utf-8")

    heldout_paths = [resolve_image_path(records[f]) for f in heldout_frames]
    heldout_tensor = preprocess_images(heldout_paths)
    common_heldout = scene_root / "common" / "heldout_images"
    heldout_names = [f"frame{f:06d}.png" for f in heldout_frames]
    heldout_images_ok = scene_root / "common" / "heldout_images.ok"
    if not heldout_images_ok.is_file():
        if common_heldout.exists():
            shutil.rmtree(common_heldout)
        save_tensor_pngs(heldout_tensor, common_heldout, heldout_names)
        heldout_images_ok.write_text("PASS\n", encoding="utf-8")

    full_source = build_train_source(scene_root, "full", full, train_tensor, train_names, common_train, stride)
    quant_source = build_train_source(scene_root, "w4a4", quant, train_tensor, train_names, common_train, stride)

    # Matched control: exact same images and deterministic same point count.
    full_meta_adapter = read_json(full_source / "adapter_meta.json")
    quant_meta_adapter = read_json(quant_source / "adapter_meta.json")
    if full_meta_adapter["point_count"] != quant_meta_adapter["point_count"]:
        die("Full/W4A4 initial point count mismatch")
    train_image_shas = [sha256_file(common_train / n) for n in train_names]

    E_gt_input = np.stack([co3d_to_opencv_camera(records[f])[0] for f in frames])
    C_gt = camera_centers(E_gt_input)

    alignments = {}
    heldout_sources = {}
    K_heldout_list = []
    E_gt_heldout_list = []
    for f in heldout_frames:
        Egt, Korig = co3d_to_opencv_camera(records[f])
        E_gt_heldout_list.append(Egt)
        K_heldout_list.append(original_to_518_affine(records[f]) @ Korig)
    E_gt_heldout = np.stack(E_gt_heldout_list)
    K_heldout = np.stack(K_heldout_list)

    for variant, arrays, train_source in [
        ("full", full, full_source),
        ("w4a4", quant, quant_source),
    ]:
        E_pred = arrays["extrinsic"].astype(np.float64)
        C_pred = camera_centers(E_pred)
        s, A, b = umeyama(C_gt, C_pred)
        C_fit = (s * (A @ C_gt.T)).T + b
        residual = np.linalg.norm(C_fit - C_pred, axis=1)
        radius = float(np.median(np.linalg.norm(C_pred - np.median(C_pred, axis=0), axis=1)))
        radius = max(radius, 1e-12)
        E_held = np.stack([transform_camera_to_variant(e, s, A, b) for e in E_gt_heldout])
        heldout_sources[variant] = build_heldout_source(
            scene_root,
            variant,
            train_source,
            E_held,
            K_heldout,
            heldout_names,
            common_heldout,
        )
        alignments[variant] = {
            "scale": s,
            "rotation": A.tolist(),
            "translation": b.tolist(),
            "camera_center_residual": residual.tolist(),
            "camera_center_residual_median": float(np.median(residual)),
            "camera_center_residual_median_scene_radius": float(np.median(residual) / radius),
        }

    manifest = {
        "category": category,
        "sequence": sequence,
        "group_dir": str(group_dir),
        "group_id": full_meta.get("group_id"),
        "input_frames": frames,
        "heldout_frames": heldout_frames,
        "full_input_semantic_sha256": full_meta.get("input_semantic_sha256"),
        "w4a4_input_semantic_sha256": wmeta.get("input_semantic_sha256"),
        "full_prediction_sha256_meta": full_meta.get("prediction_sha256"),
        "w4a4_prediction_sha256_meta": wmeta.get("prediction_sha256"),
        "full_prediction_sha256_actual": full_sha,
        "w4a4_prediction_sha256_actual": quant_sha,
        "training_image_sha256": train_image_shas,
        "sampling_stride": stride,
        "initial_point_count_each": int(full_meta_adapter["point_count"]),
        "camera_alignment": alignments,
        "full_train_source": str(full_source),
        "w4a4_train_source": str(quant_source),
        "full_heldout_source": str(heldout_sources["full"]),
        "w4a4_heldout_source": str(heldout_sources["w4a4"]),
        "preprocessing": {
            "implementation": "QuantVGGT vggt.utils.load_fn.load_and_preprocess_images",
            "mode": "pad",
            "target": TARGET,
        },
    }
    write_json(manifest_path, manifest)

    content_masks = [content_mask_from_record(records[f]) for f in heldout_frames]
    foreground_masks = [foreground_mask_from_record(records[f]) for f in heldout_frames]
    return {
        "scene_root": scene_root,
        "manifest": manifest,
        "full_source": full_source,
        "quant_source": quant_source,
        "full_heldout_source": heldout_sources["full"],
        "quant_heldout_source": heldout_sources["w4a4"],
        "heldout_image_dir": common_heldout,
        "heldout_frames": heldout_frames,
        "content_masks": content_masks,
        "foreground_masks": foreground_masks,
    }


def aggregate(scene_results: List[Tuple[str, dict]]) -> dict:
    rows = []
    for name, result in scene_results:
        s = result["summary"]
        category, sequence = name.split("/", 1)
        rows.append(
            {
                "category": category,
                "sequence": sequence,
                "n_views": int(s["n_views"]),
                "full_psnr": s["full_mean_psnr_content"],
                "quant_psnr": s["quant_mean_psnr_content"],
                "full_minus_quant_psnr": s["full_minus_quant_psnr_content"],
                "full_ssim": s["full_mean_ssim_content"],
                "quant_ssim": s["quant_mean_ssim_content"],
                "full_minus_quant_ssim": s["full_minus_quant_ssim_content"],
                "full_lpips": s["full_mean_lpips_content"],
                "quant_lpips": s["quant_mean_lpips_content"],
                "quant_minus_full_lpips": s["quant_minus_full_lpips_content"],
            }
        )
    agg = {
        "scene_count": len(rows),
        "scenes": rows,
        "mean_full_psnr": float(np.mean([r["full_psnr"] for r in rows])),
        "mean_quant_psnr": float(np.mean([r["quant_psnr"] for r in rows])),
        "mean_full_minus_quant_psnr": float(np.mean([r["full_minus_quant_psnr"] for r in rows])),
        "mean_full_ssim": float(np.mean([r["full_ssim"] for r in rows])),
        "mean_quant_ssim": float(np.mean([r["quant_ssim"] for r in rows])),
        "mean_full_minus_quant_ssim": float(np.mean([r["full_minus_quant_ssim"] for r in rows])),
        "mean_full_lpips": float(np.mean([r["full_lpips"] for r in rows])),
        "mean_quant_lpips": float(np.mean([r["quant_lpips"] for r in rows])),
        "mean_quant_minus_full_lpips": float(np.mean([r["quant_minus_full_lpips"] for r in rows])),
        "scenes_full_better_psnr": int(sum(r["full_minus_quant_psnr"] > 0 for r in rows)),
        "scenes_full_better_ssim": int(sum(r["full_minus_quant_ssim"] > 0 for r in rows)),
        "scenes_full_better_lpips": int(sum(r["quant_minus_full_lpips"] > 0 for r in rows)),
    }
    write_json(OUT_ROOT / "aggregate_metrics.json", agg)
    with (OUT_ROOT / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES, help="category/sequence entries")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--heldout-views", type=int, default=9)
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--gpu-poll-seconds", type=int, default=60)
    p.add_argument("--gpu-stable-checks", type=int, default=2)
    p.add_argument("--prepare-only", action="store_true", help="CPU preparation/validation only; no GPU work")
    return p.parse_args()


def preflight() -> None:
    required = [REPO, QUANTVGGT, GS_ROOT, CV_PY, GS_PY, GS_SITE, PRED_ROOT, CO3D_ROOT]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        die("Missing required paths:\n  " + "\n  ".join(missing))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Check the runtime that will actually run GraphDECO before consuming GPU time.
    run(
        [
            str(GS_PY),
            "-c",
            "import torch, diff_gaussian_rasterization; print('GraphDECO runtime PASS', torch.__version__)",
        ],
        cwd=GS_ROOT,
        env=gs_env(),
    )


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        die("--stride must be positive")
    if args.heldout_views <= 0:
        die("--heldout-views must be positive")
    preflight()

    log("=" * 72)
    log("DOWNSTREAM VALIDATION: Full VGGT vs W4A4 QuantVGGT -> official 3DGS")
    log("VGGT inference will NOT be rerun; frozen NPZ predictions are reused.")
    log("=" * 72)

    prepared = []
    for item in args.scenes:
        if item.count("/") != 1:
            die(f"Bad scene '{item}', expected category/sequence")
        category, sequence = item.split("/", 1)
        log(f"\n--- CPU PREP {category}/{sequence} ---")
        prepared.append((item, prepare_scene(category, sequence, args.stride, args.heldout_views)))

    log("\nCPU preparation and matched-control validation: PASS")
    for name, p in prepared:
        m = p["manifest"]
        log(
            f"  {name}: group={m['group_id']} input={m['input_frames']} heldout={m['heldout_frames']} "
            f"points={m['initial_point_count_each']}"
        )
        for variant in ("full", "w4a4"):
            a = m["camera_alignment"][variant]
            log(
                f"    {variant}: median camera-center residual = "
                f"{a['camera_center_residual_median_scene_radius']:.4f} scene radii"
            )

    if args.prepare_only:
        log("\n--prepare-only requested: stopping before all GPU work.")
        return

    results: List[Tuple[str, dict]] = []
    for name, p in prepared:
        scene_root: Path = p["scene_root"]
        log(f"\n{'=' * 72}\nGPU/3DGS {name}\n{'=' * 72}")
        full_model = scene_root / "models" / "full"
        quant_model = scene_root / "models" / "w4a4"

        # Full and Quant are trained independently but with identical 3DGS settings.
        train_3dgs(p["full_source"], full_model, args.iterations, args.gpu_poll_seconds, args.gpu_stable_checks)
        train_3dgs(p["quant_source"], quant_model, args.iterations, args.gpu_poll_seconds, args.gpu_stable_checks)

        full_renders = render_heldout(
            p["full_heldout_source"], full_model, scene_root / "heldout" / "full",
            args.iterations, args.heldout_views, args.gpu_poll_seconds, args.gpu_stable_checks,
        )
        quant_renders = render_heldout(
            p["quant_heldout_source"], quant_model, scene_root / "heldout" / "w4a4",
            args.iterations, args.heldout_views, args.gpu_poll_seconds, args.gpu_stable_checks,
        )

        result = evaluate_scene(
            scene_root,
            p["heldout_frames"],
            p["content_masks"],
            full_renders,
            quant_renders,
            p["heldout_image_dir"],
            foreground_masks=p["foreground_masks"],
        )
        s = result["summary"]
        log(
            f"RESULT {name}: Full-Quant PSNR={s['full_minus_quant_psnr_content']:+.3f} dB, "
            f"SSIM={s['full_minus_quant_ssim_content']:+.5f}, "
            f"Quant-Full LPIPS={s['quant_minus_full_lpips_content']:+.5f}"
        )
        results.append((name, result))

    agg = aggregate(results)
    log("\n" + "=" * 72)
    log("DOWNSTREAM VALIDATION COMPLETE")
    log("=" * 72)
    log(f"Scenes: {agg['scene_count']}")
    log(f"Mean Full-Quant PSNR: {agg['mean_full_minus_quant_psnr']:+.3f} dB")
    log(f"Mean Full-Quant SSIM: {agg['mean_full_minus_quant_ssim']:+.5f}")
    log(f"Mean Quant-Full LPIPS: {agg['mean_quant_minus_full_lpips']:+.5f}")
    log(f"Aggregate JSON: {OUT_ROOT / 'aggregate_metrics.json'}")


if __name__ == "__main__":
    main()
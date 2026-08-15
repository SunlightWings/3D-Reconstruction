#!/usr/bin/env python3
"""CPU-only dataset-wide Full-vs-W4A4 disagreement analysis.

This script consumes the already-completed 1330 Full + 1330 W4A4 predictions
from the frozen CO3D manifest. It NEVER initializes VGGT or CUDA models.

It validates the frozen manifest and every prediction pair, reconstructs CO3D GT
geometry with the validated float16-bit-pattern depth decoder, applies one proper
Sim(3) per six-view group and model variant, scores only primary frames, computes
scene-level disagreement/excess-error statistics, a shuffled negative control,
scene-bootstrap CIs, spatial localization statistics, compact dataset figures,
and a result-first Markdown report.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_MANIFEST_SHA = "1ce2f3f8d43cc17f61d84545b82f132a3b64d49d5acf70ff0fe0ac6aa9968e06"
EXPECTED_SCENES = 40
EXPECTED_GROUPS = 1330
EXPECTED_PRIMARY_FRAMES = 7889
EXPECTED_CONTEXT_OCCURRENCES = 91
TARGET = 518
SAMPLE_PER_FRAME = 5000
RNG_SEED = 20260815
NEGATIVE_CONTROL_PERMUTATIONS = 20
BOOTSTRAP_RESAMPLES = 10000
CAM_FLIP = np.diag([-1.0, -1.0, 1.0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/var/tmp/luli38se/quantsplat/disagreement_dataset_v1"),
    )
    p.add_argument(
        "--co3d-root",
        type=Path,
        default=Path("/var/tmp/luli38se/co3d_single_all"),
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/utn/luli38se/cv/3D-Reconstruction"),
    )
    p.add_argument(
        "--skip-file-sha",
        action="store_true",
        help="Skip expensive NPZ file SHA verification. Metadata/input contract checks still run.",
    )
    return p.parse_args()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path, block: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(text: str, extra: int = 0) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(h[:8], "little") + RNG_SEED + extra) % (2**63 - 1)


def load_jgz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        raise RuntimeError(f"Expected list in {path}")
    return value


def average_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and xs[j] == xs[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64).copy()
    y = np.asarray(y, np.float64).copy()
    if len(x) < 2:
        return float("nan")
    x -= x.mean()
    y -= y.mean()
    den = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    if den <= 0:
        return float("nan")
    return float(np.sum(x * y) / den)


def spearman_from_ranks(rx: np.ndarray, y: np.ndarray) -> float:
    return pearson(rx, average_ranks(y))


def exact_top_mask(x: np.ndarray, frac: float = 0.10) -> np.ndarray:
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(math.ceil(frac * n)))
    idx = np.argpartition(x, n - k)[n - k :]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def roc_auc_from_ranks(score_ranks_zero_based: np.ndarray, positive: np.ndarray) -> float:
    positive = np.asarray(positive, bool)
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks_one_based = np.asarray(score_ranks_zero_based, np.float64) + 1.0
    return float(
        (
            ranks_one_based[positive].sum()
            - n_pos * (n_pos + 1) / 2.0
        )
        / (n_pos * n_neg)
    )


def top_precision_exact(score: np.ndarray, target: np.ndarray, frac: float = 0.10) -> float:
    hs = exact_top_mask(score, frac)
    ht = exact_top_mask(target, frac)
    if hs.sum() == 0:
        return float("nan")
    return float(np.mean(ht[hs]))


def camera_center(E: np.ndarray) -> np.ndarray:
    R = E[:, :3]
    t = E[:, 3]
    return -R.T @ t


def centers(E: np.ndarray) -> np.ndarray:
    return np.stack([camera_center(e) for e in E])


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    ms = src.mean(axis=0)
    md = dst.mean(axis=0)
    X = src - ms
    Y = dst - md
    cov = Y.T @ X / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    A = U @ D @ Vt
    var = np.sum(X * X) / len(src)
    if var <= 0:
        raise RuntimeError("Degenerate camera trajectory")
    scale = float(np.sum(S * np.diag(D)) / var)
    b = md - scale * (A @ ms)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid Sim(3) scale")
    return scale, A, b


def pred_to_gt(P: np.ndarray, s: float, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (P - b) @ A / s


def angle_deg(R: np.ndarray) -> float:
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def co3d_to_opencv(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    vp = record["viewpoint"]
    R_p3d = np.asarray(vp["R"], dtype=np.float64)
    T_p3d = np.asarray(vp["T"], dtype=np.float64)
    if R_p3d.shape != (3, 3) or T_p3d.shape != (3,):
        raise RuntimeError("Bad CO3D camera")
    R_cv = CAM_FLIP @ R_p3d.T
    t_cv = CAM_FLIP @ T_p3d
    if abs(np.linalg.det(R_cv) - 1.0) > 1e-5:
        raise RuntimeError("Camera determinant failure")
    H, W = [int(x) for x in record["image"]["size"]]
    focal = np.asarray(vp["focal_length"], dtype=np.float64)
    principal = np.asarray(vp["principal_point"], dtype=np.float64)
    fmt = vp.get("intrinsics_format", "ndc_norm_image_bounds")
    if fmt == "ndc_norm_image_bounds":
        sx, sy = W / 2.0, H / 2.0
    elif fmt == "ndc_isotropic":
        sx = sy = min(H, W) / 2.0
    else:
        raise RuntimeError(f"Unsupported intrinsics format: {fmt}")
    fx, fy = focal[0] * sx, focal[1] * sy
    cx = W / 2.0 - principal[0] * sx
    cy = H / 2.0 - principal[1] * sy
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    E = np.concatenate([R_cv, t_cv[:, None]], axis=1)
    return E, K


def import_pad_transform(repo_root: Path):
    script_dir = repo_root / "scripts/model_tests/quant_3dgs"
    sys.path.insert(0, str(script_dir))
    import prepare_co3d_scene  # type: ignore
    return prepare_co3d_scene.pad_transform


def K_to_518(K: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    A = np.asarray(transform["original_to_vggt_affine"], dtype=np.float64)
    return A @ K


def resize_depth_and_masks(
    depth: np.ndarray,
    depth_mask: np.ndarray,
    fg_mask: np.ndarray,
    transform: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rw = int(transform["resized_width"])
    rh = int(transform["resized_height"])
    left = int(transform["padding"]["left"])
    top = int(transform["padding"]["top"])
    depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
    dm_img = Image.fromarray((depth_mask.astype(np.uint8) * 255), mode="L")
    fg_img = Image.fromarray(fg_mask.astype(np.uint8), mode="L")
    depth_r = np.asarray(depth_img.resize((rw, rh), resample=Image.Resampling.NEAREST), dtype=np.float32)
    dm_r = np.asarray(dm_img.resize((rw, rh), resample=Image.Resampling.NEAREST)) > 0
    fg_r = np.asarray(fg_img.resize((rw, rh), resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    out_depth = np.zeros((TARGET, TARGET), dtype=np.float32)
    out_dm = np.zeros((TARGET, TARGET), dtype=bool)
    out_fg = np.zeros((TARGET, TARGET), dtype=np.float32)
    out_depth[top : top + rh, left : left + rw] = depth_r
    out_dm[top : top + rh, left : left + rw] = dm_r
    out_fg[top : top + rh, left : left + rw] = fg_r
    valid = out_dm & np.isfinite(out_depth) & (out_depth > 0)
    return out_depth, valid, out_fg


def unproject_to_world(depth: np.ndarray, valid: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    z = depth.astype(np.float64)
    x = (xx - K[0, 2]) / K[0, 0] * z
    y = (yy - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], axis=-1)
    R = E[:, :3]
    t = E[:, 3]
    world = (cam - t) @ R
    world[~valid] = np.nan
    return world.astype(np.float32)


def decode_gt_frame(record: dict[str, Any], co3d_root: Path, pad_transform) -> dict[str, Any]:
    image_path = co3d_root / record["image"]["path"]
    with Image.open(image_path) as img:
        W, H = img.size
    ann_H, ann_W = [int(x) for x in record["image"]["size"]]
    if (W, H) != (ann_W, ann_H):
        raise RuntimeError(f"Image/annotation size mismatch: {image_path}")
    transform = pad_transform(W, H, TARGET)
    E, K0 = co3d_to_opencv(record)
    K = K_to_518(K0, transform)
    depth_meta = record["depth"]
    depth_path = co3d_root / depth_meta["path"]
    dm_path = co3d_root / depth_meta["mask_path"]
    fg_path = co3d_root / record["mask"]["path"]
    with Image.open(depth_path) as img:
        bits = np.array(img, dtype=np.uint16)
        depth_raw = np.frombuffer(bits.tobytes(order="C"), dtype=np.float16).astype(np.float32).reshape(bits.shape)
    with Image.open(dm_path) as img:
        dm_raw = np.asarray(img)
    with Image.open(fg_path) as img:
        fg_raw = np.asarray(img.convert("L"))
    scale_adjustment = float(depth_meta.get("scale_adjustment", 1.0))
    depth_actual = depth_raw * scale_adjustment
    depth_518, valid_518, fg_518 = resize_depth_and_masks(
        depth_actual, dm_raw != 0, fg_raw, transform
    )
    gt_valid = valid_518 & (fg_518 >= 0.5)
    world = unproject_to_world(depth_518, gt_valid, K, E)
    return {
        "image_path": image_path,
        "transform": transform,
        "E": E,
        "K": K,
        "depth": depth_518,
        "valid": gt_valid,
        "fg": fg_518,
        "world": world,
    }


def read_point_cloud(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    for key in ("x", "y", "z"):
        if key not in names:
            raise RuntimeError(f"Point cloud missing {key}: {path}")
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ],
        axis=1,
    )
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 1000:
        raise RuntimeError(f"Point cloud too small: {path}")
    return xyz


def reference_radius(points: np.ndarray) -> float:
    center = np.median(points, axis=0)
    radius = float(np.quantile(np.linalg.norm(points - center, axis=1), 0.90))
    if not np.isfinite(radius) or radius <= 0:
        raise RuntimeError("Invalid point-cloud radius")
    return radius


def load_prediction_pair(
    pred_root: Path,
    scene: dict[str, Any],
    group: dict[str, Any],
    skip_file_sha: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    category = str(scene["category"])
    sequence = str(scene.get("sequence") or scene.get("sequence_name"))
    gid = str(group["group_id"])
    root = pred_root / category / sequence / gid
    paths = {
        "full_npz": root / "full.npz",
        "w4a4_npz": root / "w4a4.npz",
        "full_meta": root / "full_meta.json",
        "w4a4_meta": root / "w4a4_meta.json",
    }
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    fm = json.loads(paths["full_meta"].read_text())
    qm = json.loads(paths["w4a4_meta"].read_text())
    frames = [int(x) for x in group["frame_numbers"]]
    primary = [int(x) for x in group["primary_frame_numbers"]]
    context = [int(x) for x in group["context_only_frame_numbers"]]
    for meta, variant in ((fm, "full"), (qm, "w4a4")):
        if meta.get("category") != category or meta.get("sequence") != sequence or meta.get("group_id") != gid:
            raise RuntimeError(f"Metadata identity mismatch: {category}/{sequence}/{gid}/{variant}")
        if [int(x) for x in meta.get("frame_numbers", [])] != frames:
            raise RuntimeError(f"Frame order mismatch: {category}/{sequence}/{gid}/{variant}")
        if [int(x) for x in meta.get("primary_frame_numbers", [])] != primary:
            raise RuntimeError(f"Primary membership mismatch: {category}/{sequence}/{gid}/{variant}")
        if [int(x) for x in meta.get("context_only_frame_numbers", [])] != context:
            raise RuntimeError(f"Context membership mismatch: {category}/{sequence}/{gid}/{variant}")
        if meta.get("frozen_manifest_sha256") != EXPECTED_MANIFEST_SHA:
            raise RuntimeError(f"Manifest provenance mismatch: {category}/{sequence}/{gid}/{variant}")
        if meta.get("variant") != variant:
            raise RuntimeError(f"Variant mismatch: {category}/{sequence}/{gid}/{variant}")
    if fm.get("input_semantic_sha256") != qm.get("input_semantic_sha256"):
        raise RuntimeError(f"Full/W4A4 input hash mismatch: {category}/{sequence}/{gid}")
    if not skip_file_sha:
        if sha256_file(paths["full_npz"]) != fm.get("prediction_sha256"):
            raise RuntimeError(f"Full prediction SHA mismatch: {category}/{sequence}/{gid}")
        if sha256_file(paths["w4a4_npz"]) != qm.get("prediction_sha256"):
            raise RuntimeError(f"W4A4 prediction SHA mismatch: {category}/{sequence}/{gid}")
    wanted = ("frame_numbers", "depth", "intrinsic", "extrinsic", "world_points_from_depth")
    out = []
    for key in ("full_npz", "w4a4_npz"):
        with np.load(paths[key], allow_pickle=False) as z:
            if set(wanted) - set(z.files):
                raise RuntimeError(f"Missing arrays in {paths[key]}")
            a = {k: z[k] for k in wanted}
        if a["frame_numbers"].tolist() != frames:
            raise RuntimeError(f"NPZ frame order mismatch: {paths[key]}")
        if a["depth"].shape != (6, TARGET, TARGET, 1):
            raise RuntimeError(f"Bad depth shape: {paths[key]} {a['depth'].shape}")
        if a["intrinsic"].shape != (6, 3, 3) or a["extrinsic"].shape != (6, 3, 4):
            raise RuntimeError(f"Bad camera shape: {paths[key]}")
        if a["world_points_from_depth"].shape != (6, TARGET, TARGET, 3):
            raise RuntimeError(f"Bad pointmap shape: {paths[key]}")
        for arrk in ("depth", "intrinsic", "extrinsic", "world_points_from_depth"):
            if not np.isfinite(a[arrk]).all():
                raise RuntimeError(f"Nonfinite {arrk}: {paths[key]}")
        out.append(a)
    return out[0], out[1], fm, qm


def camera_alignment_report(Egt: np.ndarray, Epred: np.ndarray, s: float, A: np.ndarray, b: np.ndarray, radius: float) -> dict[str, float]:
    ce = []
    re = []
    for i in range(6):
        C_expected = s * (A @ camera_center(Egt[i])) + b
        R_expected = Egt[i, :, :3] @ A.T
        ce.append(np.linalg.norm(camera_center(Epred[i]) - C_expected) / (s * radius))
        re.append(angle_deg(Epred[i, :, :3] @ R_expected.T))
    return {
        "center_median": float(np.median(ce)),
        "center_max": float(np.max(ce)),
        "rotation_median_deg": float(np.median(re)),
        "rotation_max_deg": float(np.max(re)),
    }


def frame_metrics(d: np.ndarray, qerr: np.ndarray, ferr: np.ndarray) -> dict[str, float]:
    delta = qerr - ferr
    excess = np.maximum(delta, 0.0)
    rd = average_ranks(d)
    high_d = exact_top_mask(d)
    high_ex = exact_top_mask(excess)
    high_q = exact_top_mask(qerr)
    return {
        "n": int(len(d)),
        "full_error_median": float(np.median(ferr)),
        "quant_error_median": float(np.median(qerr)),
        "disagreement_median": float(np.median(d)),
        "delta_error_median": float(np.median(delta)),
        "excess_error_median": float(np.median(excess)),
        "fraction_quant_worse": float(np.mean(delta > 0)),
        "rho_d_eQ": spearman_from_ranks(rd, qerr),
        "rho_d_delta": spearman_from_ranks(rd, delta),
        "rho_d_excess": spearman_from_ranks(rd, excess),
        "auc_top10_eQ": roc_auc_from_ranks(rd, high_q),
        "auc_top10_excess": roc_auc_from_ranks(rd, high_ex),
        "top10_precision_eQ": float(np.mean(high_q[high_d])),
        "top10_precision_excess": float(np.mean(high_ex[high_d])),
        "top10_quant_worse_fraction": float(np.mean(delta[high_d] > 0)),
    }


def scene_metrics(d: np.ndarray, qerr: np.ndarray, ferr: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    delta = qerr - ferr
    excess = np.maximum(delta, 0.0)
    rd = average_ranks(d)
    rex = average_ranks(excess)
    high_d = exact_top_mask(d)
    high_ex = exact_top_mask(excess)
    high_q = exact_top_mask(qerr)
    m = {
        "sampled_pixels": int(len(d)),
        "full_error_median": float(np.median(ferr)),
        "quant_error_median": float(np.median(qerr)),
        "disagreement_median": float(np.median(d)),
        "delta_error_median": float(np.median(delta)),
        "excess_error_median": float(np.median(excess)),
        "fraction_quant_worse": float(np.mean(delta > 0)),
        "rho_d_eQ": spearman_from_ranks(rd, qerr),
        "rho_d_delta": spearman_from_ranks(rd, delta),
        "rho_d_excess": pearson(rd, rex),
        "auc_top10_eQ": roc_auc_from_ranks(rd, high_q),
        "auc_top10_excess": roc_auc_from_ranks(rd, high_ex),
        "top10_precision_eQ": float(np.mean(high_q[high_d])),
        "top10_precision_excess": float(np.mean(high_ex[high_d])),
        "top10_quant_worse_fraction": float(np.mean(delta[high_d] > 0)),
    }
    quintiles = []
    n = len(d)
    qidx = np.minimum(4, ((rd / max(n, 1)) * 5).astype(int))
    for q in range(5):
        mask = qidx == q
        quintiles.append(
            {
                "quintile": q + 1,
                "n": int(mask.sum()),
                "disagreement_median": float(np.median(d[mask])),
                "delta_error_median": float(np.median(delta[mask])),
                "excess_error_median": float(np.median(excess[mask])),
                "fraction_quant_worse": float(np.mean(delta[mask] > 0)),
            }
        )
    return m, {"rd": rd, "rex": rex, "high_ex": high_ex, "high_d": high_d, "delta": delta, "excess": excess, "quintiles": quintiles}


def bootstrap_ci(values: np.ndarray, statistic, rng: np.random.Generator, n_resamples: int = BOOTSTRAP_RESAMPLES) -> list[float]:
    values = np.asarray(values, np.float64)
    n = len(values)
    out = np.empty(n_resamples, np.float64)
    for i in range(n_resamples):
        sample = values[rng.integers(0, n, size=n)]
        out[i] = statistic(sample)
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def rgb_518(image_path: Path, transform: dict[str, Any]) -> np.ndarray:
    rw = int(transform["resized_width"])
    rh = int(transform["resized_height"])
    left = int(transform["padding"]["left"])
    top = int(transform["padding"]["top"])
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((rw, rh), resample=Image.Resampling.BICUBIC)
        arr = np.asarray(img)
    out = np.full((TARGET, TARGET, 3), 255, dtype=np.uint8)
    out[top : top + rh, left : left + rw] = arr
    return out


def make_example_panel(
    destination: Path,
    rgb: np.ndarray,
    valid: np.ndarray,
    d: np.ndarray,
    qerr: np.ndarray,
    delta: np.ndarray,
    excess: np.ndarray,
    boundary: np.ndarray,
    high_grad: np.ndarray,
    title: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    dm = np.where(valid, d * 100.0, np.nan)
    qm = np.where(valid, qerr * 100.0, np.nan)
    de = np.where(valid, delta * 100.0, np.nan)
    ex = np.where(valid, excess * 100.0, np.nan)
    posmax = float(np.nanquantile(np.concatenate([dm[np.isfinite(dm)], qm[np.isfinite(qm)], ex[np.isfinite(ex)]]), 0.99))
    signed = float(np.nanquantile(np.abs(de[np.isfinite(de)]), 0.99))
    posmax = max(posmax, 1e-8)
    signed = max(signed, 1e-8)
    regions = np.zeros((TARGET, TARGET, 3), np.float32)
    regions[boundary, 0] = 1.0
    regions[high_grad, 2] = 1.0
    regions[boundary & high_grad, 1] = 1.0
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    axes[0].imshow(rgb); axes[0].set_title("RGB")
    im1 = axes[1].imshow(dm, cmap="magma", vmin=0, vmax=posmax); axes[1].set_title("Full-W4A4 disagreement")
    im2 = axes[2].imshow(qm, cmap="magma", vmin=0, vmax=posmax); axes[2].set_title("W4A4 GT error")
    im3 = axes[3].imshow(de, cmap="coolwarm", vmin=-signed, vmax=signed); axes[3].set_title("Signed delta e")
    im4 = axes[4].imshow(ex, cmap="magma", vmin=0, vmax=posmax); axes[4].set_title("Positive excess")
    axes[5].imshow(regions); axes[5].set_title("Boundary red / depth-grad blue")
    for ax in axes:
        ax.axis("off")
    for im, ax in ((im1, axes[1]), (im2, axes[2]), (im3, axes[3]), (im4, axes[4])):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("% GT scene radius")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def decision_from_summary(s: dict[str, Any]) -> str:
    pos = s["fraction_scenes_positive_rho_excess"]
    auc_pos = s["fraction_scenes_auc_excess_gt_0_5"]
    mr = s["median_scene_rho_excess"]
    ma = s["median_scene_auc_excess"]
    mp = s["median_scene_top10_precision_excess"]
    control_gap = ma - s["negative_control"]["median_auc_excess"]
    if pos >= 0.80 and auc_pos >= 0.85 and mr >= 0.30 and ma >= 0.70 and mp >= 0.20 and control_gap >= 0.10:
        return "STRONGLY SUPPORTED"
    if pos >= 0.65 and auc_pos >= 0.70 and mr >= 0.15 and ma >= 0.60 and mp >= 0.15 and control_gap >= 0.05:
        return "SUPPORTED WITH LIMITATIONS"
    if mr > 0 and control_gap >= 0.02:
        return "MIXED"
    return "NOT SUPPORTED"


def main() -> None:
    args = parse_args()
    exp = args.experiment_root.resolve()
    co3d = args.co3d_root.resolve()
    repo = args.repo_root.resolve()
    manifest_path = exp / "frozen_dataset_manifest.json"
    pred_root = exp / "run/predictions"
    results = exp / "run/results"
    figures = exp / "run/figures"
    report_dir = exp / "run/report"
    progress_path = exp / "run/analysis_progress.json"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise RuntimeError(f"Frozen manifest SHA mismatch: {manifest_sha}")
    manifest = json.loads(manifest_path.read_text())
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != EXPECTED_SCENES:
        raise RuntimeError(f"Expected {EXPECTED_SCENES} scenes, got {0 if not isinstance(scenes, list) else len(scenes)}")
    group_count = sum(len(s["groups"]) for s in scenes)
    primary_count = sum(len(g["primary_frame_numbers"]) for s in scenes for g in s["groups"])
    context_count = sum(len(g["context_only_frame_numbers"]) for s in scenes for g in s["groups"])
    if (group_count, primary_count, context_count) != (EXPECTED_GROUPS, EXPECTED_PRIMARY_FRAMES, EXPECTED_CONTEXT_OCCURRENCES):
        raise RuntimeError(f"Frozen denominator changed: groups={group_count}, primary={primary_count}, context={context_count}")

    pad_transform = import_pad_transform(repo)

    # Load annotations once per category.
    categories = sorted({str(s["category"]) for s in scenes})
    frame_index: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    seq_index: dict[str, dict[str, dict[str, Any]]] = {}
    wanted_by_cat: dict[str, set[str]] = defaultdict(set)
    for s in scenes:
        wanted_by_cat[str(s["category"])].add(str(s.get("sequence") or s.get("sequence_name")))
    for cat in categories:
        fr = load_jgz(co3d / cat / "frame_annotations.jgz")
        sq = load_jgz(co3d / cat / "sequence_annotations.jgz")
        frame_index[cat] = {seq: {} for seq in wanted_by_cat[cat]}
        for r in fr:
            seq = r.get("sequence_name")
            if seq in frame_index[cat]:
                frame_index[cat][seq][int(r["frame_number"])] = r
        seq_index[cat] = {r["sequence_name"]: r for r in sq if r.get("sequence_name") in wanted_by_cat[cat]}

    per_frame_rows: list[dict[str, Any]] = []
    per_scene_rows: list[dict[str, Any]] = []
    scene_quintiles: dict[str, list[dict[str, Any]]] = {}
    neg_scene_rows: list[dict[str, Any]] = []
    selected_frame_material: dict[tuple[str, str, int], dict[str, Any]] = {}

    # Dataset localization accumulators: weighted by pixels, not frame-ratio means.
    loc_totals = defaultdict(float)
    loc_counts = defaultdict(int)
    total_groups_validated = 0
    input_hash_pairs_matched = 0

    for scene_idx, scene in enumerate(scenes, start=1):
        cat = str(scene["category"])
        seq = str(scene.get("sequence") or scene.get("sequence_name"))
        records = frame_index[cat][seq]
        seq_record = seq_index[cat].get(seq)
        if seq_record is None:
            raise RuntimeError(f"Missing sequence annotation: {cat}/{seq}")
        pc_meta = seq_record.get("point_cloud")
        if not isinstance(pc_meta, dict):
            raise RuntimeError(f"Missing point cloud: {cat}/{seq}")
        pc_path = co3d / pc_meta["path"]
        pc = read_point_cloud(pc_path)
        radius = reference_radius(pc)
        pc_tree = cKDTree(pc)

        sd: list[np.ndarray] = []
        sqe: list[np.ndarray] = []
        sfe: list[np.ndarray] = []
        scene_gt_pc_samples: list[np.ndarray] = []
        scene_cam_full = []
        scene_cam_quant = []
        scene_alignment_warning_count = 0
        scene_alignment_warning_groups = []
        scene_boundary_sum = scene_interior_sum = scene_grad_sum = scene_smooth_sum = 0.0
        scene_boundary_n = scene_interior_n = scene_grad_n = scene_smooth_n = 0
        scene_global_sum = 0.0
        scene_global_n = 0
        scene_frame_rows: list[dict[str, Any]] = []

        for group_idx, group in enumerate(scene["groups"]):
            full, quant, fm, qm = load_prediction_pair(pred_root, scene, group, args.skip_file_sha)
            total_groups_validated += 1
            input_hash_pairs_matched += int(fm["input_semantic_sha256"] == qm["input_semantic_sha256"])
            frames = [int(x) for x in group["frame_numbers"]]
            Egt = np.stack([co3d_to_opencv(records[f])[0] for f in frames])
            Ef = full["extrinsic"].astype(np.float64)
            Eq = quant["extrinsic"].astype(np.float64)
            sf, Af, bf = umeyama(centers(Egt), centers(Ef))
            sq_, Aq, bq = umeyama(centers(Egt), centers(Eq))
            camf = camera_alignment_report(Egt, Ef, sf, Af, bf, radius)
            camq = camera_alignment_report(Egt, Eq, sq_, Aq, bq, radius)
            scene_cam_full.append(camf)
            scene_cam_quant.append(camq)
            alignment_warning = (
                camf["center_median"] > 0.25
                or camq["center_median"] > 0.25
                or camf["rotation_median_deg"] > 20
                or camq["rotation_median_deg"] > 20
            )
            if alignment_warning:
                scene_alignment_warning_count += 1
                scene_alignment_warning_groups.append(str(group["group_id"]))
                print(
                    f"[WARN] camera alignment residual: {cat}/{seq}/{group['group_id']} "
                    f"Full(center_med={camf['center_median']:.4f}, "
                    f"rot_med={camf['rotation_median_deg']:.2f}deg) "
                    f"W4A4(center_med={camq['center_median']:.4f}, "
                    f"rot_med={camq['rotation_median_deg']:.2f}deg)",
                    flush=True,
                )
            PF = pred_to_gt(full["world_points_from_depth"].astype(np.float64), sf, Af, bf)
            PQ = pred_to_gt(quant["world_points_from_depth"].astype(np.float64), sq_, Aq, bq)

            for frame in [int(x) for x in group["primary_frame_numbers"]]:
                i = frames.index(frame)
                gt = decode_gt_frame(records[frame], co3d, pad_transform)
                Pgt = gt["world"].astype(np.float64)
                valid = (
                    gt["valid"]
                    & np.isfinite(Pgt).all(axis=2)
                    & np.isfinite(PF[i]).all(axis=2)
                    & np.isfinite(PQ[i]).all(axis=2)
                )
                flat_valid = np.flatnonzero(valid.reshape(-1))
                if len(flat_valid) == 0:
                    raise RuntimeError(f"No valid GT pixels: {cat}/{seq}/frame {frame}")
                n_sample = min(SAMPLE_PER_FRAME, len(flat_valid))
                rng = np.random.default_rng(stable_seed(f"{cat}/{seq}/{frame}"))
                idx = rng.choice(flat_valid, n_sample, replace=False) if len(flat_valid) > n_sample else flat_valid

                pg = Pgt.reshape(-1, 3)[idx]
                pf = PF[i].reshape(-1, 3)[idx]
                pq = PQ[i].reshape(-1, 3)[idx]
                ferr = np.linalg.norm(pf - pg, axis=1) / radius
                qerr = np.linalg.norm(pq - pg, axis=1) / radius
                dis = np.linalg.norm(pq - pf, axis=1) / radius
                row = frame_metrics(dis, qerr, ferr)
                row.update({
                    "category": cat,
                    "sequence": seq,
                    "group_id": str(group["group_id"]),
                    "frame": frame,
                    "valid_pixels": int(len(flat_valid)),
                    "sampled_pixels": int(n_sample),
                    "scene_radius": radius,
                })

                # Full-resolution maps for region localization only.
                ferr_map = np.linalg.norm(PF[i] - Pgt, axis=2) / radius
                qerr_map = np.linalg.norm(PQ[i] - Pgt, axis=2) / radius
                dis_map = np.linalg.norm(PQ[i] - PF[i], axis=2) / radius
                delta_map = qerr_map - ferr_map
                excess_map = np.maximum(delta_map, 0.0)
                fg = gt["fg"] >= 0.5
                eroded = binary_erosion(fg, iterations=2)
                boundary = valid & fg & ~eroded
                interior = valid & eroded
                logd = np.full_like(gt["depth"], np.nan, dtype=np.float64)
                logd[gt["valid"]] = np.log(np.maximum(gt["depth"][gt["valid"]], 1e-12))
                gy, gx = np.gradient(logd)
                grad = np.hypot(gx, gy)
                grad_valid = valid & np.isfinite(grad)
                if grad_valid.any():
                    grad_threshold = float(np.quantile(grad[grad_valid], 0.90))
                    high_grad = grad_valid & (grad >= grad_threshold)
                    smooth = grad_valid & ~high_grad
                else:
                    high_grad = np.zeros_like(valid)
                    smooth = np.zeros_like(valid)

                for name, mask in (("boundary", boundary), ("interior", interior), ("grad", high_grad), ("smooth", smooth), ("global", valid)):
                    if mask.any():
                        val_sum = float(np.sum(excess_map[mask]))
                        val_n = int(mask.sum())
                        loc_totals[name] += val_sum
                        loc_counts[name] += val_n
                        if name == "boundary": scene_boundary_sum += val_sum; scene_boundary_n += val_n
                        elif name == "interior": scene_interior_sum += val_sum; scene_interior_n += val_n
                        elif name == "grad": scene_grad_sum += val_sum; scene_grad_n += val_n
                        elif name == "smooth": scene_smooth_sum += val_sum; scene_smooth_n += val_n
                        elif name == "global": scene_global_sum += val_sum; scene_global_n += val_n

                # Small GT sample for independent point-cloud convention sanity check.
                gt_pts = Pgt[valid]
                if len(gt_pts) > 100:
                    rr = np.random.default_rng(stable_seed(f"gtpc/{cat}/{seq}/{frame}"))
                    gt_pts = gt_pts[rr.choice(len(gt_pts), 100, replace=False)]
                scene_gt_pc_samples.append(gt_pts)

                sd.append(dis.astype(np.float32))
                sqe.append(qerr.astype(np.float32))
                sfe.append(ferr.astype(np.float32))
                per_frame_rows.append(row)
                scene_frame_rows.append(row)

                # Cache material only for candidate representative frames after metrics are known.
                selected_frame_material[(cat, seq, frame)] = {
                    "group_id": str(group["group_id"]),
                    "group_frames": frames,
                    "index": i,
                }

            if group_idx % 5 == 0:
                write_json_atomic(progress_path, {
                    "stage": "analyze",
                    "scene": scene_idx,
                    "scenes_total": len(scenes),
                    "category": cat,
                    "sequence": seq,
                    "group_in_scene": group_idx + 1,
                    "groups_in_scene": len(scene["groups"]),
                    "groups_validated": total_groups_validated,
                    "elapsed_seconds": time.time() - started,
                })

        d = np.concatenate(sd).astype(np.float64)
        qerr = np.concatenate(sqe).astype(np.float64)
        ferr = np.concatenate(sfe).astype(np.float64)
        sm, aux = scene_metrics(d, qerr, ferr)
        sm.update({
            "category": cat,
            "sequence": seq,
            "primary_frames": len(scene_frame_rows),
            "groups": len(scene["groups"]),
            "scene_radius": radius,
            "full_camera_center_median": float(np.median([x["center_median"] for x in scene_cam_full])),
            "w4a4_camera_center_median": float(np.median([x["center_median"] for x in scene_cam_quant])),
            "full_camera_rotation_median_deg": float(np.median([x["rotation_median_deg"] for x in scene_cam_full])),
            "w4a4_camera_rotation_median_deg": float(np.median([x["rotation_median_deg"] for x in scene_cam_quant])),
            "full_camera_center_max": float(np.max([x["center_max"] for x in scene_cam_full])),
            "w4a4_camera_center_max": float(np.max([x["center_max"] for x in scene_cam_quant])),
            "full_camera_rotation_max_deg": float(np.max([x["rotation_max_deg"] for x in scene_cam_full])),
            "w4a4_camera_rotation_max_deg": float(np.max([x["rotation_max_deg"] for x in scene_cam_quant])),
            "camera_alignment_warning_groups": int(scene_alignment_warning_count),
            "camera_alignment_warning_fraction": float(
                scene_alignment_warning_count / max(len(scene["groups"]), 1)
            ),
            "camera_alignment_warning_group_ids": ";".join(scene_alignment_warning_groups),
        })

        gtpc = np.concatenate(scene_gt_pc_samples, axis=0)
        if len(gtpc) > 10000:
            rr = np.random.default_rng(stable_seed(f"scene-gtpc/{cat}/{seq}"))
            gtpc = gtpc[rr.choice(len(gtpc), 10000, replace=False)]
        nn, _ = pc_tree.query(gtpc, k=1, workers=-1)
        gt_pc_median = float(np.median(nn) / radius)
        sm["gt_depth_to_pointcloud_median_fraction_radius"] = gt_pc_median
        if gt_pc_median > 0.10:
            raise RuntimeError(f"GT depth/camera point-cloud sanity failed: {cat}/{seq} {gt_pc_median:.4f}")

        global_mean = scene_global_sum / max(scene_global_n, 1)
        sm["boundary_excess_enrichment"] = (scene_boundary_sum / max(scene_boundary_n, 1)) / max(global_mean, 1e-15)
        sm["interior_excess_enrichment"] = (scene_interior_sum / max(scene_interior_n, 1)) / max(global_mean, 1e-15)
        sm["high_depth_gradient_excess_enrichment"] = (scene_grad_sum / max(scene_grad_n, 1)) / max(global_mean, 1e-15)
        sm["smooth_excess_enrichment"] = (scene_smooth_sum / max(scene_smooth_n, 1)) / max(global_mean, 1e-15)
        per_scene_rows.append(sm)
        scene_quintiles[f"{cat}/{seq}"] = aux["quintiles"]

        # Negative control: 20 deterministic within-scene permutations of disagreement ranks.
        rng = np.random.default_rng(stable_seed(f"negative/{cat}/{seq}"))
        neg_rho = []
        neg_auc = []
        neg_prec = []
        rd = aux["rd"]
        rex = aux["rex"]
        high_ex = aux["high_ex"]
        for _ in range(NEGATIVE_CONTROL_PERMUTATIONS):
            perm = rng.permutation(len(rd))
            rdp = rd[perm]
            neg_rho.append(pearson(rdp, rex))
            neg_auc.append(roc_auc_from_ranks(rdp, high_ex))
            neg_prec.append(float(np.mean(high_ex[exact_top_mask(rdp)])))
        neg_scene_rows.append({
            "category": cat,
            "sequence": seq,
            "rho_mean": float(np.mean(neg_rho)),
            "rho_std": float(np.std(neg_rho)),
            "auc_mean": float(np.mean(neg_auc)),
            "auc_std": float(np.std(neg_auc)),
            "precision_mean": float(np.mean(neg_prec)),
            "precision_std": float(np.std(neg_prec)),
        })

        print(
            f"[{scene_idx:02d}/{len(scenes)}] {cat}/{seq}: "
            f"frames={len(scene_frame_rows)} rho_ex={sm['rho_d_excess']:.4f} "
            f"auc_ex={sm['auc_top10_excess']:.4f} prec={sm['top10_precision_excess']:.4f}",
            flush=True,
        )

    if total_groups_validated != EXPECTED_GROUPS or input_hash_pairs_matched != EXPECTED_GROUPS:
        raise RuntimeError(f"Prediction validation incomplete: groups={total_groups_validated}, input_hash_pairs={input_hash_pairs_matched}")
    if len(per_frame_rows) != EXPECTED_PRIMARY_FRAMES:
        raise RuntimeError(f"Primary frame count mismatch after analysis: {len(per_frame_rows)}")

    # Write CSVs.
    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = sorted({k for r in rows for k in r.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

    write_csv(results / "per_frame.csv", per_frame_rows)
    write_csv(results / "per_scene.csv", per_scene_rows)
    write_csv(results / "negative_control_per_scene.csv", neg_scene_rows)
    write_json_atomic(results / "disagreement_quintiles.json", scene_quintiles)

    rho = np.asarray([r["rho_d_excess"] for r in per_scene_rows], np.float64)
    rho_eq = np.asarray([r["rho_d_eQ"] for r in per_scene_rows], np.float64)
    auc = np.asarray([r["auc_top10_excess"] for r in per_scene_rows], np.float64)
    prec = np.asarray([r["top10_precision_excess"] for r in per_scene_rows], np.float64)
    qworse = np.asarray([r["top10_quant_worse_fraction"] for r in per_scene_rows], np.float64)
    neg_auc_scene = np.asarray([r["auc_mean"] for r in neg_scene_rows], np.float64)
    neg_rho_scene = np.asarray([r["rho_mean"] for r in neg_scene_rows], np.float64)
    neg_prec_scene = np.asarray([r["precision_mean"] for r in neg_scene_rows], np.float64)
    boot_rng = np.random.default_rng(RNG_SEED + 999)

    global_mean = loc_totals["global"] / max(loc_counts["global"], 1)
    localization = {
        "boundary_excess_enrichment": (loc_totals["boundary"] / max(loc_counts["boundary"], 1)) / max(global_mean, 1e-15),
        "interior_excess_enrichment": (loc_totals["interior"] / max(loc_counts["interior"], 1)) / max(global_mean, 1e-15),
        "high_depth_gradient_excess_enrichment": (loc_totals["grad"] / max(loc_counts["grad"], 1)) / max(global_mean, 1e-15),
        "smooth_excess_enrichment": (loc_totals["smooth"] / max(loc_counts["smooth"], 1)) / max(global_mean, 1e-15),
        "counts": {k: int(v) for k, v in loc_counts.items()},
        "definition": {
            "boundary": "foreground minus 2-pixel-eroded foreground",
            "high_depth_gradient": "top decile of finite log-depth gradient magnitude; gradients touching invalid depth become non-finite and are excluded",
        },
    }

    summary: dict[str, Any] = {
        "version": 1,
        "manifest_sha256": manifest_sha,
        "evaluated_scenes": len(per_scene_rows),
        "primary_frames": len(per_frame_rows),
        "six_view_groups": total_groups_validated,
        "full_predictions_valid": total_groups_validated,
        "w4a4_predictions_valid": total_groups_validated,
        "input_hash_pairs_matched": input_hash_pairs_matched,
        "camera_alignment_warning_groups": int(
            sum(r["camera_alignment_warning_groups"] for r in per_scene_rows)
        ),
        "scenes_with_camera_alignment_warnings": int(
            sum(r["camera_alignment_warning_groups"] > 0 for r in per_scene_rows)
        ),
        "scene_positive_rho_excess": int(np.sum(rho > 0)),
        "scene_rho_excess_gt_0_2": int(np.sum(rho > 0.2)),
        "scene_rho_excess_gt_0_4": int(np.sum(rho > 0.4)),
        "fraction_scenes_positive_rho_excess": float(np.mean(rho > 0)),
        "median_scene_rho_excess": float(np.median(rho)),
        "iqr_scene_rho_excess": [float(np.quantile(rho, 0.25)), float(np.quantile(rho, 0.75))],
        "median_scene_rho_eQ": float(np.median(rho_eq)),
        "scene_auc_excess_gt_0_5": int(np.sum(auc > 0.5)),
        "scene_auc_excess_gt_0_75": int(np.sum(auc > 0.75)),
        "fraction_scenes_auc_excess_gt_0_5": float(np.mean(auc > 0.5)),
        "median_scene_auc_excess": float(np.median(auc)),
        "iqr_scene_auc_excess": [float(np.quantile(auc, 0.25)), float(np.quantile(auc, 0.75))],
        "median_scene_top10_precision_excess": float(np.median(prec)),
        "median_scene_top10_quant_worse_fraction": float(np.median(qworse)),
        "negative_control": {
            "permutations_per_scene": NEGATIVE_CONTROL_PERMUTATIONS,
            "median_rho_excess": float(np.median(neg_rho_scene)),
            "median_auc_excess": float(np.median(neg_auc_scene)),
            "median_top10_precision_excess": float(np.median(neg_prec_scene)),
        },
        "scene_bootstrap_95ci": {
            "median_rho_excess": bootstrap_ci(rho, np.median, boot_rng),
            "median_auc_excess": bootstrap_ci(auc, np.median, boot_rng),
            "median_top10_precision_excess": bootstrap_ci(prec, np.median, boot_rng),
            "fraction_positive_rho_excess": bootstrap_ci((rho > 0).astype(float), np.mean, boot_rng),
            "fraction_auc_excess_gt_0_5": bootstrap_ci((auc > 0.5).astype(float), np.mean, boot_rng),
        },
        "localization": localization,
        "analysis_contract": {
            "distance_normalization": "GT robust point-cloud radius (90th percentile distance from median point-cloud center)",
            "sampling": f"deterministic up to {SAMPLE_PER_FRAME} valid GT-foreground pixels per primary frame",
            "top_decile": "exact top ceil(10% * N) by deterministic array order for tie resolution",
            "alignment": "one proper Sim(3) per six-view group, independently GT camera centers -> Full and GT camera centers -> W4A4; score primary frames only",
            "depth_decoder": "uint16 PNG bytes reinterpreted as float16, cast float32, multiplied by scale_adjustment",
            "camera_alignment_warning": "A group is flagged, but not excluded, when Full or W4A4 median camera-center residual exceeds 0.25 normalized scene radius or median rotation residual exceeds 20 degrees. Nonfinite/degenerate Sim(3) remains fatal.",
        },
    }
    summary["decision"] = decision_from_summary(summary)
    write_json_atomic(results / "dataset_summary.json", summary)
    write_json_atomic(results / "negative_control.json", {
        "per_scene": neg_scene_rows,
        "dataset_medians": summary["negative_control"],
    })

    # Dataset plots.
    plt.figure(figsize=(7, 4)); plt.hist(rho, bins=12); plt.axvline(0, linewidth=1); plt.xlabel("Scene Spearman rho(d, excess)"); plt.ylabel("Scenes"); plt.title("Scene-level disagreement vs quantization excess error"); plt.tight_layout(); plt.savefig(figures / "dataset_rho_distribution.png", dpi=160); plt.close()
    plt.figure(figsize=(7, 4)); plt.hist(auc, bins=12); plt.axvline(0.5, linewidth=1); plt.xlabel("Scene AUC: disagreement -> top-decile excess"); plt.ylabel("Scenes"); plt.title("Scene-level excess-error localization AUC"); plt.tight_layout(); plt.savefig(figures / "dataset_auc_distribution.png", dpi=160); plt.close()
    plt.figure(figsize=(6, 5)); plt.scatter(neg_auc_scene, auc); lo=min(float(np.min(neg_auc_scene)),float(np.min(auc)),0.45); hi=max(float(np.max(neg_auc_scene)),float(np.max(auc)),0.55); plt.plot([lo,hi],[lo,hi]); plt.xlabel("Shuffled-control AUC"); plt.ylabel("Real disagreement AUC"); plt.title("Real vs shuffled disagreement"); plt.tight_layout(); plt.savefig(figures / "real_vs_shuffled.png", dpi=160); plt.close()
    plt.figure(figsize=(6, 5)); fmed=np.asarray([r["full_error_median"] for r in per_scene_rows]); qmed=np.asarray([r["quant_error_median"] for r in per_scene_rows]); plt.scatter(fmed*100,qmed*100); hi=max(float(fmed.max()),float(qmed.max()))*100; plt.plot([0,hi],[0,hi]); plt.xlabel("Full median GT error (% radius)"); plt.ylabel("W4A4 median GT error (% radius)"); plt.title("Full vs W4A4 scene geometry error"); plt.tight_layout(); plt.savefig(figures / "full_vs_quant_error.png", dpi=160); plt.close()

    # Aggregate scene-balanced quintiles: median of scene-level quintile medians.
    qrows=[]
    for q in range(5):
        qrows.append({
            "quintile": q+1,
            "excess_median": float(np.median([scene_quintiles[k][q]["excess_error_median"] for k in scene_quintiles])),
            "delta_median": float(np.median([scene_quintiles[k][q]["delta_error_median"] for k in scene_quintiles])),
            "quant_worse_fraction": float(np.median([scene_quintiles[k][q]["fraction_quant_worse"] for k in scene_quintiles])),
        })
    plt.figure(figsize=(7,4)); plt.plot([r["quintile"] for r in qrows],[100*r["excess_median"] for r in qrows],marker="o",label="positive excess"); plt.plot([r["quintile"] for r in qrows],[100*r["delta_median"] for r in qrows],marker="o",label="signed delta"); plt.xlabel("Disagreement quintile (low -> high)"); plt.ylabel("Median error change (% radius)"); plt.title("Scene-balanced disagreement quintiles"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "disagreement_quintiles.png", dpi=160); plt.close()

    # Deterministic strongest / typical / weakest scenes and representative frames.
    sorted_scenes = sorted(per_scene_rows, key=lambda r: r["rho_d_excess"])
    median_rho = float(np.median(rho))
    weakest = sorted_scenes[:3]
    strongest = sorted_scenes[-3:][::-1]
    typical = sorted(per_scene_rows, key=lambda r: abs(r["rho_d_excess"] - median_rho))[:3]
    chosen=[]
    seen=set()
    for label, seqs in (("weak",weakest),("typical",typical),("strong",strongest)):
        for sr in seqs:
            key=(sr["category"],sr["sequence"])
            if key in seen: continue
            seen.add(key)
            rows=[r for r in per_frame_rows if r["category"]==key[0] and r["sequence"]==key[1]]
            scene_frame_median=float(np.median([r["rho_d_excess"] for r in rows]))
            fr=min(rows,key=lambda r:abs(r["rho_d_excess"]-scene_frame_median))
            chosen.append((label,sr,fr))

    scene_map={(str(s["category"]),str(s.get("sequence") or s.get("sequence_name"))):s for s in scenes}
    for label,sr,fr in chosen:
        cat,seq,frame=sr["category"],sr["sequence"],int(fr["frame"])
        scene=scene_map[(cat,seq)]
        group=next(g for g in scene["groups"] if str(g["group_id"])==fr["group_id"])
        full,quant,_,_=load_prediction_pair(pred_root,scene,group,True)
        frames=[int(x) for x in group["frame_numbers"]]
        i=frames.index(frame)
        records=frame_index[cat][seq]
        Egt=np.stack([co3d_to_opencv(records[f])[0] for f in frames])
        sf,Af,bf=umeyama(centers(Egt),centers(full["extrinsic"].astype(np.float64)))
        sq_,Aq,bq=umeyama(centers(Egt),centers(quant["extrinsic"].astype(np.float64)))
        PF=pred_to_gt(full["world_points_from_depth"].astype(np.float64),sf,Af,bf)
        PQ=pred_to_gt(quant["world_points_from_depth"].astype(np.float64),sq_,Aq,bq)
        gt=decode_gt_frame(records[frame],co3d,pad_transform); Pgt=gt["world"].astype(np.float64)
        valid=gt["valid"]&np.isfinite(Pgt).all(2)&np.isfinite(PF[i]).all(2)&np.isfinite(PQ[i]).all(2)
        rad=float(sr["scene_radius"])
        ferr_map=np.linalg.norm(PF[i]-Pgt,axis=2)/rad; qerr_map=np.linalg.norm(PQ[i]-Pgt,axis=2)/rad; dmap=np.linalg.norm(PQ[i]-PF[i],axis=2)/rad; delta=qerr_map-ferr_map; excess=np.maximum(delta,0)
        fg=gt["fg"]>=.5; boundary=valid&fg&~binary_erosion(fg,iterations=2)
        logd=np.full_like(gt["depth"],np.nan,dtype=np.float64); logd[gt["valid"]]=np.log(np.maximum(gt["depth"][gt["valid"]],1e-12)); gy,gx=np.gradient(logd); grad=np.hypot(gx,gy); gv=valid&np.isfinite(grad); high=np.zeros_like(valid); high[gv]=grad[gv]>=np.quantile(grad[gv],.9) if gv.any() else False
        rgb=rgb_518(gt["image_path"],gt["transform"])
        dest=figures/"selected_examples"/f"{label}_{cat}_{seq}_frame_{frame:06d}.png"
        make_example_panel(dest,rgb,valid,dmap,qerr_map,delta,excess,boundary,high,f"{label.upper()} | {cat}/{seq} | frame {frame} | scene rho={sr['rho_d_excess']:.3f}")

    # Result-first Markdown report.
    s=summary
    def pct(x): return f"{100*x:.1f}%"
    lines=[
        "# Dataset-wide Full-vs-W4A4 Geometric Disagreement Validation",
        "",
        "## Main question",
        "",
        "Across the complete frozen CO3D subset, how consistently does Full-vs-W4A4 geometric disagreement identify quantization-induced excess geometric error?",
        "",
        "## Main result",
        "",
        "| Metric | Dataset result |",
        "|---|---:|",
        f"| Evaluated scenes | {s['evaluated_scenes']} |",
        f"| Primary frames | {s['primary_frames']} |",
        f"| Six-view groups | {s['six_view_groups']} |",
        f"| Camera-alignment warning groups | {s['camera_alignment_warning_groups']}/{s['six_view_groups']} |",
        f"| Scenes with camera-alignment warnings | {s['scenes_with_camera_alignment_warnings']}/{s['evaluated_scenes']} |",
        f"| Positive rho(d, excess) scenes | {s['scene_positive_rho_excess']}/{s['evaluated_scenes']} ({pct(s['fraction_scenes_positive_rho_excess'])}) |",
        f"| rho(d, excess) > 0.4 | {s['scene_rho_excess_gt_0_4']}/{s['evaluated_scenes']} |",
        f"| Median scene rho(d, excess) | {s['median_scene_rho_excess']:.4f} |",
        f"| 95% scene-bootstrap CI, median rho | [{s['scene_bootstrap_95ci']['median_rho_excess'][0]:.4f}, {s['scene_bootstrap_95ci']['median_rho_excess'][1]:.4f}] |",
        f"| Excess AUC > 0.5 | {s['scene_auc_excess_gt_0_5']}/{s['evaluated_scenes']} ({pct(s['fraction_scenes_auc_excess_gt_0_5'])}) |",
        f"| Excess AUC > 0.75 | {s['scene_auc_excess_gt_0_75']}/{s['evaluated_scenes']} |",
        f"| Median scene excess AUC | {s['median_scene_auc_excess']:.4f} |",
        f"| Median top-10% excess precision | {pct(s['median_scene_top10_precision_excess'])} |",
        f"| Shuffled median AUC | {s['negative_control']['median_auc_excess']:.4f} |",
        f"| Shuffled median top-10% precision | {pct(s['negative_control']['median_top10_precision_excess'])} |",
        f"| High-disagreement pixels where W4A4 is worse | {pct(s['median_scene_top10_quant_worse_fraction'])} (median scene) |",
        f"| Boundary excess enrichment | {s['localization']['boundary_excess_enrichment']:.2f}x |",
        f"| High GT depth-gradient excess enrichment | {s['localization']['high_depth_gradient_excess_enrichment']:.2f}x |",
        "",
        f"**Decision under the pre-specified rule:** **{s['decision']}**",
        "",
        "The decision rule was fixed in the analysis code before the dataset-wide results were inspected. Absolute W4A4-error correlation is treated as secondary; the main target is quantization-induced positive excess error.",
        "",
        "## Controlled method",
        "",
        "Every frozen six-view group was processed by both variants from the same image set. The saved metadata records a semantic input hash; all 1,330 Full/W4A4 input-hash pairs were required to match. Full and W4A4 were aligned independently to the same six GT camera centers using one proper Sim(3) per group. Only primary frames were scored; context-only frames were never double-counted.",
        "",
        "Distances are normalized by each sequence's robust GT point-cloud radius. CO3D depth PNGs are decoded as float16 bit patterns stored in uint16 containers and multiplied by `scale_adjustment`.",
        "",
        "Definitions: `d = ||P_Q-P_F||`, `eF = ||P_F-P_GT||`, `eQ = ||P_Q-P_GT||`, `delta_e = eQ-eF`, and `excess = max(delta_e,0)`.",
        "",
        "## Scene-level consistency",
        "",
        "| Scene | Frames | rho(d,eQ) | rho(d,excess) | Excess AUC | Top-10% precision | Full err %R | W4A4 err %R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(per_scene_rows,key=lambda x:x["rho_d_excess"],reverse=True):
        lines.append(f"| {r['category']}/{r['sequence']} | {r['primary_frames']} | {r['rho_d_eQ']:.3f} | {r['rho_d_excess']:.3f} | {r['auc_top10_excess']:.3f} | {100*r['top10_precision_excess']:.1f}% | {100*r['full_error_median']:.2f} | {100*r['quant_error_median']:.2f} |")
    lines += [
        "",
        "## Signed-error interpretation",
        "",
        "Disagreement is unsigned: it measures how much Full and W4A4 geometry differ, not which one is better. The signed quantity `delta_e=eQ-eF` therefore remains necessary. The report explicitly measures the fraction of the highest-disagreement pixels for which W4A4 is actually worse.",
        "",
        "## Negative control",
        "",
        f"Twenty deterministic disagreement-rank permutations were evaluated per scene. Their median scene AUC was {s['negative_control']['median_auc_excess']:.4f}, median rho was {s['negative_control']['median_rho_excess']:.4f}, and median top-decile precision was {100*s['negative_control']['median_top10_precision_excess']:.1f}%.",
        "",
        "## Spatial localization",
        "",
        f"Pixel-weighted positive-excess enrichment was {s['localization']['boundary_excess_enrichment']:.2f}x at the object-boundary band and {s['localization']['high_depth_gradient_excess_enrichment']:.2f}x in the top-decile valid GT log-depth-gradient region. These are aggregate tendencies, not universal per-scene claims.",
        "",
        "## Figures",
        "",
        "- `../figures/dataset_rho_distribution.png`",
        "- `../figures/dataset_auc_distribution.png`",
        "- `../figures/disagreement_quintiles.png`",
        "- `../figures/real_vs_shuffled.png`",
        "- `../figures/full_vs_quant_error.png`",
        "- `../figures/selected_examples/` contains deterministic strong, typical, and weak cases.",
        "",
        "## Limitations",
        "",
        "W4A4 is simulated/fake quantization rather than native packed INT4. The benchmark is a reduced CO3D subset rather than all possible object/scene distributions. Pixels within a frame are correlated, so the principal evidence is the distribution across scenes and scene-level bootstrap intervals. Full VGGT is a teacher/reference model, not ground truth, which is why the signed GT comparison is included. Six-view context can affect a frame's prediction.",
        "",
        "## Provenance",
        "",
        f"Frozen manifest SHA256: `{manifest_sha}`",
        f"Validated Full predictions: {total_groups_validated}/{EXPECTED_GROUPS}",
        f"Validated W4A4 predictions: {total_groups_validated}/{EXPECTED_GROUPS}",
        f"Matching Full/W4A4 semantic input hashes: {input_hash_pairs_matched}/{EXPECTED_GROUPS}",
        f"Analysis seed: {RNG_SEED}",
        f"Sample cap per primary frame: {SAMPLE_PER_FRAME}",
        f"Negative-control permutations per scene: {NEGATIVE_CONTROL_PERMUTATIONS}",
        f"Scene bootstrap resamples: {BOOTSTRAP_RESAMPLES}",
        "",
    ]
    report_path=report_dir/"DATASET_DISAGREEMENT_VALIDATION.md"
    report_path.write_text("\n".join(lines),encoding="utf-8")

    write_json_atomic(progress_path, {
        "stage": "done",
        "scenes": len(per_scene_rows),
        "primary_frames": len(per_frame_rows),
        "groups_validated": total_groups_validated,
        "elapsed_seconds": time.time() - started,
        "summary": str(results/"dataset_summary.json"),
        "report": str(report_path),
    })

    print("\n===== DATASET-WIDE DISAGREEMENT ANALYSIS COMPLETE =====")
    print(json.dumps({
        "status":"PASS",
        "scenes":s["evaluated_scenes"],
        "primary_frames":s["primary_frames"],
        "groups":s["six_view_groups"],
        "positive_rho_excess":f"{s['scene_positive_rho_excess']}/{s['evaluated_scenes']}",
        "rho_excess_gt_0_4":f"{s['scene_rho_excess_gt_0_4']}/{s['evaluated_scenes']}",
        "median_rho_excess":s["median_scene_rho_excess"],
        "auc_excess_gt_0_5":f"{s['scene_auc_excess_gt_0_5']}/{s['evaluated_scenes']}",
        "auc_excess_gt_0_75":f"{s['scene_auc_excess_gt_0_75']}/{s['evaluated_scenes']}",
        "median_auc_excess":s["median_scene_auc_excess"],
        "median_top10_precision_excess":s["median_scene_top10_precision_excess"],
        "shuffled_median_auc":s["negative_control"]["median_auc_excess"],
        "boundary_enrichment":s["localization"]["boundary_excess_enrichment"],
        "depth_gradient_enrichment":s["localization"]["high_depth_gradient_excess_enrichment"],
        "decision":s["decision"],
        "summary":str(results/"dataset_summary.json"),
        "report":str(report_path),
    },indent=2))


if __name__ == "__main__":
    main()

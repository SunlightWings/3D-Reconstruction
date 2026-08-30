#!/usr/bin/env python3
"""Validate a variant-neutral VGGT-to-official-3DGS adapter output."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

from vggt_to_3dgs import (
    CV_ROOT,
    DEFAULT_PREDICTION_DIR,
    OFFICIAL_3DGS_ROOT,
    converted_intrinsics,
    read_json,
    selected_points,
)
from vggt_prediction_contract import require_expected_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--official-3dgs-root", type=Path, default=OFFICIAL_3DGS_ROOT)
    parser.add_argument("--loader-smoke", action="store_true")
    return parser.parse_args()


def default_adapter_dir(prediction_dir: Path, metadata: dict) -> Path:
    return (
        prediction_dir.parents[3]
        / "3dgs_inputs"
        / metadata["sequence_name"]
        / metadata["split"]
        / metadata["variant"]
    )


def count_points(points_path: Path) -> int:
    with points_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def official_loader_smoke(adapter_dir: Path, official_root: Path) -> dict:
    """Run the unchanged official dataset-reader source without training modules.

    The top-level ``scene`` package imports CUDA training extensions. A minimal
    package shell lets Python load the exact official ``scene.dataset_readers``
    file while avoiding that unrelated package initializer.
    """
    official_root = official_root.resolve()
    dataset_source = official_root / "scene/dataset_readers.py"
    if not dataset_source.is_file():
        raise FileNotFoundError(dataset_source)
    sys.path.insert(0, str(official_root))
    package = types.ModuleType("scene")
    package.__path__ = [str(official_root / "scene")]
    previous = sys.modules.get("scene")
    previous_gaussian_model = sys.modules.get("scene.gaussian_model")
    sys.modules["scene"] = package
    # dataset_readers needs only BasicPointCloud; this avoids importing CUDA training extensions.
    from utils.graphics_utils import BasicPointCloud

    gaussian_model = types.ModuleType("scene.gaussian_model")
    gaussian_model.BasicPointCloud = BasicPointCloud
    sys.modules["scene.gaussian_model"] = gaussian_model
    try:
        spec = importlib.util.spec_from_file_location("scene.dataset_readers", dataset_source)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load official dataset_readers.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["scene.dataset_readers"] = module
        spec.loader.exec_module(module)
        scene_info = module.readColmapSceneInfo(
            str(adapter_dir), None, "", False, False
        )
    finally:
        sys.modules.pop("scene.dataset_readers", None)
        if previous_gaussian_model is None:
            sys.modules.pop("scene.gaussian_model", None)
        else:
            sys.modules["scene.gaussian_model"] = previous_gaussian_model
        if previous is None:
            sys.modules.pop("scene", None)
        else:
            sys.modules["scene"] = previous
    return {
        "status": "PASS",
        "official_reader": str(dataset_source),
        "train_cameras": len(scene_info.train_cameras),
        "test_cameras": len(scene_info.test_cameras),
        "point_cloud_points": int(len(scene_info.point_cloud.points)),
        "point_cloud_path": scene_info.ply_path,
    }


def main() -> None:
    args = parse_args()
    prediction_dir = args.prediction_dir.resolve()
    metadata = read_json(prediction_dir / "inference_meta.json")
    adapter_dir = (
        args.adapter_dir.resolve()
        if args.adapter_dir is not None
        else default_adapter_dir(prediction_dir, metadata)
    )
    adapter_meta = read_json(adapter_dir / "adapter_meta.json")
    with np.load(prediction_dir / "predictions.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    require_expected_shapes(arrays)
    preprocess = read_json(Path(metadata["input_tensor_path"]).parent / "preprocess_meta.json")
    views = preprocess["views"]
    if len(views) != arrays["world_points"].shape[0]:
        raise ValueError("Preprocessing metadata view count differs from predictions")

    intrinsics = np.stack(
        [converted_intrinsics(arrays["intrinsic"][index], view) for index, view in enumerate(views)]
    )
    extrinsics = arrays["extrinsic"].astype(np.float64)
    rotations = extrinsics[:, :, :3]
    translations = extrinsics[:, :, 3]
    camera_centers = -np.einsum("sji,sj->si", rotations, translations)
    determinants = np.linalg.det(rotations)
    if not np.isfinite(camera_centers).all():
        raise ValueError("Non-finite camera centers")
    if not np.all(np.isfinite(intrinsics)) or np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
        raise ValueError("Invalid converted focal lengths")
    if not np.allclose(determinants, 1.0, atol=2e-3):
        raise ValueError("Camera rotation determinant is not approximately +1")

    selection_config = adapter_meta["point_selection"]
    point_source = adapter_meta["source_point_array"]
    if point_source not in ("world_points_from_depth", "world_points"):
        raise ValueError(f"Unsupported adapter point source: {point_source}")
    selection = selected_points(
        arrays[point_source],
        extrinsics,
        views,
        int(selection_config["max_points"]),
        int(selection_config["seed"]),
    )
    points = np.asarray(selection["points"])
    view_indices = np.asarray(selection["view_indices"])
    camera_points = np.einsum("nij,nj->ni", rotations[view_indices], points) + translations[view_indices]
    projected_u = (
        intrinsics[view_indices, 0, 0] * camera_points[:, 0] / camera_points[:, 2]
        + intrinsics[view_indices, 0, 2]
    )
    projected_v = (
        intrinsics[view_indices, 1, 1] * camera_points[:, 1] / camera_points[:, 2]
        + intrinsics[view_indices, 1, 2]
    )
    reprojection_error = np.hypot(
        projected_u - np.asarray(selection["u_original"]),
        projected_v - np.asarray(selection["v_original"]),
    )
    if not np.isfinite(reprojection_error).all():
        raise ValueError("Non-finite reprojection error")

    sparse = adapter_dir / "sparse/0"
    if count_points(sparse / "points3D.txt") != len(points):
        raise ValueError("points3D.txt count does not match deterministic source selection")
    for index, view in enumerate(views):
        image_path = adapter_dir / "images" / view["image_name"]
        with Image.open(image_path) as image:
            if image.size != (view["original_width"], view["original_height"]):
                raise ValueError(f"Original image dimensions changed for camera {index + 1}")

    loader_result = {"status": "NOT_RUN"}
    if args.loader_smoke:
        loader_result = official_loader_smoke(adapter_dir, args.official_3dgs_root)

    report = {
        "status": "valid",
        "adapter_dir": str(adapter_dir),
        "camera_count": len(views),
        "original_image_dimensions": [
            [view["original_width"], view["original_height"]] for view in views
        ],
        "converted_intrinsics": intrinsics.tolist(),
        "initial_point_count": int(len(points)),
        "point_rejections": int(selection["rejected_count"]),
        "rotation_determinant_range": [float(determinants.min()), float(determinants.max())],
        "reprojection_error_pixels": {
            "median": float(np.median(reprojection_error)),
            "mean": float(np.mean(reprojection_error)),
            "p95": float(np.percentile(reprojection_error, 95)),
        },
        "loader_smoke": loader_result,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

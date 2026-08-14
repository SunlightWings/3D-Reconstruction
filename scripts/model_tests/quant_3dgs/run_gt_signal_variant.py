#!/usr/bin/env python3
"""Run one frozen GT-signal tensor through Full VGGT or W4A4 QuantVGGT.

This intentionally reuses the established model construction and common
inference/output conversion from ``run_vggt_variant.py``.  Unlike that
historical runner, it has no dependency on Phase-1 experiment metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from run_vggt_variant import (
    BASE_MODEL_PATH,
    QUANTVGGT_ROOT,
    git_revision,
    load_variant,
    run_shared_inference,
)
from vggt_prediction_contract import (
    ARRAY_ORDER,
    prediction_content_sha256,
    require_expected_shapes,
    sha256_file,
)


EXPECTED_SHAPE = (6, 3, 518, 518)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("full", "w4a4"))
    parser.add_argument("--input-tensor", required=True, type=Path)
    parser.add_argument(
        "--frame-numbers",
        required=True,
        help="Six frozen frame numbers, comma-separated (for example 1,41,75,123,162,202).",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_frames(text: str) -> list[int]:
    try:
        frames = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--frame-numbers must be comma-separated integers") from exc
    if len(frames) != 6 or len(set(frames)) != 6:
        raise ValueError("--frame-numbers must contain exactly six distinct frames")
    return frames


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_frozen_input(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    images = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(images, torch.Tensor):
        raise TypeError("Frozen input must be a tensor")
    if tuple(images.shape) != EXPECTED_SHAPE:
        raise ValueError(f"Expected frozen input shape {EXPECTED_SHAPE}, got {tuple(images.shape)}")
    if images.dtype != torch.float32:
        raise ValueError(f"Expected frozen input dtype torch.float32, got {images.dtype}")
    if not torch.isfinite(images).all().item():
        raise ValueError("Frozen input contains NaN or Inf")
    minimum, maximum = float(images.min()), float(images.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"Frozen input range must be [0,1], got [{minimum}, {maximum}]")
    return images.contiguous()


def main() -> None:
    args = parse_args()
    frames = parse_frames(args.frame_numbers)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)

    input_path = args.input_tensor.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / args.variant
    predictions_path = output_dir / "predictions.npz"
    metadata_path = output_dir / "inference_meta.json"
    hash_path = output_dir / "prediction_hash.sha256"
    existing = [p for p in (predictions_path, metadata_path, hash_path) if p.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing outputs without --force: {existing}")

    started = time.perf_counter()
    images = load_frozen_input(input_path)
    input_file_sha = sha256_file(input_path)
    input_content_sha = tensor_content_sha256(images)

    load_started = time.perf_counter()
    model, variant_metadata = load_variant(args.variant, args.device)
    model_load_seconds = time.perf_counter() - load_started
    if model.training:
        raise RuntimeError("Loaded model is not in eval mode")

    arrays, original_dtypes, runtime = run_shared_inference(model, images, frames, args.device)
    require_expected_shapes(arrays)
    for name in ARRAY_ORDER:
        if name != "frame_numbers" and not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"Non-finite output values in {name}")
    if arrays["frame_numbers"].tolist() != frames:
        raise RuntimeError("Saved frame order differs from requested frozen order")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(predictions_path, **{name: arrays[name] for name in ARRAY_ORDER})
    npz_sha = sha256_file(predictions_path)
    content_sha = prediction_content_sha256(arrays)
    hash_path.write_text(f"{npz_sha}  predictions.npz\n", encoding="utf-8")

    metadata = {
        "variant": args.variant,
        "ordered_frame_numbers": frames,
        "input_tensor_path": str(input_path),
        "input_tensor_file_sha256": input_file_sha,
        "input_tensor_semantic_sha256_raw_float32_c_order": input_content_sha,
        "input_tensor_shape": list(images.shape),
        "input_tensor_dtype": str(images.dtype),
        "input_tensor_range": [float(images.min()), float(images.max())],
        "preprocessing_contract": "frozen saved tensor; pad/no-crop 518-space; no JPEG reload",
        "base_checkpoint_path": str(BASE_MODEL_PATH),
        "base_checkpoint_sha256": sha256_file(BASE_MODEL_PATH),
        "quantvggt_revision": git_revision(QUANTVGGT_ROOT),
        "experiment_repository_revision": git_revision(Path(__file__).resolve().parents[3]),
        **variant_metadata,
        "inference_mode": "torch.inference_mode",
        "autocast": {"enabled": True, "device_type": "cuda", "dtype": "torch.bfloat16"},
        "cuda": {"device": args.device, "device_name": torch.cuda.get_device_name(args.device), "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda},
        "runtime": {**runtime, "model_load_seconds": model_load_seconds, "total_seconds": time.perf_counter() - started},
        "array_order": list(ARRAY_ORDER),
        "arrays": {name: {"shape": list(arrays[name].shape), "saved_dtype": str(arrays[name].dtype), "original_dtype": original_dtypes[name]} for name in ARRAY_ORDER},
        "prediction_content_sha256": content_sha,
        "predictions_npz_sha256": npz_sha,
        "camera_conversion": "vggt.utils.pose_enc.pose_encoding_to_extri_intri at image_size_hw=(518,518)",
        "point_unprojection": "vggt.utils.geometry.unproject_depth_map_to_point_map using saved depth, world-to-camera extrinsics, and pixel-space intrinsics",
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare one manifest-defined CO3D input split for controlled VGGT runs.

This script performs preprocessing only. It deliberately does not load or run a
VGGT model. The serialized tensor it creates is the single input artifact that
Full VGGT and every QuantVGGT variant must reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image


SUPPORTED_SPLITS = ("input_3", "input_6", "input_9", "input_12")
REQUESTED_TARGET_SIZE = 518
PATCH_SIZE = 14

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
CV_ROOT = REPOSITORY_ROOT.parent
DEFAULT_MANIFEST = (
    CV_ROOT / "data/co3d_pilot/quantsplat_co3dv2_pilot_v2_manifest.json"
)
DEFAULT_QUANTVGGT_ROOT = CV_ROOT / "QuantVGGT"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/model_tests/quant_3dgs/prepared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and serialize one shared CO3D tensor for VGGT variants."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"CO3D pilot manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Extracted dataset root. Defaults to <manifest-dir>/<manifest name>.",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help=(
            "Scene selector: sequence name, category/sequence name, or zero-based "
            "manifest index. Defaults to the first scene."
        ),
    )
    parser.add_argument(
        "--split",
        choices=SUPPORTED_SPLITS,
        default="input_12",
        help="Manifest input split to prepare (default: input_12).",
    )
    parser.add_argument(
        "--quant-vggt-root",
        type=Path,
        default=DEFAULT_QUANTVGGT_ROOT,
        help=f"Local QuantVGGT repository (default: {DEFAULT_QUANTVGGT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Experiment output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an already prepared artifact set for the same scene and split.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Manifest root must be an object: {path}")
    return value


def select_scene(manifest: dict[str, Any], selector: str | None) -> tuple[int, dict[str, Any]]:
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Manifest has no scenes")
    if selector is None:
        return 0, scenes[0]

    try:
        index = int(selector)
    except ValueError:
        index = -1
    if str(index) == selector and 0 <= index < len(scenes):
        return index, scenes[index]

    matches = []
    for index, scene in enumerate(scenes):
        sequence_name = str(scene.get("sequence_name", ""))
        qualified_name = f"{scene.get('category', '')}/{sequence_name}"
        if selector in (sequence_name, qualified_name):
            matches.append((index, scene))
    if len(matches) != 1:
        raise ValueError(
            f"Scene selector {selector!r} matched {len(matches)} scenes; "
            "use an exact sequence name, category/sequence name, or manifest index"
        )
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor_content(tensor: torch.Tensor) -> str:
    """Hash CPU tensor semantics independently of PyTorch ZIP serialization."""
    cpu_tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu_tensor.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(cpu_tensor.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(cpu_tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def save_tensor_atomically(tensor: torch.Tensor, destination: Path) -> None:
    """Never replace a prepared tensor until its serialized temporary file loads."""
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(tensor, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        loaded = torch.load(temporary_path, map_location="cpu", weights_only=True)
        if not torch.equal(loaded, tensor):
            raise RuntimeError("Temporary serialized tensor does not round-trip exactly")
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def import_local_preprocessor(quant_vggt_root: Path):
    quant_vggt_root = quant_vggt_root.resolve()
    load_fn_path = quant_vggt_root / "vggt/utils/load_fn.py"
    if not load_fn_path.is_file():
        raise FileNotFoundError(f"Local VGGT preprocessing module not found: {load_fn_path}")

    sys.path.insert(0, str(quant_vggt_root))
    from vggt.utils.load_fn import load_and_preprocess_images

    implementation_path = Path(inspect.getfile(load_and_preprocess_images)).resolve()
    if implementation_path != load_fn_path.resolve():
        raise RuntimeError(
            "Imported a different VGGT preprocessing implementation: "
            f"{implementation_path} (expected {load_fn_path.resolve()})"
        )
    signature = inspect.signature(load_and_preprocess_images)
    if "mode" not in signature.parameters:
        raise RuntimeError("Local VGGT preprocessor does not expose pad/crop mode")
    return load_and_preprocess_images, implementation_path, signature


def pad_transform(original_width: int, original_height: int, target_size: int) -> dict[str, Any]:
    """Mirror the geometry used by the inspected local VGGT pad implementation."""
    effective_target = int(target_size / PATCH_SIZE) * PATCH_SIZE
    if original_width >= original_height:
        resized_width = effective_target
        resized_height = round(
            original_height * (resized_width / original_width) / PATCH_SIZE
        ) * PATCH_SIZE
    else:
        resized_height = effective_target
        resized_width = round(
            original_width * (resized_height / original_height) / PATCH_SIZE
        ) * PATCH_SIZE

    padding_height = effective_target - resized_height
    padding_width = effective_target - resized_width
    if padding_height < 0 or padding_width < 0:
        raise RuntimeError("Computed negative padding for VGGT pad preprocessing")

    pad_top = padding_height // 2
    pad_bottom = padding_height - pad_top
    pad_left = padding_width // 2
    pad_right = padding_width - pad_left
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    original_to_vggt = [
        [scale_x, 0.0, float(pad_left)],
        [0.0, scale_y, float(pad_top)],
        [0.0, 0.0, 1.0],
    ]
    vggt_to_original = [
        [1.0 / scale_x, 0.0, -pad_left / scale_x],
        [0.0, 1.0 / scale_y, -pad_top / scale_y],
        [0.0, 0.0, 1.0],
    ]
    return {
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "padding": {
            "left": pad_left,
            "right": pad_right,
            "top": pad_top,
            "bottom": pad_bottom,
            "value": 1.0,
        },
        "vggt_input_width": effective_target,
        "vggt_input_height": effective_target,
        "original_to_vggt_affine": original_to_vggt,
        "vggt_to_original_affine": vggt_to_original,
        "intrinsics_mapping": {
            "original_to_vggt": (
                "fx*=scale_x; fy*=scale_y; "
                "cx=cx*scale_x+padding.left; cy=cy*scale_y+padding.top"
            ),
            "vggt_to_original": (
                "fx/=scale_x; fy/=scale_y; "
                "cx=(cx-padding.left)/scale_x; cy=(cy-padding.top)/scale_y"
            ),
        },
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_json(manifest_path)
    scene_index, scene = select_scene(manifest, args.scene)

    sequence_name = str(scene["sequence_name"])
    category = str(scene["category"])
    split_frames = scene.get("splits", {}).get(args.split)
    if not isinstance(split_frames, list):
        raise ValueError(f"Scene does not define split {args.split!r}")
    expected_views = int(args.split.removeprefix("input_"))
    if len(split_frames) != expected_views:
        raise ValueError(
            f"Split {args.split} contains {len(split_frames)} frames, expected {expected_views}"
        )
    if len(set(split_frames)) != len(split_frames):
        raise ValueError(f"Split {args.split} contains duplicate frame numbers")

    heldout_frames = set(scene.get("splits", {}).get("heldout", []))
    overlap = heldout_frames.intersection(split_frames)
    if overlap:
        raise ValueError(f"Input split contains held-out frames: {sorted(overlap)}")

    selected_frames = scene.get("selected_frames", [])
    frame_records = {record["frame_number"]: record for record in selected_frames}
    missing_manifest_frames = [frame for frame in split_frames if frame not in frame_records]
    if missing_manifest_frames:
        raise ValueError(
            f"Input frames are absent from selected_frames: {missing_manifest_frames}"
        )

    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else (manifest_path.parent / str(manifest["name"])).resolve()
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Extracted dataset root does not exist: {dataset_root}")

    ordered_images = []
    image_paths = []
    for view_index, frame_number in enumerate(split_frames):
        record = frame_records[frame_number]
        relative_path = Path(record["image_path"])
        absolute_path = (dataset_root / relative_path).resolve()
        if not absolute_path.is_file():
            raise FileNotFoundError(f"RGB image does not exist: {absolute_path}")
        ordered_images.append(str(absolute_path))
        image_paths.append(
            {
                "view_index": view_index,
                "frame_number": frame_number,
                "image_name": absolute_path.name,
                "manifest_relative_path": relative_path.as_posix(),
                "original_rgb_path": str(absolute_path),
            }
        )
    if len(ordered_images) != expected_views:
        raise RuntimeError("Resolved RGB count changed unexpectedly")

    preprocessor, implementation_path, signature = import_local_preprocessor(
        args.quant_vggt_root.expanduser()
    )
    call_kwargs: dict[str, Any] = {"mode": "pad"}
    if "target_size" in signature.parameters:
        call_kwargs["target_size"] = REQUESTED_TARGET_SIZE
    tensor = preprocessor(ordered_images, **call_kwargs).cpu().contiguous()

    if tensor.ndim != 4 or tuple(tensor.shape[:2]) != (expected_views, 3):
        raise RuntimeError(f"Unexpected VGGT input tensor shape: {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("VGGT input tensor contains non-finite values")

    transforms = []
    for image_path in ordered_images:
        with Image.open(image_path) as image:
            original_width, original_height = image.size
        transform = pad_transform(
            original_width, original_height, REQUESTED_TARGET_SIZE
        )
        if (transform["vggt_input_height"], transform["vggt_input_width"]) != tuple(
            tensor.shape[-2:]
        ):
            raise RuntimeError(
                "Recorded preprocessing geometry does not match the actual tensor shape"
            )
        transforms.append(transform)

    output_dir = args.output_root.expanduser().resolve() / sequence_name / args.split
    output_files = {
        "tensor": output_dir / "input_tensor.pt",
        "image_paths": output_dir / "image_paths.json",
        "preprocess": output_dir / "preprocess_meta.json",
        "experiment": output_dir / "experiment_meta.json",
    }
    existing = [str(path) for path in output_files.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to replace prepared artifacts without --force: " + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    save_tensor_atomically(tensor, output_files["tensor"])
    tensor_sha256 = sha256_file(output_files["tensor"])
    tensor_content_sha256 = sha256_tensor_content(tensor)

    image_paths_document = {
        "ordering": "manifest split order",
        "dataset_root": str(dataset_root),
        "images": image_paths,
    }
    preprocess_document = {
        "implementation": str(implementation_path),
        "implementation_sha256": sha256_file(implementation_path),
        "function": "vggt.utils.load_fn.load_and_preprocess_images",
        "mode": "pad",
        "requested_target_size": REQUESTED_TARGET_SIZE,
        "effective_target_size": int(tensor.shape[-1]),
        "resize_resampling": "PIL.Image.Resampling.BICUBIC",
        "tensor_value_range": [0.0, 1.0],
        "coordinate_convention": (
            "Affine matrices encode the standard resize-and-pad mapping used for "
            "camera intrinsics; original RGB pixels are never cropped."
        ),
        "views": [
            {
                **image_paths[index],
                **transform,
            }
            for index, transform in enumerate(transforms)
        ],
    }
    creation_time = datetime.now().astimezone().isoformat(timespec="seconds")
    experiment_document = {
        "dataset_name": manifest.get("name"),
        "dataset_version": manifest.get("version"),
        "manifest_path": str(manifest_path),
        "manifest_scene_index": scene_index,
        "category": category,
        "sequence_name": sequence_name,
        "split": args.split,
        "frame_numbers": split_frames,
        "image_names": [entry["image_name"] for entry in image_paths],
        "vggt_input_shape": list(tensor.shape),
        "vggt_input_dtype": str(tensor.dtype),
        "preprocessing_mode": "pad",
        "requested_target_size": REQUESTED_TARGET_SIZE,
        "input_tensor_path": str(output_files["tensor"]),
        "input_tensor_sha256": tensor_sha256,
        "input_tensor_content_sha256": tensor_content_sha256,
        "input_tensor_content_sha256_contract": (
            "SHA-256 of contiguous CPU raw bytes prefixed by torch dtype and JSON shape, "
            "each separated by a NUL byte."
        ),
        "reuse_contract": (
            "Full VGGT and every QuantVGGT variant must load this exact tensor file; "
            "they must not independently preprocess the source RGB images."
        ),
        "heldout_frames_excluded": True,
        "date": creation_time,
        "created_at": creation_time,
    }

    write_json(output_files["image_paths"], image_paths_document)
    write_json(output_files["preprocess"], preprocess_document)
    write_json(output_files["experiment"], experiment_document)

    print(f"Final tensor shape: {tuple(tensor.shape)}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Output directory: {output_dir}")
    print("Experiment metadata:")
    print(json.dumps(experiment_document, indent=2))
    print("Preprocessing metadata:")
    print(json.dumps(preprocess_document, indent=2))


if __name__ == "__main__":
    main()

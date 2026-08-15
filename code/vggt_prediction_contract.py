#!/usr/bin/env python3
"""Shared constants and deterministic hashing for frozen VGGT predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


SCENE = "602_93503_186945"
CATEGORY = "toilet"
SPLIT = "input_12"
FRAME_NUMBERS = np.asarray(
    [1, 22, 36, 56, 77, 91, 112, 126, 147, 167, 181, 202], dtype=np.int64
)
INPUT_SHAPE = (12, 3, 518, 518)
INPUT_SHA256 = "41fdcacfa3d403f6aa76dda70d22d845c44ce295eb61fb195bf77a98d744fb87"

ARRAY_ORDER = (
    "pose_enc",
    "depth",
    "depth_conf",
    "intrinsic",
    "extrinsic",
    "world_points",
    "world_points_conf",
    "world_points_from_depth",
    "frame_numbers",
)

def expected_shapes(view_count: int, height: int, width: int) -> dict[str, tuple[int, ...]]:
    return {
        "pose_enc": (view_count, 9),
        "depth": (view_count, height, width, 1),
        "depth_conf": (view_count, height, width),
        "intrinsic": (view_count, 3, 3),
        "extrinsic": (view_count, 3, 4),
        "world_points": (view_count, height, width, 3),
        "world_points_conf": (view_count, height, width),
        "world_points_from_depth": (view_count, height, width, 3),
        "frame_numbers": (view_count,),
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash array name, JSON shape, dtype name, and C-contiguous bytes in fixed order."""
    digest = hashlib.sha256()
    for name in ARRAY_ORDER:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def require_expected_shapes(arrays: Mapping[str, np.ndarray]) -> None:
    missing = [name for name in ARRAY_ORDER if name not in arrays]
    if missing:
        raise ValueError(f"Missing required prediction arrays: {missing}")
    depth_shape = tuple(arrays["depth"].shape)
    if len(depth_shape) != 4 or depth_shape[-1] != 1:
        raise ValueError(f"depth must have shape (views,height,width,1), got {depth_shape}")
    expected_shapes_for_input = expected_shapes(*depth_shape[:3])
    wrong = {
        name: {"actual": tuple(arrays[name].shape), "expected": expected}
        for name, expected in expected_shapes_for_input.items()
        if tuple(arrays[name].shape) != expected
    }
    if wrong:
        raise ValueError(f"Unexpected prediction array shapes: {wrong}")

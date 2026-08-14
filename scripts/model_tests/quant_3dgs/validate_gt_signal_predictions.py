#!/usr/bin/env python3
"""Validate one standardized GT-signal predictions.npz without historical metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vggt_prediction_contract import ARRAY_ORDER, prediction_content_sha256, require_expected_shapes, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--frame-numbers", required=True)
    args = parser.parse_args()
    frames = [int(x) for x in args.frame_numbers.split(",")]
    root = args.prediction_dir.resolve()
    npz_path = root / "predictions.npz"
    meta_path = root / "inference_meta.json"
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in ARRAY_ORDER}
    require_expected_shapes(arrays)
    for name in ARRAY_ORDER:
        if name != "frame_numbers" and not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"Non-finite values in {name}")
    if arrays["frame_numbers"].tolist() != frames:
        raise RuntimeError("Frame order mismatch")
    meta = json.loads(meta_path.read_text())
    if meta["prediction_content_sha256"] != prediction_content_sha256(arrays):
        raise RuntimeError("Prediction content SHA mismatch")
    if meta["predictions_npz_sha256"] != sha256_file(npz_path):
        raise RuntimeError("Prediction file SHA mismatch")
    print(json.dumps({"PASS": True, "shapes": {k: list(v.shape) for k, v in arrays.items()}, "prediction_content_sha256": meta["prediction_content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

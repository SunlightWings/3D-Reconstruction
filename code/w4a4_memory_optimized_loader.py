#!/usr/bin/env python3
"""Behavior-preserving W4A4 loader with bounded temporary lifetimes.

This module does not alter QuantVGGT source or quantization arithmetic. Its key
difference is running the one-time reparameterization under ``torch.no_grad()``,
which prevents WeightQuantizer.scale buffers from retaining conversion autograd
graphs. It also loads QS files explicitly on CPU and releases each dictionary
after copying it into the model.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
CV_ROOT = REPOSITORY_ROOT.parent
QUANTVGGT_ROOT = CV_ROOT / "QuantVGGT"
BASE_MODEL_PATH = QUANTVGGT_ROOT / "VGGT-1B/model_tracker_fixed_e20.pt"
QS_ROOT = QUANTVGGT_ROOT / "evaluation/outputs/w4a4/a44_model_tracker_fixed_e20.pt_sym"
FRAME_QS_PATH = QS_ROOT / "qs_frame_parameters_total.pth"
GLOBAL_QS_PATH = QS_ROOT / "qs_global_parameters_total.pth"
AUDIT_ROOT = REPOSITORY_ROOT / "outputs/model_tests/quant_3dgs/w4a4_memory_audit"
DEFAULT_INPUT = AUDIT_ROOT / "input_3_224.pt"
EQUIVALENCE_JSON = AUDIT_ROOT / "numerical_equivalence.json"


def _configure_w4a4():
    from evaluation.quarot.args_utils import get_config

    config = get_config()
    config.output_dir = str(QUANTVGGT_ROOT / "evaluation/outputs")
    config.update_from_args(
        wbit=4,
        abit=4,
        not_smooth=False,
        not_rot=False,
        lwc=True,
        lac=True,
        rv=False,
        model_id=str(BASE_MODEL_PATH),
        exp_name="a44",
    )
    config.exp_dir = str(QS_ROOT)
    return config


def _load_base_model(device: str):
    sys.path.insert(0, str(QUANTVGGT_ROOT))
    from vggt.models.vggt import VGGT

    model = VGGT()
    state_dict = torch.load(BASE_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    return model.eval().to(device)


def load_original_w4a4(device: str = "cuda:0"):
    """Load through the unmodified external QuantVGGT resume path."""
    model = _load_base_model(device)
    from evaluation.quarot.utils import quarot_smooth_quant_model

    config = _configure_w4a4()
    quarot_smooth_quant_model(
        config,
        model,
        calib_data=None,
        wbit=4,
        abit=4,
        resume_qs=True,
        exp_name="a44",
    )
    return model.eval().to(device)


def _copy_qs_checkpoint(blocks, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    devices = sorted(
        {str(tensor.device) for state in checkpoint.values() for tensor in state.values()}
    )
    for index in range(len(checkpoint.keys())):
        blocks[index].load_state_dict(checkpoint[index], strict=False)
    tensor_count = sum(len(state) for state in checkpoint.values())
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for state in checkpoint.values()
        for tensor in state.values()
    )
    del checkpoint
    gc.collect()
    return {
        "path": str(checkpoint_path),
        "source_devices": devices,
        "tensor_count": tensor_count,
        "tensor_bytes": tensor_bytes,
        "tensor_mib": tensor_bytes / (1024**2),
    }


def load_optimized_w4a4(device: str = "cuda:0"):
    """Load W4A4 without retaining one-time conversion autograd graphs."""
    model = _load_base_model(device)
    from evaluation.quarot.utils import (
        after_resume_qs,
        quantize_linear,
        set_ignore_quantize,
    )

    config = _configure_w4a4()
    set_ignore_quantize(model)
    quantize_linear(model, args=config)
    frame_info = _copy_qs_checkpoint(model.aggregator.frame_blocks, FRAME_QS_PATH)
    global_info = _copy_qs_checkpoint(model.aggregator.global_blocks, GLOBAL_QS_PATH)

    # reparameterize() computes the same values either way. The resume path is
    # inference-only, so recording an autograd graph for this one-time transform
    # is dead state. no_grad changes lifetime only, not the arithmetic performed.
    with torch.no_grad():
        after_resume_qs(model)

    gc.collect()
    torch.cuda.empty_cache()
    model._memory_optimized_loader_info = {
        "optimization": "after_resume_qs under torch.no_grad; streamed CPU QS loading",
        "frame_checkpoint": frame_info,
        "global_checkpoint": global_info,
    }
    return model.eval()


def graph_retention_summary(model: torch.nn.Module) -> dict[str, Any]:
    roots = []
    for name, buffer in model.named_buffers():
        if buffer.grad_fn is not None:
            roots.append(
                {
                    "name": name,
                    "dtype": str(buffer.dtype),
                    "device": str(buffer.device),
                    "shape": list(buffer.shape),
                    "grad_fn": type(buffer.grad_fn).__name__,
                    "bytes": buffer.numel() * buffer.element_size(),
                }
            )
    return {
        "buffer_roots_with_grad_fn": len(roots),
        "root_buffer_bytes": sum(root["bytes"] for root in roots),
        "roots": roots,
    }


def _move_predictions_to_cpu(predictions: dict[str, Any], images: torch.Tensor):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    selected = {
        "pose_enc": predictions["pose_enc"],
        "depth": predictions["depth"],
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "world_points": predictions["world_points"],
        "depth_conf": predictions["depth_conf"],
        "world_points_conf": predictions["world_points_conf"],
    }
    return {key: value.detach().cpu() for key, value in selected.items()}


def run_variant(variant: str, input_path: Path, output_path: Path, device: str) -> None:
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    model = (
        load_original_w4a4(device)
        if variant == "original"
        else load_optimized_w4a4(device)
    )
    gc.collect()
    torch.cuda.empty_cache()
    persistent_allocated = torch.cuda.memory_allocated(device)
    persistent_reserved = torch.cuda.memory_reserved(device)
    graph_summary = graph_retention_summary(model)

    images = torch.load(input_path, map_location="cpu").to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        predictions = model(images)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    outputs = _move_predictions_to_cpu(predictions, images)
    metadata = {
        "variant": variant,
        "input_path": str(input_path),
        "input_shape": list(images.shape),
        "persistent_allocated_bytes": persistent_allocated,
        "persistent_allocated_mib": persistent_allocated / (1024**2),
        "persistent_reserved_bytes": persistent_reserved,
        "persistent_reserved_mib": persistent_reserved / (1024**2),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "inference_time_seconds": elapsed,
        "graph_retention": graph_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "outputs": outputs}, output_path)
    print(json.dumps(metadata, indent=2))


def compare_outputs(original_path: Path, optimized_path: Path) -> dict[str, Any]:
    original = torch.load(original_path, map_location="cpu")
    optimized = torch.load(optimized_path, map_location="cpu")
    comparisons = {}
    for key, original_tensor in original["outputs"].items():
        optimized_tensor = optimized["outputs"][key]
        if original_tensor.shape != optimized_tensor.shape:
            raise ValueError(f"Shape mismatch for {key}")
        difference = (original_tensor.double() - optimized_tensor.double()).abs()
        denominator = original_tensor.double().abs().clamp_min(1e-12)
        relative = difference / denominator
        comparisons[key] = {
            "shape": list(original_tensor.shape),
            "bitwise_identical": bool(torch.equal(original_tensor, optimized_tensor)),
            "max_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "max_relative_difference": float(relative.max()),
            "mean_relative_difference": float(relative.mean()),
        }
    return {
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_path": original["metadata"]["input_path"],
        "original_metadata": original["metadata"],
        "optimized_metadata": optimized["metadata"],
        "comparisons": comparisons,
        "all_outputs_bitwise_identical": all(
            comparison["bitwise_identical"] for comparison in comparisons.values()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("original", "optimized"))
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def controller_main(args: argparse.Namespace) -> None:
    if not args.input_path.is_file():
        raise FileNotFoundError(args.input_path)
    outputs = {
        variant: AUDIT_ROOT / f"equivalence_{variant}_outputs.pt"
        for variant in ("original", "optimized")
    }
    for variant, output_path in outputs.items():
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--worker",
            variant,
            "--input-path",
            str(args.input_path),
            "--output-path",
            str(output_path),
            "--device",
            args.device,
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"{variant} worker failed: {completed.returncode}")
    report = compare_outputs(outputs["original"], outputs["optimized"])
    EQUIVALENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    EQUIVALENCE_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.output_path is None:
            raise ValueError("--output-path is required for worker mode")
        run_variant(args.worker, args.input_path, args.output_path, args.device)
    else:
        controller_main(args)


if __name__ == "__main__":
    main()

"""Load a QuantVGGT model at an arbitrary calibrated bit-width.

Generalises `code/w4a4_memory_optimized_loader.py`, which hard-codes W4A4 and the
`a44` experiment name, so the same proven load path serves W2A4 (and any other
bit-width whose qs_*_parameters_total.pth pair exists).

Behaviour is deliberately identical to the W4A4 loader: QS checkpoints are read on
CPU and released per block, and the one-time `after_resume_qs()` reparameterisation
runs under `torch.no_grad()` so it does not retain conversion autograd graphs. Only
the bit-width, experiment name and checkpoint directory are parameterised.

Note on `rv`: the W4A4 loader passes `rv=False` while QuantVGGT's own run_co3d.py
defaults it to True. The flag is assigned in `quarot_linear.py` and never read
anywhere in the codebase, so the two are equivalent; `rv=False` is kept here to match
the existing loader exactly.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import torch

CV_ROOT = Path("/home/utn/luli38se/cv")
QUANTVGGT_ROOT = CV_ROOT / "QuantVGGT"
BASE_MODEL_PATH = QUANTVGGT_ROOT / "VGGT-1B/model_tracker_fixed_e20.pt"

# Known calibrated variants. `exp_name` and `qs_dir` must match what produced the
# checkpoints -- for w4a4 that is the authors' HuggingFace release, for w2a4 the
# local run of code/quantization/calibrate_w2a4.py.
VARIANTS: dict[str, dict[str, Any]] = {
    "w4a4": {
        "wbit": 4, "abit": 4, "exp_name": "a44",
        "qs_dir": QUANTVGGT_ROOT / "evaluation/outputs/w4a4/a44_model_tracker_fixed_e20.pt_sym",
        "provenance": "downloaded from the QuantVGGT authors' HuggingFace release",
    },
    "w3a3": {
        "wbit": 3, "abit": 3, "exp_name": "a33",
        "qs_dir": QUANTVGGT_ROOT / "evaluation/outputs/w3a3/a33_model_tracker_fixed_e20.pt_sym",
        "provenance": "calibrated by Sharma Poudel / Prabin (UTN) on their machine, artifacts transferred",
    },
    "w2a4": {
        "wbit": 2, "abit": 4, "exp_name": "a24",
        "qs_dir": QUANTVGGT_ROOT / "evaluation/outputs/w2a4/a24_model_tracker_fixed_e20.pt_sym",
        "provenance": "calibrated locally by code/quantization/calibrate_w2a4.py",
    },
}

# --- W4A4 ablation arms (see ongoing_logs.md; calibrated by calibrate_w2a4.py) ---
# Each removes ONE component of QuantVGGT's pipeline, holding bit-width at W4A4.
# `a44_local` is the fair baseline: the authors' `w4a4` was calibrated by them with
# unknown settings, so ablations must be compared against a locally-calibrated full
# pipeline, not against the download.
for _tag, _exp, _desc in [
    ("w4a4_local",    "a44_local",    "full pipeline, calibrated locally (ablation baseline)"),
    ("w4a4_no_rot",   "a44_norot",    "ablate QuaRot Hadamard rotation"),
    ("w4a4_no_smooth","a44_nosmooth", "ablate SmoothQuant per-channel scaling"),
    ("w4a4_no_lwc",   "a44_nolwc",    "ablate learned weight clipping"),
    ("w4a4_no_lac",   "a44_nolac",    "ablate learned activation clipping"),
    ("w4a4_rtn",      "a44_rtn",      "no smooth/rot/lwc/lac: round-to-nearest, no calibration"),
]:
    VARIANTS[_tag] = {
        "wbit": 4, "abit": 4, "exp_name": _exp,
        "qs_dir": QUANTVGGT_ROOT / f"evaluation/outputs/w4a4/{_exp}_model_tracker_fixed_e20.pt_sym",
        "provenance": f"W4A4 ablation: {_desc}",
    }



def _configure(variant: str):
    """Build the quant config for `variant`, preferring the config recorded at calibration.

    The four ablation switches (smooth / rot / lwc / lac) change the MODULE STRUCTURE, so a
    loader configured differently from the calibrator would silently drop tensors through
    load_state_dict(strict=False) and evaluate a model nobody calibrated. calibrate_w2a4.py
    writes ablation_config.json next to the artifacts; when present it is authoritative.
    """
    import json

    spec = dict(VARIANTS[variant])
    sys.path.insert(0, str(QUANTVGGT_ROOT))
    from evaluation.quarot.args_utils import get_config

    # defaults = the authors' full pipeline (all four components on)
    flags = {"not_smooth": False, "not_rot": False, "lwc": True, "lac": True}
    cfg_path = Path(spec["qs_dir"]) / "ablation_config.json"
    if cfg_path.is_file():
        rec = json.loads(cfg_path.read_text())
        flags = {k: rec[k] for k in flags if k in rec}
        spec["wbit"], spec["abit"] = rec.get("wbit", spec["wbit"]), rec.get("abit", spec["abit"])
        spec["calibration_performed"] = rec.get("calibration_performed", True)
        spec["provenance"] = rec.get("provenance", spec.get("provenance", ""))
    spec.setdefault("calibration_performed", True)
    spec["flags"] = flags

    config = get_config()
    config.output_dir = str(QUANTVGGT_ROOT / "evaluation/outputs")
    config.update_from_args(
        wbit=spec["wbit"], abit=spec["abit"],
        not_smooth=flags["not_smooth"], not_rot=flags["not_rot"],
        lwc=flags["lwc"], lac=flags["lac"], rv=False,
        model_id=str(BASE_MODEL_PATH), exp_name=spec["exp_name"],
    )
    config.exp_dir = str(spec["qs_dir"])
    return config, spec


def _load_base_model(device: str):
    sys.path.insert(0, str(QUANTVGGT_ROOT))
    from vggt.models.vggt import VGGT

    model = VGGT()
    state_dict = torch.load(BASE_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    return model.eval().to(device)


def _copy_qs_checkpoint(checkpoint_path: Path, blocks) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"QS checkpoint missing: {checkpoint_path}. For w2a4 this means calibration "
            "has not finished -- run code/quantization/calibrate_w2a4.py."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    for index in range(len(checkpoint.keys())):
        blocks[index].load_state_dict(checkpoint[index], strict=False)
    tensor_bytes = sum(t.numel() * t.element_size()
                       for state in checkpoint.values() for t in state.values())
    info = {"path": str(checkpoint_path), "tensor_mib": tensor_bytes / (1024 ** 2)}
    del checkpoint
    gc.collect()
    return info


def load_quantized(variant: str, device: str = "cuda:0"):
    """Load a calibrated quantized VGGT. `variant` is a key of VARIANTS, e.g. 'w2a4'."""
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
    config, spec = _configure(variant)
    model = _load_base_model(device)
    from evaluation.quarot.utils import after_resume_qs, quantize_linear, set_ignore_quantize

    set_ignore_quantize(model)
    quantize_linear(model, args=config)
    if spec["calibration_performed"]:
        frame_info = _copy_qs_checkpoint(Path(spec["qs_dir"]) / "qs_frame_parameters_total.pth",
                                         model.aggregator.frame_blocks)
        global_info = _copy_qs_checkpoint(Path(spec["qs_dir"]) / "qs_global_parameters_total.pth",
                                          model.aggregator.global_blocks)
    else:
        # Round-to-nearest ablation: nothing was learned, so there is nothing to load.
        frame_info = global_info = {"path": None, "note": "no calibration performed (RTN)"}
    with torch.no_grad():
        after_resume_qs(model)
    gc.collect()
    torch.cuda.empty_cache()
    model._quant_loader_info = {
        "variant": variant, "wbit": spec["wbit"], "abit": spec["abit"],
        "smooth": not spec["flags"]["not_smooth"], "rot": not spec["flags"]["not_rot"],
        "lwc": spec["flags"]["lwc"], "lac": spec["flags"]["lac"],
        "calibration_performed": spec["calibration_performed"],
        "provenance": spec["provenance"],
        "frame_checkpoint": frame_info, "global_checkpoint": global_info,
    }
    return model.eval()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    m = load_quantized(a.variant, a.device)
    print(f"loaded {a.variant}: {m._quant_loader_info}")

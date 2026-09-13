"""Calibrate QuantVGGT at W2A4 (2-bit weights, 4-bit activations).

WHY THIS EXISTS
---------------
Every downstream comparison in this project so far is Full VGGT vs W4A4, and at that
bit-width the rendering difference on the object is not measurable: +0.1115 dB with
p = 0.174 and a 95% CI spanning zero (ongoing_logs.md #020). A confidence predictor has
nothing to recover from a gap that cannot be resolved. Pushing to 2-bit weights is the
cheapest way to find out whether a measurable gap exists at all.

WHAT THIS IS NOT
----------------
It is NOT a reuse of the W4A4 parameters at a different bit-width. Those live in
QuantVGGT/evaluation/outputs/w4a4/ and were downloaded from the authors' HuggingFace
repo, not calibrated here. Learned weight/activation clipping is bit-width specific;
reusing it at 2 bits would produce a model that looks quantized but whose clipping
thresholds were fitted for a different quantizer. This runs the real calibration.

WHAT IT DOES
------------
Loads the frozen VGGT-1B checkpoint and the authors' filtered CO3D calibration set, then
runs QuantVGGT's quarot smooth+rotation pipeline with w_bits=2, a_bits=4 and learned
weight/activation clipping. Calibration is block-wise (one aggregator block on the GPU at
a time), so it fits in 20 GB. Output is a new qs_*_parameters_total.pth pair under
evaluation/outputs/w2a4/.

QuantVGGT's own run_co3d.py whitelists only w4a4/w6a6/w4a8/w8a8, so this driver calls the
calibration entry point directly rather than patching their repo.
"""
import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUANTVGGT = PROJECT_ROOT / "QuantVGGT"
DEFAULT_CALIB = PROJECT_ROOT / "models/QuantVGGT/calib_data.pt"

sys.path.insert(0, str(QUANTVGGT))
sys.path.insert(0, str(QUANTVGGT / "evaluation"))

import torch  # noqa: E402

DEFAULT_MODEL = QUANTVGGT / "VGGT-1B" / "model_tracker_fixed_e20.pt"


def main():
    ap = argparse.ArgumentParser(description="Calibrate QuantVGGT at an arbitrary bit-width.")
    ap.add_argument("--wbit", type=int, default=2)
    ap.add_argument("--abit", type=int, default=4)
    ap.add_argument("--exp-name", default="a24")
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calib-path", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--output-dir", type=Path, default=QUANTVGGT / "evaluation" / "outputs")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs (default 15)")
    ap.add_argument("--nsamples", type=int, default=None, help="limit calibration samples")
    ap.add_argument("--dry-run", action="store_true", help="set up and report, then exit before calibrating")
    a = ap.parse_args()

    if not a.model_path.is_file():
        raise SystemExit(f"VGGT checkpoint missing: {a.model_path}")
    if not a.calib_path.is_file():
        raise SystemExit(f"Calibration set missing: {a.calib_path}")

    from vggt.models.vggt import VGGT  # noqa: E402
    from quarot.args_utils import get_config  # noqa: E402
    from quarot.utils import quarot_smooth_quant_model  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}")
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}  "
              f"total={torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

    print(f"loading VGGT from {a.model_path} ...")
    t0 = time.time()
    model = VGGT()
    state = torch.load(a.model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"loading calibration set from {a.calib_path} ...")
    t0 = time.time()
    calib_data = torch.load(a.calib_path, map_location="cpu")
    if a.nsamples is not None:
        calib_data = calib_data[: a.nsamples]
    print(f"  {len(calib_data)} samples in {time.time()-t0:.1f}s")

    config = get_config()
    config.output_dir = str(a.output_dir)
    config.update_from_args(
        wbit=a.wbit, abit=a.abit, model_id=str(a.model_path),
        not_smooth=False, not_rot=False, lwc=True, lac=True, rv=True,
        exp_name=a.exp_name,
    )
    config.update_nsamples(len(calib_data))
    if a.epochs is not None:
        config.epochs = a.epochs

    print(f"W-bit={config.w_bits}  A-bit={config.a_bits}  LWC={config.lwc}  LAC={config.lac}  "
          f"epochs={config.epochs}  nsamples={config.nsamples}  cali_bsz={config.cali_bsz}")
    print(f"exp_dir={config.exp_dir}")

    if a.dry_run:
        print("dry run: setup OK, exiting before calibration")
        return

    t0 = time.time()
    quarot_smooth_quant_model(
        config, model, calib_data,
        wbit=a.wbit, abit=a.abit,
        resume_qs=False,              # <- run the real calibration, do not load W4A4
        exp_name=a.exp_name,
        model_id=str(a.model_path),
    )
    mins = (time.time() - t0) / 60.0
    print(f"calibration finished in {mins:.1f} min")
    produced = sorted(Path(config.exp_dir).glob("qs_*_parameters_total.pth"))
    for p in produced:
        print(f"  wrote {p}  ({p.stat().st_size/2**30:.2f} GiB)")
    if not produced:
        raise SystemExit("calibration produced no qs_*_parameters_total.pth -- treat as FAILED")


if __name__ == "__main__":
    main()

#!/bin/bash
# W3A3 vs W4A4 vs full precision: the same three-stage comparison every other arm went through.
#
# The pipeline is already arm-generic -- `w3a3` is registered in quant_loader.py and the
# downstream scorer takes an arbitrary --arms list -- so nothing here is new code, only the
# right invocation order with the guards that caught real bugs before:
#
#   stage 0  refuse to start unless BOTH totals and ablation_config.json are present. A missing
#            config would make _configure() guess the module structure and silently evaluate a
#            model nobody calibrated (load_state_dict(strict=False) drops the mismatched tensors).
#   stage 1  inference, 40 scenes. Each scene re-checks the input semantic SHA against the `full`
#            arm and aborts on any difference, so a preprocessing mismatch on the teammate's side
#            cannot produce an incomparable prediction.
#   stage 2  geometry only, no GPU training: pose / focal / point error vs full and vs CO3D GT.
#            This is where a 3-bit model is EXPECTED to separate, and #039/#043 say the camera
#            columns are the ones that predict rendering damage.
#   stage 3  downstream 3DGS at 7000 iterations + paired tests over 40 scenes. `full` and `w4a4`
#            are in PREBUILT_ARMS and are skipped by the resume check, so only w3a3's 40 scenes
#            train: ~1.8 min each from the n=40 ablation run (433 min / 240 scene-arms) = ~1.3 h.
#   stage 4  order-ensembling spread (#045), ~8.8 s/scene = ~6 min. Tests whether the detector
#            that works at 4 bits still tracks pose error at 3 bits -- a second bit-width is the
#            cheapest generalisation evidence the proposal can get.
#
# Total ~1.6 h, so it fits any time before the 00:35 shutdown.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
QS=/home/utn/luli38se/cv/QuantVGGT/evaluation/outputs/w3a3/a33_model_tracker_fixed_e20.pt_sym
PY=/home/utn/luli38se/cv/.venv/bin/python
mkdir -p "$LOG"

echo "=== stage 0: artifact check ($(date +%F' '%H:%M)) ==="
missing=0
for f in qs_frame_parameters_total.pth qs_global_parameters_total.pth ablation_config.json; do
  if [ -s "$QS/$f" ]; then
    echo "  ok   $f  ($(stat -c %s "$QS/$f") bytes)"
  else
    echo "  MISSING $f"; missing=1
  fi
done
if [ "$missing" = 1 ]; then
  echo "ABORT: W3A3 artifacts incomplete -- not running a guessed configuration."
  exit 1
fi
echo "  base checkpoint sha256 (must match the teammate's calibration input):"
sha256sum /home/utn/luli38se/cv/QuantVGGT/VGGT-1B/model_tracker_fixed_e20.pt
cat "$QS/ablation_config.json"

echo "=== stage 1: inference, 40 scenes ($(date +%H:%M)) ==="
$PY -u "$SP/run_w2a4_inference.py" --variant w3a3 --scenes all \
    2>&1 | tee "$LOG/infer40_w3a3.log"
if ! grep -q "40/40 scenes" "$LOG/infer40_w3a3.log"; then
  echo "ABORT: inference did not complete all 40 scenes -- downstream would score a partial arm."
  exit 1
fi

echo "=== stage 2: geometry, 40 scenes ($(date +%H:%M)) ==="
$PY -u "$SP/ablation_compare.py" --scenes all --tag w3a3_40 \
    --variants full w4a4 w4a4_rtn w3a3 2>&1 | tee "$LOG/geometry_40_w3a3.log"

echo "=== stage 3: downstream 3DGS, 40 scenes ($(date +%H:%M)) ==="
$PY -u "$SP/run_w2a4_downstream.py" --scenes all --iterations 7000 \
    --arms full w4a4 w3a3 2>&1 | tee "$LOG/downstream_40_w3a3.log"

echo "=== stage 4: order-ensembling spread (#045) on w3a3 ($(date +%H:%M)) ==="
$PY -u "$SP/pose_uncertainty.py" --variant w3a3 --perms 4 --scenes all \
    2>&1 | tee "$LOG/pose_uncertainty_w3a3.log"

echo "W3A3 CHAIN COMPLETE ($(date +%F' '%H:%M))"

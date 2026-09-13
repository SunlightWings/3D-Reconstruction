#!/bin/bash
# Stage 2 of the W4A4 ablation: everything after calibration.
#
# For every arm that actually has usable artifacts:
#   1. VGGT inference on the downstream groups          (~2 s/scene)
#   2. geometry error vs full precision                 (seconds, CPU)
#   3. 3DGS train + render + PSNR/SSIM/LPIPS            (~12 min/arm on the 8-scene subset)
#
# Both levels are measured on purpose. Geometry is the sensitive instrument -- at W4A4 the
# full-vs-quant rendering difference is not resolvable (p=0.174, ongoing_logs.md #020) --
# but the ablation arms are far more degraded than w4a4 (RTN has 18x the camera error), so a
# downstream gap may well be resolvable there, and the proposal's claim is a downstream one.
# Geometry says WHICH component matters; 3DGS says whether it matters to renders.
#
# Arms are DETECTED, not hardcoded: an arm counts as available when its calibration produced
# both totals, or when it is a round-to-nearest arm that legitimately has none. This lets the
# stage run on whatever is finished rather than aborting on a missing arm, which matters when
# a scheduled reboot can interrupt a 3.5 h calibration.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
OUTROOT=/home/utn/luli38se/cv/QuantVGGT/evaluation/outputs/w4a4
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
SCENES="${1:-subset}"
mkdir -p "$LOG"

declare -A EXP=( [w4a4_local]=a44_local [w4a4_no_rot]=a44_norot [w4a4_no_smooth]=a44_nosmooth
                 [w4a4_no_lwc]=a44_nolwc [w4a4_no_lac]=a44_nolac [w4a4_rtn]=a44_rtn )

AVAIL=""
for arm in "${!EXP[@]}"; do
  d="$OUTROOT/${EXP[$arm]}_model_tracker_fixed_e20.pt_sym"
  if [ -f "$d/qs_frame_parameters_total.pth" ] && [ -f "$d/qs_global_parameters_total.pth" ]; then
    AVAIL="$AVAIL $arm"
  elif [ -f "$d/ablation_config.json" ] && grep -q '"calibration_performed": false' "$d/ablation_config.json" 2>/dev/null; then
    AVAIL="$AVAIL $arm"   # round-to-nearest: no artifacts by design
  else
    echo "skipping $arm: no usable artifacts at $d"
  fi
done
AVAIL=$(echo $AVAIL | tr ' ' '\n' | sort | tr '\n' ' ')
echo "=== stage 2 on arms:$AVAIL ($(date +%H:%M)) ==="

for arm in $AVAIL; do
  echo "=== inference $arm ($(date +%H:%M)) ==="
  python3 -u "$SP/run_w2a4_inference.py" --variant "$arm" --scenes "$SCENES" \
      >> "$LOG/infer_${arm}.log" 2>&1 || echo "  inference FAILED for $arm"
done

echo "=== geometry comparison ($(date +%H:%M)) ==="
python3 -u "$SP/ablation_compare.py" --scenes "$SCENES" --tag "$SCENES" \
    --variants w4a4 $AVAIL w2a4 2>&1 | tee "$LOG/ablation_geometry.log"

echo "=== downstream 3DGS ($(date +%H:%M)) ==="
python3 -u "$SP/run_w2a4_downstream.py" --scenes "$SCENES" --iterations 7000 \
    --arms full w4a4 $AVAIL 2>&1 | tee "$LOG/ablation_downstream.log"

echo "ABLATION STAGE 2 COMPLETE ($(date +%H:%M))"

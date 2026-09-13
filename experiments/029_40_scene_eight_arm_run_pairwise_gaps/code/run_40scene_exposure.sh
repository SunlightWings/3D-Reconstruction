#!/bin/bash
# The measurement-power run: all ablation arms, 40 scenes, exposure-corrected metrics.
#
# WHY: #027 left the rendering claim unresolved rather than refuted. Two separate defects,
# both addressed here.
#
#   1. Exposure confound. #022 measured 3.542 dB of the oracle's score as pure auto-exposure
#      mismatch, and showed raw PSNR ranks GT geometry (14.828) BELOW copying the nearest
#      training photograph (15.414) -- a metric that does that is not ranking geometry.
#      Corrected, the oracle reaches 18.370 and the ordering repairs itself. run_w2a4_downstream
#      now fits one gain+bias per image and applies it to PSNR, SSIM and LPIPS alike.
#
#   2. Power. At n=8 the exposure-corrected full-minus-rtn effect is +1.64 dB with CI
#      [-1.72,+5.01] -- a LARGE effect measured badly, not a null. Contrast full-minus-no_rot
#      at -0.07 with CI [-0.60,+0.46], which is a genuine null. Distinguishing those two cases
#      is exactly what n=40 buys; the n=40 minimum detectable effect is ~0.23 dB (#020).
#
# COST: `full` and `w4a4` already have all 40 scenes trained at 7000 iterations and are
# skipped by the resume check, so only the six ablation arms train -- 200 scene-arm runs at
# ~2.3 min = ~7.7 h. Fits the ~20 h window (machine shuts down 00:35 nightly, #027).
#
# Chained behind the running job rather than launched now: one GPU, and a 3DGS run sharing it
# with a calibration would corrupt both timings.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs

echo "=== waiting for nolac+stage2 to finish ($(date +%F' '%H:%M)) ==="
while pgrep -f "run_nolac_then_stage2.sh" > /dev/null; do sleep 120; done
echo "=== predecessor done ($(date +%H:%M)) ==="

ARMS="full w4a4 w4a4_local w4a4_no_rot w4a4_no_smooth w4a4_no_lwc w4a4_rtn"
NOLAC=/home/utn/luli38se/cv/QuantVGGT/evaluation/outputs/w4a4/a44_nolac_model_tracker_fixed_e20.pt_sym
if [ -f "$NOLAC/qs_global_parameters_total.pth" ]; then
  ARMS="$ARMS w4a4_no_lac"
else
  echo "!! w4a4_no_lac has no totals -- running seven arms, NOT substituting a guess"
fi
echo "arms: $ARMS"

# inference for every arm across all 40 scenes (~2 s/scene, cheap)
for arm in $ARMS; do
  [ "$arm" = "full" ] && continue
  echo "=== inference $arm on 40 scenes ($(date +%H:%M)) ==="
  python3 -u "$SP/run_w2a4_inference.py" --variant "$arm" --scenes all \
      >> "$LOG/infer40_${arm}.log" 2>&1 || echo "  inference FAILED for $arm"
done

echo "=== geometry, 40 scenes ($(date +%H:%M)) ==="
python3 -u "$SP/ablation_compare.py" --scenes all --tag all40 \
    --variants $ARMS w2a4 2>&1 | tee "$LOG/ablation_geometry_40.log"

echo "=== downstream 3DGS, 40 scenes ($(date +%H:%M)) ==="
python3 -u "$SP/run_w2a4_downstream.py" --scenes all --iterations 7000 \
    --arms $ARMS 2>&1 | tee "$LOG/ablation_downstream_40.log"

echo "40-SCENE EXPOSURE-CORRECTED RUN COMPLETE ($(date +%F' '%H:%M))"

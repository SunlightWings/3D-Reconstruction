#!/bin/bash
# W4A4 ablation study: calibrate one arm per component of QuantVGGT's pipeline.
#
# Each arm removes exactly ONE component, holding the bit-width at W4A4. `a44_local` is
# the baseline: the shipped `w4a4` parameters came from the authors' HuggingFace release
# with unknown calibration settings, so ablations must be compared against a locally
# calibrated FULL pipeline, not against the download. Comparing an ablation to the
# download would confound "component removed" with "different calibration".
#
# a44_rtn (all four components off) needs no calibration and is built separately in
# seconds -- QuantVGGT skips calibration when smooth/lwc/lac are all disabled.
#
# Cost: ~3.5 h per arm on a 20 GB GPU, 5 arms => ~17.5 h. Resumable: an arm whose
# qs_*_parameters_total.pth already exist is skipped.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
OUTROOT=/home/utn/luli38se/cv/QuantVGGT/evaluation/outputs/w4a4
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
mkdir -p "$LOG"

run () {
  exp=$1; shift
  dir="$OUTROOT/${exp}_model_tracker_fixed_e20.pt_sym"
  if [ -f "$dir/qs_frame_parameters_total.pth" ] && [ -f "$dir/qs_global_parameters_total.pth" ]; then
    echo "=== SKIP $exp (already calibrated) ==="
    return
  fi
  echo "=== START $exp ($(date +%H:%M)) : $* ==="
  python3 -u "$SP/calibrate_w2a4.py" --wbit 4 --abit 4 --exp-name "$exp" "$@" \
      > "$LOG/calib_${exp}.log" 2>&1
  echo "=== DONE $exp rc=$? ($(date +%H:%M)) ==="
}

# baseline first: everything downstream is compared against this
run a44_local
# then one component removed at a time
run a44_norot    --no-rot
run a44_nosmooth --no-smooth
run a44_nolwc    --no-lwc
run a44_nolac    --no-lac

echo "ABLATION CALIBRATION COMPLETE"

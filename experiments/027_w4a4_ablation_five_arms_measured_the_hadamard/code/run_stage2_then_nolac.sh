#!/bin/bash
# Ordering chosen to survive a scheduled reboot.
#
# Calibration is NOT resumable: per-layer files are written as it goes, but a run only
# becomes usable at the final merge, so a job killed at 85% loses all ~3.5 h (this already
# happened once today when the home quota filled). Stage 2, by contrast, is short (~1 h) and
# every step writes a durable result.
#
# So: as soon as a44_nolwc lands, run stage 2 and bank a complete, reportable ablation over
# the five finished arms. Only then start a44_nolac, which is the one that can afford to be
# interrupted -- losing it costs one arm, not the whole study.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
OUT=/home/utn/luli38se/cv/QuantVGGT/evaluation/outputs/w4a4/a44_nolwc_model_tracker_fixed_e20.pt_sym
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs

echo "=== waiting for a44_nolwc to finish ($(date +%H:%M)) ==="
while [ ! -f "$OUT/qs_global_parameters_total.pth" ]; do
  if ! pgrep -f "calibrate_w2a4.py --wbit 4 --abit 4 --exp-name a44_nolwc" > /dev/null; then
    sleep 20   # let a final write settle
    if [ ! -f "$OUT/qs_global_parameters_total.pth" ]; then
      echo "!! a44_nolwc process gone without producing totals -- continuing without it"
      break
    fi
  fi
  sleep 60
done
echo "=== a44_nolwc done ($(date +%H:%M)) ==="

echo "=== STAGE 2 START ($(date +%H:%M)) ==="
"$SP/run_ablation_full.sh" subset
echo "=== STAGE 2 END ($(date +%H:%M)) ==="

echo "=== a44_nolac START ($(date +%H:%M)) ==="
python3 -u "$SP/calibrate_w2a4.py" --wbit 4 --abit 4 --exp-name a44_nolac --no-lac \
    > "$LOG/calib_a44_nolac.log" 2>&1
echo "=== a44_nolac DONE rc=$? ($(date +%H:%M)) ==="
echo "ALL ABLATION WORK COMPLETE"

#!/bin/bash
# Final ablation arm + a stage-2 rerun that folds it in.
#
# WHY THIS EXISTS: a44_nolac was killed 1h07m in (14/48 blocks) by the machine's nightly
# shutdown. `last` shows the shutdown is at 00:35 every night, not 12:30 as previously
# assumed, with the machine back around 04:33 -- a usable window of ~20 h starting 04:33.
# A 3.5 h calibration must therefore begin before ~21:00 to be safe.
#
# NOT RESUMED, deliberately restarted from scratch. Upstream `resume_qs` means "load
# finished totals", not "continue mid-run"; block-wise calibration also feeds each block's
# output into the next, so a hand-rolled resume would have to replay layers 0-6 anyway and
# would risk diverging from a clean run. Every other arm is a clean run, and an arm that is
# not comparable to the others is worth nothing here. The partial files are kept under
# killed_run_2026-09-09T2328/ rather than deleted.
#
# Stage 2 then reruns over all six arms so the final table is internally consistent
# (one iteration count, one metric pass) rather than stitched from two dates.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
OUT=/home/utn/luli38se/cv/QuantVGGT/evaluation/outputs/w4a4/a44_nolac_model_tracker_fixed_e20.pt_sym

echo "=== a44_nolac START ($(date +%F' '%H:%M)) ==="
python3 -u "$SP/calibrate_w2a4.py" --wbit 4 --abit 4 --exp-name a44_nolac --no-lac \
    > "$LOG/calib_a44_nolac.log" 2>&1
rc=$?
echo "=== a44_nolac DONE rc=$rc ($(date +%H:%M)) ==="

if [ ! -f "$OUT/qs_global_parameters_total.pth" ]; then
  echo "!! a44_nolac produced no totals -- stage 2 will run on five arms, not six"
fi

echo "=== STAGE 2 RERUN START ($(date +%H:%M)) ==="
"$SP/run_ablation_full.sh" subset
echo "=== STAGE 2 RERUN END ($(date +%H:%M)) ==="
echo "ALL ABLATION WORK COMPLETE"

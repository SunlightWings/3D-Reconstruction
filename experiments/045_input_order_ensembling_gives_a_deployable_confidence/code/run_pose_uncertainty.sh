#!/bin/bash
# Order-ensembling pose uncertainty: w4a4 (the signal) and full precision (the control).
# The control decides the interpretation: if full's spread predicts full's pose error just as well,
# the signal is about VGGT's own ambiguity on a scene, not about quantization damage.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
for v in w4a4 full w4a4_rtn; do
  echo "=== $v start ($(date +%H:%M)) ==="
  /home/utn/luli38se/cv/.venv/bin/python -u "$SP/pose_uncertainty.py" --variant $v --perms 4 --scenes all \
      > "$LOG/pose_uncertainty_$v.log" 2>&1
  echo "=== $v done rc=$? ($(date +%H:%M)) ==="
done
echo "POSE UNCERTAINTY COMPLETE"

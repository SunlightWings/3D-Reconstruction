#!/bin/bash
# Does averaging input orderings correct the pose error that #045's spread detects?
# w4a4_rtn carries measurable rendering damage and is the arm a correction can be tested on;
# w4a4 and full are controls (full should show no gain vs itself -- its target is ~0 by construction).
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
for v in w4a4_rtn w4a4 full; do
  echo "=== $v start ($(date +%H:%M)) ==="
  /home/utn/luli38se/cv/.venv/bin/python -u "$SP/pose_ensemble.py" --variant $v --perms 4 --scenes all \
      --save-extrinsics > "$LOG/pose_ensemble_$v.log" 2>&1
  echo "=== $v done rc=$? ($(date +%H:%M)) ==="
done
echo "POSE ENSEMBLE COMPLETE"

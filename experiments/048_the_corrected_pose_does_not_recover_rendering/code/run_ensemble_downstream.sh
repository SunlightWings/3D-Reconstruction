#!/bin/bash
# Waits for the pose-ensemble inference chain, then measures whether the corrected pose recovers
# rendering quality. One condition per scene (~2.3 min); stops cleanly at 00:15.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
while pgrep -f run_pose_ensemble.sh > /dev/null; do sleep 30; done
echo "=== ensemble inference done ($(date +%H:%M)) ==="
/home/utn/luli38se/cv/.venv/bin/python -u "$SP/ensemble_downstream.py" --arm w4a4_rtn --perms 4 \
    --iterations 7000 --deadline 00:15 > "$LOG/ensemble_downstream.log" 2>&1
echo "=== ensemble downstream rc=$? ($(date +%H:%M)) ==="

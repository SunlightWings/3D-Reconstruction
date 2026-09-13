#!/bin/bash
# Queued behind run_probes_day.sh (one GPU). Splits the swap's camera damage into pose vs focal
# length -- the CPU analysis found w4a4_rtn focal lengths off by a median 18.65%, which the
# swap's camera condition carried along with the pose. Stops cleanly at 00:15.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
echo "=== waiting for day queue ($(date +%F' '%H:%M)) ==="
while pgrep -f run_probes_day.sh > /dev/null; do sleep 60; done
echo "=== CAMSPLIT start ($(date +%H:%M)) ==="
/home/utn/luli38se/cv/.venv/bin/python -u "$SP/geometry_probes.py" camsplit --arm w4a4_rtn \
    --scenes 40 --deadline 00:15 > "$LOG/probe_camsplit_rtn.log" 2>&1
echo "=== CAMSPLIT done rc=$? ($(date +%H:%M)) ==="

#!/bin/bash
# 2026-09-11 queue. Swap first: it decides whether point-level confidence can matter at all
# (#030). Sweep second: it places W4A4 on a damage axis. Both stop cleanly at 00:15.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
PY=/home/utn/luli38se/cv/.venv/bin/python
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
echo "=== SWAP start ($(date +%F' '%H:%M)) ==="
$PY -u "$SP/geometry_probes.py" swap --arm w4a4_rtn --scenes 40 --deadline 00:15 > "$LOG/probe_swap_rtn.log" 2>&1
echo "=== SWAP done rc=$? ($(date +%H:%M)) ==="
echo "=== SWEEP start ($(date +%H:%M)) ==="
$PY -u "$SP/geometry_probes.py" sweep --scenes 12 --deadline 00:15 > "$LOG/probe_sweep.log" 2>&1
echo "=== SWEEP done rc=$? ($(date +%H:%M)) ==="
echo "DAY QUEUE COMPLETE"

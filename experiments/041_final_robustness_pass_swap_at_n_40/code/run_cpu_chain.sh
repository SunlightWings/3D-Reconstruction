#!/bin/bash
# CPU-only experiments beside the GPU queue. nice 15, 8 workers of 24 cores: never competes with 3DGS.
# gaussians waits for the swap probe, because its mechanism test reads the swap's trained models;
# robustness runs twice -- once now, once after the swap completes so the final n=40 swap is covered.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
PY="nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u"
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
run () { echo "=== $1 start ($(date +%H:%M)) ==="; $PY "$SP/cpu_probes.py" "$1" --workers 8 > "$LOG/cpu_$1${2:-}.log" 2>&1; echo "=== $1 done rc=$? ($(date +%H:%M)) ==="; }
run robustness _pass1
run depthgt
run predict
echo "=== waiting for swap probe ($(date +%H:%M)) ==="
while pgrep -f "geometry_probes.py swap" > /dev/null; do sleep 60; done
run gaussians
run robustness _final
echo "CPU CHAIN COMPLETE ($(date +%H:%M))"

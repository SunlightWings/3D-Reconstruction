#!/bin/bash
# Queued behind the pose-only downstream test. Two follow-ups the #047 result implies:
#   1. correct the INTRINSICS too. #047 ensembles pose only, so ensK still carries the arm's 18.652 %
#      focal error, worth 0.4876 dB on its own (#043) -- a ceiling on what a pose-only fix can recover.
#      The p4 JSON predates the intrinsics patch, so the ensemble is recomputed (same seeds, same
#      orderings; it gains an `ensemble_intrinsic` field and should otherwise reproduce).
#   2. does 2 orderings work as well as 4? The proposal claims inference-time efficiency, and 4x is the
#      honest cost of the current detector+corrector. If 2x recovers most of the gain, that halves it.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
PY=/home/utn/luli38se/cv/.venv/bin/python
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
while pgrep -f run_ensemble_downstream.sh > /dev/null; do sleep 30; done
echo "=== pose-only downstream finished ($(date +%H:%M)) ==="

echo "=== re-run rtn p4 with intrinsics ($(date +%H:%M)) ==="
$PY -u "$SP/pose_ensemble.py" --variant w4a4_rtn --perms 4 --scenes all --save-extrinsics \
    > "$LOG/pose_ensemble_w4a4_rtn_p4b.log" 2>&1

for v in w4a4_rtn w4a4; do
  echo "=== $v with 2 orderings ($(date +%H:%M)) ==="
  $PY -u "$SP/pose_ensemble.py" --variant $v --perms 2 --scenes all > "$LOG/pose_ensemble_${v}_p2.log" 2>&1
done

echo "=== downstream with pose+focal correction ($(date +%H:%M)) ==="
$PY -u "$SP/ensemble_downstream.py" --arm w4a4_rtn --perms 4 --iterations 7000 --use-intrinsics \
    --deadline 00:15 > "$LOG/ensemble_downstream_posefocal.log" 2>&1
echo "=== FOLLOWUPS COMPLETE ($(date +%H:%M)) ==="

#!/bin/bash
# User instruction 2026-09-11 22:2x: let the pose-vs-focal split complete all 40 scenes rather than stop at
# its 00:15 guard. The running process's guard cannot be changed in flight, so this waits for it to exit and,
# if any scenes are missing, resumes ONLY those (completed scenes are kept, never re-scored), running until
# 00:31 -- the last start that still lets a ~4 min scene finish before the nightly 00:35 shutdown. If the
# shutdown still wins, the same --resume command completes the rest the next morning.
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
while pgrep -f "geometry_probes.py camsplit --arm w4a4_rtn --scenes 40 --deadline 00:15" > /dev/null; do sleep 20; done
n=$(/home/utn/luli38se/cv/.venv/bin/python -c "import json;print(len(json.load(open('/var/tmp/luli38se/quantsplat/oracle_sweep/results/geometry_probe_camsplit_w4a4_rtn_it7000.json'))['per_scene']))")
echo "=== first run exited ($(date +%H:%M)) with $n/40 scenes ==="
if [ "$n" -lt 40 ]; then
  /home/utn/luli38se/cv/.venv/bin/python -u "$SP/geometry_probes.py" camsplit --arm w4a4_rtn --scenes 40 \
      --deadline 00:31 --resume >> "$LOG/probe_camsplit_rtn_resume.log" 2>&1
  echo "=== resume exited rc=$? ($(date +%H:%M)) ==="
fi
echo "CAMSPLIT FINISH DONE"

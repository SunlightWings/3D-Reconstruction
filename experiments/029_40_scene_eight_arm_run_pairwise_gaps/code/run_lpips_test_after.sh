#!/bin/bash
# Chained behind the 40-scene job. Re-scores existing renders only -- no training, no GPU --
# so it costs minutes, not hours. See lpips_clip_test.py for what it resolves (#028).
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs

echo "=== waiting for the 40-scene job ($(date +%F' '%H:%M)) ==="
while pgrep -f "run_40scene_exposure.sh" > /dev/null; do sleep 120; done
echo "=== 40-scene job done ($(date +%H:%M)) ==="

python3 -u "$SP/lpips_clip_test.py" --scenes all --iterations 7000 \
    --arms full w4a4 w4a4_local w4a4_no_rot w4a4_rtn 2>&1 | tee "$LOG/lpips_clip_test.log"
echo "LPIPS CLIP TEST COMPLETE ($(date +%F' '%H:%M))"

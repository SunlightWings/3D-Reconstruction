#!/bin/bash
# Does more averaging cross the rendering-damage threshold, or does the gain saturate?
# 4 orderings take w4a4_rtn from 6.07 to ~4.9-5.2 deg; #042 puts the threshold at ~2-4 deg, and the
# downstream test found no recovery -- consistent with landing just above it. If 8 or 16 orderings push
# the median under 4 deg, a downstream re-test is justified; if the gain saturates, the method has a
# ceiling and that is the result. Seeds are now crc32-based, so these runs are reproducible (the earlier
# hash()-based seeding was not, which is why the p4 rerun differed from #047).
set -u
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
PY=/home/utn/luli38se/cv/.venv/bin/python
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
for n in 2 4 8 16; do
  for v in w4a4_rtn w4a4; do
    echo "=== $v perms=$n ($(date +%H:%M)) ==="
    $PY -u "$SP/pose_ensemble.py" --variant $v --perms $n --scenes all --save-extrinsics \
        > "$LOG/pose_ensemble_${v}_p${n}_seeded.log" 2>&1
  done
done
echo "ENSEMBLE SCALING COMPLETE ($(date +%H:%M))"

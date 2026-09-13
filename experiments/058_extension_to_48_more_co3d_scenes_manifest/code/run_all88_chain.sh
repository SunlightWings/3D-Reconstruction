#!/bin/bash
# 48 new CO3D scenes: manifest (now, CPU) -> wait for phase old40 -> full+W4A4 VGGT -> scene prep -> corrected-rotation 3DGS.
set -u
PY=/home/utn/luli38se/cv/.venv/bin/python
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
NEW=/var/tmp/luli38se/quantsplat/dataset_v2_new48
if [ -s "$NEW/new48.txt" ] && [ "$(wc -l < "$NEW/new48.txt")" = 48 ]; then
  echo "PHASE new48-manifest: reusing existing manifest ($(date +%H:%M))"
else
  echo "PHASE new48-manifest ($(date +%H:%M))"
  $PY -u $SP/extend_new48.py manifest --min-valid 1 2>&1 | grep -vE "RuntimeWarning|nanmedian"
fi
[ -s "$NEW/new48.txt" ] && [ "$(wc -l < "$NEW/new48.txt")" = 48 ] || { echo "ERROR: manifest step did not produce 48 scenes"; exit 1; }
echo "=== waiting for phase old40 to finish ($(date +%H:%M))"
while pgrep -f "run_qfix_a[l]l.py --old40" >/dev/null; do sleep 60; done
echo "PHASE new48-infer ($(date +%H:%M))"
$PY -u $SP/extend_new48.py infer 2>&1 | grep -vE "RuntimeWarning|UserWarning|warnings.warn" || true
n=$(ls $NEW/../disagreement_dataset_v1/run/predictions/*/*/g0000/w4a4_meta.json 2>/dev/null | wc -l)
echo "w4a4 predictions present for g0000 groups (old+new): $n"
echo "PHASE new48-prepare ($(date +%H:%M))"
$PY -u $SP/extend_new48.py prepare 2>&1 | grep -aE "^\[prepare|Traceback|Error|die|SystemExit" || true
$PY -u $SP/run_qfix_all.py --scenes-file "$NEW/new48.txt" --arms full w4a4 --source-mode direct --tag new48 \
  || { echo "ERROR: new48 3DGS phase failed"; exit 1; }
echo "=== ALL-88 DONE ($(date '+%F %H:%M')) ==="

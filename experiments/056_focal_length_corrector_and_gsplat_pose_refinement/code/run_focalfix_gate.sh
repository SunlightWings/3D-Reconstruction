#!/bin/bash
# Gate between phase 1 (W3A3, n = 8) and phase 3 (W4A4-RTN, n = 40) of the focal-corrector plan.
# Rule fixed BEFORE seeing the result: launch phase 3 only if perfect focal correction (oracle) recovers
# >= 0.2 dB foreground raw PSNR of W3A3's gap. Below that, no focal predictor can matter and GPU time is not spent.
set -u
PY=/home/utn/luli38se/cv/.venv/bin/python
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
RES=/var/tmp/luli38se/quantsplat/oracle_sweep/results
J1=$RES/arms_full_w4a4_w3a3_w3a3_Koracle_w3a3_Kconst_w3a3_Kiqr_subset_it7000.json
DEC=$LOG/focalfix_gate_decision.txt
rm -f "$DEC"
echo "=== waiting for phase 1 ($(date +%H:%M))"
while [ ! -f "$J1" ]; do
  if ! pgrep -f "run_w2a4_downstream.py" >/dev/null && [ ! -f "$J1" ]; then
    sleep 60; [ -f "$J1" ] || { echo "STOP: phase-1 process gone and no results JSON" | tee "$DEC"; exit 1; }
  fi
  sleep 30
done
gain=$($PY - "$J1" <<'P'
import json,sys
d=json.load(open(sys.argv[1]))
g=lambda k: d["pairs"][k]["mean_delta"]
# gap closed by the oracle = (full - w3a3) - (full - w3a3_Koracle)
print(f'{g("full_minus_w3a3__foreground__psnr") - g("full_minus_w3a3_Koracle__foreground__psnr"):.4f}')
P
)
echo "oracle focal correction recovers $gain dB of W3A3's foreground PSNR gap"
if $PY -c "import sys; sys.exit(0 if float('$gain') >= 0.2 else 1)"; then
  echo "GO: gain $gain >= 0.2 -> launching phase 3 at $(date +%H:%M)" | tee "$DEC"
  $PY -u $SP/run_w2a4_downstream.py --scenes all --iterations 7000 \
      --arms full w4a4_rtn w4a4_rtn_Koracle w4a4_rtn_Kconst > $LOG/downstream_rtn_focalfix_40.log 2>&1
  echo "PHASE 3 DONE $(date +%H:%M) rc=$?" | tee -a "$DEC"
else
  echo "STOP: gain $gain < 0.2 -> focal line closed, phase 3 not launched ($(date +%H:%M))" | tee "$DEC"
fi

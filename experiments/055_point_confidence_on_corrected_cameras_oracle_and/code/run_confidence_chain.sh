#!/bin/bash
# W3A3 + confidence -> 3DGS, on camera-corrected sources. Waits for the baseline, then oracle arms, then predictor arms, then scores.
set -u
PY=/home/utn/luli38se/cv/.venv/bin/python
SP=/home/utn/luli38se/cv/3D-Reconstruction/code/quantization
LOG=/var/tmp/luli38se/quantsplat/w2a4_logs
RES=/var/tmp/luli38se/quantsplat/oracle_sweep/results
DONE=$LOG/confidence_chain.done; rm -f "$DONE"
echo "=== waiting for baseline ($(date +%H:%M))"
while pgrep -f "qfix_down[s]tream.py" >/dev/null; do sleep 30; done
n=$($PY -c "import json;print(len(json.load(open('$RES/qfix_baseline_full_w3a3_subset_it7000.json'))['per_scene']))" 2>/dev/null || echo 0)
echo "baseline scenes complete: $n"
[ "$n" = 8 ] || { echo "ABORT: baseline incomplete ($n/8)" | tee "$DONE"; exit 1; }
echo "=== oracle confidence arms ($(date +%H:%M))"
$PY -u $SP/confidence_3dgs.py --arm w3a3 --signal oracle --modes prune weight random > $LOG/confidence_oracle_qfix.log 2>&1 || echo "oracle rc=$?"
echo "=== waiting for predictor confidences ($(date +%H:%M))"
for i in $(seq 1 120); do [ "$(ls $RES/conf_predictor/*_w3a3.npy 2>/dev/null | wc -l)" = 8 ] && break; sleep 30; done
if [ "$(ls $RES/conf_predictor/*_w3a3.npy 2>/dev/null | wc -l)" = 8 ]; then
  echo "=== predictor confidence arms ($(date +%H:%M))"
  $PY -u $SP/confidence_3dgs.py --arm w3a3 --signal predictor --modes prune weight > $LOG/confidence_predictor_qfix.log 2>&1 || echo "predictor rc=$?"
else
  echo "predictor confidences missing - predictor arms skipped"
fi
echo "=== scoring ($(date +%H:%M))"
$PY -u $SP/score_arms.py --scenes subset --out confidence_w3a3_qfix \
  --arms full_qfix w3a3_qfix w3a3_qfix_oracle_prune w3a3_qfix_oracle_weight w3a3_qfix_rand_prune w3a3_qfix_predictor_prune w3a3_qfix_predictor_weight \
  --pairs full_qfix:w3a3_qfix w3a3_qfix_oracle_prune:w3a3_qfix w3a3_qfix_oracle_weight:w3a3_qfix w3a3_qfix_rand_prune:w3a3_qfix \
          w3a3_qfix_oracle_prune:w3a3_qfix_rand_prune w3a3_qfix_predictor_prune:w3a3_qfix w3a3_qfix_predictor_weight:w3a3_qfix \
          w3a3_qfix_predictor_prune:w3a3_qfix_rand_prune \
  > $LOG/confidence_score.log 2>&1 || echo "score rc=$?"
echo "CHAIN DONE $(date +%H:%M)" | tee "$DONE"

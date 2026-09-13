#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Usage:
#
# bash run_one_scene_timing.sh CATEGORY SEQUENCE [ITERATIONS]
#
# Example:
#
# bash run_one_scene_timing.sh \
#   broccoli 412_56288_108844 7000
#
# ============================================================

CATEGORY="${1:?CATEGORY required}"
SEQUENCE="${2:?SEQUENCE required}"
ITERATIONS="${3:-7000}"

GS_ROOT="/home/utn/luli38se/cv/gaussian-splatting"
GS_PY="/opt/saltstack/salt/bin/python3.10"
GS_SITE="/var/tmp/luli38se/3dgs-site"

# These are the already-prepared, matched Full/Quant 3DGS inputs.
SCENE_ROOT="/var/tmp/luli38se/quantsplat/downstream_validation_v1/${CATEGORY}/${SEQUENCE}"

FULL_SOURCE="${SCENE_ROOT}/sources/full/train"
QUANT_SOURCE="${SCENE_ROOT}/sources/w4a4/train"

# Completely separate timing outputs.
# Existing 40-scene experiment is NOT touched.
OUT_ROOT="/var/tmp/luli38se/quantsplat/one_scene_timing_v1/${CATEGORY}/${SEQUENCE}/${ITERATIONS}"

FULL_MODEL="${OUT_ROOT}/models/full"
QUANT_MODEL="${OUT_ROOT}/models/w4a4"

RESULTS="${OUT_ROOT}/timing_results.csv"

mkdir -p "${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}/timing"

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

echo "============================================================"
echo "ONE-SCENE FULL vs QUANT 3DGS TIMING"
echo "============================================================"
echo "Scene      : ${CATEGORY}/${SEQUENCE}"
echo "Iterations : ${ITERATIONS}"
echo
echo "Full source : ${FULL_SOURCE}"
echo "Quant source: ${QUANT_SOURCE}"
echo

if [ ! -d "${FULL_SOURCE}" ]; then
    echo "ERROR: missing Full source:"
    echo "${FULL_SOURCE}"
    exit 1
fi

if [ ! -d "${QUANT_SOURCE}" ]; then
    echo "ERROR: missing Quant source:"
    echo "${QUANT_SOURCE}"
    exit 1
fi

if [ ! -f "${GS_ROOT}/train.py" ]; then
    echo "ERROR: GraphDECO train.py missing"
    exit 1
fi

if [ ! -x "${GS_PY}" ]; then
    echo "ERROR: 3DGS Python missing"
    exit 1
fi

# ------------------------------------------------------------
# GPU gate
#
# We only start when there is no CUDA compute process.
# We do not kill anybody else's job.
# ------------------------------------------------------------

wait_for_gpu()
{
    echo
    echo "Checking GPU..."

    while true; do

        COMPUTE="$(
            nvidia-smi \
              --query-compute-apps=pid,process_name,used_memory \
              --format=csv,noheader,nounits 2>/dev/null || true
        )"

        if [ -z "${COMPUTE}" ]; then
            echo "GPU gate: free -> proceed"
            break
        fi

        echo "GPU gate: another compute process is active:"
        echo "${COMPUTE}"
        echo "Rechecking in 10 seconds..."
        sleep 10
    done
}

# ------------------------------------------------------------
# GPU monitor
#
# Records GPU utilization, VRAM and power during each training.
# ------------------------------------------------------------

start_gpu_monitor()
{
    local FILE="$1"

    (
        echo "timestamp,gpu_util_percent,memory_used_mib,memory_total_mib,power_w"

        while true; do
            nvidia-smi \
              --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
              --format=csv,noheader,nounits 2>/dev/null || true

            sleep 2
        done
    ) > "${FILE}" &

    MONITOR_PID=$!
}

stop_gpu_monitor()
{
    if [ -n "${MONITOR_PID:-}" ]; then
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        unset MONITOR_PID
    fi
}

# ------------------------------------------------------------
# Run one variant
# ------------------------------------------------------------

run_variant()
{
    local NAME="$1"
    local SOURCE="$2"
    local MODEL="$3"

    local LOG="${OUT_ROOT}/logs/${NAME}.log"
    local TIMEFILE="${OUT_ROOT}/timing/${NAME}.time.txt"
    local GPUFILE="${OUT_ROOT}/timing/${NAME}_gpu.csv"

    echo
    echo "============================================================"
    echo "RUNNING: ${NAME}"
    echo "============================================================"

    echo "Source: ${SOURCE}"
    echo "Model : ${MODEL}"
    echo

    # Fresh timing run.
    # Only this dedicated timing model directory is deleted.
    rm -rf "${MODEL}"
    mkdir -p "${MODEL}"

    wait_for_gpu

    echo
    echo "GPU state immediately before ${NAME}:"
    nvidia-smi \
      --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total \
      --format=csv

    start_gpu_monitor "${GPUFILE}"

    START_EPOCH="$(date +%s)"
    START_TEXT="$(date --iso-8601=seconds)"

    echo
    echo "${NAME} start: ${START_TEXT}"
    echo

    cd "${GS_ROOT}"

    set +e

    PYTHONPATH="${GS_SITE}:${PYTHONPATH:-}" \
    /usr/bin/time \
      -o "${TIMEFILE}" \
      -f "elapsed_seconds=%e
user_cpu_seconds=%U
system_cpu_seconds=%S
max_rss_kb=%M
exit_status=%x" \
      "${GS_PY}" -u train.py \
        -s "${SOURCE}" \
        -m "${MODEL}" \
        --iterations "${ITERATIONS}" \
        --data_device cpu \
        --resolution 1 \
        2>&1 | tee "${LOG}"

    TRAIN_EXIT="${PIPESTATUS[0]}"

    set -e

    END_EPOCH="$(date +%s)"
    END_TEXT="$(date --iso-8601=seconds)"

    stop_gpu_monitor

    WALL_SECONDS=$((END_EPOCH - START_EPOCH))

    echo
    echo "${NAME} end  : ${END_TEXT}"
    echo "${NAME} wall : ${WALL_SECONDS} seconds"
    echo

    if [ "${TRAIN_EXIT}" -ne 0 ]; then
        echo "ERROR: ${NAME} training failed with exit ${TRAIN_EXIT}"
        exit "${TRAIN_EXIT}"
    fi

    if [ ! -f "${MODEL}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]; then
        echo "ERROR: ${NAME} did not produce iteration_${ITERATIONS}"
        exit 1
    fi

    echo "${WALL_SECONDS}"
}

# ------------------------------------------------------------
# Run Full and Quant sequentially
# ------------------------------------------------------------

FULL_SECONDS="$(run_variant \
    full \
    "${FULL_SOURCE}" \
    "${FULL_MODEL}" \
    | tee /dev/tty \
    | tail -1)"

# Small cooldown so the second measurement does not start
# immediately after the first GPU workload.
echo
echo "Cooling down for 30 seconds before Quant run..."
sleep 30

QUANT_SECONDS="$(run_variant \
    w4a4 \
    "${QUANT_SOURCE}" \
    "${QUANT_MODEL}" \
    | tee /dev/tty \
    | tail -1)"

# ------------------------------------------------------------
# Final results
# ------------------------------------------------------------

FULL_MIN="$(
python - <<PY
print(${FULL_SECONDS} / 60.0)
PY
)"

QUANT_MIN="$(
python - <<PY
print(${QUANT_SECONDS} / 60.0)
PY
)"

DELTA="$(
python - <<PY
print(${QUANT_SECONDS} - ${FULL_SECONDS})
PY
)"

RATIO="$(
python - <<PY
f = float(${FULL_SECONDS})
q = float(${QUANT_SECONDS})
print(q / f if f > 0 else float("nan"))
PY
)"

cat > "${RESULTS}" <<CSV
category,sequence,variant,iterations,wall_seconds,wall_minutes
${CATEGORY},${SEQUENCE},full,${ITERATIONS},${FULL_SECONDS},${FULL_MIN}
${CATEGORY},${SEQUENCE},w4a4,${ITERATIONS},${QUANT_SECONDS},${QUANT_MIN}
CSV

cat > "${OUT_ROOT}/summary.txt" <<EOF
Scene: ${CATEGORY}/${SEQUENCE}
Iterations: ${ITERATIONS}

Full 3DGS time:
  ${FULL_SECONDS} seconds
  ${FULL_MIN} minutes

Quant 3DGS time:
  ${QUANT_SECONDS} seconds
  ${QUANT_MIN} minutes

Quant - Full:
  ${DELTA} seconds

Quant / Full time ratio:
  ${RATIO}x
EOF

echo
echo "============================================================"
echo "TIMING COMPLETE"
echo "============================================================"

cat "${OUT_ROOT}/summary.txt"

echo
echo "Results:"
echo "${RESULTS}"

echo
echo "Detailed timing:"
echo "${OUT_ROOT}/timing/full.time.txt"
echo "${OUT_ROOT}/timing/w4a4.time.txt"

echo
echo "GPU monitoring:"
echo "${OUT_ROOT}/timing/full_gpu.csv"
echo "${OUT_ROOT}/timing/w4a4_gpu.csv"

echo
echo "Logs:"
echo "${OUT_ROOT}/logs/full.log"
echo "${OUT_ROOT}/logs/w4a4.log"

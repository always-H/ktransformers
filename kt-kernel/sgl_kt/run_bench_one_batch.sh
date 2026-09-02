#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing model weights}"

PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_NAME=${MODEL_NAME:-DeepSeek-V2-Lite}
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/${MODEL_NAME}"}
LOG_ROOT=${LOG_ROOT:-"${PWD}/logs"}
METHOD_NAME=${METHOD_NAME:-LayerScopeV2}
FILE_NAME=${FILE_NAME:-DSv2}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-32}
NUM_GPU_EXPERTS=${NUM_GPU_EXPERTS:-6}
CPU_THREADS=${CPU_THREADS:-32}
CUDA_DEVICE=${CUDA_DEVICE:-0}
BATCH_SIZES=${BATCH_SIZES:-"4 8 16 32 64"}

LOG_DIR="${LOG_ROOT}/${MODEL_NAME}-${METHOD_NAME}"
mkdir -p "${LOG_DIR}"

for batch_size in ${BATCH_SIZES}; do
  log_file="${LOG_DIR}/${FILE_NAME}_Batch_Size_${batch_size}_${METHOD_NAME}.log"
  echo "Running batch_size=${batch_size}; log=${log_file}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -m sglang.bench_one_batch \
    --model "${MODEL_PATH}" \
    --kt-weight-path "${MODEL_PATH}" \
    --kt-cpuinfer "${CPU_THREADS}" \
    --kt-threadpool-count 1 \
    --kt-num-gpu-experts "${NUM_GPU_EXPERTS}" \
    --kt-method BF16 \
    --kt-gpu-prefill-token-threshold 32769 \
    --attention-backend flashinfer \
    --trust-remote-code \
    --mem-fraction-static 0.98 \
    --chunked-prefill-size 32769 \
    --max-running-requests 1024 \
    --max-total-tokens 40000 \
    --watchdog-timeout 3000 \
    --enable-mixed-chunk \
    --tensor-parallel-size 1 \
    --batch "${batch_size}" \
    --input-len "${INPUT_LEN}" \
    --output-len "${OUTPUT_LEN}" \
    --log-decode-step 1 \
    --disable-cuda-graph \
    >"${log_file}" 2>&1
done

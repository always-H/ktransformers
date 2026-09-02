#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing model weights}"

PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_NAME=${MODEL_NAME:-Qwen2-57B-A14B-Instruct}
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/${MODEL_NAME}"}
CUDA_DEVICE=${CUDA_DEVICE:-0}
CPU_THREADS=${CPU_THREADS:-64}
NUM_GPU_EXPERTS=${NUM_GPU_EXPERTS:-60}
INPUT_LEN=${INPUT_LEN:-4096}
OUTPUT_LEN=${OUTPUT_LEN:-32}
BATCH_SIZE=${BATCH_SIZE:-1}

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -m sglang.bench_one_batch \
  --model "${MODEL_PATH}" \
  --kt-weight-path "${MODEL_PATH}" \
  --kt-cpuinfer "${CPU_THREADS}" \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts "${NUM_GPU_EXPERTS}" \
  --kt-method BF16 \
  --kt-gpu-prefill-token-threshold 4097 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.98 \
  --chunked-prefill-size 4096 \
  --max-running-requests 32 \
  --max-total-tokens 40000 \
  --watchdog-timeout 3000 \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --batch "${BATCH_SIZE}" \
  --input-len "${INPUT_LEN}" \
  --output-len "${OUTPUT_LEN}" \
  --disable-cuda-graph

#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing model weights}"
: "${CMMLU_DATA_DIR:?Set CMMLU_DATA_DIR to the CMMLU dataset directory}"

PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_NAME=${MODEL_NAME:-DeepSeek-V2-Lite}
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/${MODEL_NAME}"}
CUDA_DEVICE=${CUDA_DEVICE:-0}
CPU_THREADS=${CPU_THREADS:-64}

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -m sglang.collect_trace \
  --model-path "${MODEL_PATH}" \
  --dataset-type cmmlu \
  --cmmlu-data-dir "${CMMLU_DATA_DIR}" \
  --cmmlu-split test \
  --cmmlu-task all \
  --apply-chat-template \
  --batch-size 32 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --shard-id 0 \
  --num-shards 1 \
  --ignore-eos \
  --flush-every 512 \
  --kt-weight-path "${MODEL_PATH}" \
  --kt-cpuinfer "${CPU_THREADS}" \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 0 \
  --kt-method BF16 \
  --kt-gpu-prefill-token-threshold 4096 \
  --disable-cuda-graph \
  --attention-backend triton \
  --watchdog-timeout 1200

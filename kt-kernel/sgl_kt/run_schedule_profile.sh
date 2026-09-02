#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing model weights}"

PYTHON_BIN=${PYTHON_BIN:-python3}
LOG_ROOT=${LOG_ROOT:-"${PWD}/logs"}
CUDA_DEVICE=${CUDA_DEVICE:-0}
CPU_THREADS=${CPU_THREADS:-32}
BATCH_SIZES=${BATCH_SIZES:-"4 8 16 32 64"}
INPUT_LEN=${INPUT_LEN:-32}
OUTPUT_LEN=${OUTPUT_LEN:-8}

run_profile() {
  local model_name=$1
  local label=$2
  local num_gpu_experts=$3
  local model_path="${MODEL_ROOT}/${model_name}"
  local log_dir="${LOG_ROOT}/${model_name}-schedule-profile"
  mkdir -p "${log_dir}"

  for batch_size in ${BATCH_SIZES}; do
    local log_file="${log_dir}/${label}_Batch_Size_${batch_size}.log"
    echo "Profiling ${model_name}, batch_size=${batch_size}; log=${log_file}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -m sglang.bench_one_batch \
      --model "${model_path}" \
      --kt-weight-path "${model_path}" \
      --kt-cpuinfer "${CPU_THREADS}" \
      --kt-threadpool-count 1 \
      --kt-num-gpu-experts "${num_gpu_experts}" \
      --kt-method BF16 \
      --kt-gpu-prefill-token-threshold 32769 \
      --attention-backend flashinfer \
      --trust-remote-code \
      --mem-fraction-static 0.98 \
      --chunked-prefill-size 32769 \
      --max-running-requests 64 \
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
}

run_profile "${DSV2_MODEL_NAME:-DeepSeek-V2-Lite}" DSv2 "${DSV2_NUM_GPU_EXPERTS:-6}"
run_profile "${QWEN2_MODEL_NAME:-Qwen2-57B-A14B-Instruct}" Qw2 "${QWEN2_NUM_GPU_EXPERTS:-8}"

#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing model weights}"

PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_NAME=${MODEL_NAME:-Qwen2-57B-A14B-Instruct}
MODEL=${MODEL:-"${MODEL_ROOT}/${MODEL_NAME}"}
KT_WEIGHT_PATH=${KT_WEIGHT_PATH:-"${MODEL}"}
KT_CPUINFER=${KT_CPUINFER:-32}
KT_THREADPOOL_COUNT=${KT_THREADPOOL_COUNT:-1}
KT_NUM_GPU_EXPERTS=${KT_NUM_GPU_EXPERTS:-8}
KT_CHUNKED_PREFILL_SIZE=${KT_CHUNKED_PREFILL_SIZE:-64}
WARMUP_ITERS=${WARMUP_ITERS:-10}
BENCH_ITERS=${BENCH_ITERS:-100}
MAX_TOKENS=${MAX_TOKENS:-64}
GPU_SLOTS_PER_TOKEN=${GPU_SLOTS_PER_TOKEN:-7}

for num_tokens in $(seq 1 "${MAX_TOKENS}"); do
  gpu_slots=$((num_tokens * GPU_SLOTS_PER_TOKEN))
  "${PYTHON_BIN}" -m sglang.srt.layers.moe.benchmark_kt_ep \
    --model "${MODEL}" \
    --kt-weight-path "${KT_WEIGHT_PATH}" \
    --kt-cpuinfer "${KT_CPUINFER}" \
    --kt-threadpool-count "${KT_THREADPOOL_COUNT}" \
    --kt-num-gpu-experts "${KT_NUM_GPU_EXPERTS}" \
    --kt-method BF16 \
    --kt-chunked-prefill-size "${KT_CHUNKED_PREFILL_SIZE}" \
    --num-tokens "${num_tokens}" \
    --warmup-iters "${WARMUP_ITERS}" \
    --bench-iters "${BENCH_ITERS}" \
    --gpu-slots "${gpu_slots}" \
    --gpu-experts-active "${KT_NUM_GPU_EXPERTS}" \
    --cpu-experts-active 1
done

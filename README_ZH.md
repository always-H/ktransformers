# LayerScope 与 LayerScopeV2

[English](README.md) | [简体中文](README_ZH.md)

面向显存受限混合专家（MoE）推理的跨层预测调度与 AMX 加速 CPU-GPU 协同执行。

## 概览（Overview）

MoE 模型虽然只为每个 token 激活少量专家，但完整专家权重仍须全部保存在内存中。当显存放不下全部专家时，系统必须在“将专家传输到 GPU”与“直接在 CPU 上计算”之间做出选择。因此，端到端性能不仅取决于 GPU 计算，还取决于专家放置、PCIe 传输、CPU/GPU 协同执行和调度开销。

LayerScope 与 LayerScopeV2 从两个相继演进的层面解决这一问题。

### LayerScope（ICS 2026）

LayerScope 是面向商用及存量 CPU-GPU 服务器上多 batch MoE 推理的预测驱动专家调度系统。它利用未来层的路由信息，跨层协调计算与数据迁移：

- **LLaPor 层感知预测器：** 区分靠近输入、靠近输出和中间层的层组，利用各组不同的路由与隐状态特征预测后续层所需专家，并支持在线持续学习。
- **PreSched 预取感知跨层调度器：** 联合考虑当前层需求与预测的下一层需求，在共享 PCIe 预算下，根据估计的全局收益选择 CPU 计算、GPU 按需加载或跨层预取。
- **AsyncIO 异步 I/O 优化器：** 通过固定传输缓存区、细粒度专家分块和多流并行，将专家传输与 CPU/GPU 计算流水重叠。

LayerScope 主要解决显存有限、PCIe 时延较高的平台上，朴素按需加载专家效率低下的问题。

### LayerScopeV2（submitted to TACO）

LayerScopeV2 将 LayerScope 的预测调度基础引入 KTransformers+SGLang（KT+SGL），并针对 AMX 加速的 CPU 执行重新设计。CPU 专家计算变快后，权重布局要求和调度收益临界点都会改变，因此 LayerScopeV2 新增：

- **checkpoint 原生 BF16 AMX 内核：** 手写 AVX-512 gather 与 AMX GEMM 内核直接读取行主序 BF16 checkpoint 权重，只重排生命周期很短的激活数据，避免运行时权重量化或重打包，也无需持久保存两套权重布局。
- **AMX 感知的成本感知调度（Cost-Aware Scheduling）：** 通过聚合 AMX 线程池时延模型，仅在预测收益高于计算、传输和调度成本时，才启用动态专家放置与跨层预取；无收益的调度会被直接绕过。
- **调度重叠（Scheduler Overlap）：** 在其余放置决策完成前，提前提交确定留在 CPU 上执行的专家，将大部分调度时延隐藏在有效 CPU 计算之后。

本仓库发布 LayerScopeV2 的 KTransformers 侧实现，包括原生 BF16 AMX 内核、CPU/GPU 专家掩码接口以及启动与分析脚本。配套的 [always-H/sglang](https://github.com/always-H/sglang) 仓库已作为 [`third_party/sglang`](third_party/sglang) submodule 随本仓库提供，用于实现跨层预测器、成本感知运行时调度、预取和 Scheduler Overlap 集成。下面的快速开始运行的是两者组合后的完整 LayerScopeV2 系统。

## 关键结果（Key Results）

### LayerScope：端到端吞吐

在 NVIDIA V100 和 A100 GPU、24 GB 显存预算下，针对 DeepSeek-V2-Lite、Mixtral-8x7B、Qwen3-30B-A3B 和 Moonlight-16B-A3B，以及 4-64 的 batch size，LayerScope 的端到端吞吐最高比 Klotski **提升 141%**，最高比 HybriMoE **提升 70.8%**。

![LayerScope 在 V100 和 A100 上的吞吐对比](doc/assets/layerscope-throughput.png)

*LayerScope 论文图 11：四个 MoE 模型在两代 GPU 平台上的端到端吞吐。*

### LayerScopeV2：prefill 与 decoding

在 NVIDIA RTX A6000 和 Intel Xeon Gold 6430 平台上，LayerScopeV2 将 prefill 的首 token 时延（TTFT）**降低 13.4%-48.0%**（最高 **1.92x 加速**），端到端解码吞吐最高达到 **KT+SGL 的 2.16x**。评测覆盖 Qwen2-57B-A14B、DeepSeek-V2-Lite、Qwen3-30B-A3B 和 Mixtral-8x7B。

![LayerScopeV2 的 prefill TTFT 与解码吞吐](doc/assets/layerscope-v2-end-to-end.png)

*LayerScopeV2 论文图 8：prefill TTFT 与解码吞吐；虚线表示相对 KT+SGL 的加速比及调度策略切换位置。*

## 快速开始（Quick Start）

以下示例在单 GPU 上以 BF16 运行 LayerScopeV2 + DeepSeek-V2-Lite。CPU 和 GPU 均以原始 BF16 checkpoint 为权重来源，不需要量化或转换 CPU 权重布局。

### 1. 环境要求

- Linux x86-64、Python 3.10 或更新版本（推荐 Python 3.11），以及可用的 C/C++ 编译工具链。
- 支持 AMX-BF16 和 AVX-512 的 Intel CPU、NVIDIA CUDA GPU，以及足以容纳模型专家权重的主机内存。
- 已下载到本地的模型权重；两个代码仓库均不包含模型文件。

下面的命令使用 Conda 和 Debian/Ubuntu 包名，其他环境请按需调整。当前调度与校准路径面向 BF16 SiLU 专家、单 GPU 和 TP=1/PP=1。下面以 DeepSeek-V2-Lite 为参考配置；其他模型和硬件需要调整常驻 GPU 专家数量、重新校准并独立验证。

### 2. 构建 LayerScopeV2 内核

```bash
git clone --branch master https://github.com/always-H/ktransformers.git
cd ktransformers
git submodule update --init --recursive

conda create -n layerscope python=3.11 -y
conda activate layerscope

cd kt-kernel
sudo apt update
sudo apt install -y build-essential libhwloc-dev pkg-config
python -m pip install --upgrade pip setuptools wheel cmake pybind11
./install.sh
python -c "from kt_kernel import kt_kernel_ext; print('kt-kernel import OK')"
```

`install.sh` 会检测本机 CPU 并构建原生扩展。

### 3. 安装随仓库提供的 SGLang 运行时

第 2 步的递归 submodule 初始化已将 LayerScopeV2 运行时拉取到 `third_party/sglang`；请使用这个随仓库提供的 fork，而不是原版 SGLang。从上面的 `kt-kernel` 目录继续执行：

```bash
cd ../third_party/sglang
pip install -e python
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

校准、服务和 benchmark 应在同一个 Python 环境中使用这套内核与运行时。

### 4. 配置并校准调度器

将 `MODEL_ROOT` 替换为本地模型所在目录。下面的配置使用一个 GPU、32 个 CPU worker、一个 NUMA pool，并为 DeepSeek-V2-Lite 在 GPU 上常驻 6 个专家。

```bash
export CUDA_VISIBLE_DEVICES=0
export SGLANG_KT_ENABLE_SCHEDULER=1
export SGLANG_KT_ENABLE_EARLY_CPU=1
export SGLANG_KT_ENABLE_PREFETCH=1
unset SGLANG_KT_BUDGET_PROFILE SGLANG_KT_SCHEDULER_TRANSFER_LIMIT

export MODEL_ROOT=/path/to/models
export MODEL_NAME=DeepSeek-V2-Lite
export MODEL_PATH="$MODEL_ROOT/$MODEL_NAME"
export KT_WEIGHT_PATH="$MODEL_PATH"
export CPU_THREADS=32
export NUMA_POOLS=1
export NUM_GPU_EXPERTS=6
export BUDGET_PROFILE="$PWD/kt-budget.json"

python -m sglang.srt.layers.moe.kt_budget \
  --model-config "$MODEL_PATH/config.json" \
  --kt-cpuinfer "$CPU_THREADS" \
  --kt-threadpool-count "$NUMA_POOLS" \
  --kt-num-gpu-experts "$NUM_GPU_EXPERTS" \
  --batch-size 4 \
  --budget-max-tokens 512 \
  --budget-output "$BUDGET_PROFILE"
```

校准过程会测量本机 AMX、GPU、PCIe 和调度路径，生成可复用的成本感知调度配置。本示例从 4 个 token 开始逐次加倍至 512，覆盖 4、8、16、32、64、128、256 和 512。请通过 `--batch-size` 和 `--budget-max-tokens` 覆盖预期的 MoE token 负载。重新校准时请使用新的输出路径，因为已有配置文件不会被覆盖。若未提供配置文件或显式的诊断传输上限，动态调度将保持关闭。

### 5. 启动服务并发送请求

请保持校准步骤中的环境变量和数值配置不变：

```bash
export SGLANG_KT_BUDGET_PROFILE="$BUDGET_PROFILE"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name layerscope-demo \
  --host 0.0.0.0 \
  --port 8000 \
  --kt-weight-path "$KT_WEIGHT_PATH" \
  --kt-method BF16 \
  --kt-cpuinfer "$CPU_THREADS" \
  --kt-threadpool-count "$NUMA_POOLS" \
  --kt-num-gpu-experts "$NUM_GPU_EXPERTS" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --disable-cuda-graph
```

在另一个终端发送 OpenAI 兼容请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "layerscope-demo",
    "messages": [{"role": "user", "content": "简要介绍一下 MoE 推理。"}],
    "max_tokens": 32
  }'
```

这个单请求调用仅用于验证 API 服务，不用于测试调度性能。它的 decode 负载通常只有 1 个 token，不在上述校准范围内，因此不应期望它触发动态调度。请使用下面已完成负载校准的固定 batch benchmark 评估调度行为。

### 6. 运行固定 batch benchmark（可选）

在同一 GPU 上运行 benchmark 前，请先停止服务。下面的命令会沿用上面导出的模型与 worker 配置，并将各 batch 的日志写入 `kt-kernel/logs/`：

```bash
cd ../../kt-kernel
BATCH_SIZES="4 8 16 32 64" INPUT_LEN=512 OUTPUT_LEN=32 \
  bash sgl_kt/run_bench_one_batch.sh
```

更多启动与分析入口见 [`kt-kernel/sgl_kt/`](kt-kernel/sgl_kt/)。如需对比 static、on-demand 和 prefetch 模式，请使用运行时仓库中的 [`multi_batch_benchmark.sh`](https://github.com/always-H/sglang/blob/main/benchmark/kt/multi_batch_benchmark.sh)。

## 引用（Citation）

如果在研究中使用了 LayerScope 或 LayerScopeV2，请引用对应论文。

### LayerScope

```bibtex
@inproceedings{yu2026layerscope,
  title={LayerScope: Predictive Cross-Layer Scheduling for Efficient Multi-Batch MoE Inference on Legacy Servers},
  author={Yu, Enda and Dong, Dezun and Zhang, Zhaoning and Bai, Zhe and Yang, Weiling and Wang, Haojie and Li, Dongsheng and Wu, Yongwei and Liao, Xiangke},
  booktitle={Proceedings of the 40th ACM International Conference on Supercomputing},
  pages={881--892},
  year={2026}
}
```

### LayerScopeV2

```bibtex
@inproceedings{bai2026layerscopev2,
  title={LayerScopeV2: Predictive Cross-Layer Scheduling with AMX Acceleration for Memory-Bound MoE Inference},
  author={Bai, Zhe and Yu, Enda and Yang, Weiling and Dong, Dezun},
  booktitle={submitted to TACO},
  year={2026}
}
```

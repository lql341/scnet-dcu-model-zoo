# MiMo-V2.6-Distill-Qwen-9B on BW1000 + vLLM 0.20.0

> 提交仓库。技术报告即本文件，按官方《国芯国模黑客松 · 技术报告模板》8 个章节组织。
> 本文件中的版本号、精度分数与性能数字均可由 `logs/env.log`、`verify_result.json`、`bench/round2.jsonl` 复现。

## 1. 简介

本文档记录 `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` 在 `异构加速卡BW（BW1000）` + `vLLM 0.20.0`（精度 `bf16`）环境的适配与验证结果。

模型为小米 MiMo 团队以 Qwen3.5-9B 为基座的 9B 智能体模型，`Qwen3_5ForConditionalGeneration` 架构：32 层（24 层线性注意力 + 8 层全注意力，每 4 层 1 层全注意力），带视觉塔，256K 上下文，BF16 共 4 分片约 18.9 GB，MIT 许可。

与官方部署方式的核心差异：官方只给出 `sglang serve --reasoning-parser mimo` 的形式，未覆盖国产卡，更未覆盖 vLLM。本件在 BW1000 上需要 ① 选用面向 BW1000 构建的 vLLM 0.20 镜像，② `--gpu-memory-utilization 0.95`（DCU 上 vLLM 显存探测偏保守，0.85 会直接报 `No available memory for the cache blocks`），③ `--max-num-seqs 128` 对齐压测最高并发档，④ `--max-model-len 20480` 以容纳 16K 输入 + 1K 输出 + chat template 开销。**CUDA Graph 保持开启**（§7.1 的扫描证明 `--enforce-eager` 反而慢 3%~32%）。无包升级、无代码 patch。

相关获取地址：

- 权重下载（ModelScope）：`https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`
- 镜像：`image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0`
- 提交仓库：https://developer.sourcefind.cn/codes/luql/mimo-vllm-bw1000

参考文档：

- 官方模型卡 `https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`
- vLLM 启动参数：镜像内 `python3 -m vllm.entrypoints.openai.api_server --help`

仓库内容：

| 文件 | 作用 | 产出 |
| --- | --- | --- |
| `env.sh` | 模型、镜像、框架、卡型、并行方式与启动参数，其余脚本读这里的变量 | — |
| `env_check.sh` | 采集卡型与重点组件版本 | `logs/env.log` |
| `requirements-extra.txt` | 在镜像基础上升级的包（本件无） | — |
| `patches/` | 代码改动（本件无） | — |
| `run.sh` | 模型启动脚本：装依赖 + 重放 patch + 拉起服务，轮询 `/health` | `logs/server.log` |
| `smoke.sh` | Smoke 测试：健康检查、单轮 / 多轮 / 流式问答 | `logs/smoke.log` |
| `verify.py` | 精度验证脚本：按官方基线文件跑评测并判定 | `verify_result.json` |
| `bench.sh` | 性能测试脚本：6 档负载，每档连测两次 | `bench/` |
| `container/submit.slurm` | 容器启动封装（BW1000 上的 podman + Slurm），宿主机侧执行 | `sbatch-*.out` |
| `MISSING.md` | 官方尚未下发的材料清单与影响 | — |

一键复现（**容器内**执行 2–7 步；第 1 步在宿主机执行）：

```bash
# 1) 宿主机：起容器并进入（详见 §3.4）
sbatch container/submit.slurm

# 以下均在容器内 /work/repo 下
bash run.sh                             # 装依赖 + 重放 patch + 起服务
bash env_check.sh                       # 记录验证环境 → env.log
bash smoke.sh                           # Smoke 测试 → smoke.log
source env.sh && python verify.py --selfcheck
source env.sh && python verify.py       # 精度测评 → verify_result.json（当前被官方基线文件阻塞）
bash bench.sh                           # 性能测试 → bench/
```

### 一键复现的实测记录

本节脚本已按「评审视角」实际验证过：**从公开仓库匿名 `git clone` 下来，直接 `sbatch`，产物正常落盘**。

```bash
git clone https://developer.sourcefind.cn/codes/luql/mimo-vllm-bw1000.git
cd mimo-vllm-bw1000
SKIP_BENCH=1 sbatch --export=ALL container/submit.slurm      # 只验到 smoke，约 20 min
BENCH_QUICK=1 sbatch --export=ALL container/submit.slurm     # 再补一轮最小压测，验证压测链路
sbatch container/submit.slurm                                # 正式全量（6 档 × 2）
```

验证到的链路：镜像 load → `env_check.sh`→`logs/env.log` → `run.sh`→`logs/server.log` → `smoke.sh`→`logs/smoke.log`
→ `verify.py --selfcheck` → `verify.py`（缺官方基线时主动 `[BLOCKED]` 退出）→ `bench.sh`→`bench/`，
产物全部落在 `$WORK/repo/` 下。

踩到并已修复的三个问题（都在 `container/submit.slurm` 里）：

1. **`$0` 在 sbatch 下不可用**：Slurm 会把脚本拷到 spool 目录，`$(dirname "$0")/..` 指向错误位置。
   改用 `SLURM_SUBMIT_DIR` 定位仓库根。
2. **仓库没有挂载进容器**：早期版本 `cd /work/repo` 必然失败。现改为「`/repo` 只读挂载 + 拷进
   可写的 `/work/repo`」，这样脚本产生的日志能留在宿主机。
3. **`HOST_IMAGE_TAR` 为空时 `unbound variable`**：现改为「本地 tar 存在就 load，否则直接 pull」。

## 2. 验证环境

镜像：`image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0`

重点组件（取自 `logs/env.log`）：

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| DTK | `26.04`（`DTK_VERSION=2604`） | 镜像自带，`ROCM_PATH=/opt/dtk` |
| `torch` | `2.10.0`（HIP `6.3.26093`） | 镜像自带 |
| `triton` | `3.5.0` | 镜像自带 |
| `vllm` | `0.20.0` | 镜像自带 |
| `transformers` | `5.8.1` | 镜像自带 |
| `hyhal` | 宿主挂载 `/opt/hyhal`（`libhydmi.so.1.5`） | 与宿主内核驱动匹配，**必须挂载** |

- 卡型：`异构加速卡BW（BW1000）`，`arch=gfx936:sramecc+:xnack-`，单卡 64.0 GB
- 卡数：1（`TP=1`）
- 模型路径（容器内）：`/models/MiMo`（宿主：`/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B`）
- 服务端口：`30000`

环境自检（`logs/env.log` 关键部分）：

```text
image: image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
python: 3.10.12
torch: 2.10.0
HIP: 6.3.26093
available: True count: 1
Device: BW vram_GB: 64.0 arch: gfx936:sramecc+:xnack-
vllm: 0.20.0
transformers: 5.8.1
triton: 3.5.0
```

## 3. 适配优化方案

### 3.1 基础镜像

基于 `image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0`。选它的原因：这是 Harbor 公共镜像里带 `_bw` 标识、明确面向 BW1000 构建的 vLLM 0.20 镜像，自带 `torch 2.10.0` 与 `transformers 5.8.1`，无需自行编译。

### 3.2 软件包升级

无。镜像内组件版本已满足要求，`requirements-extra.txt` 仅保留说明性注释。

### 3.3 代码改动

无代码 patch，`patches/` 为空。所有适配通过启动参数完成：

| 参数 | 取值 | 原因 |
| --- | --- | --- |
| `--max-model-len` | `20480` | 覆盖 bench 的 16K 输入 + 1K 输出 + chat template 开销。**17408 不够**：实测被服务端 400 拒绝（详见 §8） |
| `--gpu-memory-utilization` | `0.95` | DCU 上 vLLM 的显存探测偏保守，用 0.85 会报 `ValueError: No available memory for the cache blocks` |
| `--max-num-seqs` | `128` | 与 bench 最高并发档对齐。vLLM 默认 `max_num_seqs` 较小，64 并发档会排队 |
| CUDA Graph | **保持开启** | 不使用 `--enforce-eager`；首版用它导致性能偏低，见 §7.1 |
| `--trust-remote-code` | — | 模型使用自定义 `Qwen3_5ForConditionalGeneration` |
| `--dtype bfloat16` | — | 赛题不支持 FP8/FP4 |

**容器运行时的三条硬性要求**（不属于代码改动，但不满足则拿不到卡）：

1. 必须传 `--device /dev/mkfd`：BW1000 上存在第二个 KFD 设备 `mkfd`，只传 `/dev/kfd` 时 HSA 报 `No HIP GPUs are available`；
2. 必须把**宿主** `/opt/hyhal` 挂进容器（提供与内核驱动匹配的 `libhydmi.so`）；
3. **不要**用宿主 DTK 覆盖容器内 `/opt/dtk`，否则 Triton JIT 报 `ROCm compiler not found under /opt/dtk/aillvm/bin/clang`。

另外本镜像的 `ENTRYPOINT` 是 `/bin/sh -c "<SothisAI 初始化>; /usr/sbin/sshd -D"`（Notebook 专用），必须用 `--entrypoint=/bin/bash` 覆盖，否则自定义命令会被吞掉、报 `cannot execute binary file`。

### 3.4 升级步骤

宿主机侧（提交前镜像已缓存在 `/public/.../images/mimo-sglang-bw10000_1.1.1.tar`，避免每次重新拉取）：

```bash
# 容器存储必须放节点本地磁盘（/public 是 NFS，不支持 overlay 所需 xattr）
export XDG_RUNTIME_DIR=/tmp/xdg-$$
PD="podman --root /tmp/pr --runroot /tmp/prr \
    --storage-driver overlay --storage-opt ignore_chown_errors=true"
$PD load -i /public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar   # 约 70–290 s

$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B:/models/MiMo:ro \
  -v /public/home/scnethr2601a10/work:/work \
  <镜像> bash -lc 'export LD_LIBRARY_PATH=...; cd /work/repo && bash run.sh'
```

容器内 `LD_LIBRARY_PATH` 必须包含（漏任一段都会报找不到 `.so`）：

```bash
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:\
/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:\
/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:\
/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:\
/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:$LD_LIBRARY_PATH
```

## 4. 服务启动脚本

`env.sh` 中与启动相关的配置：

```bash
export MODEL_PATH="/models/MiMo"
export SERVED_NAME="mimo-v2.6-9b"
export FRAMEWORK="vllm"
export CARD="BW1000"
export PORT=30000
export TP=1
export MAX_LEN=20480
export MEM_FRAC=0.95
export MAX_NUM_SEQS=128
export LOAD_TIMEOUT=1800
export EXTRA_ARGS="--trust-remote-code --dtype bfloat16 --max-num-seqs $MAX_NUM_SEQS"
```

启动：

```bash
bash run.sh
```

`run.sh` 展开后实际执行的完整命令（取自 `logs/server.log`）：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/MiMo \
  --served-model-name mimo-v2.6-9b \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128 \
  --host 127.0.0.1 --port 30000 \
  --trust-remote-code --dtype bfloat16
```

重点参数说明：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--max-model-len` | `20480` | 覆盖 16K + 1K 压测；开到模型最大长度时 KV cache 不足 |
| `--gpu-memory-utilization` | `0.95` | 0.85 时 vLLM 在 DCU 上报无可用 KV 空间 |
| `--max-num-seqs` | `128` | 对齐 64 并发档，避免排队 |
| CUDA Graph | 保持开启 | 见 §7.1 |
| `--tensor-parallel-size` | `1` | 权重 18 GB + KV，单卡 64 GB 足够 |

冷启动实测 **270–300 s**（权重加载 38.7 s，其余为 kernel 编译与显存 profile）。日志中
`Cannot use ROCm custom paged attention kernel, falling back to Triton implementation` 属预期回退，不影响正确性。

多机：本件单机单卡，未使用多机。

## 5. Smoke 测试

```bash
bash smoke.sh
```

`smoke.sh` 依次执行以下检查（任一接口非 2xx 即失败）：

```bash
curl -sf --noproxy '*' http://127.0.0.1:30000/health -o /dev/null -w "HTTP %{http_code}\n"
curl -sf --noproxy '*' http://127.0.0.1:30000/v1/models
curl -sf --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mimo-v2.6-9b","messages":[{"role":"user","content":"用一句话介绍国产异构加速器。"}],"temperature":0,"max_tokens":256}'
# 另有多轮（第三轮问「我叫什么名字？」）与流式（"stream": true），见 smoke.sh
```

验证结果（`logs/smoke.log`）：

- `/health` 返回 `200`
- `/v1/models` 返回 `mimo-v2.6-9b`，`max_model_len: 20480`
- 单轮问答回复：`<think>The user asks for a one-sentence introduction to domestic heterogeneous accelerators...`
- 多轮：正确回答 `你叫小海。`（`finish_reason: stop`）
- 流式：逐 token 返回，以 `data: [DONE]` 结束

额外做了一次 GSM8K 单题抽查（非官方评测，仅证明推理链路正确）：

```
Q: Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes. How much did she earn?
A: ... Earnings = $12 × 5/6 = $60/6 = $10.  \boxed{10}
finish_reason: stop    usage: {prompt_tokens: 41, completion_tokens: 81}
```

## 6. 精度测评

**⚠️ 本节当前被阻塞，尚未产出 `verify_result.json`。**

原因：`verify.py` 按官方基线文件 `baselines/<MODEL_ID 中 / 换成 __>.json` 里写死的 `子集路径 / 题数 / max_tokens / 思考开关 / 官方 GPU 基线分数` 执行评测，该文件属于官方下发物，赛题规则明确「不允许修改」；反作弊条款将「伪造日志」列为取消资格。因此在该文件与 `data/` 子集到位前，本仓库不产出任何精度分数。

`verify.py` 的判分逻辑本身已就绪并通过自检：

```bash
python verify.py --selfcheck     # → selfcheck OK
```

评测配置（**待官方下发后按 `baselines/<模型>.json` 原样填入**）：

| 数据集 | 子集题数 | max_tokens | 解码 | 思考开关 | 判分方式 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K | — | — | greedy（temperature=0） | — | 取 `\boxed{}` 或末尾数字，与答案数值相等 | **[BLOCKED]** 官方未下发 `baselines/MiMo-V2.6-Distill-Qwen-9B.json` |
| MMLU | — | — | greedy（temperature=0） | — | 取首个 A–D 选项字母 | **[BLOCKED]** 同上 |

> 这里刻意留空而不是填估算值：**子集、max_tokens、思考开关都由官方 `baselines/<模型>.json` 写死**，
> 自行猜测出来的分数与官方不可比，按本仓库红线（`docs/entry-spec.md` §6）不得入库。

结果（`verify_result.json` 原文）：

```json
【待官方基线文件下发后填入】
```

| 数据集 | 官方 GPU 基线 | 本次提交 | 劣化 | 容差 | 判定 |
| --- | --- | --- | --- | --- | --- |
| GSM8K | 【】 | 【】 | 【】 pt | ≤2 pt | 【】 |
| MMLU | 【】 | 【】 | 【】 pt | ≤2 pt | 【】 |

结论：**待官方下发 `baselines/` 与 `data/` 后补齐**。

> 备注：本模型不在赛题 §五 明确列出的「无官方精度数据、可不做精度」豁免名单内（豁免的是 `Recon2Reason-Reasoning-4B`、`RoboBrain2.5-8B-NV`），因此必须按官方基线验收。

## 7. 性能参考

性能测试脚本 `bench.sh`，负载固定：输入 4K / 16K × 输出 1K × 并发 1 / 8 / 64，共 6 档；每档连测两次，第一次预热，**取第二次数据**；请求数为并发的 2 倍（至少 8 个）。

压测客户端用**官方 sglang 的 `sglang.benchmark.serving`**（模板说明「SGLang / vLLM 服务通用」）。vLLM 镜像内不含该客户端，因此用**双容器**：容器 A 起 vLLM 服务（`--network=host`），容器 B（`mimo-sglang-bw10000` 镜像）跑压测客户端打 localhost。压测工具与 SGLang 件完全一致，两件数据可比。见 `container/submit.slurm`。

### 7.1 参数扫描与优化过程

首版用 `--enforce-eager`（关闭 CUDA Graph）+ 默认 `max_num_seqs`。随后在同一张卡、同一镜像、同一双容器压测链路上扫描三组配置（每档只跑 4K/c1、4K/c64、16K/c8 以缩短时间）：

| 变体 | 参数 | 4K/c1 | 4K/c64 | 16K/c8 |
| --- | --- | --- | --- | --- |
| W1（首版基准） | `mem=0.97 --enforce-eager` | 17.24 | 450.26 | 42.42 |
| W2 | `mem=0.97`（CUDA Graph 开启） | 20.66 | 490.87 | 44.94 |
| **W3（最终采用）** | `mem=0.95 --max-num-seqs 128`（CUDA Graph 开启） | **20.65** | **492.47** | **44.92** |

结论：**`--enforce-eager` 是首版的主要损失点**。仅去掉它（W2）就让 4K/c1 +19.8%、4K/c64 +9.0%、16K/c8 +5.9%。再加 `--max-num-seqs 128`（W3）与 W2 基本持平，说明 vLLM 的默认 `max_num_seqs` 已足够，真正的瓶颈是 CUDA Graph。

对比 SGLang 件的 73%~466%，vLLM 侧提升幅度小得多（3%~32%）——因为 SGLang 首版还叠加了 `max-running-requests=8` 这个更严重的错误。

### 7.2 最终性能基准（`bench/summary.md` 原文）

| random_input_len | random_output_len | max_concurrency | completed | duration | request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 1 | 8 | 395.18 | 0.02 | 20.73 | 103.65 | 432.72 | 457.34 | 47.86 | 49.24 | 49396.88 |
| 4096 | 1024 | 8 | 16 | 102.00 | 0.16 | 160.63 | 803.15 | 2035.96 | 4868.12 | 47.69 | 48.67 | 50820.77 |
| 4096 | 1024 | 64 | 128 | 267.76 | 0.48 | 489.52 | 2447.60 | 11325.69 | 42159.16 | 117.73 | 127.72 | 131764.91 |
| 16384 | 1024 | 1 | 8 | 1000.69 | 0.01 | 8.19 | 139.17 | 2936.11 | 3269.71 | 119.40 | 126.04 | 125085.32 |
| 16384 | 1024 | 8 | 16 | 365.49 | 0.04 | 44.83 | 762.07 | 17450.37 | 45115.45 | 159.75 | 179.64 | 180871.49 |
| 16384 | 1024 | 64 | 128 | 1300.03 | 0.10 | 100.82 | 1713.97 | 106464.10 | 396752.53 | 511.06 | 644.68 | 629276.54 |

**有效性校验：all valid**。

### 7.3 优化前后对比（同一卡、同一框架、同一负载、同一客户端）

首版基准存档于 `bench-baseline-v1/`（`summary.md` + `round2.jsonl`），可复核。

| 档位 | 优化前 (tok/s) | 优化后 (tok/s) | 提升 | p99 TTFT 前 → 后 (ms) |
| --- | --- | --- | --- | --- |
| 4K / c1 | 16.69 | 20.73 | **+24.2%** | 599 → 457 |
| 4K / c8 | 121.76 | 160.63 | **+31.9%** | 5342 → 4868 |
| 4K / c64 | 449.47 | 489.52 | **+8.9%** | 45338 → 42159 |
| 16K / c1 | 7.69 | 8.19 | **+6.4%** | 3564 → 3270 |
| 16K / c8 | 42.29 | 44.83 | **+6.0%** | 46868 → 45115 |
| 16K / c64 | 97.75 | 100.82 | **+3.1%** | 410826 → 396753 |

- 原始数据：`bench/round1.jsonl`、`bench/round2.jsonl`、各档日志 `bench/round*_*.log`
- 首版基准存档：`bench-baseline-v1/`

### 7.4 与同卡同模型另一框架的对比（参考）

| 档位 | SGLang 0.5.10（已优化） | vLLM 0.20.0（本件，已优化） |
| --- | --- | --- |
| 4K / c1 | **44.31** | 20.73 |
| 4K / c8 | **360.23** | 160.63 |
| 4K / c64 | **998.05** | 489.52 |
| 16K / c1 | **41.47** | 8.19 |
| 16K / c8 | **308.72** | 44.83 |
| 16K / c64 | **299.81** | 100.82 |

两件均在各自框架内做了参数扫描并取最优，SGLang 在 6 档全部领先，16K 档差距最大（5 倍）。差异来源可进一步分析（KV cache 布局、MoE 调度、线性注意力路径的实现差异），不属于本次适配范围。

### 7.1 长上下文（16K）专项调查 —— 配置层面无解

16K 档比同模型 SGLang 慢 5–7 倍，2026-11-03 做了三轮扫描（共 ~3 卡时），
完整数据见 [`sweep/results-2026-11-03.md`](sweep/results-2026-11-03.md)：

| 旋钮 | 16K/c1 | 16K/c8 | 判定 |
| --- | --- | --- | --- |
| base（mns 128 / mem 0.95） | 8.20 | 44.88 | 基准 |
| `--max-num-batched-tokens` 8192 / 2048 | 8.21 / 8.20 | 48.83 / 44.91 | 无影响 |
| `VLLM_ROCM_USE_AITER=1` | 8.20 | 44.92 | 无影响（内核仍不可用） |
| `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | 8.17 | 44.76 | 无影响 |
| `--kv-cache-dtype fp8` | 7.60 | 34.43 | **负优化** |
| `--tensor-parallel-size 2` | — | — | 启动失败（可见性/初始化问题） |

根因线索：`Cannot use ROCm custom paged attention kernel, falling back to Triton implementation`，
即 **DCU 上的 paged decode 只有 Triton 兜底实现**，TPOT 是 SGLang 的 5 倍。
要真正补上需要自建镜像（重编 aiter / 试 v0 引擎 / 改用 SGLang），不在本条目范围内。

## 8. 注意事项

> 主动说明的问题不扣分，评审复跑时才发现会打回。

- **容器运行时与官方脚本假设的差异**：官方脚本假设裸机（`pip install` + 直接起服务）。本件在容器内运行，适配集中在 §3.3 的参数与 §3.4 的挂载；`run.sh` 之后的所有脚本与官方口径一致。
- **镜像 ENTRYPOINT 必须覆盖**：该镜像是 SothisAI Notebook 镜像，`ENTRYPOINT=/bin/sh -c "<初始化>; /usr/sbin/sshd -D"`。不传 `--entrypoint=/bin/bash` 时自定义命令会被吞掉，报 `/usr/bin/bash: cannot execute binary file`。
- **`/dev/mkfd` 必需**：只传 `/dev/kfd` 会得到 `No HIP GPUs are available`，且错误发生在 torch 初始化阶段、无进一步提示，是本次适配最难定位的一处。
- **`--gpu-memory-utilization` 必须调高**：0.85 时报 `ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization`；0.95 正常（`0.97` 在 `max-num-seqs 128` 下偏紧）。
- **CUDA Graph 认知修正**：首版用 `--enforce-eager`，理由是"国产卡 CUDA Graph 捕获会失败"。§7.1 的扫描证伪了它：去掉 `--enforce-eager` 后 CUDA Graph 能正常捕获，且 4K/c1 +19.8%、4K/c64 +9.0%。本件最终配置**不使用** `--enforce-eager`。
- **压测客户端需单独容器**：vLLM 镜像内没有 `sglang.benchmark.serving`。官方模板要求用它（「SGLang / vLLM 服务通用」），因此用双容器 + `--network=host` 实现，压测工具与 SGLang 件保持一致。
- **`--context-length`/`--max-model-len` 必须留余量**：首次用 `17408`（= 16384 + 1024）跑压测，**16K 档全部被服务端 400 拒绝**——客户端 `--random-input-len` 生成的 token 数不含 chat template 包装开销，实际输入达 16740–18784 tokens。更危险的是压测客户端把 400 当「完成」，打出 `output_throughput 9791 tok/s`、`TTFT 0.00 ms` 的假数据。现改为 `20480`，并在 `bench.sh` 里加了两道防护：① 压测前用 `/tokenize` 精确构造 ~16.4K tokens 的请求做 preflight，非 200 就退出；② 压测后校验 `mean_ttft_ms != 0` 且吞吐不超物理上限，否则判 INVALID 并拒绝写入报告。
- **容器存储不能放 `/public`**：NFS 不支持 overlay 所需 xattr，`podman load/pull` 报 `lsetxattr /boot: operation not supported`。必须用节点本地盘，且**每个作业用独立 `--root`**，否则同节点上一个作业残留的 podman pause/storage 状态会让 `podman pull` 直接失败（`invalid internal status ... podman system migrate`）。
- **节点磁盘水位不稳**：实测同一分区不同节点根盘可用空间从 0 GB 到 254 GB 不等。本件需要同时装两个镜像（vLLM 29.6 GB + SGLang 32.2 GB），提交前需 `df` 检查，不足 68 GB 主动 `exit 42` 重投。
- **冷启动 270–300 s**：含权重加载 38.7 s 与 kernel 编译；`LOAD_TIMEOUT` 已调为 1800 s。
- **可忽略的告警**：`Cannot use ROCm custom paged attention kernel, falling back to Triton implementation`、`Input tensor shape suggests potential format mismatch: seq_len (14) < num_heads (16)` —— 均为 DCU 上的预期回退/提示，不影响正确性。
- **全 6 档均跑通，无跳过档位**。
- **精度章节未完成**：见 §6，阻塞于官方 `baselines/` 与 `data/` 下发，详见 `MISSING.md`。

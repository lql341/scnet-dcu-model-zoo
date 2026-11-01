# Kimi-Linear-48B-A3B-Instruct on BW1000 + SGLang 0.5.10

> 提交仓库。技术报告即本文件，按官方《国芯国模黑客松 · 技术报告模板》8 个章节组织。
> 本文件中的版本号、精度分数与性能数字均可由 `logs/env.log`、`verify_result.json`、`bench/round2.jsonl` 复现。

## 1. 简介

本文档记录 `moonshotai/Kimi-Linear-48B-A3B-Instruct` 在 `异构加速卡BW（BW1000）` + `SGLang 0.5.10`（精度 `bf16`，`TP=4`）环境的适配与验证结果。

模型是月之暗面开源的混合线性注意力 MoE 模型（Instruct 指令版）。核心是 **Kimi Delta Attention（KDA）**——对 Gated DeltaNet 门控机制的改进，与全局 MLA 按 3:1 混合。本件实测的架构参数（取自 `config.json`）：`KimiLinearForCausalLM`（`kimi_linear`），27 层，其中 **20 层 KDA**（`kda_layers`）+ **7 层全注意力**（`full_attn_layers = [4,8,12,16,20,24,27]`），256 专家，`hidden_size=2304`，32 头 / `head_dim=128`，`short_conv_kernel_size=4`，词表 163840；BF16 权重 20 分片共 **91.5 GB**，MIT 许可。

**与官方部署方式的核心差异：无。**

这一点值得单独说明。官方模型卡建议 `vllm serve ... --tensor-parallel-size 4 --trust-remote-code`，并提到需要 `fla-core ≥ 0.4.0`；模板里的 patch 示例也是为它在国产卡上改 KDA kernel（`BT=64 → 32` 以适配 LDS 容量）。但本件实测发现：

- **vLLM 路线不可用**：vLLM 0.20 的内置模型列表里没有 `kimi_linear`（只有 `kimi_k25` / `kimi_vl` / `kimi_audio`），该架构只能走 SGLang；
- **SGLang 0.5.10rc0 已内置全部所需实现**：`sglang.srt.models.kimi_linear`、`sglang.srt.configs.kimi_linear`，KDA kernel 以 **vendored** 形式放在 `sglang/srt/layers/attention/fla/{kda,chunk,fused_norm_gate}.py`；
- 因此**不需要安装外部 `fla-core`**（实测 `import fla` 为 `ModuleNotFoundError`，但服务正常启动），**也不需要任何 patch**。

本件的适配全部通过启动参数完成：`--tp-size 4`（91.5 GB 切成 ~22.9 GB/卡）、`--attention-backend triton`、`--disable-cuda-graph`、`--context-length 20480`、`--max-running-requests 64`。

相关获取地址：

- 权重下载（ModelScope）：`https://www.modelscope.cn/models/moonshotai/Kimi-Linear-48B-A3B-Instruct`
- 镜像：`image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1`
- 提交仓库：https://developer.sourcefind.cn/codes/luql/kimi-linear-48b-bw1000

参考文档：

- 官方模型卡 `https://www.modelscope.cn/models/moonshotai/Kimi-Linear-48B-A3B-Instruct`
- SGLang 启动参数：镜像内 `python3 -m sglang.launch_server --help`

仓库内容：

| 文件 | 作用 | 产出 |
| --- | --- | --- |
| `env.sh` | 模型、镜像、框架、卡型、并行方式与启动参数，其余脚本读这里的变量 | — |
| `env_check.sh` | 采集卡型与重点组件版本 | `logs/env.log` |
| `requirements-extra.txt` | 在镜像基础上升级的包（**本件无**，见 §3.2） | — |
| `patches/` | 代码改动（**本件无**，见 §3.3） | — |
| `run.sh` | 模型启动脚本：装依赖 + 重放 patch + 拉起服务，轮询 `/health` | `logs/server.log` |
| `smoke.sh` | Smoke 测试：健康检查、单轮 / 多轮 / 流式问答 | `logs/smoke.log` |
| `verify.py` | 精度验证脚本：按官方基线文件跑评测并判定 | `verify_result.json` |
| `bench.sh` | 性能测试脚本：6 档负载，每档连测两次 | `bench/` |
| `container/submit.slurm` | 容器启动封装（BW1000 上的 podman + Slurm，4 卡） | `sbatch-*.out` |
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
git clone https://developer.sourcefind.cn/codes/luql/kimi-linear-48b-bw1000.git
cd kimi-linear-48b-bw1000
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

镜像：`image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1`

重点组件（取自 `logs/env.log`）：

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| DTK | `26.04`（`DTK_VERSION=2604`） | 镜像自带，`ROCM_PATH=/opt/dtk` |
| `torch` | `2.9.0+das.opt1.dtk2604` | 镜像自带，HIP `6.3.26113` |
| `triton` | `3.5.1+das.opt1.dtk2604` | 镜像自带 |
| `sglang` | `0.5.10rc0+das.opt2.alpha.dtk2604.torch290` | 镜像自带（das 优化版） |
| `transformers` | `5.3.0` | 镜像自带 |
| `hyhal` | 宿主挂载 `/opt/hyhal`（`libhydmi.so.1.5`） | 与宿主内核驱动匹配，**必须挂载** |

- 卡型：`异构加速卡BW（BW1000）`，`arch=gfx936:sramecc+:xnack-`，单卡 64.0 GB
- 卡数：**4**（`TP=4`），单节点 8 卡节点上用 4 卡
- 模型路径（容器内）：`/models/Kimi`（宿主：`/public/home/scnethr2601a10/models/Kimi-Linear-48B-A3B-Instruct`）
- 服务端口：`30000`

环境自检（`logs/env.log` 关键部分）：

```text
image: image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
dtk: DTK_VERSION=2604
HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%      Mode
0       54.0C    94.0W      manual   1000.0W    86%        0.0%      Normal
1       51.0C    148.0W     auto     1000.0W    86%        0.0%      Normal
2       53.0C    113.0W     auto     1000.0W    86%        0.0%      Normal
3       54.0C    83.0W      auto     1000.0W    86%        0.0%      Normal
python: 3.10.12
torch: 2.9.0
HIP: 6.3.26113
available: True count: 4
Device: BW x 4 vram_GB: 64.0 arch: gfx936:sramecc+:xnack-
sglang: 0.5.10rc0+das.opt2.alpha.dtk2604
triton: 3.5.1
transformers: 5.3.0
```

超算互联网平台（参赛算力统一使用超算互联网）：

| 项 | 值 |
| --- | --- |
| 区域 | 华中一区【A区】（郑州，`hx1hdnormal` 分区） |
| 作业规格 | 4 × BW1000，32 CPU，`-t 03:30:00` |
| 加速卡与显存 | BW1000 × 4；64 GB/卡 |
| 开发镜像 | `dcu/linshp/base/mimo-sglang-bw10000:1.1.1` |
| 模型获取方式 | ModelScope 直连下载（`snapshot_download`），17 分 43 秒 / 39 文件 |
| 服务映射 | 容器内 `127.0.0.1:30000`；Slurm 作业内自测，未对外暴露（「无」） |
| 平台证据 | `platform/`（作业提交与 `hy-smi` 截图待补） |

## 3. 适配优化方案

### 3.1 基础镜像

基于 `image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1`。选它的原因：它是 Harbor 公共镜像里同时满足「SGLang das 版 + DTK 26.04 + gfx936 可跑」的组合，且 `sglang 0.5.10rc0` 已内置 `kimi_linear` 实现与 vendored KDA kernel（见 §3.3 实测记录）。

### 3.2 软件包升级

**无。** `requirements-extra.txt` 仅保留说明性注释。

关键实测结论：官方模型卡建议的 `fla-core ≥ 0.4.0` 在本件**不需要**。在容器内逐个 import 的结果：

```text
  OK    sglang.srt.models.kimi_linear
  OK    sglang.srt.configs.kimi_linear
  OK    sglang.srt.layers.attention.fla.kda
  OK    sglang.srt.layers.attention.fla.chunk
  OK    sglang.srt.layers.attention.fla.fused_norm_gate
  FAIL  fla  ModuleNotFoundError: No module named 'fla'
  OK    einops
```

`kimi_linear.py` 的 KDA 相关 import 全部指向 `sglang.srt.layers.attention.fla.*`（vendored），外部 `fla` 包不存在也不影响；`einops` 镜像已自带。服务在 `fla` 缺失的情况下正常启动并通过全部 smoke。

### 3.3 代码改动

**无代码 patch**，`patches/` 为空。

模板中的 patch 示例是为本模型准备的（`sglang/srt/layers/attention/fla/chunk.py` 里 `BT = 64 → 32`，规避国产卡 LDS 容量小于 A100 共享内存导致 kernel 申请超限）。本件实测**该 patch 不需要**：镜像内的 das 优化版已在 vendored kernel 中处理好分块尺寸，`--attention-backend triton` 下可直接运行，未出现 `out of resource: shared memory`。

适配全部通过启动参数完成：

| 参数 | 取值 | 原因 |
| --- | --- | --- |
| `--tp-size` | `4` | 91.5 GB BF16 权重切成 ~22.9 GB/卡，每卡余 ~31 GB 给 KV cache |
| `--context-length` | `20480` | 覆盖 bench 的 16K 输入 + 1K 输出 + chat template 开销。**17408 不够**：实测被服务端 400 拒绝（详见 §8） |
| `--mem-fraction-static` | `0.85` | 实测 4 卡 VRAM 占用 86%，留余量避免 OOM |
| `--max-running-requests` | `64` | 与 bench 的 64 并发档对齐。若沿用 8，64 并发请求会在服务端排队，测出的是排队延迟而非框架能力（对照见 §7） |
| `--attention-backend` | `triton` | gfx936 上默认 flash-attention 路径无可用 kernel |
| `--disable-cuda-graph` | — | 国产卡 CUDA Graph 捕获阶段进程静默退出 |
| `--trust-remote-code` | — | 模型仓库带 `configuration_kimi.py` 自定义代码 |

**容器运行时的三条硬性要求**（不属于代码改动，但不满足则拿不到卡）：

1. 必须传 `--device /dev/mkfd`：BW1000 上存在第二个 KFD 设备 `mkfd`，只传 `/dev/kfd` 时 HSA 报 `No HIP GPUs are available`；
2. 必须把**宿主** `/opt/hyhal` 挂进容器（提供与内核驱动匹配的 `libhydmi.so`）；
3. **不要**用宿主 DTK 覆盖容器内 `/opt/dtk`，否则 Triton JIT 报 `ROCm compiler not found under /opt/dtk/aillvm/bin/clang`。

### 3.4 升级步骤

宿主机侧（镜像已缓存在 `/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar`）：

```bash
export XDG_RUNTIME_DIR=/tmp/xdg-$$
JID="${SLURM_JOB_ID:-$$}"
# 容器存储必须放节点本地磁盘（/public 是 NFS，不支持 overlay 所需 xattr）；
# 且每个作业要用独立 --root，否则同节点上一个作业残留的 podman 状态会让 pull 直接失败
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID \
    --storage-driver overlay --storage-opt ignore_chown_errors=true"
$PD load -i /public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar   # 166–270 s

$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined \
  --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /public/home/scnethr2601a10/models/Kimi-Linear-48B-A3B-Instruct:/models/Kimi:ro \
  -v /public/home/scnethr2601a10/work/kimi:/work \
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
export MODEL_PATH="/models/Kimi"
export SERVED_NAME="kimi-linear-48b"
export FRAMEWORK="sglang"
export CARD="BW1000"
export PORT=30000
export TP=4
export MAX_LEN=20480
export MEM_FRAC=0.85
export MAX_RUNNING_REQUESTS=64
export LOAD_TIMEOUT=1800     # 实测冷启动 280–360 s（TP=4）
export EXTRA_ARGS="--attention-backend triton --disable-cuda-graph"
```

启动：

```bash
bash run.sh
```

`run.sh` 展开后实际执行的完整命令（取自 `logs/server.log` 首行）：

```bash
python3 -m sglang.launch_server \
  --model-path /models/Kimi \
  --served-model-name kimi-linear-48b \
  --tp-size 4 --context-length 20480 \
  --mem-fraction-static 0.85 \
  --max-running-requests 64 \
  --host 127.0.0.1 --port 30000 \
  --trust-remote-code \
  --attention-backend triton --disable-cuda-graph
```

重点参数说明：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--tp-size` | `4` | 权重 91.5 GB；`TP=2` 时每卡 46 GB，KV cache 余量不足 20480 上下文 |
| `--context-length` | `20480` | 覆盖 16K + 1K 压测；开到模型最大长度（1M）显存不够 |
| `--mem-fraction-static` | `0.85` | 实测 VRAM 86%，再高易 OOM |
| `--max-running-requests` | `64` | 对齐 64 并发档；见 §7 的排队对照 |
| `--disable-cuda-graph` | — | 国产卡必需 |

多机：本件单机 4 卡，未使用多机。

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
  -d '{"model":"kimi-linear-48b","messages":[{"role":"user","content":"用一句话介绍国产异构加速器。"}],"temperature":0,"max_tokens":256}'
# 另有多轮（第三轮问「我叫什么名字？」）与流式（"stream": true），见 smoke.sh
```

验证结果（`logs/smoke.log`）：

- `/health` 返回 `200`
- `/v1/models` 返回 `kimi-linear-48b`，`max_model_len: 20480`
- 单轮问答回复：`国产异构加速器是以专用计算架构为核心，融合CPU与GPU/FPGA/ASIC等异构单元，实现高性能、高能效计算的中国自主技术平台。`
- 多轮：正确识别「小海」（`你刚刚说：“我叫小海。” 所以我知道你的名字是 **小海**。`），`finish_reason: stop`
- 流式：逐 token 返回，以 `data: [DONE]` 结束

## 6. 精度测评

**⚠️ 本节当前被阻塞，尚未产出 `verify_result.json`。**

原因：`verify.py` 按官方基线文件 `baselines/<MODEL_ID 中 / 换成 __>.json` 里写死的 `子集路径 / 题数 / max_tokens / 思考开关 / 官方 GPU 基线分数` 执行评测，该文件属于官方下发物，赛题规则明确「不允许修改」；反作弊条款将「伪造日志」列为取消资格。因此在该文件与 `data/` 子集到位前，本仓库不产出任何精度分数。

`verify.py` 的判分逻辑本身已就绪并通过自检：

```bash
python verify.py --selfcheck     # → selfcheck OK
```

评测配置（**待官方下发后按 `baselines/<模型>.json` 原样填入**）：

| 数据集 | 子集题数 | max_tokens | 解码 | 思考开关 | 判分方式 |
| --- | --- | --- | --- | --- | --- |
| GSM8K | 【待填】 | 【待填】 | greedy（temperature=0） | 【待填】 | 取 `\boxed{}` 或末尾数字，与答案数值相等 |
| MMLU | 【待填】 | 【待填】 | greedy（temperature=0） | 【待填】 | 取首个 A–D 选项字母 |

结果（`verify_result.json` 原文）：

```json
【待官方基线文件下发后填入】
```

| 数据集 | 官方 GPU 基线 | 本次提交 | 劣化 | 容差 | 判定 |
| --- | --- | --- | --- | --- | --- |
| GSM8K | 【】 | 【】 | 【】 pt | ≤2 pt | 【】 |
| MMLU | 【】 | 【】 | 【】 pt | ≤2 pt | 【】 |

结论：**待官方下发 `baselines/` 与 `data/` 后补齐**。

> 备注：本模型不在赛题 §五 明确列出的「无官方精度数据、可不做精度」豁免名单内（豁免的是 `Recon2Reason-Reasoning-4B`、`RoboBrain2.5-8B-NV`）。行业现状见 `MISSING.md`：截至 2026-10-30，公开提交仓库中**没有一个**拿到 `baselines/`，所有人都在同一处阻塞。

## 7. 性能参考

性能测试脚本 `bench.sh` 使用 SGLang 自带压测客户端，随机 token 输入、固定输入输出长度，负载如下：

| 输入 | 输出 | 并发 |
| --- | --- | --- |
| 4K（4096） | 1K（1024） | 1 / 8 / 64 |
| 16K（16384） | 1K（1024） | 1 / 8 / 64 |

每档连测两次，第一次预热，**取第二次数据**；请求数为并发的 2 倍（至少 8 个）。

```bash
bash bench.sh
```

单档命令（`bench.sh` 内部对 6 档 × 2 次循环执行）：

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --model kimi-linear-48b --tokenizer /models/Kimi \
  --dataset-name random-ids \
  --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 \
  --max-concurrency 8 --num-prompts 16 --seed 1 \
  --output-file bench/round2.jsonl
```

第二次结果（`bench/summary.md` 原文）：

| random_input_len | random_output_len | max_concurrency | completed | duration | request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 1 | 8 | 564.15 | 0.01 | 14.52 | 72.60 | 260.29 | 269.51 | 68.68 | 71.28 | 70516.59 |
| 4096 | 1024 | 8 | 16 | 147.59 | 0.11 | 111.01 | 555.03 | 1704.18 | 2600.70 | 70.46 | 73.70 | 73788.59 |
| 4096 | 1024 | 64 | 128 | 215.50 | 0.59 | 608.23 | 3041.13 | 11441.21 | 21179.62 | 94.09 | 117.27 | 107692.25 |
| 16384 | 1024 | 1 | 8 | 584.55 | 0.01 | 14.01 | 238.24 | 1759.22 | 2062.27 | 69.70 | 79.09 | 73066.82 |
| 16384 | 1024 | 8 | 16 | 202.95 | 0.08 | 80.73 | 1372.41 | 15800.85 | 25744.48 | 83.74 | 103.08 | 101463.96 |
| 16384 | 1024 | 64 | 128 | 524.51 | 0.24 | 249.89 | 4248.20 | 100050.92 | 183974.57 | 158.51 | 248.50 | 262209.38 |

**有效性校验：all valid**（6 档 `mean_ttft_ms` 均非 0，吞吐均在物理上限内）。

几点说明：

- **6 档全部跑通**，`completed` 与 `num-prompts` 一致（8 / 16 / 128），无请求丢弃。
- **单流（并发 1）稳定在 14 tok/s 左右**，4K 与 16K 输入几乎无差异（14.52 vs 14.01），说明线性注意力对长上下文的解码开销影响很小。
- **4K 并发 64 达到 608.23 tok/s**，是单流的 42 倍；`mean_ttft` 11.4 s、`p99_ttft` 21.2 s，属于 64 路并发下的正常排队。
- **16K 并发 64 为 249.89 tok/s**，`mean_ttft` 100 s。16K 档的瓶颈在 prefill：128 个请求 × 16384 tokens 的 prefill 总量远超 KV 读取。
- 4 卡 VRAM 稳定在 88%，功耗在 82–529 W 之间波动。

### 并发上限对性能基准的影响（重要）

本件把 `--max-running-requests` 设为 **64**，与 bench 的最高并发档一致。若不设此项（SGLang 默认较小），64 档的请求会在服务端排队，测出的将是排队延迟而非框架能力。作为对照，同卡同框架下把该值设为 8 时（另一模型 MiMo 的件），4K×64 档只有 176 tok/s，而本件放开后为 608 tok/s。**建议评审复跑时保持 `MAX_RUNNING_REQUESTS=64`**，否则数字不可比。

- 原始数据：`bench/round1.jsonl`、`bench/round2.jsonl`、各档日志 `bench/round*_*.log`
- 本结果作为该件「模型 × 卡型 × 框架 × 精度」的性能基准；后续优化在同卡型、同框架、同负载下对比

## 8. 注意事项

> 主动说明的问题不扣分，评审复跑时才发现会打回。

- **本模型只有 SGLang 一条路**：vLLM 0.20 的内置模型列表里没有 `kimi_linear`（只有 `kimi_k25` / `kimi_vl` / `kimi_audio`），官方模型卡给的 `vllm serve` 命令在 vLLM 上无法直接加载该架构。
- **不需要 `fla-core`，也不需要 patch**：见 §3.2 / §3.3 的实测记录。这与官方模型卡和模板 patch 示例的描述不同，是本件最重要的发现之一。
- **`/tokenize` 端点在本模型上不可用**：调用会返回 500，根因是 sglang 用 orjson 序列化 token id 时报 `TypeError: Integer exceeds 64-bit range`（词表 163840 的 id 在某些路径下越界）。因此 `bench.sh` 的前置校验改为「直接发 chat 请求 + 长度阶梯探测」，不依赖 `/tokenize`。
- **`--context-length` 必须留余量**：`17408`（= 16384 + 1024）不够——客户端 `--random-input-len` 不计 chat template 包装开销，实测被服务端 400 拒绝（`Requested token count exceeds the model's maximum context length`）。更危险的是压测客户端把 400 当「完成」，会打出 `TTFT 0.00 ms` 的假数据。现改为 `20480`，并在 `bench.sh` 里加了两道防护：① 前置长度阶梯探测；② 压测后校验 `mean_ttft_ms != 0` 且吞吐不超物理上限，否则判 INVALID 并拒绝写入报告。
- **`/dev/mkfd` 必需**：只传 `/dev/kfd` 会得到 `No HIP GPUs are available`，错误发生在 torch 初始化阶段、无进一步提示，是最难定位的一处。
- **`--disable-cuda-graph` 必需**：否则进程在 CUDA Graph 捕获阶段静默退出，`logs/server.log` 无 traceback。
- **容器存储不能放 `/public`**：NFS 不支持 overlay 所需 xattr，报 `lsetxattr /boot: operation not supported`。必须用节点本地盘，且**每个作业用独立 `--root`**，否则同节点上一个作业残留的 podman pause/storage 状态会让 `podman load/pull` 直接失败。
- **节点磁盘水位不稳**：实测同一分区不同节点根盘可用空间从 0 GB 到 278 GB 不等。本件镜像解压后约 32 GB，提交前需 `df` 检查，不足 45 GB 主动 `exit 42` 重投。
- **冷启动 280–360 s**：TP=4 下含 91.5 GB 权重加载与 Triton kernel JIT。`LOAD_TIMEOUT` 已调为 1800 s。
- **可忽略的告警**：`Using a slow tokenizer. This might cause a significant slowdown.` —— 该模型只有 slow tokenizer，属预期。
- **全 6 档均跑通，无跳过档位**。
- **精度章节未完成**：见 §6，阻塞于官方 `baselines/` 与 `data/` 下发，详见 `MISSING.md`。

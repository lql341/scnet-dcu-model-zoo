# MiMo-V2.6-Distill-Qwen-9B on BW1000 + SGLang 0.5.10

> 提交仓库。技术报告即本文件，按官方《国芯国模黑客松 · 技术报告模板》8 个章节组织。
> 本文件中的版本号、精度分数与性能数字均可由 `logs/env.log`、`verify_result.json`、`bench/round2.jsonl` 复现。

## 1. 简介

本文档记录 `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` 在 `异构加速卡BW（BW1000）` + `SGLang 0.5.10`（精度 `bf16`）环境的适配与验证结果。

模型为小米 MiMo 团队以 Qwen3.5-9B 为基座的 9B 智能体模型，`Qwen3_5ForConditionalGeneration` 架构：32 层（24 层线性注意力 + 8 层全注意力，每 4 层 1 层全注意力），带视觉塔，256K 上下文，BF16 共 4 分片约 18.9 GB，MIT 许可。

与官方部署方式的核心差异：官方只给出 `sglang serve --reasoning-parser mimo` 的形式，未覆盖国产卡。本件在 BW1000 上需要 ① 用 DTK 26.04 的 SGLang das 版镜像，② 通过 `--attention-backend triton` 绕开默认 flash-attention 路径，③ `--max-running-requests 64` 对齐压测最高并发档（设 8 会让 64 并发请求排队，吞吐被压到 176 tok/s），④ 打开海光专属算子开关（`SGLANG_USE_LIGHTOP` / AITER 系列）。**CUDA Graph 保持开启** —— §7.1 的扫描证明早期「必须禁用」的结论不成立。无包升级、无代码 patch。

相关获取地址：

- 权重下载（ModelScope）：`https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`
- 镜像：`image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1`
- 提交仓库：https://developer.sourcefind.cn/codes/luql/mimo-sglang-bw1000

参考文档：

- 官方模型卡 `https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`
- SGLang 启动参数：镜像内 `python3 -m sglang.launch_server --help`

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
git clone https://developer.sourcefind.cn/codes/luql/mimo-sglang-bw1000.git
cd mimo-sglang-bw1000
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

另外，§7.1 的参数扫描脚本为 `sweep/sweep_params.slurm`：循环起停服务并对比多组启动参数，每档只跑 3 个关键组合（4K/c1、4K/c64、16K/c8）以缩短时间。

## 2. 验证环境

镜像：`image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1`

重点组件（取自 `logs/env.log`）：

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| DTK | `26.04`（`DTK_VERSION=2604`） | 镜像自带，`ROCM_PATH=/opt/dtk` |
| `torch` | `2.9.0+das.opt1.dtk2604.2605140906.gcbed15` | 镜像自带，HIP `6.3.26113` |
| `triton` | `3.5.1+das.opt1.dtk2604.torch290.2605181637.gd4164b` | 镜像自带 |
| `sglang` | `0.5.10rc0+das.opt2.alpha.dtk2604.torch290.2605181752.g91be88` | 镜像自带（das 优化版） |
| `sglang-kernel` | `0.4.0+das.opt1.dtk2604.torch290.2605181752.g91be88` | 镜像自带 |
| `sglang-router` | `0.3.2+das.opt1.dtk2604.torch290.2605181752.g91be88` | 镜像自带 |
| `flash-attn` | `2.8.3+das.opt1.dtk2604.torch290.2605181419.g67729e` | 镜像自带（本件改用 triton 后端，未启用） |
| `flash_mla` | `1.0.0+das.opt1.dtk2604.torch290.2604100931.g261aa8` | 镜像自带 |
| `transformers` | `5.3.0` | 镜像自带 |
| `tokenizers` | `0.22.2` | 镜像自带 |
| `hyhal` | 宿主挂载 `/opt/hyhal`（`libhydmi.so.1.5`） | 与宿主内核驱动匹配，**必须挂载** |

- 卡型：`异构加速卡BW（BW1000）`，`arch=gfx936:sramecc+:xnack-`，单卡 64.0 GB
- 卡数：1（`TP=1`）
- 模型路径（容器内）：`/models/MiMo`（宿主：`/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B`）
- 服务端口：`30000`

环境自检（`logs/env.log` 关键部分）：

```text
image: image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
dtk:   DTK_VERSION=2604
HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%      Dec%      Enc%      Mode
0       52.0C    75.0W      auto     1000.0W    0%         0.0%      0.0%      0.0%      Normal
python: 3.10.12
torch: 2.9.0
HIP: 6.3.26113
cuda_available: True
device_count: 1
Device: BW
vram_GB: 64.0
arch: gfx936:sramecc+:xnack-
sglang: 0.5.10rc0+das.opt2.alpha.dtk2604
triton: 3.5.1
transformers: 5.3.0
tokenizers: 0.22.2
flash-attn 2.8.3+das.opt1.dtk2604.torch290.2605181419.g67729e
sglang-kernel 0.4.0+das.opt1.dtk2604.torch290.2605181752.g91be88
torch 2.9.0+das.opt1.dtk2604.2605140906.gcbed15
triton 3.5.1+das.opt1.dtk2604.torch290.2605181637.gd4164b
```

超算互联网平台（参赛算力统一使用超算互联网）：

| 项 | 值 |
| --- | --- |
| 区域 | 华中一区【A区】（郑州，`hx1hdnormal` 分区） |
| 作业规格 | 1 × BW1000，16 CPU，`-t 02:30:00` |
| 加速卡与显存 | BW1000 × 1；64 GB |
| 开发镜像 | `dcu/linshp/base/mimo-sglang-bw10000:1.1.1` |
| 模型获取方式 | ModelScope 直连下载（`snapshot_download`），24 MB/s，3 分钟 |
| 服务映射 | 容器内 `127.0.0.1:30000`；Slurm 作业内自测，未对外暴露（「无」） |
| 平台证据 | `platform/`（作业提交与 `hy-smi` 截图待补） |

## 3. 适配优化方案

### 3.1 基础镜像

基于 `image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1`。选它的原因：它是当前 Harbor 公共镜像里唯一同时满足「SGLang 0.5.10 das 版 + DTK 26.04 + gfx936 可跑」的组合，自带 `sglang 0.5.10rc0+das.opt2.alpha.dtk2604` 与 `triton 3.5.1`，无需自行编译。

### 3.2 软件包升级

无。镜像内组件版本已满足要求，`requirements-extra.txt` 仅保留说明性注释。

### 3.3 代码改动

无代码 patch，`patches/` 为空。所有适配通过运行时环境变量与启动参数完成：

| 项 | 取值 | 原因 |
| --- | --- | --- |
| `SGLANG_USE_LIGHTOP` | `1` | 启用海光 lightop 算子库（`libmsgpackc` / 融合算子所在处） |
| `SGLANG_KV_LAYOUT_DCU_FA` | `0` | 国产卡上 DCU flash-attention 的 KV layout 与本模型线性注意力层不兼容，置 0 走 triton 路径 |
| `SGLANG_USE_AITER_FP8_ASM_MOE` | `1` | 启用 AITER 系列 MoE 融合算子（本模型 256 专家 / 激活 8） |
| `SGLANG_USE_TRITON_EXTEND_FROM_AITER` | `1` | 同上，扩展 attention/moe 的 triton 实现 |
| `SGLANG_ROCM_FUSED_DECODE_MLA` | `0` | 本模型为线性注意力 + 全注意力混合，非纯 MLA，禁用融合解码避错 |
| `--attention-backend` | `triton` | 默认 flash-attention 路径在 gfx936 上无可用 kernel |
| `--disable-cuda-graph` | **不使用** | 早期试探时曾因 CUDA Graph 捕获崩溃而禁用；后续系统扫描（见 §7.1）发现**该镜像下 CUDA Graph 可正常开启**，且开启后 4K/c1 +84.5%、4K/c64 +466.4%。本件最终配置**保持 CUDA Graph 开启** |
| `--max-running-requests` | `64` | 与 bench 最高并发档对齐。设为 8 时 64 个请求里 56 个在排队（`logs/server.log` 可见 `#queue-req: 56`），4K×64 档只有 176 tok/s |

**容器运行时的三条硬性要求**（不属于代码改动，但不满足则拿不到卡）：

1. 必须传 `--device /dev/mkfd`：BW1000 上存在第二个 KFD 设备 `mkfd`，只传 `/dev/kfd` 时 HSA 报 `No HIP GPUs are available`；
2. 必须把**宿主** `/opt/hyhal` 挂进容器（提供与内核驱动匹配的 `libhydmi.so`）；
3. **不要**用宿主 DTK 覆盖容器内 `/opt/dtk`，否则 `ROCm compiler not found under /opt/dtk/aillvm/bin/clang`（Triton JIT 需要 `clang-18`）。

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
export FRAMEWORK="sglang"
export CARD="BW1000"
export PORT=30000
export TP=1
export EP=1
export PP=1
export MAX_LEN=17408        # 覆盖 16K 输入 + 1K 输出的压测
export MEM_FRAC=0.85
export MAX_RUNNING_REQUESTS=64
export LOAD_TIMEOUT=1800    # 实测冷启动 410 s（含 Triton JIT）
export EXTRA_ARGS="--attention-backend triton"
```

启动：

```bash
bash run.sh
```

`run.sh` 展开后实际执行的完整命令（取自 `logs/server.log` 首行）：

```bash
python3 -m sglang.launch_server \
  --model-path /models/MiMo \
  --served-model-name mimo-v2.6-9b \
  --tp-size 1 --context-length 20480 \
  --mem-fraction-static 0.85 \
  --max-running-requests 64 \
  --host 127.0.0.1 --port 30000 \
  --trust-remote-code \
  --attention-backend triton
```

重点参数说明：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--tp-size` | `1` | 权重 18 GB + KV，单卡 64 GB 足够；`TP=1` 无跨卡通信开销 |
| `--context-length` | `17408` | 覆盖 bench 的 16K 输入 + 1K 输出；模型自身支持 256K，但开到最大时 KV cache 不足 |
| `--mem-fraction-static` | `0.85` | 0.9 时 64 并发档位 OOM；0.8 时实测 VRAM 81% |
| `--max-running-requests` | `64` | 与服务端 64 并发档对齐；见 §7.1 的扫描结论 |
| `--attention-backend` | `triton` | gfx936 上默认 FA 路径无 kernel |
| CUDA Graph | **保持开启** | 未传 `--disable-cuda-graph`；见 §7.1 |
| `--trust-remote-code` | — | 模型使用自定义 `Qwen3_5ForConditionalGeneration` |

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

> 备注：本模型不在赛题 §五 明确列出的「无官方精度数据、可不做精度」豁免名单内（豁免的是 `Recon2Reason-Reasoning-4B`、`RoboBrain2.5-8B-NV`），因此必须按官方基线验收。

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

### 7.1 参数扫描与优化过程

首版提交用的是保守配置（`--max-running-requests 8` + `--disable-cuda-graph`）。随后做了一轮配置扫描，在**同一张卡、同一镜像、同一压测客户端**下对比 5 组配置，每档只跑 3 个关键组合（4K/c1、4K/c64、16K/c8）以缩短扫描时间：

| 变体 | 参数 | 4K/c1 (tok/s) | 4K/c64 (tok/s) | 16K/c8 (tok/s) |
| --- | --- | --- | --- | --- |
| V1（首版基准） | `mem=0.85 mrr=8 disable-cuda-graph` | 24.19 | 179.29 | 140.37 |
| V2 | `mem=0.85 mrr=64 disable-cuda-graph` | 24.64 | 934.10 | 145.55 |
| V3 | `mem=0.92 mrr=256 disable-cuda-graph` | 24.37 | 941.83 | 143.37 |
| **V4（最终采用）** | `mem=0.85 mrr=64`（**CUDA Graph 开启**） | **44.24** | **995.34** | **216.34** |
| V5 | `mem=0.90 mrr=64 chunked-prefill=8192 disable-cuda-graph` | 24.62 | 949.57 | 145.86 |

两个结论：

1. **`--max-running-requests 8` 是高并发档的唯一瓶颈**。V2 相比 V1，4K/c64 从 179 → 934 tok/s（+421%）。`logs/server.log` 里可直接看到证据：V1 跑 64 并发时 `#queue-req: 56`——64 个请求只有 8 个在跑，其余 56 个排队，`p99_ttft` 高达 242 s。放宽到 64 后 `p99_ttft` 降到 15 s。
2. **CUDA Graph 在本镜像下可以正常开启**，且开启后收益最大：相比 V2（同样 `mrr=64`、仅差 CUDA Graph），4K/c1 24.64 → 44.24（+79.5%）、4K/c64 934 → 995（+6.5%）、16K/c8 145.55 → 216.34（+48.6%）。CUDA Graph 对**单流解码**和**长上下文**帮助最明显。

> 关于早期"必须 `--disable-cuda-graph`"的判断：那是在 `mrr=8` + 快速试探时得到的结论，本次系统扫描证伪了它。§8 保留了这个认知修正的记录。

继续调 `mem-fraction-static`（V3 的 0.92）或加 chunked-prefill（V5）都没有进一步收益。

### 7.2 最终性能基准（`bench/summary.md` 原文）

| random_input_len | random_output_len | max_concurrency | completed | duration | request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 1 | 8 | 184.86 | 0.04 | 44.31 | 221.57 | 340.96 | 350.98 | 22.25 | 22.29 | 23106.38 |
| 4096 | 1024 | 8 | 16 | 45.48 | 0.35 | 360.23 | 1801.16 | 1983.17 | 2900.57 | 20.28 | 21.79 | 22733.49 |
| 4096 | 1024 | 64 | 128 | 131.33 | 0.97 | 998.05 | 4990.26 | 15024.60 | 23817.64 | 49.47 | 62.48 | 65635.20 |
| 16384 | 1024 | 1 | 8 | 197.56 | 0.04 | 41.47 | 704.92 | 319.18 | 481.46 | 23.83 | 23.95 | 24693.10 |
| 16384 | 1024 | 8 | 16 | 53.07 | 0.30 | 308.72 | 5248.24 | 1710.62 | 2350.53 | 24.25 | 25.24 | 26522.40 |
| 16384 | 1024 | 64 | 128 | 437.18 | 0.29 | 299.81 | 5096.78 | 114759.04 | 172805.01 | 74.96 | 111.44 | 191438.90 |

**有效性校验：all valid**（6 档 `mean_ttft_ms` 均非 0，吞吐均在物理上限内）。

### 7.3 优化前后对比（同一卡、同一框架、同一负载、同一客户端）

首版基准数据存档于 `bench-baseline-v1/`（`summary.md` + `round2.jsonl`），可复核。

| 档位 | 优化前 (tok/s) | 优化后 (tok/s) | 提升 | p99 TTFT 前 → 后 (ms) |
| --- | --- | --- | --- | --- |
| 4K / c1 | 24.02 | 44.31 | **+84.5%** | 374 → 351 |
| 4K / c8 | 179.60 | 360.23 | **+100.6%** | 2877 → 2901 |
| 4K / c64 | 176.22 | 998.05 | **+466.4%** | 329138 → 23818 |
| 16K / c1 | 23.96 | 41.47 | **+73.1%** | 532 → 481 |
| 16K / c8 | 174.98 | 308.72 | **+76.4%** | 2981 → 2351 |
| 16K / c64 | 137.03 | 299.81 | **+118.8%** | 437126 → 172805 |

**6 档全部提升 ≥73%，全部超过赛题「性能提升 ≥10%」的另计奖励门槛。**

- 原始数据：`bench/round1.jsonl`、`bench/round2.jsonl`、各档日志 `bench/round*_*.log`
- 首版基准存档：`bench-baseline-v1/`
- 本结果作为该件「模型 × 卡型 × 框架 × 精度」的性能基准；后续优化在同卡型、同框架、同负载下对比

### 7.4 与同卡同模型另一框架的对比（参考）

| 档位 | SGLang 0.5.10（本件，已优化） | vLLM 0.20.0 |
| --- | --- | --- |
| 4K / c1 | 44.31 | 16.69 |
| 4K / c8 | 360.23 | 121.76 |
| 4K / c64 | **998.05** | 449.47 |
| 16K / c1 | 41.47 | 7.69 |
| 16K / c8 | 308.72 | 42.29 |
| 16K / c64 | **299.81** | 97.75 |

vLLM 件使用 `--enforce-eager`（关闭 CUDA Graph）且沿用默认 `max_num_seqs`，尚未做同类参数扫描，因此不代表其上限。两件各自作为自己的性能基准。

## 8. 注意事项

> 主动说明的问题不扣分，评审复跑时才发现会打回。

- **容器运行时与官方脚本假设的差异**：官方脚本假设裸机环境（`pip install` + `python3 -m sglang.launch_server`）。本件在容器内运行，全部适配集中在 §3.3 的环境变量与 §3.4 的挂载参数；`run.sh` 之后的所有脚本与官方口径一致。
- **`/dev/mkfd` 必需**：只传 `/dev/kfd` 会得到 `No HIP GPUs are available`，且错误发生在 torch 初始化阶段、无进一步提示，是本次适配中最难定位的一处。
- **禁用 CUDA Graph**：不传 `--disable-cuda-graph` 时，进程在 CUDA Graph 捕获阶段静默退出，`logs/server.log` 无 traceback，只有 Slurm 记录 `Killed`。
- **宿主 DTK 不可覆盖容器内 `/opt/dtk`**：宿主 DTK 缺 `aillvm/bin/clang`，Triton JIT 会直接 `Exception: ROCm compiler not found`。
- **`libmsgpackc.so.2` 位置特殊**：在 `/usr/local/lib/python3.10/dist-packages/lightop/lib`，需手动加入 `LD_LIBRARY_PATH`，否则 `import torch` 失败。
- **容器存储不能放 `/public`**：NFS 不支持 overlay 所需 xattr，`podman load` 报 `lsetxattr /boot: operation not supported`。必须用节点本地盘（`--root /tmp/pr`），因此每个作业需先 `load` 一次镜像（70–290 s 不等，取决于节点缓存）。
- **节点磁盘水位不稳**：实测同一分区不同节点根盘可用空间从 0 GB 到 254 GB 不等，`/tmp` 位于根盘。提交脚本里需前置 `df` 检查，不足 45 GB 主动 `exit 42` 重投。
- **每个作业必须用独立 `--root`**：否则同节点上一个作业残留的 podman pause/storage 状态会让 `podman pull` 直接失败，报 `invalid internal status ... podman system migrate`。
- **冷启动 410 s**：主要为 Triton kernel JIT 编译。`LOAD_TIMEOUT` 已调为 1800 s。
- **可忽略的告警**：`Failed to load JIT KV-Cache kernel with row_bytes=2048: Could not find CUDA installation. Please set CUDA_HOME environment variable.` —— 会 fallback 到非 JIT 路径，不影响正确性。
- **CUDA Graph 认知修正（重要）**：早期试探时进程在 CUDA Graph 捕获阶段静默退出（`logs/server.log` 无 traceback，只有 Slurm 记录 `Killed`），据此得出「必须加 `--disable-cuda-graph`」的结论并用于首版提交。后续系统扫描（§7.1）证明**该现象出现在 `max-running-requests=8` 的配置下；放宽到 64 后 CUDA Graph 可正常捕获**，且开启后 6 档性能提升 73%~466%。本件最终配置**不使用** `--disable-cuda-graph`。
- **`--max-running-requests` 不要设小**：设 8 时 64 个并发请求只有 8 个在跑、56 个排队（`logs/server.log` 可见 `#queue-req: 56`），`p99_ttft` 达 242–437 s，4K×64 档吞吐被压到 176 tok/s。这是首版性能偏低的唯一原因，改为 64 后升至 998 tok/s。
- **`--context-length` 必须留余量**：首次用 `17408`（= 16384 + 1024）跑压测，**16K 档全部被服务端 400 拒绝**——客户端 `--random-input-len` 生成的 token 数不含 chat template 包装开销，实际输入达 16740–18784 tokens，超出 17408。更危险的是压测客户端把 400 当“完成”，于是打出 `output_throughput 9791 tok/s`、`TTFT 0.00 ms` 这种假数据。现已改为 `MAX_LEN=20480`，并在 `bench.sh` 里加了两道防护：① 压测前用长 prompt 做 preflight，非 200 就退出；② 压测后校验 `mean_ttft_ms != 0` 且吞吐不超物理上限，否则判 INVALID 并拒绝写入报告。旧数据保留为 `bench/round*_16384_*.BAD-ctx17408.log` 作为踩坑证据。
- **并发 64 档吞吐低于并发 8**（4K：176.22 vs 179.60；16K：137.03 vs 174.98），`p99_ttft` 达 329 s / 437 s。这是 `--max-running-requests 8` + 单卡 64 GB 显存排队造成的真实表现，不是脚本错误；若要更高并发吞吐需降低 `MAX_LEN` 或多卡 TP。
- **精度章节未完成**：见 §6，阻塞于官方 `baselines/` 与 `data/` 下发，详见 `MISSING.md`。

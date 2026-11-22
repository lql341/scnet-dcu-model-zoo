# Qwen3-VL-8B-Instruct on BW1000 × **vLLM 0.21** —— ✅ 服务跑通 + 6 档完成

> 本条目与 `BAAI__RoboBrain2.5-8B-NV__bw1000__vllm` **同底座**（`Qwen3VLForConditionalGeneration`），
> 两者参数量同为 8B 级、同一镜像、同一卡——所以 §7 的对照是**同条件 A/B**，
> 差异只来自后训练（权重 16.34 GiB vs ~9 GiB，本模型视觉编码器更大）。
> **没有 `__sglang` 配对条目**：同底座的 SGLang 侧已因 Triton JIT VMFault 封存（gotchas #57/#58）。

## 1. 简介

| 项 | 值 |
| --- | --- |
| 任务 | T02 视觉多模态理解（图片问答/推理/OCR） |
| 架构 | `Qwen3VLForConditionalGeneration`（`model_type: qwen3_vl`） |
| 权重 | 16.34 GiB / 17 文件（4 × safetensors，`model-00001..00004`） |
| 卡型 | BW1000 × 1（TP=1，bf16 约 16.3 GB 显存，64 GB 卡余量充足） |
| 来源 | ModelScope `Qwen/Qwen3-VL-8B-Instruct` |

**与官方部署方式的差异**：无。Qwen3-VL 在 vLLM 0.21 有原生实现，无需 patch、无需额外依赖。
唯一的差异是**权重是离线搬运的**（集群无外网，见 §8.1）。

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `dcu/wqxyyt06/base/vllm0.21.0-ubuntu22.04-dtk2604-py3.10:v1`（digest `sha256:c6a2f554…`） |
| 框架 | **vLLM 0.21.0**（Ubuntu 22.04 / DTK 26.04 / Python 3.10） |
| 卡 | BW1000 × 1（TP=1，`gfx936:sramecc+:xnack-`，64 GB） |
| 节点 | 华中一区【A区】郑州 `hx1hdnormal` |
| 端口 | 127.0.0.1:30000（作业内自测，不对外暴露） |
| 冷启动 | **360–420 s**（实测 360 s 到 `/health` 可用；拆解见 §8.5） |

> 版本表取自 `logs/env.log`（本次运行后回填）。

## 3. 适配优化方案

### 3.1 基础镜像

`image.ac.com:5000/dcu/wqxyyt06/base/vllm0.21.0-ubuntu22.04-dtk2604-py3.10:v1`

选 0.21 而不是 0.20 是**硬约束**：0.20 的 `qwen3_vl.py::_get_deepstack_input_embeds` 会硬性断言
`num_tokens=72 > 67`，任何带图请求都 500。0.21 已删除该断言（改为切片），实测单图/多图均正常。

### 3.2 软件包升级

无。`requirements-extra.txt` 里只有说明性注释。

### 3.3 代码改动或参数

| 参数 | 值 | 理由 |
| --- | --- | --- |
| `--max-model-len` | 32768 | 与同底座 RoboBrain 条目一致，留足多图 token |
| `--gpu-memory-utilization` | 0.95 | DCU 上 vLLM 探测偏保守 |
| `--max-num-seqs` | 128 | — |
| `--dtype` | bfloat16 | 权重本身是 bf16 |
| `--trust-remote-code` | 是 | 与该镜像其它条目一致 |
| **不启用** `--enforce-eager` | — | CUDA Graph 实测更快（同底座条目已验） |

### 3.4 升级步骤

```bash
bash container/submit.slurm                        # 6 档基准
SKIP_BENCH=1 bash container/submit.slurm           # 只到 smoke
BENCH_QUICK=1 bash container/submit.slurm          # 自检：4K/c1/1 轮
```

## 4. 服务启动脚本

双镜像编排（与 `BAAI__RoboBrain2.5-8B-NV__bw1000__vllm` 一致）：

| 阶段 | 在哪跑 | 做什么 |
| --- | --- | --- |
| 容器 A（vLLM 镜像） | `--network=host` | `env_check.sh` → `vllm.entrypoints.openai.api_server` |
| 宿主机 | 只需 curl + python3 标准库 | `smoke.sh`、`verify.py` |
| 容器 B（**同一个** vLLM 镜像） | `--network=host` | `bench_vllm.sh` → `vllm bench serve` |

> 与仓库其它条目不同：**不加载 SGLang 客户端镜像**。原因见 §8.4（节点磁盘放不下两个镜像）。
> `bench.sh`（SGLang 版）保留作参照，实际执行的是 `bench_vllm.sh`。

`run.sh` 展开后的完整命令：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/M --served-model-name qwen3-vl-8b-instruct \
  --host 127.0.0.1 --port 30000 --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 --max-num-seqs 128
```

## 5. Smoke 测试

**2026-11-22 实测全绿**（作业 1192284 / 1192374，原始输出见 `logs/smoke.log`）：

| 检查项 | 结果 | 关键证据 |
| --- | --- | --- |
| 服务启动 | ✅ `/health` HTTP 200 | `max_model_len=32768` |
| `/v1/models` | ✅ | `id=qwen3-vl-8b-instruct`，`root=/models/M` |
| 纯文本 | ✅ | 正确解释"空间推理"，`prompt_tokens=15` |
| **单图** | ✅ **答"红色"**（正确） | `prompt_tokens=99` |
| **多图（2 张）** | ✅ **"第一张红色、第二张蓝色"**（均正确） | `prompt_tokens=179` |
| `verify.py --selfcheck` | ✅ `selfcheck OK` | 逻辑自检通过 |
| `verify.py` | `[BLOCKED]` 缺官方基线 | 按红线不编造分数，见 §6 |

单图/多图是本条目的**核心判据**——同底座（Qwen3-VL 系）在 vLLM **0.20** 上正是这一步 500
（deepstack 断言），0.21 已修。

环境（`logs/env.log`）：torch 2.10.0 / HIP 6.3.26113 / transformers 5.5.0 / triton 3.6.0，
BW1000 64.0 GB，`gfx936:sramecc+:xnack-`。

## 6. 精度测评

`[BLOCKED]`。原因不是"等材料"，而是**口径不可比 + 数据不可得**：

1. T02 官方公开分数是 8-shot/5-shot、base/EM 口径，本仓库 `verify.py` 是 0-shot greedy，相减无意义；
2. OCRBench / MMMU 数据集在本集群不可得（无外网）。

详见 `docs/t01-t02-baseline-investigation.md`。按红线不填任何估算值。

## 7. 性能参考

6 档 × 2 轮（单位 output tok/s，取 `bench/round2.jsonl`，作业 1193165）：

| 档位 | token/s | TPOT (ms) | TTFT (ms) |
| --- | ---: | ---: | ---: |
| 4K / c1 | **62.54** | 15.95 | 50.6 |
| 4K / c8 | **387.30** | 20.47 | 200.9 |
| 4K / c64 | **688.50** | 75.91 | 8340.7 |
| 16K / c1 | **51.26** | 19.39 | 135.4 |
| 16K / c8 | **230.34** | 34.03 | 662.8 |
| 16K / c64 | **184.06** | 85.23 | 205161.3 |

有效性校验 `all valid`（TTFT 均 > 0、吞吐未超物理上限）。完整明细见 [`bench/summary.md`](bench/summary.md)。

**与同底座 RoboBrain2.5-8B-NV 对照**（同卡同镜像，同为 Qwen3-VL 系）：

| 档位 | Qwen3-VL-8B | RoboBrain2.5-8B-NV | 差异 |
| --- | ---: | ---: | ---: |
| 4K / c1 | 62.54 | 64.01 | −2.3% |
| 4K / c8 | 387.30 | 387.93 | −0.2% |
| 4K / c64 | 688.50 | 667.69 | +3.1% |
| 16K / c1 | 51.26 | 51.14 | +0.2% |
| 16K / c8 | 230.34 | 225.76 | +2.0% |
| 16K / c64 | 184.06 | 175.71 | +4.8% |

六档最大差异 4.8%，**基本重合**。这符合预期（同为 8B 级 Qwen3-VL 底座），
也说明这套 6 档文本负载主要反映**架构与引擎**，对同底座的不同后训练不敏感。

**16K/c64 的异常**：TTFT 达 205 s、p99 TTFT 304 s，是本次唯一"难看"的档位。
同底座 RoboBrain 该档更差（TTFT 212.8 s、吞吐 175.71），说明这是 **Qwen3-VL 系在 16K 长上下文 + 64 并发下的共同特征**，
不是本条目独有。未做进一步优化（本条目不追求调优，见 §8.6）。

> ⚠️ 这些数字是**引擎能力与吞吐证据**，不是官方评分依据（T02 评测器未下发）。

## 8. 注意事项

### 8.1 集群已无外网，权重只能离线搬运（本条目最费时的一环）

2026-11-22 实测：登录节点与计算节点 **DNS 均不可用**（`getent www.modelscope.cn` 失败），
公网 IP 的 443 端口（223.5.5.5 / 114.114.114.114 / 47.96.8.1 / 8.8.8.8 / 202.96.128.86）**全部超时**，
网关代理口与内网 pip 源也全部 closed。这不是 DNS 故障，是**没有出网路由**，硬编码 IP 也救不了。

对比：`luql:~/work/dl-logs/dl_curl.sh` 在 2026-11-15 22:48 还能直连下载 85 个 DeepSeek 权重文件
（`.dl.progress` 有逐文件 OK 记录）。说明这是 **11-15 之后新出现的变化**，历史条目的下载流程不再适用。

现行做法：本机下载（16 MB/s）→ `scnet.py upload` 单文件上传（10 MB/s），16.34 GiB 约 50 分钟。
不要试图在集群上 `snapshot_download`，会白等一轮调度。

### 8.2 为什么没有 `__sglang` 配对条目

同底座（Qwen3-VL 系）的 SGLang 条目 `BAAI__RoboBrain2.5-8B-NV__bw1000__sglang` 已封存：
服务能起但 warmup 时 `_fwd_grouped_kernel_stage1` 在 gfx936 上 KERNEL VMFault，
且已排除节点/参数/显存/cuda-graph/架构/镜像（gotchas #57/#58）。
本条目不重复投卡时验证同一失败。

### 8.3 Harbor 已不可达，镜像只能从本地 tar 加载

2026-11-22 实测：`podman pull image.ac.com:5000/...` 报

```
pinging container registry image.ac.com:5000:
  dial tcp 12.13.17.201:5000: connect: connection timed out   → 重试后 no route to host
```

同时 `dcu/wqxyyt06` 的 vllm0.21 仓库另报 `artifact ... not found`（上游可能已删）。
**所以本条目不能靠 pull**，`env.sh` 里显式指向本地归档：

| 用途 | 本地归档 | 形态 |
| --- | --- | --- |
| vLLM 0.21（服务） | `luql:~/work/imgstore/vllm021.tar` | 31 GB，**未压缩**，`podman load -i` |
| ~~SGLang CLI（压测）~~ | ~~`scnethr2601a10:~/zoo-images/mimo-sglang-bw10000_1.1.1.tar.gz`~~ | 不用了，见 8.4 |

tar 里 `repositories` 记录的 tag 与 `image.lock` 完全一致（config `7bfe7ebf…`），load 后无需 retag。

> ⚠️ 一个教训：**不要用 `podman manifest inspect` 判断镜像是否还在**。那次它四个镜像全报 OK，
> 而真实 pull 全部失败——超时文本里不含 `not found`，被"只要没看到某几种错误就算成功"的判据漏掉了（gotcha #61）。

### 8.4 压测改用同镜像内的 `vllm bench serve`，不再加载 SGLang 客户端

原计划沿用其它条目的双镜像编排（vLLM 起服务 + SGLang 镜像跑 `sglang.benchmark.serving`）。
但两个镜像共存需节点 `/tmp` ≥145 GB，而本分区各节点在 **21–141 GB** 间波动，
实测 103 GB / 99 GB 两次都在 load 第二个镜像时 `no space left on device`（写 rocblas TensileLibrary 失败），
等 20 分钟也不见释放。

改法：用 vLLM 镜像自带的 `vllm bench serve` 在**同一容器内**对 `127.0.0.1` 发压，只需一个镜像。
实测链路可行，4K/c1 与同底座 RoboBrain 条目（64.01 tok/s）同量级。

踩到的两个参数坑：

| 现象 | 根因 | 修法 |
| --- | --- | --- |
| `OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/models/M'` | 压测容器的 `$PD run` 漏了 `-v $HOST_MODEL_PATH:/models/M:ro`，本地路径不存在被当 repo id | 补回挂载 |
| `ValueError: OpenAI Chat Completions API URL must end with ...` | `--backend openai-chat` 的默认 endpoint 是 `/v1/completions` | 显式 `--endpoint /v1/chat/completions` |
| `ValueError: --random-input-len is too small` | 新版 `--random-range-ratio 1` 语义是"全随机"，下限可为 0 | 去掉该参数 |

### 8.5 冷启动构成（实测拆解）

| 阶段 | 耗时 |
| --- | --- |
| 镜像 `podman load`（31 GB 未压缩 tar） | ~180 s |
| 权重加载（16.33 GiB，NFS） | 15.3 s |
| **torch.compile**（compile range 1–2048） | **123.0 s** ← 最大头 |
| 引擎 init 合计（含 profiling/warmup） | 141.5 s |
| 服务 `/health` 可用 | 启动后 **360 s** |

torch.compile 占了冷启动的一半以上，且缓存落在容器内 `/root/.cache/vllm`，
容器销毁即丢，所以**每次作业都要重付这 123 s**。若将来要反复起停，值得把它挂到宿主机持久化。

KV cache：`Available 40.57 GiB` / `295,424 tokens` / 32K 上下文下最大并发 9.02×。

### 8.6 其它（沿用同底座条目的经验）

- 多卡时（TP>1）必须显式 `-e HIP_VISIBLE_DEVICES=0,1,...`，否则容器只见 1 张卡（gotchas #21）
- NFS 上 `cp -rf` 会漏拷/不覆盖 → 同步一律用 `rsync`（gotchas #26）
- 收容器前把 `podman logs` 落盘，否则 500 的根因随容器消失
- 每张 DCU 需 ≥8 CPU，否则 GRES 静默降级成 1 张（gotchas #30/#31）

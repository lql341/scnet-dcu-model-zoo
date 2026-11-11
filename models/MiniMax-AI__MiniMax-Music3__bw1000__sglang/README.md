# MiniMax-Music3 on BW1000 × 镜像自带 Hygon SGLang —— 🟡 进行中

## 1. 简介

| 项 | 值 |
| --- | --- |
| 模型 | `MiniMaxAI/MiniMax-Music3`（8B Global LLM + 0.6B Local LLM + 2.4B Flow-Matching + Flow-VAE） |
| 任务 | T03 音乐生成：`input`（带段落标签的歌词）+ `instructions`（音乐描述）→ 32 kHz 立体声 |
| 权重 | 57.4 GB（ModelScope `MiniMax/MiniMax-Music3` 与 HF `MiniMaxAI/MiniMax-Music3` 均可得） |
| 卡型 | BW1000 单卡（64 GB），TP=1；镜像内实测峰值显存 ~33.7 GB |

**与官方部署方式的差异（三条，都是"必须"）**：

1. **不是常规 LLM serving**：官方 GitHub 仓只有 README + prompt 改写 skill，推理脚本在
   ModelScope 仓的 `scripts/end_to_end/`，而且它是一个**客户端** —— 必须有一个
   SGLang-Omni 服务端，接口是 OpenAI 兼容的 `POST /v1/audio/speech`。
2. 上游说「需要两张 CUDA GPU」；本条目用 Harbor 上 **BW 专用镜像 + 镜像作者的 harness**，
   在**单卡**上跑通（输出规格与官方一致：32 kHz 立体声、750 声学帧）。
3. 上游 SGLang-Omni 的调度/扩散图接口与本镜像的 Hygon 构建不匹配 → harness 里有一份兼容 patch
   （改成 eager 组件、token-major KV、`num_stages=1` 等，详见 §3.3）。

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `dcu/acizv7iztz/base/minimax-music3-bw-sglang:20261031-rc1`（digest 见 `image.lock`） |
| OS / Python | Ubuntu 22.04.5 / Python 3.10.12 |
| torch | 2.10.0+das（HIP 6.3.26113） |
| SGLang | 0.5.12（镜像内 Hygon 构建） |
| transformers / vllm | 5.6.0 / 0.21.0+das |
| 卡 | BW，64 GiB，`gfx936:sramecc+:xnack-` |
| 端口 | 127.0.0.1:8000（容器内） |
| 权重路径 | 宿主 `~/models/MiniMax-Music3` → 容器 `/models/Music3` |

## 3. 适配优化方案

### 3.1 基础镜像

`dcu/acizv7iztz/base/minimax-music3-bw-sglang:20261031-rc1`（压缩 12.27 GB）。
这是 Harbor 上唯一带 Music3 + Hygon SGLang 的 BW 镜像；上游没有 compact 的官方推理仓，自建成本远高。

**镜像 `/tmp/music3-hygon/` 里带着镜像作者的适配工作区**（`serve.py`、`music3_hygon/` runtime、
`patches/0001-hygon-eager-components.patch`、`evaluate.py`、`env.sh`）。`/tmp` 会随容器消失，
因此我们把它归档为 `/public/home/luql/archive/zoo-dcu-2026-11/assets/music3-hygon-harness.tar.gz`（sha256 `d5e119a5…`），
并在启动时挂载到容器的 `/work/harness`。

> **镜像归属说明**：本条目使用的镜像属于**第三方账号**（`dcu/linshp` 或 `dcu/acizv7iztz`），
> 非本项目构建。已用 `image.lock` 锁定 digest，并在
> `/public/home/luql/archive/zoo-dcu-2026-11/images/` 保留 tar.gz 归档（sha256 已核对、还原验证通过），
> 以应对上游删除仓库。推送到自有命名空间需要 Harbor 凭据，平台不提供自助获取接口（2026-11-11 已确认无法推进）。

### 3.2 软件包升级

**不装任何包**：torch / SGLang / transformers / vllm / RVQ / flow-matching / DAV 全在镜像里。
刻意不装 `sglang-omni`（与镜像内 Hygon 构建冲突）与 `torchaudio`（sinc 重采样直接拷函数，见 gotchas #19）。

### 3.3 代码改动或参数

harness 自己打一份兼容 patch（我们**不改它**，用归档原件），关键点：

| 改动 | 原因 |
| --- | --- |
| 改用 eager 组件替换上游 Omni 调度/扩散图接口 | 镜像的 Hygon SGLang 没有那套接口 |
| `SGLANG_KV_LAYOUT_HCU_FA=0` | 镜像默认的 FA 专用 paged KV 按固定 64 token 索引，与 Triton allocator 的 1-token page 不匹配 |
| 注意力走 ordinary SDPA；确定性 matmul `num_stages=1` | 三阶段 BF16 要 96 KiB 共享内存，BW 只有 64 KiB |
| 走 native Q/K norm + rotary，不用 fused 版 | 镜像的 fused RoPE 起 512 线程，而 kernel bound 是 256 |
| 声学精度 fp32（`ACOUSTIC_PRECISION=fp32`） | 与上游适配一致；backbone/RVQ 仍 bf16 |

### 3.4 升级步骤

```bash
sbatch container/submit.slurm                     # 官方 60 条
PROBE_ONLY=2 sbatch container/submit.slurm        # 只跑 2 条验管线（约 8 min）
```

## 4. 服务启动脚本

`container/inner.sh` 依次执行：清掉 harness 里悬空的 `runtime/` 符号链接 → `env_check.sh`
→ `bash run.sh`（打 patch、建 `--system-site-packages` venv、起 `serve.py --attention-backend triton`、
轮询 `/health`）→ `python3 evaluate_official.py`（本条目自己的客户端）→ 产物规格抽检。

## 5. Smoke 测试

`PROBE_ONLY=2`（官方条件前 2 条）实测：

| 检查项 | 结果 |
| --- | --- |
| 服务启动 | ✅ `Server ready`（长期等待窗口 25 min，实测 model load + vene 约 10 min） |
| 条件 1 | ✅ `music_zh_001_s42`：30.02 s、32 kHz / 2ch、3.84 MB、RTF ≈ 7.2（216 s 出 30 s 音频） |
| 条件 2 | ✅ `music_zh_001_s2026`：同上，207 s |
| 规格 | ✅ `sr=32000 ch=2 frames=960749 dur=30.02s`，与官方 `expected` 一致 |
| 显存 | 服务自报 `x-peak-allocated-bytes: 33.67 GB`（64 GB 单卡有余量） |

## 6. 精度测评（官方条件）

官方条件：`conditions/music3.jsonl`，**60 条**（中英各 10 段歌词 × seed 42/2026/1234）。
参数取自清单：`max_new_tokens=750`、`dit_steps=30`、`dit_cfg_scale=1.7`、`ar_guidance_scale=1.5`、`top_k=50`、`stream=false`。

官方明示「不进行统一评分和性能比较」→ 不计算任何指标，只交原始产物 + 规格统计。

### 结果：**60 / 60 成功，0 失败**

| 指标 | 实测 |
| --- | --- |
| 评测总耗时 | **12179.1 s（3.4 h）**，单条平均 **203 s** |
| 输出规格 | 32 kHz / 立体声 / 16 bit（60 条全部符合，抽检 0 违规） |
| 时长 | 17.65 – 30.02 s，合计 **1778.2 s** 音频 |
| 产物体积 | 227.6 MB（不入库；`official-eval/samples/` 存 1 条前 5 秒片段 640 KB） |
| 峰值显存 | 33.7 GB / 64 GB |

产物与反馈表：`official-eval/{反馈表.csv, eval_result.json, 环境说明.txt}`。
**官方不进行统一评分和性能比较** → 不计 FAD/CLAP 等任何指标。

## 7. 性能参考

官方 T03 不评分 → `meta.bench = not-required`。

| 场景 | 实测（单卡） |
| --- | --- |
| 单条（750 声学帧 ≈ 30 s 音频） | 203 s 平均（RTF ≈ 6.8，范围 6.9–7.2） |
| 60 条全量 | **12179 s（3.4 h）/ 单卡** |
| 服务启动（model load + venv） | ~10 min |
| 峰值显存 | 33.7 GB / 64 GB |

## 8. 注意事项

- **「服务在监听」≠「能出声」**：镜像作者自己的最后一版日志里有 HW queue VMFault + 14 GB core dump
  （我们已删除），说明这条路径并不稳定。本条目以**产物规格校验**为准（32 kHz/2ch/时长），不看 `/health`。
- harness 里的 `runtime/` 是作者环境里的**悬空符号链接**，直接复用会 `FileExistsError`；必须清空重建。
- 该 harness 是**第三方个人工作区**，不是官方发布物；升级镜像后其内部路径可能变，届时以
  `image.lock` 记录的 digest 为准复现，或从归档 tar 里取同一份。
- 官方 GitHub 仓不含推理代码；客户端协议见 ModelScope 仓 `scripts/end_to_end/minimax_ttm_test.py`。

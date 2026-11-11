# Tencent-Hunyuan/AuK on BW1000 + Transformers —— 🟡 进行中

> **状态：探针作业已提交，等待集群资源。** 先记录已确认的事实与版本冲突。

## 1. 简介

`Tencent-Hunyuan/AuK` 在 `异构加速卡BW（BW1000）` 上的适配。

腾讯混元开源的 1.5B 语音生成与编辑基础模型，基于数百万小时音频训练，通过统一自然语言指令接口支持：
零样本 TTS、指令 TTS（仅凭音色描述）、语音内容编辑与歌词编辑、音高/语速/音量编辑、
情感/音色/去口音/非语言声音/耳语转换、语音增强、语音/音乐分离与目标说话人提取。
1.69B 参数以 FP32 存储约 6.8 GB，BF16 约 3.4 GB，MIT。MLLM 编码器需另下 `Qwen/Qwen2.5-Omni-3B`。

**与官方部署方式的差异（已确认的硬冲突）**：

官方 `pyproject.toml` 钉死：

```
"torch>=2.7,<2.8"
"torchaudio>=2.7,<2.8"
"transformers>=4.52.0,<5"
```

而国产卡镜像是 **DCU 版 torch 2.9 / 2.10 + transformers 5.x**。
**PyPI 上没有 DCU 版 torch 2.7**，无法降级到官方 pin。本条目因此采用「忽略 pin + 用镜像自带 torch」，
能否跑通以实测为准。这是本条目最大的未知。

## 2. 验证环境

```text
image: image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
torch: 2.9.0+das.opt1.dtk2604   torchaudio: 2.9.0+das（DCU 版，必需）
transformers: 5.3.0（镜像自带，官方要求 <5）
Device: BW × 1，64.0 GB，gfx936
```

**为什么用 SGLang 镜像而不是 vLLM 镜像**：AuK 的 `infer_cli` 需要 `torchaudio`，
SGLang 镜像自带 **DCU 版 torchaudio 2.9.0**；vLLM 镜像没有 torchaudio，
而 PyPI 的 torchaudio 是 CUDA/CPU 构建，与 DCU 版 torch 不匹配（实测会失败）。

## 3. 适配优化方案

### 3.1 基础镜像

SGLang 镜像（唯一带 DCU 版 torchaudio 的可用镜像）。

> **镜像归属说明**：本条目使用的镜像属于**第三方账号**（`dcu/linshp` 或 `dcu/acizv7iztz`），
> 非本项目构建。已用 `image.lock` 锁定 digest，并在
> `/public/home/luql/archive/zoo-dcu-2026-11/images/` 保留 tar.gz 归档（sha256 已核对、还原验证通过），
> 以应对上游删除仓库。推送到自有命名空间需要 Harbor 凭据，平台不提供自助获取接口（2026-11-11 已确认无法推进）。

### 3.2 软件包升级

**不装 AuK 的 pinned 依赖**（会破坏镜像）。改为：

```bash
pip install -q --no-deps --target /opt/auk-libs \
  x_transformers torchdiffeq omegaconf qwen-omni-utils safetensors accelerate
export PYTHONPATH=/opt/auk-libs:/auk-src/src:$PYTHONPATH
```

AuK 是 pyproject 项目，`pip install --target` 装不进；直接把源码 `src/` 加进 `PYTHONPATH`。

### 3.3 代码改动

无 patch（目前）。若镜像 torch 2.9 与 AuK 不兼容，可能需要在 `patches/` 里改版本校验。

### 3.4 运行步骤

```bash
sbatch container/submit.slurm    # 内含导入测试 + 单条推理探针
```

## 4. 运行方式

一次性批处理（`auk-infer` CLI），无常驻服务。入口 `container/submit.slurm`。

## 5. Smoke 测试

`probe_one.py` / `PROBE_ONLY=N` 打印：条件首条关键字段、参考音路径与存在性、torch 版本、
`auk` 与 `infer_cli` 导入结果。

**2026-11-03 实测（= 官方 120 条全量中的前若干条）**：

| 检查项 | 结果 |
| --- | --- |
| 引擎加载 | ✅ 46.8 s（`Loaded EMA weights | missing=924 (text_encoder.*=924, other=0)`，与官方说明一致） |
| 参考音 | ✅ 120 条引用的 21 个 prompt wav 全部存在，sha256 已校验 |
| 依赖 | ✅ librosa 0.10.2 / numba 0.61.2 / llvmlite 0.44 / scipy 1.15.3 / soundfile / soxr 全部可导入（**cp310** 轮子） |
| 音频 I/O | ✅ 读 `soundfile.read`，写 `soundfile.write` —— **完全不碰 torchaudio**（见 §9） |
| 单条输出 | ✅ 24 kHz / 单声道 / PCM_16，时长与 `params.gen_seconds` 一致 |

## 6. 精度测评（官方条件）

官方条件：`official/t03-t04/conditions/auk.jsonl`，共 **120 条**（中英各 20 组参考音 × 3 seed；
实测 21 个唯一参考音被 120 条复用，**零缺失**）。

每条的输入字段（实测）：

```json
{"id":"auk_zh_00000309-00000009_s42","model":"auk","task":"tts","seed":42,
 "params":{"gen_seconds":6.07,"nfe":32,"cfg_strength":2.0,"sway_sampling_coef":-1.0},
 "assets":{"reference_audio":{"path":"../../data/seedtts/zh/prompt-wavs/...wav","sha256":"..."}},
 "text":"...","language":"zh","instructions":"...","ref_text":"..."}
```

**全部生成参数逐条取自条件清单**（`gen_seconds` / `nfe` / `cfg_strength` / `sway_sampling_coef` / `seed`），
不写死任何默认值。

### 结果：**120 / 120 成功，0 失败**

| 指标 | 实测 |
| --- | --- |
| 引擎加载 | 46.8 s |
| 评测总耗时 | **531.2 s**（单条 **4.4 s**） |
| 输出现格 | 24 kHz / 单声道 / 16 bit |
| 时长 | 3.08 / **5.86**（中位）/ 12.46 s，合计 749.5 s 音频 |
| 时长合规 | 120 条全部 ≥ `expected.duration_min_s`（3.03–12.41 s） |
| 产物体积 | 36 MB（不入库；`official-eval/samples/` 存中英各 1 条，`stats.json` 存逐条时长） |

产物与反馈表：`official-eval/{反馈表.csv, eval_result.json, stats.json, 环境说明.txt}`。

**官方不进行统一评分和性能比较**，因此本条目**不计算**任何相似度/自然度指标 —— 不编造分数。

## 7. 性能参考

官方 T03/T04 明确不评分 → `meta.bench = not-required`。可对照的实测吞吐：

| 场景 | 数值 |
| --- | --- |
| 单卡 bf16，nfe=32，6 s 音频 | **4.4 s/条**（≈0.23 条/s ≈ 1.35 s 音频/s） |
| 120 条全量 | 8.9 min（不含 47 s 加载与 5 min 镜像 load） |

官方给出的 A800-80GB 参考点：峰值显存约 24.8–25.0 GiB；加 `--cpu_offload` 约 16.8–17.0 GiB
（官方注明**仅 CUDA**）。DCU 上本次未开 offload，64 GB 单卡直接跑通。

## 8. 注意事项

- **torch 版本 pin 无法满足**：官方要 `>=2.7,<2.8`，DCU 只有 2.9/2.10。用 2.9 实测，
  若失败需评估"改 AuK 源码的版本检查"是否可行。
- **`transformers<5` 也无法满足**：镜像是 5.3.0。`x_transformers` 与 `transformers` 是不同包，
  但 AuK 同时依赖两者，需注意混淆。
- **`torchaudio` 只能用 DCU 版**：选镜像时就因此排除了 vLLM 镜像。
- **`--cpu_offload` 官方标注仅 CUDA**，DCU 上需实测。
- **权重 FP32 存储**（6.8 GB），加载时转 BF16。
- **MLLM 编码器需单独下载** `Qwen/Qwen2.5-Omni-3B`（12 GB，已下载）。

## 9. 阻塞实况 → 已定位并解除（2026-11-03）

此前 7 轮迭代都卡在「音频解码」，日志写着「librosa 装不进 / 与已存在包冲突」。
复查 `反馈表.csv` 与构建日志后，真实根因是**两个工程性错误**，不是 DCU 或框架限制：

| # | 症状 | 真实根因 | 修法 |
| --- | --- | --- | --- |
| ① | 120/120 `ModuleNotFoundError: No module named 'librosa'` | 离线 wheel 在**本机（py3.12）**下载，全是 `cp312` 轮子；镜像是 **py3.10**，pip 判定 tag 不兼容 → 静默拒装 | `pip download --python-version 310 --platform manylinux2014_x86_64 --abi cp310` 重下 53 个轮子 |
| ② | 单条耗时 4.5 s（说明推理已跑通）但 120 条全失败 | `torchaudio.save()` 与 `load()` 一样走 **torchcodec**，DCU 版 torch 加载不了 `libtorchcodec*.so` | 写盘改 `soundfile.write(out, arr.squeeze(), sr, subtype="PCM_16")` |

**证据链**：`反馈表.csv` 的 `error` 列（120 行同一条 `librosa` 缺失）；
装好 cp310 轮子后同一条命令 120 条仍全失败，但每条耗时从 0.04 s 变成 4.5 s（推理在跑），
错误变成 `Could not load libtorchcodec`；改 `soundfile` 写盘后开始产出 `<id>.wav`
（6.08 s @ 24 kHz 单声道，符合官方 `expected.duration_min_s`）。

**耗时以全量实测为准：120 条 531 s，单条 4.4 s**（nfe=32、单卡 bf16）。

> 一处需要如实记录的观察：同为 soundfile 写盘的上一趟作业（1022209）里，前 6 条间隔约
> 150 s/条，据此曾把墙钟设成 7 h；随后同一配置全量跑出 4.4 s/条。**两个数字都来自实测**，
> 差异原因未查明（嫌疑是当时节点 I/O 抖动或首轮 kernel 编译），此处以全量结果为准，
> 不猜原因。教训：`submit.slurm` 保留宽墙钟 + `evaluate_official.py` 支持按产物续跑，
> 这样估算错了也不会白烧卡时。

这两条已写进 `docs/gotchas.md` #18 / #19：**「装不上」先查轮子 tag 与 `requires_dist`，
不要先怀疑国产卡。**

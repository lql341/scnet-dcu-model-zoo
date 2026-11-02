# Qwen3-ASR-1.7B on BW1000 + vLLM 0.20.0

> 语音识别（T04）。本条目验证「音频输入 → 文本输出」全链路在国产卡上可用，并给出实测的识别结果与错误。

## 1. 简介

`Qwen/Qwen3-ASR-1.7B` 在 `异构加速卡BW（BW1000）` + `vLLM 0.20.0`（精度 `bf16`）上的适配验证。

模型是通义千问的语音识别模型，`Qwen3ASRForConditionalGeneration`（`qwen3_asr`）：音频编码器 + 文本解码器，合计 2.35B 参数；支持 30 种语言与 22 种中文方言，单模型统一流式 / 离线推理。BF16 权重 2 分片约 4.7 GB，Apache-2.0。

**与官方部署方式的差异：无。** 官方 `vllm serve` 命令可直接使用，只需补 `--max-model-len` / `--gpu-memory-utilization` / `--max-num-seqs` 三个容量参数。无包升级、无代码 patch。

相关地址：

- 权重：`https://www.modelscope.cn/models/Qwen/Qwen3-ASR-1.7B`
- 镜像：`image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0`
- 提交仓库：https://developer.sourcefind.cn/codes/luql/qwen3-asr-1.7b-bw1000-vllm

## 2. 验证环境

镜像：`image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0`

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

- 卡型：BW1000，单卡 64.0 GB
- 卡数：1（`TP=1`）
- 冷启动：**150–200 s**
- 模型路径（容器内）：`/models/ASR`

## 3. 适配优化方案

**无软件包升级、无代码 patch**，全部通过启动参数完成：

| 参数 | 取值 | 原因 |
| --- | --- | --- |
| `--max-model-len` | `8192` | 语音任务输出短，8K 足够；不必按 20480 分配 KV |
| `--gpu-memory-utilization` | `0.85` | 模型仅 4.7 GB，显存压力小 |
| `--max-num-seqs` | `16` | 语音请求耗时长，16 路并发已足够 |
| CUDA Graph | 保持开启 | 未传 `--enforce-eager`；见 `docs/gotchas.md` #2 |

容器运行时要求同其它条目：必须传 `/dev/mkfd`、必须挂宿主 `/opt/hyhal`、容器存储放节点本地盘。见 `docs/gotchas.md`。

## 4. 服务启动脚本

```bash
bash run.sh
```

实际执行：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/ASR --served-model-name qwen3-asr-1.7b \
  --host 127.0.0.1 --port 30000 \
  --tensor-parallel-size 1 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code --dtype bfloat16 --max-num-seqs 16
```

## 5. Smoke 测试

`smoke.sh` 用 `assets/asr_test.wav`（macOS TTS 合成，已知转写）验证两条路径。

### 结果

`/v1/models` 返回 `qwen3-asr-1.7b`，`max_model_len: 8192`。

**路径 A：`/v1/chat/completions` + `audio_url`**

```json
{"choices":[{"message":{"content":"language Chinese<asr_text>甚至出现交易几乎停顿的情况。"},
 "finish_reason":"stop"}],"usage":{"prompt_tokens":56,"completion_tokens":12}}
```

**路径 B：`/v1/audio/transcriptions`**

```json
{"text":"甚至出现交易几乎停顿的情况。","usage":{"type":"duration","seconds":4}}
```

**参考文本**：`甚至出现交易几乎停滞的情况`

### 单样本 CER

| 参考 | 识别 | 差异 |
| --- | --- | --- |
| 甚至出现交易几乎停**滞**的情况 | 甚至出现交易几乎停**顿**的情况 | 1 字 / 12 字 |

单样本 **CER ≈ 8.3%**。需要注意：**这不是官方评测**——音频来自 macOS TTS，发音与真实录音差异大，且只有一条。它证明的是**链路可用**（音频解码 → 特征 → 编码器 → 解码器 → 文本后处理全部走通，`language` 前缀正确识别为 Chinese），不是模型精度。官方 WER/CER 需按分题材料下发的测试集跑。

## 6. 精度测评（官方条件，已完成）

已按官方 `T03_T04模型可行性方案_框架数据接入评测` 的固定条件执行**全部 300 条**，**300 成功 / 0 失败**。

### 执行口径（严格按官方规定）

| 项 | 官方要求 | 本件执行 |
| --- | --- | --- |
| 输入 | **只传 `assets.audio`**，不传参考 `text`、不传 `language` 标签 | ✅ 一致 |
| 解码 | `temperature=0`（greedy） | ✅ |
| 生成长度 | `max_new_tokens=1024` | ✅ |
| 语言 | `language=null`（自动识别） | ✅ 未传语言 |
| 精度 | `dtype=bf16` | ✅ |
| 输出 | `<id>.txt`（UTF-8） | ✅ 300 个，见 `official-eval/outputs/` |
| 反馈表 | `condition_id,model,status,output_path,error,parameter_changes` | ✅ `official-eval/反馈表.csv` |
| 环境说明 | 模型/权重版本、框架版本、设备型号与数量 | ✅ `official-eval/环境说明.txt` |

### 测试集构成

| 数据源 | 语言 | 条数 | 成功 |
| --- | --- | ---: | ---: |
| AISHELL-1 test | zh | 100 | 100 |
| LibriSpeech test-clean | en | 100 | 100 |
| LibriSpeech test-other | en | 100 | 100 |
| **合计** | | **300** | **300** |

### 结果（`official-eval/eval_result.json` 原文）

```json
{
  "total": 300,
  "ok": 300,
  "fail": 0,
  "groups": {
    "AISHELL-1":   { "n": 100, "CER": 1.21, "WER": 0.0 },
    "LibriSpeech": { "n": 200, "CER": 0.65, "WER": 2.86 }
  }
}
```

**关于这些数字的性质，必须说清**：

- 官方明写「**不进行统一评分和性能比较**」，因此上面的 CER/WER 是**本仓库自算**的，
  仅用于内部记录与横向参考，**不是官方评分**。算法见 `evaluate_official.py` 的 `cer()` / `wer()`：
  - 中文：NFKC 归一化后去掉空白与标点，按**字符**算编辑距离
  - 英文：NFKC + 大写 + 去标点 + 压缩空白，按**词**算 WER、按**字符**算 CER
  - 聚合方式：CER 按字符数加权，WER 按句平均
- 与官方模型卡给出的参考值对照（口径不同，仅供感知量级）：
  模型卡报告 LibriSpeech clean/other WER 1.63 / 3.38、AISHELL-2 2.71；
  本件 LibriSpeech 200 条（clean+other 各半）WER 2.86，量级一致。

### 抽样核对（人工）

| id | 参考文本 | 识别结果 |
| --- | --- | --- |
| `BAC009S0764W0121` | 甚至出现交易几乎停滞的情况 | `甚至出现交易几乎停滞的情况。` ✅ |
| `1089-134686-0000` | HE HOPED THERE WOULD BE STEW FOR DINNER… | `He hoped there would be stew for dinner, turnips and carrots, …` ✅ |

英文条目我们保留输出原文大小写与标点，仅用于展示；上述 CER/WER 计算前已做归一化。

## 7. 性能参考

**官方明确不要求**：T04 主文档原文「不限定框架或调用方式，不要求提供服务接口，**不进行统一评分和性能比较**」。
因此本条的 `meta.bench` 取值为 `not-required`（不是缺材料）。

已实测的运行相关数据（仅供参考）：

| 指标 | 值 |
| --- | --- |
| 冷启动 | 150–200 s |
| 显存 | 模型 4.7 GB，`--gpu-memory-utilization 0.85` |
| 300 条评测耗时 | **35 s**（4 并发） |
| 单条 4 s 音频 | 见 `logs/smoke.log` 的 `usage.duration` |

## 7. 性能参考

**未做 6 档基准。** 赛题规定「文本 / 多模态理解类用 `bench.sh`，**生成类与语音类的测法随分题材料下发**」——语音类的性能测法尚未下发，因此本条目只有冷启动时间（150–200 s）与显存占用（single-card，`mem-fraction 0.85`）。

后续拿到官方测法后补。

## 8. 注意事项

- **base64 不能内联进 `curl -d`**：103 KB 音频编码后约 138 KB，会触发 `Argument list too long`。`smoke.sh` 改为先写 `/tmp/asr_req.json` 再 `--data @file`。
- **容器主进程不能先退出**：`run.sh` 把服务放后台并轮询 health 后 `exit 0`，若容器以它为主进程会导致「服务刚 ready 容器就退出」。`container/submit.slurm` 里改成容器内保活（`exec tail -f /dev/null`），再用 `podman exec` 跑 smoke。
- **测试音频放在仓库里**：计算节点无公网，`smoke.sh` 不能依赖外部下载；103 KB 的 wav 直接入库换来可复现性。
- **官方评测设定**：BF16、vLLM、greedy、`max_new_tokens=1024`、不指定语种。本条目 smoke 用的 `max_tokens=256` 更短，仅够单句。

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

## 6. 精度测评

**未完成。** T04 的 WER/CER 评测集与 `baselines/` 为官方下发物，目前未获取；`verify.py` 在该文件缺失时主动 `[BLOCKED]` 退出，不产出任何分数。

## 7. 性能参考

**未做 6 档基准。** 赛题规定「文本 / 多模态理解类用 `bench.sh`，**生成类与语音类的测法随分题材料下发**」——语音类的性能测法尚未下发，因此本条目只有冷启动时间（150–200 s）与显存占用（single-card，`mem-fraction 0.85`）。

后续拿到官方测法后补。

## 8. 注意事项

- **base64 不能内联进 `curl -d`**：103 KB 音频编码后约 138 KB，会触发 `Argument list too long`。`smoke.sh` 改为先写 `/tmp/asr_req.json` 再 `--data @file`。
- **容器主进程不能先退出**：`run.sh` 把服务放后台并轮询 health 后 `exit 0`，若容器以它为主进程会导致「服务刚 ready 容器就退出」。`container/submit.slurm` 里改成容器内保活（`exec tail -f /dev/null`），再用 `podman exec` 跑 smoke。
- **测试音频放在仓库里**：计算节点无公网，`smoke.sh` 不能依赖外部下载；103 KB 的 wav 直接入库换来可复现性。
- **官方评测设定**：BF16、vLLM、greedy、`max_new_tokens=1024`、不指定语种。本条目 smoke 用的 `max_tokens=256` 更短，仅够单句。

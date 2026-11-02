# Qwen3-ForcedAligner-0.6B on BW1000 + Transformers

> T04 词级强制对齐。按官方 30 条英文条件执行，**30/30 成功**。
> ⚠️ 官方标注 `BOUNDARIES_PENDING`：**无人工词边界，只收集预测，不计算准确率**。

## 1. 简介

`Qwen/Qwen3-ForcedAligner-0.6B` 在 `异构加速卡BW（BW1000）` + `Transformers`（bf16）上的适配验证。

Qwen3-ASR 系列配套的强制对齐模型：输入音频与对应文本，以**非自回归（NAR）**方式输出词级 / 字级时间戳，
支持 11 种语言、最长 5 分钟语音。`config.json` 实测：`architectures=['Qwen3ASRForConditionalGeneration']`，
`thinker_config.model_type = qwen3_forced_aligner`，`classify_num=5000`，BF16 单文件约 1.8 GB，Apache-2.0。

**为什么不用 SGLang / vLLM**：官方 `align()` 接口只存在于 `qwen-asr` 包（Transformers 后端）。
vLLM 也支持，但要通过 `forced_aligner_kwargs`，且本任务的产物是**时间戳数组**而不是生成文本，
没有 OpenAI serving 语义可用。故本条目是**一次性批处理**，不提供常驻服务。

**与官方用法的差异（必须说明）**：

1. `qwen-asr` 钉死 `transformers==4.57.6`，镜像内是 **5.8.1**，`check_model_inputs` 签名已变 →
   装到独立目录 `/opt/fa-libs` 并用 `PYTHONPATH` 前置，**不改动镜像本体**（镜像内的 vLLM 0.20 依赖 5.8.1）。
2. `nagisa`（日语分词）依赖 **dyNET** 重 C++ 库。本条目评测集为 30 条英文，不触达该路径 →
   用 `wheels/stub/nagisa.py` 替换；若真跑日语会**显式抛错**而非静默出错。

- 权重：`https://www.modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B`
- 镜像与 digest：见 `image.lock`
- 提交仓库：https://developer.sourcefind.cn/codes/luql/qwen3-forcedaligner-06b-bw1000-transformers

## 2. 验证环境

```text
image: image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
torch: 2.10.0   HIP: 6.3.26093   triton: 3.5.0
Device: BW × 1，vram 64.0 GB，arch gfx936:sramecc+:xnack-
镜像内 transformers: 5.8.1（被隔离目录的 4.57.6 覆盖，vLLM 0.20.0 不受影响）
隔离目录 /opt/fa-libs: transformers 4.57.6 + huggingface_hub 0.36.2 + tokenizers 0.22.2
```

- 卡型 / 卡数：BW1000 × 1
- 模型加载：**8.5 s**
- 30 条推理：**10.7 s**

## 3. 适配优化方案

### 3.1 基础镜像

复用 `vllm0.20-…-qwen_asr1.7_bw` 镜像：它已带 DCU 适配的 torch 2.10 与音频处理栈（librosa / soundfile）。

### 3.2 软件包升级（隔离安装，不改镜像）

镜像内 `transformers 5.8.1` 与 `qwen-asr` 要求的 `4.57.6` 冲突。**不做原地降级**，改为：

```bash
pip install -q --no-deps --target /opt/fa-libs \
  transformers-4.57.6 huggingface_hub-0.36.2 tokenizers-0.22.2 \
  qwen_asr-0.0.6 soynlp-0.0.493 qwen_omni_utils-0.0.9
cp stub/nagisa.py /opt/fa-libs/nagisa.py
export PYTHONPATH=/opt/fa-libs:$PYTHONPATH
```

完整依赖与**平台匹配坑**见 `wheels/README.md`（登录节点是 py3.12、容器是 py3.10，
直接 `pip download` 会拿到 cp312 wheel）。

### 3.3 代码改动

无 patch。适配集中在上面的隔离安装与 stub。

### 3.4 升级步骤

见 `container/eval_official.slurm`，脚本自带上述全部步骤。

## 4. 运行方式

本条目**没有常驻服务**：

```bash
sbatch container/eval_official.slurm     # 官方 30 条一条命令跑完
```

单条 smoke（验证链路）：

```bash
bash smoke.sh
```

## 5. Smoke 测试

`smoke.sh` 对一条 LibriSpeech 音频做对齐，打印前 5 个词的起止时间。

```
音频: /assets/data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac | 存在: True
前 5 个词: [('HE', 0.3, 0.46), ('HOPED', 0.46, 0.82), ('THERE', 0.82, 1.06), ...]
```

## 6. 精度测评（官方条件，已完成预测）

按官方 `formal/conditions/forced_aligner.jsonl` 的 30 条执行：**30/30 成功，0 失败**。

| 项 | 值 |
| --- | --- |
| 条件 | LibriSpeech test-clean，按 ID 排序、时长 ≤15 s 的前 30 条 |
| 输入 | `assets.audio` + **正确 text** + `language`（官方允许传正确文本） |
| 输出 | 每词 `start` / `end`，单位**秒**（`official-eval/outputs/<id>.json` 与 `.tsv`） |
| 预测词数合计 | **572** |
| 参考词数合计 | **572** |

### 关于"精度"必须说清

官方在条件清单里标记 **`BOUNDARIES_PENDING`**，并写明：

> 目前没有人工词边界，清单标记 `BOUNDARIES_PENDING`，**仅收集词级起止时间预测，不计算边界准确率**，
> 也不要求用户补标。

**所以本条目没有任何"准确率"数字，也不应该编一个。** 上文"预测词数 = 参考词数 = 572"
只能说明**分词切分与参考文本完全对齐**（没有漏词/多词），**不等于边界时刻准确**。

样例（`1089-134686-0001`）：

```
word	start_s	end_s
STUFF	0.32	0.8
IT	0.8	0.88
INTO	0.88	1.44
YOU	1.44	1.44
HIS	1.6	1.76
BELLY	1.76	2.16
```

## 7. 性能参考

官方 T04 明确「不进行统一评分和性能比较」→ `meta.bench = not-required`。

已实测（仅供参考）：

| 指标 | 值 |
| --- | --- |
| 模型加载 | 8.5 s |
| 30 条推理 | 10.7 s（约 0.36 s/条，含音频解码） |
| 显存 | 模型 1.8 GB，单卡 64 GB |

## 8. 注意事项

- **transformers 版本必须隔离，不能原地降级**：镜像内 `vllm 0.20.0` 依赖 `transformers 5.8.1`，
  装 4.57.6 会连带破坏 vLLM。用 `--target /opt/fa-libs` + `PYTHONPATH` 前置。
- **wheel 平台要对齐**：登录节点 py3.12、容器 py3.10，`pip download` 默认给 cp312 的 wheel
  （nagisa 就是这样踩到的），必须 `--python-version 310 --platform manylinux2014_x86_64`。
- **`huggingface_hub` 也要隔离**：transformers 4.57.6 要求 `<1.0`，镜像是 `1.11.0`。
- **`nagisa` 用 stub 替换**：真实版依赖 dyNET；仅日语路径需要，本条目英文评测不触达，
  且 stub 在真被调用时显式抛错。
- **不是 serving 任务**：产物是时间戳文件，没有 `/v1/chat/completions` 可用，
  因此 `run.sh` 只打印用法；评测入口是 `container/eval_official.slurm`。
- **`eval_result.json` 里故意没有准确率字段**：见 §6，官方不评分，编一个就是伪造数据。

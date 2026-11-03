# Wan2.1-I2V-14B-480P × BW1000 × 官方 Wan2.1 代码

> 状态：**进行中**（2026-11-03 起）。本文件在探针跑通后按实测回填 §2/§5/§6/§7，未跑通前不写任何数字。

## 1. 简介

| 项 | 值 |
| --- | --- |
| 模型 | `Wan-AI/Wan2.1-I2V-14B-480P`（图生视频，14B） |
| 任务 | T03 图生视频：输入首帧 + prompt（+negative_prompt），输出 832×480 / 81 帧 / 16 fps |
| 权重 | 82.3 GB（7 个 DiT 分片 ~68 GB + UMT5-XXL 文本编码器 11.4 GB + Wan2.1 VAE） |
| 卡型 | BW1000（单卡 64 GB）× 1 进程/卡，8 卡并行分片 |

**与官方部署方式的差异**：
- 官方要求 `flash_attn`（`INSTALL.md` 明示）；DCU 无可用构建 →
  `wan/modules/attention.py` 在 `flash_attn` 缺失时**自动回落** `torch.nn.functional.scaled_dot_product_attention`。
  代价是长序列注意力更慢，正确性不受影响（该分支同时会丢弃 padding mask，见 §8）。
- 不用 `--use_prompt_extend`（官方默认关闭；开启会调用 DashScope 或本地 Qwen 扩写 prompt，
  会改变输入文本，评测中禁止）。

## 2. 验证环境

⚠️ 探针跑通后从 `logs/env.log` 逐项回填（镜像 / torch / HIP / 卡型 / 驱动 / 权重路径）。

## 3. 适配优化方案

### 3.1 基础镜像

`image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1`（DTK 26.04 / DCU torch 2.9.0+das）。
理由：这是集群里唯一自带 DCU 版 torch + torchvision 的镜像；Wan2.1 不做 serving，只要张量算子在卡上。

### 3.2 软件包升级

- 纯 Python 依赖隔离装到 `/opt/wan-libs`（`--no-deps` + cp310 轮子）：`easydict`、`ftfy`、`imageio`、
  `imageio-ffmpeg`、`regex`
- `diffusers 0.41.0` + `huggingface_hub 1.33.0` 复用 Qwen-Image 条目归档的轮子集（`wan/utils/fm_solvers.py`
  只用到 `ConfigMixin` / `register_to_config` / `scheduling_utils` / `randn_tensor` 等稳定 API）
- **刻意不装**：`flash_attn`（DCU 无构建）、`dashscope`（prompt 扩写，评测不用）、`gradio`

### 3.3 代码改动或参数

- 用官方源码原样（`9737cba`），**不改模型代码**
- 并行方式：官方给的 `ulysses_size` / `ring_size`（xfuser 序列并行）**不使用**；
  改为**数据并行分片**——60 条条件按 `index % 8` 分给 8 个进程，每进程一张卡
  （理由：xfuser 依赖未在 DCU 验证；分片能线性摊薄，且单条失败不影响其他条）
- `--offload_model True` 默认开启，避免 14B + T5 顶上 64 GB

### 3.4 升级步骤

```bash
sbatch container/submit.slurm                                  # 全量 60 条（8 卡）
PROBE_ONLY=1 sbatch container/submit.slurm                     # 只跑 1 条，验管线
```

## 4. 服务启动脚本

本条目**无常驻服务**（一次性批处理）。`container/submit.slurm` 展开后的核心容器命令见该文件；
`container/inner.sh` 负责装依赖 → 导入校验 → 启动 N 个分片进程 → 合并反馈表 → 校验产物规格。

## 5. Smoke 测试

⚠️ 待 `PROBE_ONLY=1` 跑完后填入真实返回（起服务时长、单条耗时、产物帧数/fps/大小）。

## 6. 精度测评

官方条件：`conditions/wan_i2v.jsonl`，**60 条**（20 张首帧 × seed 42/2026/1234）。
EMA 无、固定条件全部取自清单 `params`（832×480 / 81 帧 / 40 步 / unipc / shift 3.0 / guidance 5.0）。

官方明示「不进行统一评分和性能比较」→ 本条目**不计算 VBench 等任何分数**，只交原始产物 + 规格统计
（帧数、fps、时长、体积）。`verify.py` 在没有官方评测器时主动 `[BLOCKED]` 退出。

## 7. 性能参考

⚠️ 待测。将记录：单条视频耗时（8 卡分片下的墙钟）、总耗时、峰值显存、产物体积中位数。

## 8. 注意事项

- **`flash_attn` 缺失是预期而非故障**：`wan/modules/attention.py` 会 warn
  `Padding mask is disabled when using scaled_dot_product_attention`。若后续发现"多图/不同长度"场景出现串扰，
  需在此分支自行加 mask（当前 I2V 单序列，不受影响）。
- **T5 用 transformers 5.3.0 加载 UMT5-XXL**：官方 pin 是 `>=4.49`，5.x 是否兼容以探针为准；
  若报 `T5Tokenizer` 相关错误，退路是把 `fa310` 那套 transformers 4.57.6 隔离装上（已验证可用）。
- **权重 82 GB**：磁盘/配额敏感，跑完如需腾空间可删（ModelScope 可重下），但**镜像与 wheel 必须归档**。
- 分片策略依赖 `index % N` 稳定：**不要中途改 N**，否则续跑会错位（产物按 id 命名，故实际不会错，只是分片不均）。

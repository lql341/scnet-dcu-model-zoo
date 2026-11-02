# scnet-dcu-model-zoo

国产异构加速卡（海光 DCU / BW1000 / K100-AI）上的**大模型推理适配实测库**。

每个条目是一个「**模型 × 卡型 × 框架 × 精度**」组合，包含：可一键复现的启动脚本、环境自检、Smoke 测试、6 档性能基准与原始日志。
**所有数字都来自实测**，不是从模型卡抄的；跑不通的配置会写明失败原因而不是删掉。

## 目录约定

```
models/<org>__<model>__<card>__<framework>[/__<precision>]/
├── README.md         # 8 章技术报告（同黑客松模板结构）
├── env.sh            # 唯一需要改的参数文件
├── env_check.sh      # → logs/env.log
├── run.sh            # → logs/server.log
├── smoke.sh          # → logs/smoke.log
├── verify.py         # → verify_result.json（精度，需官方基线）
├── bench.sh          # → bench/（6 档 × 2 轮）
├── logs/             # 原始日志，不删减
└── bench/            # summary.md + round1/round2.jsonl + 分档日志
```

没有走官方模板的临时实验放在 `sweeps/`，工具脚本在 `tools/`。

每个条目附 `image.lock`（镜像 digest），**本项目不构建镜像**，全部复用 Harbor 现成镜像。
为什么用单仓库、体积怎么控制、什么时候该拆：见 [docs/repo-layout.md](docs/repo-layout.md)。
规范与守门（提交前必跑）：

镜像归档与迁移（防止上游删除导致不可复现）：

- 归档位置与还原方式：[docs/image-archive.md](docs/image-archive.md)
- 推送到自有命名空间（手动）：[docs/manual-push.md](docs/manual-push.md)

```bash
./tools/lint_entry.sh      # 按 docs/entry-spec.md 校验所有条目
python3 tools/gen_index.py # 从 meta.json 重生成 INDEX.md 与下方总矩阵
./tools/check_quota.sh     # 体积守门：单条目 ≤2 MB，全库 ≤25 MB
```

## 跑通矩阵

<!-- MATRIX:BEGIN （由 tools/gen_index.py 生成，勿手改） -->
| 模型 | 参数/权重 | 卡型 | 框架 | 精度 | 状态 | 4K/c64 | 16K/c8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Recon2Reason-Reasoning-4B](models/BAAI__Recon2Reason-Reasoning-4B__bw1000__sglang/) | BW1000 ×1 | sglang | bf16 | ✅ 跑通（bench 口径未下发） | — | — |
| [RoboBrain2.5-8B-NV](models/BAAI__RoboBrain2.5-8B-NV__bw1000__sglang/) | BW1000 ×1 | sglang | bf16 | ✅ 跑通（bench 口径未下发） | — | — |
| [Trident-Qwen3VL-2B](models/OpenBMB__Trident-Qwen3VL-2B__bw1000__sglang/) | BW1000 ×1 | sglang | bf16 | blocked | — | — |
| [Trident-Qwen3VL-2B](models/OpenBMB__Trident-Qwen3VL-2B__bw1000__vllm/) | BW1000 ×1 | vllm | bf16 | blocked | — | — |
| [Qwen-Image-2.1-Turbo](models/Qwen__Qwen-Image-2.1-Turbo__bw1000__diffusers/) | BW1000 ×1 | diffusers | bf16 | ✅ 跑通（官方不要求性能对比） | — | — |
| [Qwen3-ASR-1.7B](models/Qwen__Qwen3-ASR-1.7B__bw1000__vllm/) | BW1000 ×1 | vllm | bf16 | ✅ 官方 300/300 条完成 | — | — |
| [Qwen3-ForcedAligner-0.6B](models/Qwen__Qwen3-ForcedAligner-0.6B__bw1000__transformers/) | BW1000 ×1 | transformers | bf16 | ✅ 官方 30/30 条完成 | — | — |
| [AuK](models/Tencent-Hunyuan__AuK__bw1000__transformers/) | BW1000 ×1 | transformers | bf16 | ✅ 跑通（官方不要求性能对比） | — | — |
| [MiMo-V2.6-Distill-Qwen-9B](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__sglang/) | BW1000 ×1 | sglang | bf16 | ✅ 6 档基准 | 998.05 | 308.72 |
| [MiMo-V2.6-Distill-Qwen-9B](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__vllm/) | BW1000 ×1 | vllm | bf16 | ✅ 6 档基准 | 489.52 | 44.83 |
| [Kimi-Linear-48B-A3B-Instruct](models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang/) | BW1000 ×4 | sglang | bf16 | ✅ 6 档基准 | 839.00 | 119.21 |
<!-- MATRIX:END -->

| 模型 | 参数/权重 | 卡型 | 框架 | 精度 | 状态 | 4K/c64 | 16K/c8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Kimi-Linear-48B-A3B](models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang/) | 48B/A3B · 91.5 GB | BW1000 ×4 | SGLang 0.5.10 | bf16 | ✅ 跑通 | 608.23 | 80.73 |
| [MiMo-V2.6-Distill-Qwen-9B](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__sglang/) | 9.4B · 18 GB | BW1000 ×1 | SGLang 0.5.10 | bf16 | ✅ 跑通 | 998.05 | 308.72 |
| [MiMo-V2.6-Distill-Qwen-9B](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__vllm/) | 9.4B · 18 GB | BW1000 ×1 | vLLM 0.20.0 | bf16 | ✅ 跑通 | 489.52 | 44.83 |
| [Qwen3-ASR-1.7B](models/Qwen__Qwen3-ASR-1.7B__bw1000__vllm/) | 2.35B · 4.7 GB | BW1000 ×1 | vLLM 0.20.0 | bf16 | ✅ 语音链路通（bench 口径未下发） | — | — |

吞吐单位 tok/s，取 `bench/round2.jsonl` 的 `output_throughput`。

## 三条最有价值的实测结论

1. **BW1000 上必须传 `/dev/mkfd`**。只传 `/dev/kfd` 时 HSA 初始化失败，torch 报 `No HIP GPUs are available`，且错误发生在初始化阶段、没有更多线索 —— 这是整套适配里最难定位的一处。详见 [docs/gotchas.md](docs/gotchas.md)。
2. **CUDA Graph 不是禁区**。早期在 `max-running-requests=8` 配置下遇到捕获阶段静默退出，据此得出"必须 `--disable-cuda-graph`"的结论；放宽并发后 CUDA Graph 可正常捕获，且 SGLang 侧带来 **+73%~+466%**、vLLM 侧 **+3%~+32%**。两个件都保留了这个认知修正的记录。
3. **并发上限设小会把性能基准测废**。`--max-running-requests 8` 时 64 并发请求只有 8 个在跑、56 个排队（`server.log` 里 `#queue-req: 56`），4K/c64 只有 176 tok/s；改为 64 后 998 tok/s。**压测参数的每一个都要和被测负载对齐。**

## 快速上手

```bash
git clone https://github.com/lql341/scnet-dcu-model-zoo
cd scnet-dcu-model-zoo/models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang
# 宿主机（Slurm 上，4 卡）
sbatch container/submit.slurm
# 或只验到 smoke
SKIP_BENCH=1 sbatch --export=ALL container/submit.slurm
```

## 相关

- 赛事提交仓库（光合代码托管）：<https://developer.sourcefind.cn/codes/luql>
- 集群操作 skill：[scnet-hpc](https://github.com/lql341/scnet-hpc)
- 科学计算移植 playbook：[scientific-dcu-porting](https://github.com/lql341/scientific-dcu-porting)

## 许可

各模型的权重与代码遵循其原始许可；本仓库的脚本与文档为 MIT。

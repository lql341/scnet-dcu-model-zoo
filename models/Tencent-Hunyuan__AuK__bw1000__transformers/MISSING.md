# 材料状态

## ✅ 已到

| 材料 | 位置 |
| --- | --- |
| 条件清单 120 条（中英各 20 参考音 × 3 seed） | `official/t03-t04/conditions/auk.jsonl` |
| 参考音 21 个（被 120 条复用） | `data/seedtts/zh|en/prompt-wavs/`，sha256 已校验，零缺失 |
| 反馈表模板 | `official/t03-t04/反馈表.csv` |

## ⚠️ 技术阻塞（非材料问题）

| # | 问题 | 说明 |
| --- | --- | --- |
| ① | **torch pin 不可满足** | 官方 `torch>=2.7,<2.8`；DCU 只有 2.9/2.10，PyPI 无 DCU 版 2.7 |
| ② | **transformers<5 不可满足** | 镜像是 5.3.0 |

这两条决定了本条目「能否跑通」本身就有风险，不以材料齐备为前提。

## 已完成部分

- ✅ 权重（AuK 6.3 GB + Qwen2.5-Omni-3B 12 GB）
- ✅ AuK 源码（30 MB，从 GitHub 本机 clone 后上传——集群直连 GitHub 不通）
- ✅ 依赖 wheel 离线包（x_transformers / torchdiffeq / omegaconf / qwen-omni-utils / safetensors / accelerate）
- ✅ 镜像选择（SGLang 镜像，因自带 DCU 版 torchaudio）
- ⬜ 探针（作业已提交，排队中）

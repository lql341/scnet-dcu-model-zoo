# 材料状态

## ✅ 已到

| 材料 | 位置 |
| --- | --- |
| 条件清单 120 条（中英各 20 参考音 × 3 seed） | `official/t03-t04/conditions/auk.jsonl` |
| 参考音 21 个（被 120 条复用） | `data/seedtts/zh|en/prompt-wavs/`，sha256 已校验，零缺失 |
| 反馈表模板 | `official/t03-t04/反馈表.csv` |

## ⚠️ 官方 pin 不可满足（以实测为准，不阻塞）

| # | 问题 | 实测结论 |
| --- | --- | --- |
| ① | 官方 `torch>=2.7,<2.8`；DCU 只有 2.9/2.10 | **不是硬要求**：torch 2.9.0 下权重加载、推理均正常 |
| ② | 官方 `transformers<5`；镜像是 5.3.0 | 同上，实测可跑（`x_transformers` 是另一个包，勿混淆） |

## ✅ 2026-11-03 真实根因（此前 7 轮都被误导）

| 症状 | 根因 | 修法 |
| --- | --- | --- |
| 120/120 `ModuleNotFoundError: librosa` | 离线 wheel 是按 **cp312** 下的，镜像是 **py3.10**，pip 判 tag 不兼容直接拒装 | 用 `--python-version 310 --platform manylinux2014_x86_64 --abi cp310` 重下，见 `docs/gotchas.md` #18 |
| 单条耗时 4.5 s、仍全部失败 | 推理已跑通，死在**写盘**：`torchaudio.save()` 也走 torchcodec | 改用 `soundfile.write`，见 `docs/gotchas.md` #19 |

## 已完成

- ✅ 权重（AuK 6.3 GB + Qwen2.5-Omni-3B 12 GB）
- ✅ AuK 源码（30 MB，本机 clone 后上传——集群直连 GitHub 不通）
- ✅ 依赖：cp310 轮子 53 个（含 librosa 0.10.2 / numba 0.61.2 / scipy 1.15.3 / soxr / soundfile）
- ✅ 引擎加载 46.8 s；单卡 bf16；`missing=924 (text_encoder.*)` 与官方说明一致
- ✅ 官方 120 条：**120/120 成功**，531 s（单条 4.4 s，nfe=32）；产物 36 MB 不入库，见 `official-eval/`

## 体积提示

产物 120 × ~300 KB ≈ 36 MB，**不入库**（超单条目 2 MB 上限），只入库 `反馈表.csv` 与统计。

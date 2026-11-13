# 材料状态

## ✅ 已到

| 材料 | 位置 | 校验 |
| --- | --- | --- |
| 条件清单 5 条（DTU scan1/4/9/10/11） | `conditions/hy_world.jsonl` | 与官方同源 |
| 素材 `data/dtu`（123 文件，30 MB） | 集群 `~/work/hyw/eval-materials/data/dtu`；来自赛事 zip | 条件里的 sha256 逐条校验通过 |
| WorldMirror 权重 5.1 GB | `~/models/HY-World-2.0/HY-WorldMirror-2.0` | ModelScope 快照 |
| `skyseg.onnx` 176 MB | `harness/assets/skyseg.onnx` | sha256 `ab9c34c6…` 与 provenance 一致 |

## ⬜ 官方未下发 / 不适用

- `baselines/*.json`（T01/T02 精度基线）—— T03 不提供
- DTU 官方评价脚本（Chamfer / PSNR）—— 官方明示不评分，故不计算

## ⚠️ 资产缺口

- `hy-world2-bw-sglang` 镜像**尚未归档**（在 `dcu/acizv7iztz` 名下）。上游删除即不可复现，
  下次有卡时应 `podman save | gzip` 到 `luql:archive/zoo-dcu-2026-11/images/`。
- harness 已归档（`assets/hyworld2-bw-harness.tar.gz`），但**没有它就没法复现**，与镜像同等重要。

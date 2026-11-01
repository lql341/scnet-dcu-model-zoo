# SCNet 侧原始产物位置

所有数据目前落在临时 Slurm 账号名下。**GitHub 是本仓库的权威副本**，下列路径仅用于回溯。

## 郑州集群（BW1000，Slurm 账号 `scnethr2601a10`）

```
/public/home/scnethr2601a10/
├── models/
│   ├── Kimi-Linear-48B-A3B-Instruct/          92 GB，20 分片
│   └── MiMo-V2.6-Distill-Qwen-9B/             18 GB
├── images/
│   └── mimo-sglang-bw10000_1.1.1.tar          31 GB（podman load 用，避免每次 pull）
├── work/
│   ├── kimi/           Kimi 件的 env.log / smoke.log / server.log / bench/
│   ├── opt/            MiMo SGLang 优化后的全量基准
│   ├── vopt/           MiMo vLLM 优化后的全量基准
│   ├── sweep/          MiMo SGLang 参数扫描（5 变体）
│   ├── vsweep2/        MiMo vLLM 参数扫描（3 变体）
│   ├── ksweep/         Kimi CUDA Graph 扫描（未完成）
│   └── clone-test/     从光合仓库匿名 clone 复现测试
└── probe/              最早的裸机探测日志
```

## 迁移现状

- ✅ **有效产物已随本仓库进入 Git**（`models/*/logs/`、`models/*/bench/`）
- ⬜ 权重与镜像 tar **未迁移**：均可重新获取（ModelScope 24 MB/s、Harbor 内网 pull），
  且合计 141 GB，迁移成本高于重建成本
- ⬜ 若要整体归档到永久账号：需先恢复永久账号在该集群的 SSH 密钥
  （实测 `luql@zzeshell.scnet.cn` 报 `Permission denied (publickey)`）

## 重建成本

| 资产 | 重建方式 | 耗时 |
| --- | --- | --- |
| Kimi 权重 92 GB | `modelscope download --model moonshotai/Kimi-Linear-48B-A3B-Instruct` | 18 min |
| MiMo 权重 18 GB | 同上（XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B） | 3 min |
| SGLang 镜像 | `podman pull --tls-verify=false image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1` | 4–6 min |
| vLLM 镜像 | 同上（`vllm0.20-...-_bw:1.0.0`） | 4–6 min |

结论：**权重与镜像不值得备份**，脚本 + 日志 + 基准数据才是需要持久化的部分，已全部入库。

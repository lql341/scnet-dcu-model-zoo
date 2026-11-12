# 镜像归档（防上游删除）

## 为什么要归档

本项目 4 个条目引用的镜像属于**第三方账号** `dcu/linshp`。上游删仓库时，
`image.lock` 里的 digest 也救不回来（blob 随仓库消失）。归档是这道风险的兜底。

同时它是**后续手动 push 到自有命名空间**的物料来源，见 `docs/manual-push.md`。

## 归档位置

```
/public/home/luql/archive/zoo-dcu-2026-11/images/  ← 项目结项后的唯一位置（原 scnethr 账号已被回收）
（历史位置：/public/home/scnethr2601a10/zoo-images/）
├── mimo-sglang-bw10000_1.1.1.tar.gz             7.9 GB  sha256 f16125ba…
├── vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz         8.7 GB  sha256 986f95a3…
├── MANIFEST.md                                  digest + sha256 + 还原命令
├── sha256-all.txt                               两个 tar 的实测摘要
├── assets/                                      依赖与源码（体积小，重建难）
│   ├── wheels-auk-cp310.tar.gz   104 MB         librosa 链 + AuK 依赖，全部 cp310
│   ├── wheels-qi-cp310.tar.gz      6.6 MB       diffusers 0.41.0 + hub 1.33.0（可解析的版本对）
│   ├── wheels-fa310.tar.gz        37 MB         ForcedAligner 的 transformers/hub/qwen_asr
│   ├── AuK-src.tar.gz             30 MB         AuK 源码（集群直连 GitHub 不通）
│   └── assets-sha256.txt
├── archive_images.slurm / archive_vllm.slurm    生成脚本
└── verify_archive.slurm                         还原验证脚本
```

> 位置在临时 Slurm 账户名下。**待 `luql` 账号恢复该集群 SSH 后应搬到 `/public/home/luql/zoo-images/`**
> （见 `ARTIFACTS.md`）。当前该目录 `chmod 755` + 文件 `644`，`luql` 可直接 `cp` 走。
>
> **为什么把 wheel 也归档**：这两个镜像里没有 librosa / diffusers，而“按 cp312 下载的轮子
> 装进 py3.10 镜像”曾把两个条目误判成 blocked（`docs/gotchas.md` #18）。镜像 + 正确的 wheel
> 集合才是完整可复现单元，缺一个就退化成“能跑但跑不出东西”。

### 为什么没有第二份离线副本

实测从本机拉归档的速度：**16.4 MB / 289 s ≈ 57 KB/s**，17 GB 需 ~80 小时，不可行。
所以集群内 `zoo-images/` 是**唯一**归档副本，抗风险靠三件事，不靠“再拷一份到同一块盘”：

1. `sha256` + manifest（本文件）—— 发现损坏
2. `verify_archive.slurm` 的 load + smoke —— 证明可还原
3. **推到自己命名的 Harbor 命名空间**（`docs/manual-push.md`）—— 真正脱离临时账号，**等 P0-6 的 Harbor 凭据**

反向说明：权重（194 GB）**不归档**，ModelScope 可 24 MB/s 重下；镜像是上游账号的、随时可能消失，
必须归档。

## 四个镜像

| 文件名 | 源镜像 | manifest digest | 压缩后 | tar sha256 |
| --- | --- | --- | --- | --- |
| `mimo-sglang-bw10000_1.1.1.tar.gz` | `dcu/linshp/base/mimo-sglang-bw10000:1.1.1` | `sha256:db981c46…` | 7.9 GB | `f16125ba6dcb3020aaef33b89d390508f99aad980d7436286f043ed68596d77a` |
| `vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz` | `dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0` | `sha256:07930260…` | 8.7 GB | `986f95a3e5599f024797e41faa614f72cbd5a6c87e5e97f71c172f730269d45f` |
| `minimax-music3-bw-sglang_20261031-rc1.tar.gz` | `dcu/acizv7iztz/base/minimax-music3-bw-sglang:20261031-rc1` | `sha256:20f071aa…` | 13.7 GB | `b61b05bd7df45fdfe81885cb898f88572860a37734ececc23018dbfd531d7d87` |
| `vllm0.21.0-dtk2604-py3.10_20261112.tar.gz` | `dcu/wqxyyt06/base/vllm0.21.0-ubuntu22.04-dtk2604-py3.10:v1` | `sha256:c6a2f554…` | 9.5 GB | `c3b22bccb1e5d9c0af818452da4a5af554013d3cea02547c8c80bd87173728ab` |

（tar sha256 已于 2026-11-03 全量重算写入集群 `zoo-images/sha256-all.txt`；
完整清单（含 assets/ 与还原命令）见集群 `zoo-images/MANIFEST.md`。）

没有做任何修改：`load` 后的镜像名与上游完全相同（`dcu/linshp/base/...`），
说明 tar 直接来自上游镜像，未经重新构建。

> **第三个（Music3）是怎么来的**：Harbor 上有社区用户自己构建的 BW 版 Music3 + SGLang 镜像
> （`dcu/acizv7iztz/...`），里面还留着他自己的适配工作区（`/tmp/music3-hygon/`，已单独归档为
> `assets/music3-hygon-harness.tar.gz`）。上游删除该仓库的代价对我们来说等于「Music3 条目报废」，
> 所以即便它比前两个大（13.7 GB），也归档了。

### 关于 digest：两个不同的东西，不要混淆

| 名称 | 值 | 含义 |
| --- | --- | --- |
| Harbor **manifest digest** | `sha256:db981c466e74…`（SGLang）<br>`sha256:0793026099b5…`（vLLM） | registry 上 manifest 的摘要，用于 `podman pull image@sha256:…` 精确拉取 |
| 本地 `podman images --format {{.Digest}}` | `sha256:9d16c7f3aec3…`（SGLang）<br>`sha256:641a79d516b1…`（vLLM） | **本地 image config 的 digest**，与上面**本来就不相等** |

这是实测踩到的一个坑：一开始想用本地 `.Digest` 去比对 Harbor 的 manifest digest，发现对不上，
这是正常的 —— 两者哈希的输入不同。

**正确的完整性校验方式**（`verify_archive.slurm` 采用后者）：

1. 归档 tar 的 `sha256sum` 与 `MANIFEST.md` 记录一致（防传输/存储损坏）
2. `load` 成功且镜像名为上游原名（证明未经重建）
3. 起服务跑 smoke 通过（功能性验证）

不要用 `podman images --format {{.Digest}}` 去比 Harbor 的 manifest digest。

## 还原

```bash
export XDG_RUNTIME_DIR=/tmp/xdg-$$ && mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
JID=${SLURM_JOB_ID:-$$}
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID \
    --storage-driver overlay --storage-opt ignore_chown_errors=true"

gunzip -c zoo-images/mimo-sglang-bw10000_1.1.1.tar.gz | $PD load
gunzip -c zoo-images/vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz | $PD load
$PD images --format '{{.Repository}}:{{.Tag}}|{{.Digest}}'
```

`load` 恢复的是**原名**（`dcu/linshp/...`）；要推到自己名下必须先 `podman tag`，见 `manual-push.md`。

## 还原验证结果（实测）

| 镜像 | `load` 结果 | 体积 |
| --- | --- | --- |
| `mimo-sglang-bw10000:1.1.1` | ✅ `Loaded image` | 32.2 GB |
| `vllm0.20-...-_bw:1.0.0` | ✅ `Loaded image` | 29.6 GB |

两者恢复的镜像名都是**上游原名**（`dcu/linshp/base/...`），证明归档未经重建。

## 生成时的坑

- `podman save --compress` 在本机 podman 4.9 上**只支持 `--format docker-dir`**，
  直接加 `--compress` 报 `Error: --compress can only be set when --format is 'docker-dir'`。
  改用 `podman save <img> | gzip -1 > out.tar.gz`。
- 节点根盘水位不稳：两次提交遇到 `tmp free=7G` 被 `df` 前置检查拦下（`exit 42`）。
  归档单个镜像需 ~35 GB 临时空间（解压镜像 + tar 输出）。
- `gzip` 压缩比约 3.8:1（32 GB 未压缩 → 7.9 GB），耗时约 22 分钟。
- **同一作业里连续 `load` 两个镜像会失败**：解压后共需 ~62 GB 临时空间，
  实测在 100 GB 可用空间的节点上跑到第二个时仍报
  `no space left on device ... rocfft_kernel_cache.db`。
  应分两次作业验证（`ONLY=sgl` / `ONLY=vllm`）。

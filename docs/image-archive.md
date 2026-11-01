# 镜像归档（防上游删除）

## 为什么要归档

本项目 4 个条目引用的镜像属于**第三方账号** `dcu/linshp`。上游删仓库时，
`image.lock` 里的 digest 也救不回来（blob 随仓库消失）。归档是这道风险的兜底。

同时它是**后续手动 push 到自有命名空间**的物料来源，见 `docs/manual-push.md`。

## 归档位置

```
/public/home/scnethr2601a10/zoo-images/          (chmod 755，同集群账号可读)
├── mimo-sglang-bw10000_1.1.1.tar.gz             7.9 GB
├── vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz         8.7 GB
├── MANIFEST.md                                  digest + sha256 + 还原命令
├── archive_images.slurm / archive_vllm.slurm    生成脚本
└── verify_archive.slurm                         还原验证脚本
```

> 位置在临时 Slurm 账户名下。**待 `luql` 账号恢复该集群 SSH 后应搬到 `/public/home/luql/zoo-images/`**
> （见 `ARTIFACTS.md`）。当前该目录 `chmod 755` + 文件 `644`，`luql` 可直接 `cp` 走。

## 两个镜像

| 文件名 | 源镜像 | digest | 压缩后 |
| --- | --- | --- | --- |
| `mimo-sglang-bw10000_1.1.1.tar.gz` | `dcu/linshp/base/mimo-sglang-bw10000:1.1.1` | `sha256:db981c466e74…` | 7.9 GB |
| `vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz` | `dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0` | `sha256:0793026099b5…` | 8.7 GB |

没有做任何修改：`load` 后的镜像名与上游完全相同（`dcu/linshp/base/...`），
说明 tar 直接来自上游镜像，未经重新构建。

### 关于 digest：两个不同的东西，不要混淆

| 名称 | 值 | 含义 |
| --- | --- | --- |
| Harbor **manifest digest** | `sha256:db981c466e74…`（SGLang）<br>`sha256:0793026099b5…`（vLLM） | registry 上 manifest 的摘要，用于 `podman pull image@sha256:…` 精确拉取 |
| 本地 `podman images --format {{.Digest}}` | `sha256:9d16c7f3aec3…`（SGLang，实测） | **本地 image config 的 digest**，与上面**本来就不相等** |

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

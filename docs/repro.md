# 从归档重建一个条目（端到端）

目的：**换机器 / 上游镜像被删 / 临时账号回收**之后，仍能把任意条目跑到「出真产物」。
所需素材只有两处：本仓库（脚本 + 日志 + 基准）与 `/public/home/luql/archive/zoo-dcu-2026-11/`（镜像 + wheel 集 + 官方产物）。
权重不归档（ModelScope 可重下），但要确认版本。

## 0. 前置检查（每次都做，30 s）

```bash
ssh zzeshell-hz1
squeue -u scnethr2601a10                 # 分区只允许 hx1hdnormal
df -h /public/home/scnethr2601a10 | tail -1   # 需要 155 G 以上余量才动镜像解包
cd /public/home/luql/archive/zoo-dcu-2026-11 && sha256sum -c SHA256SUMS
```

## 1. 还原镜像（一次一个，别连两个）

```bash
export XDG_RUNTIME_DIR=/tmp/xdg-$$ && mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
JID=${SLURM_JOB_ID:-$$}
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
gunzip -c /public/home/luql/archive/zoo-dcu-2026-11/images/mimo-sglang-bw10000_1.1.1.tar.gz | $PD load     # 32.2 GB
gunzip -c /public/home/luql/archive/zoo-dcu-2026-11/images/vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz | $PD load # 29.6 GB
$PD images --format '{{.Repository}}:{{.Tag}}'
```

> 同一作业里连续 load 两个会 `no space left on device`（解压共需 62 GB）。分两次作业。

## 2. 还原依赖 wheel（只有 AuK / Qwen-Image / ForcedAligner 需要）

```bash
mkdir -p ~/wheels
tar xzf /public/home/luql/archive/zoo-dcu-2026-11/assets/wheels-auk-cp310.tar.gz  -C ~/wheels   # -> ~/wheels/auk-fix
tar xzf /public/home/luql/archive/zoo-dcu-2026-11/assets/wheels-qi-cp310.tar.gz   -C ~/wheels   # -> ~/wheels/qi2
tar xzf /public/home/luql/archive/zoo-dcu-2026-11/assets/wheels-fa310.tar.gz      -C ~/wheels   # -> ~/wheels/fa310
```

**这些是 cp310 轮子**（镜像 py3.10）。若自己重下，必须：

```bash
python3 -m pip download --no-deps --dest wheels --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 310 --implementation cp --abi cp310 <pkgs>
```

## 3. 取条目脚本并提交

```bash
# 本机 → 集群
rsync -az --exclude logs --exclude bench --exclude official-eval \
  -e "ssh -o BatchMode=yes" \
  models/<条目>/ zzeshell-hz1:/public/home/scnethr2601a10/work/zoo-<名>/

# 提交（带磁盘重投循环：节点 /tmp 水位极不稳）
ssh zzeshell-hz1 'cd /public/home/scnethr2601a10/work/zoo-<名> && \
  for i in $(seq 1 12); do J=$(sbatch --parsable container/submit.slurm); sleep 25; \
    if grep -qE "NOT_ENOUGH_SPACE|no space" slurm-$J.out 2>/dev/null; then echo "try$i space"; \
    else echo "KEEP $J"; break; fi; done'
```

## 4. 验收（别只看「作业 COMPLETED」）

| 检查 | 命令 / 判据 |
| --- | --- |
| 逐条状态 | `awk -F, 'NR>1{print $3}' 反馈表.csv \| sort \| uniq -c` → 应全为「成功」 |
| 产物规格 | 音频：`wave` 读 24000 Hz / 1 ch / 16 bit；图像：全部 2048×2048 |
| 时长/尺寸合规 | 与条件清单 `expected.duration_min_s` 逐条比对 |
| 假数据探测 | 压测类必须过 `tools/preflight_lib.py`；`TTFT=0` 或吞吐 >9000 tok/s 一律当失败 |
| 入档 | 反馈表 + eval_result + stats + 环境说明 → `official-eval/`；大产物不入库 |

## 5. 归档新资产（用过就归档，不等「以后」）

```bash
# 新镜像
podman save <img> | gzip -1 > /public/home/luql/archive/zoo-dcu-2026-11/images/<repo-basename>_<tag>.tar.gz
# 并把新文件行追加到 /public/home/luql/archive/zoo-dcu-2026-11/SHA256SUMS

# 新 wheel 集
tar czf /public/home/luql/archive/zoo-dcu-2026-11/assets/wheels-<用途>-cp310.tar.gz -C ~/wheels <dir>
cd /public/home/luql/archive/zoo-dcu-2026-11 && sha256sum images/*.tar.gz assets/*.tar.gz > SHA256SUMS
```

然后更新 `/public/home/luql/archive/zoo-dcu-2026-11/SHA256SUMS` 与 `docs/image-archive.md` 的表。

## 已知的坑（照抄会踩）

- `--entrypoint=/bin/bash` + 挂载 `container/inner.sh`，不要用 `podman run -c "..."`（`$` 被展开）
- 必须传 `/dev/mkfd`，否则 HSA 枚举不到设备
- `--root /tmp/pr-$JOBID` 每作业独立，否则上一个作业的 pause 状态会让 pull 秒失败
- SGLang 加 `--attention-backend triton`（默认选到残包 aiter，首个 decode 就崩）
- 压测 `MAX_LEN ≥ 20480`，否则 16K 档全 400 且被当成成功

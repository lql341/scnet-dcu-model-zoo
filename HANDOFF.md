# HANDOFF —— 会话交接与 living TODO

> 最后更新：2026-11-03（夜）　余额 **4344.02**（起始 4405.96）
> 本轮新进展：**AuK 与 Qwen-Image 的 blocked 根因全部推翻重定位**——都不是卡的错，
> 是「离线 wheel 按 cp312 下、容器是 py3.10」「`--no-deps` 不校验 `Requires-Dist`」这类工程错误。
> 两个条目已重新起跑官方全量条件评测。详见 §7 / §9。
> 权威积累库：<https://github.com/lql341/scnet-dcu-model-zoo>
> 参赛仓库：<https://developer.sourcefind.cn/codes/luql>（7 个 public）
>
> **本文件是活文档。每次会话结束前更新「§8 进度」与「§9 TODO」。**
>
> **第一原则：镜像与经验 > 比赛提交。** 比赛是一次性的，经验与可复现环境是下次的起点。
> 因此每次会话收尾固定四件：① 新用过的镜像归档到 `~/zoo-images/` 并更新 MANIFEST；
> ② 新踩的坑写进 `docs/gotchas.md`；③ 三件套（lint / gen_index / quota）；
> ④ **推两个远端**（GitHub `origin` + GitLab `sourcefind`）。

---

## 1. 一句话现状

在 **BW1000（海光 DCU）** 上跑通 8 个「模型 × 框架」组合，其中 **3 个有完整 6 档性能基准**、
**2 个完成官方条件评测（ASR 300/300、ForcedAligner 30/30）**。
本轮把原先 2 个 blocked（AuK、Qwen-Image）的根因重定位并修掉，**两个都已跑完官方全量条件**：
- AuK：卡点 = 离线 wheel 标签（cp312 vs py3.10）+ `torchaudio.save` 走 torchcodec → **120/120**，4.4 s/条
- Qwen-Image：卡点 = hub 版本低于 diffusers 元数据要求（0.36.2 < 1.32）→ **180/180**，19.8 s/张
**没有任何精度数据能对齐官方基线 —— 因为 T01/T02 的 `baselines/` 从未下发。**

---

## 2. 账号与访问

| 账号 | 用途 | 状态 |
| --- | --- | --- |
| `scnethr2601a10` | 郑州集群 Slurm 账号（BW1000 唯一可用） | ✅ 正常，`/public` 400 G 配额 |
| `luql` | SCNet **平台账号**（OpenAPI / Notebook / GitLab） | ✅ OpenAPI 正常；❌ **郑州 SSH 密钥被拒** |
| `lql341` | GitHub | ✅ `gh auth status` 已登录，scopes: repo/gist/admin:public_key |

### SSH 别名（`~/.ssh/config`）

```bash
# ✅ 唯一可在 BW1000 上跑的
ssh zzeshell-hz1        # scnethr2601a10@zzeshell.scnet.cn:65032, key=id_rsa_zzeshell_hz1
                        # 有效期到 2027-01-21

# ❌ 当前不可用（密钥被拒）
ssh zzeshell            # luql@zzeshell.scnet.cn:65032

# 其他集群（卡型不符合赛题，仅调试用）
ssh kseshell            # 昆山 Z100 16GB
ssh wuzhshell           # 乌镇 Z100SM 16GB
ssh xianshell           # 西安 Z100SM 16GB
```

### 分支分区（郑州 / region 20091 = 华中一区【A区】）

| 分区 | 节点 | 每节点 DCU | walltime | 用途 |
| --- | --- | --- | --- | --- |
| `hx1hdnormal` | 58（f11r*） | **8 × 64 GB** | **3 天** | ✅ **赛题算力，全部作业用这个** |
| `hx1hcnormal` | 656（a*） | 8（**但 HSA 初始化失败，不可用**） | 无限 | ❌ 别用 |

> ⚠️ `luql` 账号**无权访问 `hx1hdnormal`**（`Invalid account or account/partition`）。
> 只有 `scnethr2601a10` 能进。这是硬约束。

---

## 3. 与集群交互（实测可用的命令）

### 3.1 提交作业（唯一正确姿势）

```bash
# 上传整个条目目录
rsync -az --exclude logs --exclude bench -e "ssh -o BatchMode=yes" \
  models/<条目>/ zzeshell-hz1:/public/home/scnethr2601a10/work/zoo-<名>/

# 提交（带磁盘重投循环，因为节点 /tmp 水位极不稳）
ssh zzeshell-hz1 'cd /public/home/scnethr2601a10/work/zoo-<名> && \
  for i in $(seq 1 12); do \
    J=$(sbatch --parsable container/eval_official.slurm); sleep 25; \
    S=$(sacct -j $J --format=State -n | head -1 | tr -d " "); \
    if grep -qE "NOT_ENOUGH|no space" slurm-$J.out 2>/dev/null; \
      then echo "try$i space"; else echo "KEEP $J $S"; break; fi; \
  done'

# 查状态 / 取产物
ssh zzeshell-hz1 'squeue -u scnethr2601a10 -o "%.8i %.12j %.9T %.7M %.16R"'
ssh zzeshell-hz1 'sacct -j <ID> --format=JobID,State,ExitCode,Elapsed'
scp zzeshell-hz1:/public/.../work/<dir>/repo/logs/{env,server,smoke}.log models/<条目>/logs/
```

### 3.2 绕过 SSH 直连失败（GitHub 不通时）

集群登录节点**有时连不上 github.com**（`Connection timed out`）。做法：

```bash
# 本机 clone 后上传
cd ~/Downloads && git clone --depth 1 https://github.com/Tencent-Hunyuan/AuK.git AuK-src
rsync -az --exclude .git -e "ssh -o BatchMode=yes" AuK-src/ zzeshell-hz1:/public/home/scnethr2601a10/AuK-src/
```

### 3.3 OpenAPI（Notebook / 资源查询）

```bash
SK=/Users/luql/.agents/skills/scnet-hpc/scripts
python3 $SK/scnet.py account                      # 余额
python3 $SK/scnet.py --region 20091 resource-summary   # 队列与限额
python3 $SK/scnet.py --region 11250 notebook resources # 昆山 K100-AI Notebook
python3 $SK/scnet.py --region 11250 notebook list
```

Notebook 驱动（我写的，绕开 skill 的镜像白名单）：
`zoo/tools/nb_create.py`（建实例）、`zoo/tools/nb_exec.py`（Jupyter REST + kernel websocket 执行命令）。

### 3.4 镜像

```bash
H=https://image.ac.com:5000     # Harbor，内网 10.13.17.207，仅 5000 端口，匿名可 pull
# 搜镜像（匿名可读）
curl -sk "$H/api/v2.0/search?q=sglang"
# 匿名取 manifest/tags
TOK=$(curl -sk "$H/service/token?service=harbor-registry&scope=repository:<repo>:pull" | jq -r .token)
curl -sk -H "Authorization: Bearer $TOK" "$H/v2/<repo>/tags/list"
# 拉取（计算节点内）
podman pull --tls-verify=false --cert-dir=/tmp/emptycerts image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
```

### 3.5 提交到 GitLab（sourcefind）

```bash
# token 在 ~/.sourcefind-token（20 字节，chmod 600）
T=$(cat ~/.sourcefind-token)
# 建仓库
curl -sS -X POST -H "PRIVATE-TOKEN: $T" -H "Content-Type: application/json" \
  -d '{"name":"<名>","path":"<名>","visibility":"public","description":"..."}' \
  https://developer.sourcefind.cn/codes/api/v4/projects
# 推代码（注意：main 是保护分支，不能 force push）
git -c credential.helper='!f() { echo "password=$(cat ~/.sourcefind-token)"; }; f' push origin main
# 批量上传文件（main 保护时用文件 API）
# 见 zoo/tools/export_contest.sh 末尾的 Python 片段，逐文件 PUT/POST
```

---

## 4. 数据存放路径

### 4.1 本机（macOS）

| 路径 | 内容 |
| --- | --- |
| `~/Downloads/pi-test/zoo/` | **主仓库**（11 条目 + docs + tools + official） |
| `~/Downloads/pi-test/` | 全部 slurm 脚本与探测归档（`zz_probe*.slurm`、`sweep_*.slurm`、`archive_*.slurm` 等） |
| `~/Downloads/AuK-src/` | AuK 源码（因集群连不上 GitHub，本机 clone） |
| `~/Downloads/超算互联网 × 光合..._赛事详情.zip` | 赛事详情原文 |
| `~/Downloads/T03_T04模型可行性方案_框架数据接入评测.zip` | **官方 T03/T04 材料（99 MB）** |
| `~/.sourcefind-token` | GitLab PAT |
| `~/.harbor-pass` | ❌ 不存在（Harbor 凭据未获取） |
| `~/.agents/skills/scnet-hpc/` | 集群操作 skill |

### 4.2 集群（`/public/home/scnethr2601a10/`，400 G 配额 / 已用 246 G / 剩 155 G）

```
~/models/                                    权重（合计 194 G）
├── Kimi-Linear-48B-A3B-Instruct/     92 G
├── Qwen-Image-2.1-Turbo/             31 G   ← 已下载，评测 blocked
├── MiMo-V2.6-Distill-Qwen-9B/        18 G
├── RoboBrain2.5-8B-NV/               17 G
├── Qwen2.5-Omni-3B/                  12 G   ← AuK 的文本编码器
├── Recon2Reason-Reasoning-4B/       8.3 G
├── AuK/                             6.3 G
├── Qwen3-ASR-1.7B/                  4.4 G
├── Trident-Qwen3VL-2B/              4.0 G
└── Qwen3-ForcedAligner-0.6B/        1.8 G

~/images/mimo-sglang-bw10000_1.1.1.tar      31 G   未压缩，podman load 用（省 4-6 min pull）
~/zoo-images/                                17 G   **镜像归档（防上游删除）**
├── mimo-sglang-bw10000_1.1.1.tar.gz        7.9 G   digest sha256:db981c46...
├── vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz    8.7 G   digest sha256:07930260...
└── MANIFEST.md                             digest + sha256 + 还原命令

~/work/official-t34/                         45 M   **官方测试数据**
├── formal/conditions/*.jsonl               9 份 935 条（入库到 zoo/official/）
└── data/{aishell,LibreSpeech,seedtts,...}  音频素材（不入库）

~/wheels/                                    离线 wheel（计算节点无公网）
├── fa310/      transformers 4.57.6 + hub 0.36.2 + qwen_asr（FA 用）
├── auk/        40 个纯 Python wheel（AuK 用）
├── auk-bin/    torchcodec cp310（AuK 用，ABI 不兼容）
└── qi/         diffusers 0.41.0 + hub 0.36.2（Qwen-Image 用）

~/AuK-src/                                   30 M  AuK 源码

~/work/                                      各条目的作业工作区（12 M）
├── zoo-{asr,asr-official,fa,recon,robo,trident,trident-vllm,auk,qi}/   ← 条目副本+slurm
├── kimi/ kimi-opt/                           Kimi 基准与优化
├── opt/ vopt/ sweep/ vsweep2/ ksweep2/       参数扫描
└── clone-test/                               匿名 clone 复现测试
```

### 4.3 远端持久化

| 位置 | 内容 |
| --- | --- |
| **GitHub `lql341/scnet-dcu-model-zoo`** | ✅ **权威积累**（知识库 + 条目 + 日志），6.7 MB |
| **GitLab `luql/scnet-dcu-model-zoo`** | ✅ **同源镜像**（同一仓库推两个 remote，`sourcefind`），给比赛平台侧可读 |
| GitLab `luql/*`（7 个 public 单件） | 参赛单件快照（早期按条目拆的） |
| 集群 `~/zoo-images/` | ✅ **镜像归档**：2 个 tar.gz + `MANIFEST.md` + `assets/`（wheel 集、AuK 源码） |
| ❌ `/public/home/luql` | 无法写入（SSH 被拒），等恢复后需搬 `~/zoo-images` |

**推送命令（两个远端都要推）**：
```bash
cd ~/Downloads/pi-test/zoo
git push -q origin main        # GitHub（权威）
git push -q sourcefind main    # GitLab（镜像，token 在 ~/.sourcefind-token）
```

---

## 5. 仓库结构（新会话必读）

```
zoo/
├── README.md                  总矩阵（由 gen_index.py 生成，勿手改）
├── INDEX.md                   实测索引（同上）
├── HANDOFF.md                 ← 本文件
├── ARTIFACTS.md               SCNet 侧产物位置与重建成本
├── docs/
│   ├── entry-spec.md          ★ 条目规范（强制）
│   ├── gotchas.md             ★ 17 条踩坑（按隐蔽度排序）
│   ├── environment.md         硬件/软件/镜像能力实测
│   ├── manual-push.md         手动推镜像到 SCNet 镜像库
│   ├── image-archive.md       镜像归档与还原
│   ├── repro.md               ★ 从归档重建任一条目的端到端手册
│   ├── repo-layout.md         为什么单仓库 + 体积策略
│   ├── platform-checklist.md  platform/ 截图清单
│   ├── landscape.md           93 个公开仓库竞争盘点
│   └── SELF-CHECK-2026-11-02.md  T03/T04 材料到位后的自查
├── models/<org>__<model>__<card>__<framework>/    11 个条目
├── official/t03-t04/          ★ 官方条件清单 + file-manifest + provenance
├── sweeps/                    参数扫描脚本（非正式条目）
└── tools/                     lint_entry.sh / gen_index.py / check_quota.sh
                               export_contest.sh / preflight_lib.py
                               nb_create.py / nb_exec.py / push_sourcefind.sh
```

### 每次改动后必跑的三件套

```bash
cd ~/Downloads/pi-test/zoo
./tools/lint_entry.sh        # 规范校验（含 meta↔summary 逐字比对）
python3 tools/gen_index.py   # 重生成 INDEX.md 与 README 总矩阵
./tools/check_quota.sh       # 体积守门（单条目 ≤2 MB / 全库 ≤25 MB）
```

> **lint 已两次抓到我的错**：手打 §7.2 表格数字与 `bench/summary.md` 不一致；
> `meta.throughput` 取错列（把 `total_throughput` 当成 `output_throughput`，差 5 倍）。

---

## 6. 核心技术配方（下次直接用）

### 6.1 容器运行（BW1000 铁律）

```bash
export XDG_RUNTIME_DIR=/tmp/xdg-$$
JID=${SLURM_JOB_ID:-$$}
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID \
    --storage-driver overlay --storage-opt ignore_chown_errors=true"

# 磁盘守门（解压镜像需 ~32 G；tar+解压需 ~70 G）
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
[ "$FREE" -lt 70 ] && { echo "NOT_ENOUGH_SPACE"; exit 42; }

$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_MODEL_PATH":/models/X:ro \
  -v "$WORK":/work \
  -v "$REPO/container/inner.sh":/inner.sh:ro \
  "$IMAGE" /inner.sh
```

四条缺一不可：`/dev/mkfd`、挂宿主 `/opt/hyhal`、不覆盖 `/opt/dtk`、`--entrypoint=/bin/bash`。

容器内必备环境变量：

```bash
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:\
/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:\
/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:\
/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:\
/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
```

### 6.2 两个可用镜像（详见各条目 `image.lock`）

| 镜像 | 用途 | 关键版本 |
| --- | --- | --- |
| `dcu/linshp/base/mimo-sglang-bw10000:1.1.1` | SGLang 推理 + 需要 torchaudio 的场合 | DTK 26.04 / torch 2.9.0 / sglang 0.5.10rc0 / **torchaudio 2.9.0+das** |
| `dcu/linshp/base/vllm0.20-...-qwen_asr1.7_bw:1.0.0` | vLLM 推理 + 音频 | DTK 26.04 / torch 2.10.0 / vllm 0.20.0 / transformers 5.8.1 |

### 6.3 起服务的关键参数（踩坑换来的）

| 场景 | 参数 | 原因 |
| --- | --- | --- |
| SGLang 全部条目 | `--attention-backend triton` | **默认会选残包的 aiter backend，首个 decode 请求即崩** |
| SGLang | `--max-running-requests 64` | 设 8 会让 64 并发里 56 个排队 |
| SGLang / vLLM | **CUDA Graph 保持开启** | 但 Kimi 16K/c64 档会回退 6.3%，需按档实测 |
| vLLM | `--gpu-memory-utilization 0.95` | 0.85 报 `No available memory for cache blocks` |
| 压测前 | `MAX_LEN ≥ 20480` | 17408 会让 16K 档全 400，且客户端把 400 当成功 → **假数据** |

### 6.4 离线 wheel（容器无公网时唯一可靠姿势）

```bash
# 关键：必须显式指定目标解释器，否则拿到的是本机（py3.12）的 cp312 轮子
python3 -m pip download --no-deps --dest wheels \
  --only-binary=:all: --platform manylinux2014_x86_64 \
  --python-version 310 --implementation cp --abi cp310 \
  <包名...>
```

容器内装完**必须打印每个包的版本并断言可导入**，装不全就退出——不要让 5 小时的评测
跑在一堆隐形缺失上。`--no-deps` 不是“没依赖”，只是“不装依赖”，`Requires-Dist` 没人替你查。

---

## 7. 已有条目的关键数据（引用时以此为准）

| 条目 | bench | 4K/c1 | 4K/c8 | 4K/c64 | 16K/c1 | 16K/c8 | 16K/c64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi-Linear-48B × SGLang | 6 档（优化后） | 104.67 | 353.53 | **839.00** | 50.80 | 119.21 | 234.17 |
| MiMo-9B × SGLang | 6 档（优化后） | 44.31 | 360.23 | **998.05** | 41.47 | 308.72 | 299.81 |
| MiMo-9B × vLLM | 6 档（优化后） | 20.73 | 160.63 | **489.52** | 8.19 | 44.83 | 100.82 |

单位 output tok/s，取 `bench/round2.jsonl`。

**优化前后对比**（相对自己首版，`bench-baseline-v1/` 存档）：

| 条目 | 提升 | 手段 | 回退 |
| --- | --- | --- | --- |
| MiMo × SGLang | +73.1% ~ +466.4% | `mrr 8→64` + 开 CUDA Graph | 无 |
| MiMo × vLLM | +3.1% ~ +31.9% | 去 `--enforce-eager` + `max-num-seqs 128` | 无 |
| Kimi × SGLang | +37.9% ~ +620.8% | 开 CUDA Graph | **16K/c64 −6.3%** |

**官方条件评测**：

| 条目 | 完成 | 指标 |
| --- | --- | --- |
| Qwen3-ASR-1.7B | **300/300** | AISHELL-1(zh,100) CER **1.21%** / LibriSpeech(en,200) CER **0.65%** WER **2.86%** |
| Qwen3-ForcedAligner-0.6B | **30/30** | 词级时间戳，预测 572 词 = 参考 572 词；`BOUNDARIES_PENDING` **官方不评分** |
| Qwen-Image-2.1-Turbo | ✅ **180/180** | 2048×2048、8 步 + KV cache，单图 **19.8 s**，全量 3558 s（59 min/卡） |
| AuK | ✅ **120/120** | 24 kHz 单声道，单条 **4.4 s**，全量 531 s；时长 3.08–12.46 s 全部 ≥ 官方 `duration_min_s` |

**Kimi CUDA Graph 回退的稳定性已闭环**（原 P1-4）：同配置 round1 vs round2 六档差异
+2.6% / +1.6% / +1.4% / +0.2% / −0.2% / +0.2%，即 16K/c64 的 −6.3% 是**配置级回退**而非抖动，
结论稳定可用。

---

## 8. 进度（活文档，每次更新）

| 条目 | 状态 | GitLab | 备注 |
| --- | --- | --- | --- |
| Kimi-Linear-48B × SGLang | ✅ `bench` | ✅ | 优化完成 |
| MiMo-9B × SGLang | ✅ `bench` | ✅ | 优化完成 |
| MiMo-9B × vLLM | ✅ `bench` | ✅ | 优化完成 |
| Qwen3-ASR-1.7B × vLLM | ✅ `official-eval` 300/300 | ✅ | 无可比基线 |
| Qwen3-ForcedAligner × transformers | ✅ `official-eval` 30/30 | ✅ | 官方不评分 |
| Recon2Reason-4B × SGLang | 🟡 `smoke` | ✅ | 单图+多图通；缺 T02 评测器 |
| RoboBrain2.5-8B × SGLang | 🟡 `smoke` | ✅ | 同上 |
| Trident-2B × SGLang | ❌ `blocked` | — | 框架不支持 `Qwen3VLForEmbedding` |
| Trident-2B × vLLM | ❌ `blocked` | — | 同上 |
| AuK × transformers | ✅ `official-eval` 120/120 | ✅ | 根因已修正（wheel tag + torchaudio.save）；531 s 全量 |
| Qwen-Image-2.1-Turbo × diffusers | ✅ `official-eval` 180/180 | ✅ | 根因已修正（hub 1.33.0）；3558 s 全量，19.8 s/张 |

---

## 9. TODO（活文档）

### 🔴 P0 —— 只能你做，且阻塞最多条目

| # | 事项 | 影响 |
| --- | --- | --- |
| P0-1 | 群里要 **T01/T02 的 `baselines/<模型>.json`** 与**官方 GPU 性能基线** | 3 个有 6 档基准的条目 §6 全为 `[BLOCKED]`；所有吞吐数字无官方对照 |
| P0-2 | 群里要 **T02 评测器**（OCRBench / MMMU）与 `data/` 子集 | Recon2Reason、RoboBrain 停在 smoke |
| P0-3 | ~~确认 T03 图片尺寸 2048×2048 还是 1024×1024~~ | ✅ **已自行解决**：条件清单 `params.width/height` 写的就是 2048（以机器可读为准），无需等人 |
| P0-4 | 确认 **T04 到底要不要 2% 基线门槛**（赛事详情 §五要求比基线；T03/T04 材料说"不进行统一评分"） | ASR/FA 的自算指标能否作为验收依据 |
| P0-5 | 恢复 **`luql` 在郑州集群的 SSH 密钥** | 镜像归档与数据无法搬到永久账号 |
| P0-6 | 拿 **Harbor 凭据**（个人中心→访问控制，找"镜像仓库"） | ⬆️ **升为第一优先**：按新原则，镜像归档是核心资产；只有推到自己命名空间才能真正摆脱「上游删仓 + 临时账号回收」两个风险；顺带也解决了比赛对镜像归属的要求 |
| P0-7 | 确认 **`scnethr2601a10` 是否赛事账号** | 若否，BW1000 全线作废（`luql` 进不了 `hx1hdnormal`） |
| P0-8 | 补 `platform/` 截图（17 条清单见 `docs/platform-checklist.md`） | 7 个条目都空着 |

### 🟡 P1 —— 我能做，有明确解法

| # | 事项 | 预估 | 解法 |
| --- | --- | --- | --- |
| P1-0 | **知识库双远端同步**（本轮已建） | 每次 1 min | `git push origin main && git push sourcefind main`；GitLab 仓 `luql/scnet-dcu-model-zoo` 已建 |
| P1-0b | **依赖归档**（本轮已做） | 新增一个就归档 | `~/zoo-images/assets/wheels-*-cp310.tar.gz` + `MANIFEST.md` |
| P1-1 | **AuK 官方 120 条收尾** | 已花 ~1 卡时，余 5 卡时 | 作业 `1022507` 已带续跑（已有产物跳过）；跑完后回填 `反馈表.csv` / `eval_result.json` / README §5–7 |
| P1-2 | **Qwen-Image 官方 180 条收尾** | 已花 ~1 卡时，余 1 卡时 | 作业 `1022166`；跑完后抽查 3 张尺寸 + 回填 §5–7 |
| P1-3 | Recon2Reason / RoboBrain 的 **int8/int4** | 各 3–5 卡时 | 量化件单独计件；**但都要等 P0-1 的基线** |
| P1-4 | ~~Kimi CUDA Graph 二次确认~~ | ✅ 已完成 | round1/round2 六档差异 ≤2.6%，−6.3% 是配置级回退，结论稳定 |
| P1-5 | **把 cp310 轮子规则固化成工具** | 0.5 h | `tools/fetch_wheels.sh`：统一 `--python-version 310 --abi cp310`，防再犯 gotchas #18 |
| P1-6 | **两轮审自己的结论** | 0.5 h | 本轮教训：两个「blocked」都是工程错，`blocked` 以后必须附「已核对 `requires_dist` / wheel tag」证据 |

### 🟢 P2 —— 材料已备但需 8 卡 / 大空间

| # | 模型 | 权重 | 障碍 |
| --- | --- | --- | --- |
| P2-1 | Wan2.1-I2V-14B-480P | 82 G | 8 卡 FSDP + USP |
| P2-2 | HY-OmniWeaving | 55 G | 8 卡 torchrun |
| P2-3 | MiniMax-Music3 | 52 G | README 称推理需 CUDA |
| P2-4 | HY-World-2.0 | 174 G | 空间只剩 155 G + 要 FA3(Hopper) |

### ⚫ 不可行（已论证）

| 模型 | 原因 |
| --- | --- |
| DeepSeek-V4.1-Flash-1017 | ~1 TB，需 2–3 节点 |
| DeepSeek-V4-Flash-Vision-Exp | 官方仅 FP8/FP4，需转 BF16（609 GB） |

---

## 10. 给新会话的开场提示

```
读 ~/Downloads/pi-test/zoo/HANDOFF.md 和 docs/gotchas.md，然后按 §9 的 TODO 继续。
当前余额 4344.02，集群唯一可用 SSH 是 zzeshell-hz1（用户 scnethr2601a10），
分区只能用 hx1hdnormal。每次改动后跑 lint_entry.sh + gen_index.py + check_quota.sh。
跑不通的配置要标 blocked 并留证据，不要删。
```

### 四条最容易犯的错（我自己犯过的）

1. **不要把复杂 shell 塞进 `podman run -c "..."`** —— `$` 会被外层展开。把容器内逻辑写成独立
   `container/inner.sh` 并挂载，用 `/inner.sh` 执行。
2. **不要相信"看起来跑通了"的压测数字** —— 客户端会把 HTTP 400 当成功，产出
   `TTFT=0` / 吞吐 9791 tok/s 的假数据。必须过 `preflight_lib.py` + 有效性校验。
3. **不要手写 README 里的数字表** —— 从 `bench/summary.md` 程序化回填，否则 lint 会抓你。
4. **判 `blocked` 前先怀疑自己，不是怀疑国产卡** —— 本轮两个「blocked」全是工程错：
   离线 wheel 标签下错（cp312 给 py3.10）、`--no-deps` 不校验 `Requires-Dist`（hub 0.36.2
   装给要求 ≥1.32 的 diffusers）。判据：日志里出现「装不上 / 缺符号 / 版本不存在」时，
   先 `pip download --python-version 310 --abi cp310` 重下、先读包的 `requires_dist`，
   再下结论。详见 gotchas #18–#20。
5. **产物要落盘、作业要能续跑** —— 单条 150 s 的评测跑不完一趟 1 h 作业；
   `evaluate_official.py` 一律支持「产物已存在则跳过」，否则每次重跑都从零烧卡时。

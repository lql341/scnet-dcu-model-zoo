# HANDOFF —— 会话交接与 living TODO

> ✅ **项目已结项（2026-11-09）**。`luql` 有 SSH 但**没有被授权 GPU 卡**，因此无法再提交任何作业。
> 所有数据、镜像、经验均已归档完毕：代码在 GitHub+GitLab 双远端，二进制在
> `luql:/public/home/luql/archive/zoo-dcu-2026-11/`（31 GB，sha256 全核对）。
> 收尾报告见 `docs/REPORT.md`（含 §9 结项说明），归档细节见 `docs/ARCHIVE-SUMMARY.md`。
>
> ⚠️ **2026-11-07 14:12 CST 起，`hx1hdnormal` 提交被拒**（`Invalid account or account/partition combination`），
> 正在跑的 OmniWeaving 作业被 CANCELLED（22 h，**53/60 产物已在盘上**）。集群范围内 `squeue -a` 为 0 条作业、
> OpenAPI 的 `queues` 也返回空 → 判断为账号/分区权限变更（正是 §9 的 P0-7）。详见 §11「事故记录」。
>
> 最后更新：2026-11-09（结项）　余额 **4275.25**　状态：**15 个条目**——7 个官方评测全绿、1 个 partial（OmniWeaving 53/60）、3 个 bench、2 个 smoke、2 个 blocked
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

在 **BW1000（海光 DCU）** 上跑通 **15 个条目**（模型 × 框架 × 精度），其中
**7 个完成官方条件评测且全绿**（923 条真实产物）、**1 个部分完成（53/60）**、**3 个有完整 6 档基准**。
本轮把原先 2 个 blocked（AuK、Qwen-Image）的根因重定位并修掉，**两个都已跑完官方全量条件**：
- AuK：卡点 = 离线 wheel 标签（cp312 vs py3.10）+ `torchaudio.save` 走 torchcodec → **120/120**，4.4 s/条
- Qwen-Image：卡点 = hub 版本低于 diffusers 元数据要求（0.36.2 < 1.32）→ **180/180**，19.8 s/张
**没有任何精度数据能对齐官方基线 —— 因为 T01/T02 的 `baselines/` 从未下发。**

---

## 2. 账号与访问

| 账号 | 用途 | 状态 |
| --- | --- | --- |
| `scnethr2601a10` | 郑州集群 Slurm 账号（BW1000 唯一可用） | ⚠️ **2026-11-07 起提交被拒**（Invalid account/partition），SSH 仍可登录、文件可读写 |
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
cd ~/Downloads/pi-test/sources && git clone --depth 1 https://github.com/Tencent-Hunyuan/AuK.git AuK-src
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
**约定：本机所有与本项目相关的东西都在 `~/Downloads/pi-test/` 下，不再往 `~/Downloads/` 顶层散放。**

| 路径 | 内容 |
| --- | --- |
| `~/Downloads/pi-test/zoo/` | **主仓库**（13 条目 + docs + tools + official + sweeps），GitHub/GitLab 双远端 |
| `~/Downloads/pi-test/sources/` | 上游源码 clone（`AuK-src`、`Wan2.1-src`、`OmniWeaving-src`；集群连不上 GitHub 时靠这里上传） |
| `~/Downloads/pi-test/slurm-archive/` | 早期 48 个实验 slurm + `RECIPE.md` + 竞品盘点（**副本已入 `zoo/sweeps/2026-10-11-probes/`**，这里是原件） |
| `~/Downloads/pi-test/scripts/` | 本机侧工具（`nb_create.py`、`nb_exec.py`、`push_sourcefind.sh`、`dl_t03.slurm`） |
| `~/Downloads/pi-test/assets/` | 官方原始包（`T03_T04模型可行性方案_框架数据接入评测.zip` 99 MB；`赛事详情.zip` 仍在 Downloads，属离线材料） |
| `~/Downloads/pi-test/{bw-vllm,kimi,mimo}-repo/`、`k100ai-repo-artifacts/`、`out/` | 早期按条目拆的仓库快照，仅作回溯，勿再写入 |
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
| 集群 `~/models/` 权重 | 🔁 用完即删（策略见 `docs/weights-policy.md`），配额仅 400 G |
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
│   ├── gotchas.md             ★ 24 条踩坑（按隐蔽度排序）
│   ├── environment.md         硬件/软件/镜像能力实测
│   ├── manual-push.md         手动推镜像到 SCNet 镜像库
│   ├── image-archive.md       镜像归档与还原
│   ├── repro.md               ★ 从归档重建任一条目的端到端手册
│   ├── repo-layout.md         为什么单仓库 + 体积策略
│   ├── platform-checklist.md  platform/ 截图清单
│   ├── landscape.md           93 个公开仓库竞争盘点
│   └── SELF-CHECK-2026-11-02.md  T03/T04 材料到位后的自查
├── models/<org>__<model>__<card>__<framework>/    15 个条目
├── official/t03-t04/          ★ 官方条件清单 + file-manifest + provenance
├── sweeps/                    参数扫描脚本（非正式条目）
└── tools/                     lint_entry.sh / gen_index.py / check_quota.sh
                               export_contest.sh / preflight_lib.py
                               nb_create.py / nb_exec.py / push_sourcefind.sh
```

### 每次改动后必跑的三件套

```bash
cd ~/Downloads/pi-test/zoo
./tools/lint_entry.sh            # 规范校验（含 meta↔summary 逐字比对）
python3 tools/gen_index.py       # 重生成 INDEX.md 与 README 总矩阵
./tools/check_quota.sh           # 体积守门（单条目 ≤2 MB / 全库 ≤25 MB）
python3 tools/check_products.py  # 产物规格守门（stats.json 必须与条件清单 expected 一致）
```

> `check_products.py` 是 2026-11-05 那次事故的产物：OmniWeaving 跑完 60 条后只验"文件存在且 >200 KB"，
> 实际全是 848×480（要 832×480），26 卡时白烧。现在"规格"是入库的硬门槛。

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
| SGLang / vLLM | **CUDA Graph 保持开启** | 关掉在 Kimi 16K/c8 上会掉 31%；16K 档另加 `--chunked-prefill-size 4096`（+5~8%） |
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
| Kimi × SGLang | +37.9% ~ +620.8% | 开 CUDA Graph | 无（原记的 16K/c64 −6.3% 已复测为噪声） |

**官方条件评测**：

| 条目 | 完成 | 指标 |
| --- | --- | --- |
| Qwen3-ASR-1.7B | **300/300** | AISHELL-1(zh,100) CER **1.21%** / LibriSpeech(en,200) CER **0.65%** WER **2.86%** |
| Qwen3-ForcedAligner-0.6B | **30/30** | 词级时间戳，预测 572 词 = 参考 572 词；`BOUNDARIES_PENDING` **官方不评分** |
| AuK-Flash | ✅ **120/120** | nfe=4 蒸馏版，单条 **1.0 s**（AuK 4.4 s，快 4.4×），121 s 全量 |
| MiniMax-Music3 | ✅ **60/60** | 32 kHz 立体声，单条 203 s（RTF 6.8），12179 s 全量；**单卡**（上游称需 2 卡） |
| Wan2.1-I2V-14B-480P | 🟡 60 条在跑 | 4 卡 FSDP，单条 ~18.5 min，已完成 ~52/60 |
| Qwen-Image-2.1-Turbo | ✅ **180/180** | 2048×2048、8 步 + KV cache，单图 **19.8 s**，全量 3558 s（59 min/卡） |
| AuK | ✅ **120/120** | 24 kHz 单声道，单条 **4.4 s**，全量 531 s；时长 3.08–12.46 s 全部 ≥ 官方 `duration_min_s` |

**Kimi CUDA Graph 回退的稳定性已闭环**（原 P1-4）：同配置 round1 vs round2 六档差异
+2.6% / +1.6% / +1.4% / +0.2% / −0.2% / +0.2% —— 但那两轮之间恰好没覆盖到 16K/c64 的取值区间；
2026-11-05 用 5 变体复测（§7.4）才确认该档单轮波动 ±7%，**原记的 −6.3% 是噪声，不是配置回退**。

### 7.0 今天的两个「不可行」也是有价值的结论

| 实验 | 结论 | 证据 |
| --- | --- | --- |
| Qwen-Image 合批（同 prompt 3 seed） | ❌ **显存不可行**：batch=2 即 OOM（已分配 46.56 GB / 剩 3.84 / 需 6.75） | `models/Qwen__*diffusers/sweep/results-batching-2026-11-04.md` |
| ASR fp8 量化 | ❌ `--quantization fp8` engine core 初始化即失败（无 DCU kernel） | `docs/gotchas.md` #22；bf16 子集基线 CER 0.40%/0.10% |

**意义**：这两条把「不是没试」变成「试过、有证据、知道边界在哪」，避免下个会话再烧一遍卡时。

### 7.1 vLLM 16K 档：配置层面无解（2026-11-03，三轮扫描）

同模型同卡，vLLM 16K 比 SGLang 慢 5–7 倍，**五条旋钮全试过、全无效**：

| 旋钮 | 结果 |
| --- | --- |
| `--max-num-batched-tokens` 2048 / 8192 | 无变化 |
| `VLLM_ROCM_USE_AITER=1` | 无变化（内核仍不可用） |
| `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | 无变化 |
| `--kv-cache-dtype fp8` | **负优化**（16K/c8 44.9→34.4） |
| `--tp 2` | 启动失败（podman 不继承 `HIP_VISIBLE_DEVICES`，已记 gotchas #21） |

根因：`Cannot use ROCm custom paged attention kernel → Triton 兜底`。
完整数据 `models/XiaomiMiMo__*__vllm/sweep/results-2026-11-03.md`。
**这条结论的价值**：要补上只能自建镜像（重编 aiter / 试 v0 引擎 / 或改用 SGLang），
正好指向「镜像归档 → 自有镜像」的长期方向。

### 7.2 T03/T04 官方 8 个模型的覆盖

| 模型 | 状态 | 备注 |
| --- | --- | --- |
| Qwen-Image-2.1-Turbo | ✅ 180/180 | 19.8 s/张 |
| AuK | ✅ 120/120 | 4.4 s/条 |
| Qwen3-ASR-1.7B | ✅ 300/300 | CER 1.21% / 0.65% |
| Qwen3-ForcedAligner | ✅ 30/30 | 官方不评分 |
| **Wan2.1-I2V-14B-480P** | 🟡 条目已就绪，卡在显存 | 单卡 64 GB OOM（`--offload_model` + `--t5_cpu` 都不够）→ 转 8 卡 FSDP，作业 1026191 排队中 |
| **HY-OmniWeaving** | 🟡 权重 52 GB 已下 | 官方代码 `Tencent-Hunyuan/OmniWeaving`；依赖 pins 与镜像冲突（torch>=2.6 / transformers 4.57.1 / diffusers 0.32.2） |
| MiniMax-Music3 | ⬜ 源未找到 | ModelScope 404；官方 README 标 CUDA |
| HY-World-2.0 | ⬜ 基本不可行 | 174.7 GB > 配额，且需 FA3(Hopper) |

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
| Wan2.1-I2V-14B-480P × 官方代码 | 🟡 `official-eval` 进行中 | — | 4 卡 FSDP 跑通（单条 19.4 min），全量 60 条在跑 |
| AuK-Flash × transformers | ✅ `official-eval` 120/120 | — | 4 步蒸馏版，单条 1.0 s |
| MiniMax-Music3 × sglang | ✅ `official-eval` 60/60 | — | 32 kHz 立体声；镜像 + harness 都已归档 |
| HY-OmniWeaving × 官方代码 | 🟡 探针中 | — | 需补 llm/byt5/Glyph-SDXL-v2/siglip 四份依赖（正在下）；byT5 缺失已定位 |

---

## 8.5 结项状态（2026-11-09）

| 项 | 数值 |
| --- | --- |
| 条目总数 | **15**（7 个 official-eval 全绿 / 1 个 partial 53-60 / 3 个 bench / 3 个 smoke / 2 个 blocked） |
| 官方条件真实产物 | **923 条**（官方清单 935 条 − hy_world 5 − omni 7） |
| 归二进制归档 | 31 GB @ `luql:archive/zoo-dcu-2026-11/` |
| 踩坑 | 24 条（`docs/gotchas.md`） |
| 硬约束 | 5 条（`docs/REPORT.md` §4） |
| Git 提交 | 64+（GitHub 与 GitLab 完全同步） |

**不能再做的**：提交作业（两个账号都没卡权限）；T01/T02 精度（官方 baselines 从未下发）。
**随时可做的**：拿到任何有 `hx1hdnormal` 权限的账号后，按 `docs/repro.md` 用归档镜像重跑，
OmniWeaving 只需 `sbatch --export=ALL,PROBE_ONLY=0 container/submit.slurm` 补 7 条（3.5 h）。

## 9. 机时恢复后怎么重启（runbook）

> 触发条件：平台把某个账号加入 `hx1hdnormal` 的 `AllowAccounts` **且**绑定
> `QoS=partition_hx1hdnormal`。先做 §9.0 的判定，通过后再往下走。

### 9.0 先判定"真的能提交了"（3 分钟）

```bash
ssh <账号>@zzeshell.scnet.cn -p 65032        # 或 zzeshell-hz1 / luql 的别名
# 1) 是否在分区白名单（有输出才算通过）
scontrol -a show partition hx1hdnormal | tr ' ' '\n' | grep '^AllowAccounts=' \
  | cut -d= -f2 | tr ',' '\n' | grep -x "$USER"
# 2) 最小作业能否起来
printf '#!/bin/bash\n#SBATCH -p hx1hdnormal\n#SBATCH -N 1\n#SBATCH --gres=dcu:1\n#SBATCH -t 00:05:00\nhostname\n' > ~/t.slurm
sbatch ~/t.slurm && sleep 20 && squeue -u $USER
```

失败表现与含义见 `docs/gotchas.md` #24。

### 9.1 恢复数据（不需要重下权重）

```bash
# 归档唯一位置
B=/public/home/luql/archive/zoo-dcu-2026-11

# 1) 镜像（3 个，30 GB，sha256 已核对）
cd $B/images && sha256sum -c <(cat <<'EOF'
f16125ba6dcb3020aaef33b89d390508f99aad980d7436286f043ed68596d77a  mimo-sglang-bw10000_1.1.1.tar.gz
986f95a3e5599f024797e41faa614f72cbd5a6c87e5e97f71c172f730269d45f  vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz
b61b05bd7df45fdfe81885cb898f88572860a37734ececc23018dbfd531d7d87  minimax-music3-bw-sglang_20261031-rc1.tar.gz
EOF
)

# 2) 依赖与源码（166 MB）
mkdir -p ~/wheels && tar xzf $B/assets/wheels-auk-cp310.tar.gz -C ~/wheels   # 其余同理

# 3) 代码
git clone https://github.com/lql341/scnet-dcu-model-zoo.git ~/zoo   # 或 GitLab

# 4) 权重（ModelScope，按需下载；清单见 docs/weights-policy.md）
```

### 9.2 补完唯一未完成的条目（OmniWeaving 剩 7 条，约 3.5 h）

```bash
cd ~/zoo/models/Tencent-Hunyuan__HY-OmniWeaving__bw1000__hyvideo
rsync -az --exclude logs --exclude official-eval -e "ssh -o BatchMode=yes" \
  ./ <账号>@zzeshell.scnet.cn:/public/home/<账号>/work/zoo-omni/
# 先确认产物已就位（否则会从零重跑）
rsync -az $B/products/omniweaving/ <账号>@...:/public/home/<账号>/work/omnirun/repo/official-eval/
ssh <账号>@... 'cd ~/work/zoo-omni && sbatch --export=ALL,PROBE_ONLY=0 container/submit.slurm'
```

`evaluate_official.py` 会跳过已有的 53 个产物，只补 7 条。

### 9.3 恢复后值得做的（按性价比）

| # | 事项 | 成本 | 为什么 |
| --- | --- | --- | --- |
| 1 | **把 3 个镜像推到自有 Harbor 命名空间** | 0.5 卡时 | 只有这一步能真正摆脱"上游删仓 + 临时账号回收"两个风险（需 Harbor 凭据） |
| 2 | 量化条目（AWQ 已验证可跑） | 4–8 卡时 | 全库唯一空白；AWQ 慢 5× 但显存收益真实，适合"装不下 bf16"的场景 |
| 3 | WorldMirror（需编 ROCm 版 gsplat） | 8+ 卡时，可能失败 | T03 第 5 个模型 |
| 4 | T01/T02 精度 | 需官方 baselines | 材料从未下发，无法自证 |
| 5 | `platform/` 截图 17 项 | 无卡时 | 只有人能在平台上点 |

### 9.4 明确不必再做的（已有结论，别再烧卡时）

- vLLM 16K 调参：chunked 切片 / AITER / backend 覆盖 / fp8 KV / TP=2 / **v0 引擎** —— 全部无效（gotchas #21）
- Qwen-Image 合批：2048² 显存无余量，batch=2 即 OOM（`sweep/results-batching-*.md`）
- Kimi 16K/c64 "回退"：是噪声，不是配置问题（§8 的 §7.4）
- fp8 在线量化：DCU 无 kernel（gotchas #22）

---

## 10. 给新会话的开场提示

```
读 ~/Downloads/pi-test/zoo/HANDOFF.md、docs/REPORT.md 和 docs/gotchas.md。
**项目已结项**（2026-11-09）：两个账号都没有 hx1hdnormal 的提交权限（详见 §11）。
若机时恢复，按 §9 的 runbook 走（§9.0 先判定、§9.2 补 OmniWeaving 剩 7 条）。
余额约 4275；数据全在 luql:/public/home/luql/archive/zoo-dcu-2026-11/（31 GB）。
每次改动后跑四件套：lint_entry.sh + gen_index.py + check_quota.sh + check_products.py。
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


---

## 11. 事故记录

### 2026-11-07 14:12 CST：BW1000 提交权限丢失（P0-7 的现实版）

| 时间 | 事件 |
| --- | --- |
| 14:12 | OmniWeaving 全量作业 `1048411` 在跑 22:00 后被 `CANCELLED`（产物 53/60 已落盘） |
| 之后 | `sbatch` 一律报 `Invalid account or account/partition combination specified`（最小作业也一样） |
| 诊断 | ① `squeue -a` 全集群 0 条作业；② `sinfo`（默认视图）空，`scontrol -a show partition hx1hdnormal` 仍可见（QoS=partition_hx1hdnormal）；③ 我们的 assoc QOS 是 `user_scnethr2601a10`，`--qos=partition_hx1hdnormal` 报 `Invalid qos specification`；④ OpenAPI `resource-summary` 的 `queues` 返回 `[]` |
| 判断 | **分区/账号的 QOS 绑定被改**（或临时账号到期）。SSH 与 `/public` 读写仍正常，只是不能提交作业 |
| 未受影响 | 53 条 OmniWeaving 产物、3 个镜像归档、全部仓库数据（GitHub/GitLab 已同步） |
| 恢复后要做的第一件事 | `cd ~/work/zoo-omni && sbatch --export=ALL container/submit.slurm`（`evaluate_official.py` 会跳过已存在产物，只补剩下 7 条，约 3.5 h），然后回填 README/meta |

### 2026-11-07 冒烟测试结论（六项，全部实测）

| # | 检查 | 结果 |
| --- | --- | --- |
| 1 | SSH 登录 | ✅ 正常（`scnethr2601a10@zz-login02`） |
| 2 | `/public` 读写 | ✅ 正常（写入/删除测试通过） |
| 3 | `sbatch` 提交 | ❌ `Invalid account or account/partition combination specified`（最小作业也一样） |
| 4 | 16 个分区逐个查 `AllowAccounts` | ❌ **没有任何一个分区包含 `scnethr2601a10`** |
| 5 | OpenAPI 提交（`scnet.py submit`） | ❌ 同一个 Slurm 错误（说明不是提交通道的问题） |
| 6 | 数据保全 | ✅ OmniWeaving 53/60 产物、3 个镜像归档 30 G、仓库双远端全部完好 |

**决定性证据**：`hx1hdnormal` 的 `AllowAccounts` 有 **10552** 条，包含 `scnethr2601a05`、`scnethr2601a07`、
`scnethr202601`，**独独没有 `scnethr2601a10`** → 我们的账号是被**从分区的允许列表里移除**了（权限回收/试用到期），
不是集群故障、不是脚本问题。

**问平台时要问的三句**：① `scnethr2601a10` 是否仍是有效赛事账号；② 它是否仍被允许提交到 `hx1hdnormal`；
③ 若已停用，`luql` 账号能否获得该分区权限（同时解决 §9 的 P0-5「luql 郑州 SSH 被拒」）。

### 2026-11-09 归档完成（详见 `docs/ARCHIVE-SUMMARY.md`）

`luql` 郑州 SSH 已恢复 → 改为**服务端直拷**，一次性搬完，本机中转副本已删除。

**唯一二进制归档：`luql:/public/home/luql/archive/zoo-dcu-2026-11/`（31 GB）**

| 内容 | 数量 | 校验 |
| --- | --- | --- |
| `images/` 3 个镜像 tar.gz | 30 GB | sha256 与仓库登记值**逐条一致** |
| `products/` 各模型产物 | asr 626 · auk-flash 120 · fa 93 · music3 60 · omniweaving 107 · wan21 60 | 与 `meta.json` 的 ok 数对应 |
| `products/optimizations/` 调优原始 jsonl | **34 个**（mimo-16k 8+8、kimi-16k 10、awq 3、asr-tput 1、vllm-v0 4） | 优化结论的原始证据 |
| `assets/` 轮子/源码包/harness | 7 个 166 MB | 包内 sha256 清单 |
| `sources/` AuK/Wan2.1/OmniWeaving | 3 份 | — |
| `VERIFY.md` | 校验记录与还原命令 | — |

已清理：顶层孤儿目录 `zoo-archive-2026-11`、OpenAPI 分块残留（`*.efilePart`）、本机 30 GB 中转副本。
`scnethr2601a10` 的 `~` 权限已收回 `700`。

**恢复后的第一件事**（产物续跑，只补剩下 7 条，约 3.5 h）：
```bash
ssh zzeshell-hz1 'cd ~/work/zoo-omni && sbatch --export=ALL,PROBE_ONLY=0 container/submit.slurm'
```

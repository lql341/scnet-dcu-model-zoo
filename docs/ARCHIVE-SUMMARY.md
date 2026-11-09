# 归档总结（2026-11-09）

> 触发：`scnethr2601a10`（郑州 BW1000 临时账号）**2026-11-07 14:12 被移出分区 AllowAccounts**，
> 提交作业全面被拒（详见 `HANDOFF.md` §11 冒烟测试）。本文件记录"东西都搬去哪了、怎么还原、还缺什么"。

---

## 1. 三层保全（同一份数据，三个位置）

| 层 | 位置 | 内容 | 状态 |
| --- | --- | --- | --- |
| ① 代码与经验（权威） | **GitHub `lql341/scnet-dcu-model-zoo`** + **GitLab `luql/scnet-dcu-model-zoo`** | 15 个条目、docs/gotchas（23 条坑）、REPORT.md、sweeps（调优过程）、official（官方条件清单） | ✅ 双远端同步 |
| ② 产物与归档（二进制） | **`luql:/public/home/luql/archive/zoo-dcu-2026-11/`（唯一副本，31 GB）** | 产物 + 3 个镜像 + wheel 集 + 源码 | ✅ 已完成（服务端直拷 + sha256 校验） |
| ③ 原集群（即将失效） | `scnethr2601a10:~/work`、`~/zoo-images`、`~/models` | 同上 + 权重（可重下） | ⚠️ 账号被回收，随时可能清空 |

---

## 1.5 最终目录（2026-11-09 实际状态）

`luql` 的郑州 SSH 恢复后，改为**服务端直拷**（比「拉到本机再上传」快一个数量级，也不在本机留副本）：

```text
/public/home/luql/archive/zoo-dcu-2026-11/        31 GB
├── images/            3 个镜像 tar.gz（sha256 与仓库登记值逐一核对通过）
├── products/          按模型命名的全部产物
│   ├── asr/ 626 files · auk-flash/ 120 · fa/ 93 · music3/ 60 · omniweaving/ 107 · wan21/ 60
│   └── optimizations/ 调优实验原始 jsonl（34 个）
│       mimo-16k-tuning(8) · mimo-16k-confirm(8) · kimi-16k-tuning(10)
│       awq-qwen3-8b(3) · asr-throughput(1) · vllm-v0-compare(4)
├── assets/            4 套 cp310 轮子 + 源码包 + Music3 harness（166 MB）
├── sources/           AuK / Wan2.1 / OmniWeaving 源码
└── VERIFY.md          校验记录（镜像 sha256、各目录条数与体积、还原说明）
```

**做法**：`scnethr2601a10` 侧 `chmod o+rx` 开放读取 → `luql` 侧 `rsync -a` 直拷 → **立即 chmod 700 收回**。
（注意：chmod 会被后续操作覆盖，拷贝窗口内不要并行做别的 chmod；本次踩过一次，重开即可。）

```bash
# 服务端直拷（当时用的命令，可复用）
scnethr$ chmod o+rx ~; chmod -R o+rX ~/zoo-images ~/work/<产物目录>
luql$    rsync -a /public/home/scnethr2601a10/zoo-images/*.tar.gz ~/archive/zoo-dcu-2026-11/images/
luql$    rsync -a /public/home/scnethr2601a10/work/omnirun/repo/official-eval/ .../products/omniweaving/
scnethr$ chmod 700 ~
```

## 1.6 历史做法（已废弃）

**二进制归档只放一处**：`luql:/public/home/luql/archive/zoo-dcu-2026-11/`（与 luql 既有的
`archive/` 归档放一起）。本机 `~/Downloads/pi-test/archive/` 只是**上传中转**，校验通过后即删除。

```text
/public/home/luql/archive/zoo-dcu-2026-11/
├── zoo-products-2026-11.tar.gz      628 MB  990 条官方条件产物 + 统计 + MANIFEST
├── zoo-assets-2026-11.tar.gz        177 MB  轮子/源码/harness
├── omniweaving-products.tar.gz       17 MB  OmniWeaving 产物（单抽）
└── images/                          30.3 GB 3 个镜像，500 MB 分块（*.pNNN），拼回后校验 sha256
```

> 早期误传到 `/public/home/luql/zoo-archive-2026-11/`（顶层目录）的 3 个包 + 14 个镜像分块，
> 因 OpenAPI 无删除/移动接口，需在平台网页端手动删除，避免留成第二份。

## 2. 产物清单（已上传 luql 郑州目录）

| 包 | 内容 | 大小 | 校验 |
| --- | --- | ---: | --- |
| `zoo-products-2026-11.tar.gz` | 6 个模型 × 官方条件的全部真实产物 + 统计 + MANIFEST | 628 MB | 见 §4 |
| `zoo-assets-2026-11.tar.gz` | 4 套 cp310 轮子、AuK/Wan/OmniWeaving 源码、Music3 harness 归档 | 177 MB | sha256 清单在包内 |
| `omniweaving-products.tar.gz` | 单独抽出的 OmniWeaving 产物（便于快速取用） | 17 MB | — |
| `images/*.partNNN` | 3 个镜像归档的 500 MB 分块（OpenAPI 单文件大上传会 write timeout，故分块） | 30.3 GB / 61 块 | 拼回后比 `MANIFEST.md` 的 sha256 |

具体到模型：

| 模型 | 产物 | 条数 | 规格 |
| --- | --- | ---: | --- |
| Qwen3-ASR-1.7B | `<id>.txt` | 300 | UTF-8 文本；CER 1.21%(zh) / 0.65%(en) |
| Qwen-Image-2.1-Turbo | `<id>.png` | 180 | 2048×2048（产物 1.1 GB 未随包上传，见 §5） |
| AuK / AuK-Flash | `<id>.wav` | 120 + 120 | 24 kHz 单声道 |
| Qwen3-ForcedAligner | 词级时间戳 | 30 | 572 词 = 572 词 |
| MiniMax-Music3 | `<id>.wav` | 60 | 32 kHz 立体声，17.65–30.02 s |
| Wan2.1-I2V-14B | `<id>.mp4` | 60 | 832×480 / 81 帧 / 16 fps |
| HY-OmniWeaving | `<id>.mp4` | **53/60** | 832×480 / 81 帧 / 16 fps（账号被回收时中断，剩 7 条） |

---

## 3. 还原步骤

```bash
# 1) 镜像（在集群上，拿到任意可用账号后）
cd /public/home/luql/zoo-archive-2026-11/images
cat minimax-music3-bw-sglang_20261031-rc1.tar.gz.part* > m3.tar.gz
sha256sum m3.tar.gz      # 应等于 MANIFEST.md 里 b61b05bd…
gunzip -c m3.tar.gz | podman load

# 2) 产物
tar xzf zoo-products-2026-11.tar.gz -C ~/work/products

# 3) 依赖与源码
tar xzf zoo-assets-2026-11.tar.gz -C ~/wheels   # wheels-auk-cp310 / wheels-qi-cp310 / …

# 4) 代码与经验（随时可 clone）
git clone https://github.com/lql341/scnet-dcu-model-zoo.git   # 或 GitLab 镜像
```

---

## 4. 完整性校验

- 镜像 tar：`sha256-all.txt`（SGLang `f16125ba…`、vLLM `986f95a3…`、Music3 `b61b05bd…`）
- 依赖包：`assets-sha256.txt`（4 个 tar.gz 逐个 sha256）
- 产物：每个条目 `official-eval/stats.json` 记录文件数、体积、规格违规数；`tools/check_products.py` 可复核
- 代码：`tools/lint_entry.sh`（规范）+ `check_quota.sh`（体积）+ `check_products.py`（规格）

---

## 5. 明确未归档（有意为之）

| 项 | 原因 | 重建成本 |
| --- | --- | --- |
| 模型权重（约 300 GB 在集群） | ModelScope 可重下（40–50 MB/s） | 10–40 min/个 |
| Qwen-Image 产物 1.1 GB | 单体过大、可由 180 条条件重跑（3558 s） | 1 卡时 |
| 未压缩镜像 tar（31 GB） | 已有 .tar.gz 归档，冗余 | — |
| OmniWeaving 剩余 7 条 | 账号被回收时中断 | 3.5 h（`evaluate_official.py` 支持续跑） |

---

## 6. 这次归档暴露的三个问题（已写进 gotchas / HANDOFF）

1. **临时账号不是存储**：`~/zoo-images` 在 Slurm 临时账号下，账号一被回收，30 GB 归档跟着消失。
   正确做法是第一时间推到自己的命名空间（Harbor）或自己的目录 —— 这正是 P0-6。
2. **OpenAPI 单文件上传有写超时**：7.9 GB 直传失败（`write operation timed out`），
   628 MB 可以；**大于 ~1 GB 必须分块**（本归档用 500 MB 分块 + 幂等重试）。
3. **跨账号同机搬数据没有通道**：`/public/home/luql` 是 700，`scnethr2601a10` 写不进去；
   只能「拉到本机 → 再上传」。这也说明 `luql` 的郑州 SSH（P0-5）恢复后应第一时间做服务端直拷。

---

## 7. 项目时间线（能力沉淀的时间轴）

| 日期 | 里程碑 |
| --- | --- |
| 2026-10-28 ~ 31 | 裸机探测（`/dev/mkfd`、HSA、镜像能力）、3 个 6 档基准条目落地 |
| 2026-11-01 ~ 03 | 镜像归档（2 个）、T03/T04 官方材料接入、ASR/FA 完成官方评测 |
| 2026-11-03 | **AuK / Qwen-Image 两个 blocked 条目根因推翻重定位**（cp312 轮子、hub 版本），双双跑通 |
| 2026-11-04 | AuK-Flash 120/120（4.4× 加速）、Music3 60/60、Wan2.1 60/60、Music3 镜像归档 |
| 2026-11-05 | 16K 调参定论（chunked-prefill 4096 + CUDA Graph，+5~8%）、vLLM v0 无效、分辨率桶事故与 `check_products.py` |
| 2026-11-06 | AWQ 可行性（能跑但慢 5×）、ASR 批吞吐（−10%） |
| 2026-11-07 | 账号被回收；冒烟测试定位到 `AllowAccounts` 移除 |
| 2026-11-09 | 三层归档（本文件） |

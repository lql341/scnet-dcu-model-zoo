# 条目规范（normative）

本文件是**强制性**规范。`tools/lint_entry.sh` 按此校验，不合规的条目 PR 不予接受。

## 1. 目录命名

```
models/<org>__<model>__<card>__<framework>[__<precision>]/
```

| 段 | 取值 | 示例 |
| --- | --- | --- |
| `org` | 模型组织 | `moonshotai`、`Qwen`、`XiaomiMiMo` |
| `model` | 模型名，去掉组织前缀 | `Kimi-Linear-48B-A3B-Instruct` |
| `card` | 小写卡型标识 | `bw1000`、`k100ai` |
| `framework` | 小写，可带版本 | `sglang`、`vllm`、`sglang-omni` |
| `precision` | **仅当非 bf16 时出现** | `int8`、`int4` |

用 `__`（双下划线）分隔，避免模型名里的 `-` `.` 造成歧义。

## 2. 必备文件

| 文件 | 必需 | 说明 |
| --- | --- | --- |
| `README.md` | ✅ | 8 章技术报告，见 §3 |
| `env.sh` | ✅ | **唯一需要按模型修改的参数文件** |
| `image.lock` | ✅ | 镜像 digest 与拉取命令，见 §5 |
| `env_check.sh` | ✅ | 采集环境 → `logs/env.log` |
| `run.sh` | ✅ | 起服务 → `logs/server.log` |
| `smoke.sh` | ✅ | Smoke 测试 → `logs/smoke.log` |
| `bench.sh` | ⚠️ | `bench = yes` 的条目必需；否则放 `bench/NA.md` 说明原因 |
| `bench/NA.md` | ⚠️ | `meta.bench = no-material` 时必需 |
| `verify.py` | ✅ | 精度验证；无官方基线时须主动 `[BLOCKED]` 退出 |
| `preflight_lib.py` | ⚠️ | 用 `bench.sh` 的条目必需（压测前置校验） |
| `container/submit.slurm` | ✅ | 一键复现入口 |
| `requirements-extra.txt` | ✅ | 无额外依赖时保留说明性注释，不要放空文件 |
| `patches/` | ✅ | 无 patch 时留空目录 + `.gitkeep` |
| `MISSING.md` | ✅ | 本条目缺失的官方材料清单 |
| `meta.json` | ✅ | 机器可读元数据，见 §4 |
| `logs/env.log` | ✅ | 明文 |
| `logs/smoke.log` | ✅ | 明文 |
| `logs/server.log` | ✅ | **> 100 KB 时压缩为 `.log.gz`** |
| `bench/summary.md` | ⚠️ | 有基准时必需 |
| `bench/round1.jsonl`、`round2.jsonl` | ⚠️ | 有基准时必需，**保持明文** |

## 3. README 的 8 章契约

章节标题必须严格为下列形式（`lint` 按前缀匹配）：

| 章 | 标题 | 强制内容 |
| --- | --- | --- |
| 1 | `## 1. 简介` | 模型规格 + **与官方部署方式的差异**（无差异就写"无"） |
| 2 | `## 2. 验证环境` | 版本表（取自 `logs/env.log`）+ 卡型/卡数/路径/端口 |
| 3 | `## 3. 适配优化方案` | 3.1 基础镜像 / 3.2 软件包升级 / 3.3 代码改动或参数 / 3.4 升级步骤 |
| 4 | `## 4. 服务启动脚本` | `run.sh` 展开后的完整命令 + 参数说明表 |
| 5 | `## 5. Smoke 测试` | 实际输出摘录，**必须是真实返回** |
| 6 | `## 6. 精度测评` | 结果；阻塞时写 `[BLOCKED]` 与原因，**不得编造分数** |
| 7 | `## 7. 性能参考` | 6 档表（原文照抄 `bench/summary.md`）+ 优化前后对比（若做过扫描） |
| 8 | `## 8. 注意事项` | 踩坑、已知限制、**认知修正**（曾经判断错又改正的，必须留痕） |

## 4. `meta.json` 字段

```json
{
  "model_id": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
  "org": "moonshotai",
  "model": "Kimi-Linear-48B-A3B-Instruct",
  "card": "BW1000",
  "cards_used": 4,
  "tp": 4,
  "framework": "sglang",
  "framework_version": "0.5.10rc0+das.opt2.alpha.dtk2604",
  "precision": "bf16",
  "status": "bench",              // smoke | bench | blocked | partial
  "bench": "yes",                  // yes | no-material（测法未下发）| not-required（官方不要求）
  "bench_note": "",                // bench=no-material 时写明原因
  "cold_start_s": [280, 360],
  "throughput": {                  // bench=yes 时必填，单位 output tok/s
    "4k_c1": 14.52, "4k_c8": null, "4k_c64": 608.23,
    "16k_c1": 14.01, "16k_c8": 80.73, "16k_c64": 249.89
  },
  "optimized": true,               // 是否做过参数扫描并留了 bench-baseline-v1/
  "verified_on": "2026-10-31"
}
```

`tools/gen_index.py` **从 `meta.json` 生成 `INDEX.md` 与 `README.md` 的总矩阵**，不手写。
改数字只改 `meta.json`，避免文档漂移。

`tools/lint_entry.sh` 会校验 `meta.throughput` 与 `bench/summary.md` 的 `output_throughput`
**逐字一致**——这是防漂移的关键一环（该机制在首次生成时即发现过一次取列错误：
误取了 `total_throughput` 导致 Kimi 的 4K/c64 显示成 3041.13，正确值是 608.23）。

## 5. `image.lock` 要求

- 记录 `repository` / `tag` / **`digest`**
- 说明是否匿名可 pull、是否需要 `--tls-verify=false`
- **本项目不自建镜像**；若某条目必须自建，在该条目放 `Dockerfile` 并在 README §3.1 说明
- `lint` 会校验 `image.lock` 里的镜像路径与 `env.sh` 的 `IMAGE` 一致

## 6. 数据可信规则（红线）

1. **不得编造任何数字**。精度分数必须有官方 `baselines/`；没有就写 `[BLOCKED]`
2. **作废数据必须留痕**，放在 `bench/<vN>-discarded/` 并写明作废判据（例：`bench/failed-ctx17408/`）
3. **压测结果必须过有效性校验**（`TTFT != 0` 且吞吐不超物理上限），否则不得写入 README §7
4. **性能数字必须与 `bench/summary.md` 逐字一致**，`lint` 会比对
5. 同一卡型/框架/负载下的"优化前"数据存 `bench-baseline-v1/`，供复核

## 6.5 `bench` 三种取值的判据

| 取值 | 何时用 | 依据 |
| --- | --- | --- |
| `yes` | 已完成 6 档 ×2 基准 | T01/T02 文本类，官方要求 `bench.sh` |
| `no-material` | 测法或评测集尚未下发，**暂时做不了** | 等待官方 |
| `not-required` | 官方明确**不做**统一评分与性能比较 | T03/T04：「不进行统一评分和性能比较」 |

**不要把"做不了"和"不需要做"混为一谈**：前者是阻塞，后者是已解释清楚的不适用。
T04 语音类条目应从 `no-material` 改成 `not-required`。

## 7. 体积

| 规则 | 阈值 | 校验 |
| --- | --- | --- |
| 单条目 | ≤ 2 MB | `tools/check_quota.sh` |
| 全库 | ≤ 25 MB | 同上 |
| 单文件（除 `bench/round*.jsonl`） | ≤ 1 MB | 同上 |
| `server.log` | > 100 KB 压缩 | 同上提示 |
| 禁止入库 | 模型权重 / 镜像 tar / `__pycache__` | `.gitignore` |

## 8. 新增条目的流程

```bash
# 1. 复制最接近的现有条目作为模板
cp -r models/<existing>/ models/<new>/

# 2. 只改这几处
vi models/<new>/env.sh          # 模型路径 / TP / MAX_LEN / EXTRA_ARGS
vi models/<new>/image.lock      # 镜像 digest
vi models/<new>/meta.json       # 元数据
vi models/<new>/README.md       # 技术报告

# 3. 在集群上跑通
rsync -az models/<new>/ <cluster>:<work>/     # 上传
sbatch container/submit.slurm                 # 复现
scp <cluster>:<work>/repo/logs/* models/<new>/logs/   # 回收日志

# 4. 校验后入库
./tools/lint_entry.sh models/<new>/
python3 tools/gen_index.py        # 重生成 INDEX 与总矩阵
./tools/check_quota.sh
git add -A && git commit && git push
```

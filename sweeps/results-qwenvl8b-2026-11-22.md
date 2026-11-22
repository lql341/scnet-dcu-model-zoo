# Qwen3-VL-8B × vLLM 0.21 首次跑通 + 6 档（2026-11-22）

## 结论

| 项 | 结果 |
| --- | --- |
| 服务 | ✅ `/health` 200，`max_model_len=32768`，KV cache 40.57 GiB / 295,424 tokens |
| 带图 | ✅ **单图答"红色"、双图答"红/蓝"，均正确**（同底座在 vLLM 0.20 上此步 500） |
| 6 档 | ✅ 全部有效，作业 1193165 |
| 精度 | `[BLOCKED]`（T02 评测器未下发，按红线不编造） |
| 冷启动 | 360 s 到可用 |

## 6 档数据（output tok/s，round2）

| 档位 | token/s | TPOT (ms) | TTFT (ms) |
| --- | ---: | ---: | ---: |
| 4K/c1 | 62.54 | 15.95 | 50.6 |
| 4K/c8 | 387.30 | 20.47 | 200.9 |
| 4K/c64 | 688.50 | 75.91 | 8340.7 |
| 16K/c1 | 51.26 | 19.39 | 135.4 |
| 16K/c8 | 230.34 | 34.03 | 662.8 |
| 16K/c64 | 184.06 | 85.23 | 205161.3 |

与同底座 RoboBrain2.5-8B-NV 六档最大差异 4.8%，基本重合（符合"同架构只换后训练"的预期）。

## 本轮绕开的三个环境障碍（都已独立成 gotcha）

| # | 障碍 | 处置 | gotcha |
| --- | --- | --- | --- |
| 1 | 集群无外网，不能 `snapshot_download` | 本机下载（16 MB/s）→ `scnet.py upload`（8–10 MB/s），16.34 GiB 约 30 min | #60 |
| 2 | Harbor 不可达（`no route to host`），vllm0.21 另报 `artifact not found` | 从 `luql:~/work/imgstore/vllm021.tar`（31 GB）`podman load` | #62 |
| 3 | 双镜像共存需 ≥145 GB `/tmp`，各节点 21–141 GB 波动 | 改用同镜像内 `vllm bench serve`，只需一个镜像 | #62 |

## 压测方法变更（重要，别照旧写）

本条目**不用**仓库其它条目的双镜像编排（vLLM 起服务 + SGLang 镜像跑 `sglang.benchmark.serving`）。
改用 vLLM 自带的 `vllm bench serve`，在服务容器里对 `127.0.0.1` 发压：

```bash
vllm bench serve \
  --backend openai-chat --host 127.0.0.1 --port 30000 \
  --endpoint /v1/chat/completions \
  --model qwen3-vl-8b-instruct --tokenizer /models/M \
  --dataset-name random --random-input-len 4096 --random-output-len 1024 \
  --max-concurrency 1 --num-prompts 8 --seed 1 \
  --save-result --result-dir bench/raw --result-filename round1_4096_1024_c1.json
```

三个参数坑（都实测踩过）：

| 现象 | 根因 | 修法 |
| --- | --- | --- |
| `OSError: Repo id must be in the form 'repo_name'...` | 容器漏挂 `-v $HOST_MODEL_PATH:/models/M:ro`，本地路径当成了 HF repo id | 补挂载 |
| `OpenAI Chat Completions API URL must end with ...` | `--backend openai-chat` 默认 endpoint 仍是 `/v1/completions` | 显式 `--endpoint /v1/chat/completions` |
| `--random-input-len is too small` | 新版 `--random-range-ratio 1` 表示"全随机"，下限可为 0 | 去掉该参数 |

产物口径：`vllm bench` 的 json 字段名与 sglang 不同（`total_token_throughput` 而非 `total_throughput`，
且**没有** `mean_e2e_latency_ms`）。`bench_vllm.sh` 里做了映射，
e2e 用 `TTFT + TPOT×(output_len−1)` 还原（无排队时为确定性关系）。

## 冷启动拆解（实测）

| 阶段 | 耗时 |
| --- | --- |
| `podman load` 31 GB tar | ~180 s |
| 权重加载（16.33 GiB，NFS） | 15.3–18.6 s |
| **torch.compile**（range 1–2048） | **123.0–124.3 s** |
| 引擎 init 合计 | 141.5–143.8 s |
| 启动 → `/health` 可用 | **360 s** |

torch.compile 缓存落在容器 `/root/.cache/vllm`，容器销毁即丢，所以每次作业重付这 ~124 s。

## 复现

```bash
# 条目已入库：models/Qwen__Qwen3-VL-8B-Instruct__bw1000__vllm/
tar czf /tmp/e.tgz -C models/Qwen__Qwen3-VL-8B-Instruct__bw1000__vllm .
S=~/Downloads/agent/skills/scnet-hpc/scripts/scnet.py
python3 $S --region 20091 upload --cover /tmp/e.tgz /public/home/luql/work/qwenvl8b/
python3 $S --region 20091 submit --name qwenvl8b --queue hx1hdnormal \
  --nodes 1 --dcus 1 --cpus 16 --memory 32GB --walltime 05:00:00 \
  --work-dir /public/home/luql/work/qwenvl8b \
  --command 'bash /public/home/luql/work/qwenvl8b/qwenvl_entry.sh'
```

权重在 `luql:/public/home/luql/models/Qwen3-VL-8B-Instruct/`（16.34 GiB，17 文件，
逐文件大小已校验一致）。

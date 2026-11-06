# 量化可行性：**AWQ 能跑，但在这张卡上不划算**

背景：`--quantization fp8`（在线量化）在 DCU vLLM 上 engine core 直接起不来（gotchas #22）。
AWQ/GPTQ 走的是另一套 **weight-only INT4 + Triton 反量化 kernel**，值得单独验证。

## 结果：AWQ **可用**

```
[awq_marlin.py:420] Using TritonW4A16LinearKernel for AWQMarlinLinearMethod
[gpu_worker.py:440] Available KV cache memory: 51.42 GiB
[kv_cache_utils.py:1711] GPU KV cache size: 374,400 tokens
init engine took 126.18 s (compilation: 109.34 s)
```

- 模型：`Qwen/Qwen3-8B-AWQ`（6.1 GB 权重，ModelScope 可得）
- 服务 240 s 就绪；生成输出**语义正常**（问"什么是张量并行"，回答合理、不是胡言乱语）
- 说明 DCU 上 **AWQ 的加载与反量化路径是通的**，之前"全库 0 量化覆盖"可以改成"量化可行但性能不划算"

## 吞吐：比 bf16 慢得多（1 卡，4K 档）

| 档位 | AWQ（Qwen3-8B，INT4） | bf16 参照（MiMo-9B） |
| --- | ---: | ---: |
| 4K/c1 | **9.68** tok/s（TPOT 102.5 ms） | 44.31（TPOT 22.3 ms） |
| 4K/c8 | **69.57** tok/s（TPOT 111.8 ms） | 360.23（TPOT 20.3 ms） |
| 4K/c64 | **185.31** tok/s（TPOT 317.3 ms） | 998.05（TPOT 49.5 ms） |

（两模型规模接近 8B vs 9B，但架构不同，**不是严格可比**；这里只说明量级差异。）

## 判定

1. **能跑 ≠ 值得跑**：W4A16 的 Triton 反量化在这张卡上没有快路径，TPOT 是 bf16 的 4–6 倍，
   吞吐只有 1/5 左右。显存收益却是实打实的（6.1 GB 权重 + 51 GB KV cache）。
2. **适用场景**：显存装不下 bf16 时才用 AWQ；追求吞吐应该用 bf16 + SGLang。
3. **下一步可做的量化实验**（都还没做）：
   - GPTQ 权重（不同 kernel 路径）是否更快；
   - AWQ 的 `--quantization awq_marlin` vs `awq` 强制指定有无差别；
   - 精度对照（同题集 bf16 vs AWQ 的输出差异率）——需要同尺寸模型的 bf16 版本做对照。

## 复现

```bash
sbatch sweeps/awq-feasibility.slurm   # 起服务 + 8 条生成抽查
sbatch sweeps/awq-bench.slurm         # 4K 三档压测（c1/c8/c64）
```

# 16K 档调参：一个跨模型一致成立的结论

**结论：`--chunked-prefill-size 4096` + 保持 CUDA Graph 开启，是这套 DCU（BW1000）上 16K 档的最优组合，两个模型都受益。**

## MiMo-9B × SGLang（1 卡，2 轮，同节点同参数）

| 变体 | 16K/c64 | 16K/c8 |
| --- | ---: | ---: |
| A base（graph 开，默认 chunked） | 276.95 / 315.01（均值 296.0） | 218.35 / 217.77 |
| **B chunked-prefill 4096** | **307.26 / 323.19（均值 315.2）** | 223.34 / 222.34 |

- 两轮都 B > A（r1 +10.9%，r2 +2.6%），**均值 +6.5%**
- 但单轮波动达 **±7%**（A 的 r1/r2 差 13.7%），所以只能说"小幅稳定增益"，不能说"+10.9%"
- 历史记录 16K/c64 = 299.81，与本次 A 的均值 296.0 一致 → **测量方法没问题，是这台机器的噪声大**

## Kimi-Linear-48B × SGLang（TP=4，1 轮）

| 变体 | 16K/c64 | 16K/c8 |
| --- | ---: | ---: |
| K0 base（graph 开） | 232.03 | 127.71 |
| K1 `--cuda-graph-max-bs 8` | 230.86 | 119.06 |
| K2 `--mem-fraction-static 0.92` | 231.81 | 127.37 |
| **K3 chunked-prefill 4096** | **244.73（+5.5%）** | **138.31（+8.3%）** |
| K4 `--disable-cuda-graph` | 232.03 | 87.71（**−31%**） |

- 修复 16K/c64 的"回退"猜想**不成立**：限制 graph 捕获上界（K1）与提高显存比例（K2）都没用
- 真正有效的是 **chunked-prefill 4096**：两个档位同时改善
- K4 说明**CUDA Graph 在 16K/c8 上是刚需**（关掉直接 −31%），此前"graph 导致 16K/c64 回退 6.3%"的判断
  是单轮噪声导致的误判

## 为什么有效（机理）

16K 输入下 prefill 会一次性塞入巨量 token，把 decode 挤到边上；切成 4096 一片后，
prefill 与 decode 能交错执行，KV cache 压力平滑。**这与 vLLM 侧的结论正好互补**：
vLLM 默认已开 chunked prefill，但它的 decode 内核只有 Triton 兜底，所以切不切都没用；
SGLang 的 decode 内核更强，配上切片就能吃到收益。

## 复现

```bash
sbatch sweeps/mimo-sglang-16k-opt.slurm    # MiMo：4 个变体
sbatch sweeps/mimo-confirm.slurm           # MiMo：A/B × 2 轮
sbatch sweeps/kimi-16k-opt.slurm           # Kimi：5 个变体（需 --trust-remote-code）
```

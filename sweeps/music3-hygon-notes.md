# 别人的 BW 适配产物：MiniMax-Music3 镜像里的 `/tmp/music3-hygon`

2026-11-04 发现：Harbor 上 `dcu/acizv7iztz/base/minimax-music3-bw-sglang:20261031-rc1`
（digest `sha256:20f071aa…`，压缩 12.27 GB）里**留了一套完整的 BW 适配工作区**在 `/tmp/music3-hygon/`。
我们用 `podman create` + `podman cp` 把它捞了出来（集群 `~/work/music3-hygon/`），
**只记录结论，不复刻代码**（上游带自己的 LICENSE-SGLANG-OMNI / LICENSE-TORCHAUDIO 声明）。

## 这套东西是什么

- `serve.py`：兼容 OpenAI `/v1/audio/speech` 的 FastAPI 服务（127.0.0.1:8000），
  内部调用 `music3_hygon.runtime.Music3Runtime`
- `music3_hygon/`：他们自己写的 runtime（DiT/RVQ/flow-matching/DAV 的封装）
- `patches/0001-hygon-eager-components.patch`：把上游 sgl-omni 里依赖新接口的部分改成 eager 实现
- `evaluate.py` / `infer.py` / `bench.sh` / `env_check.sh`：官方条件评测与压测脚本
- `ar-probe.log`、`core.*`（5.5 G + 5.7 G + 2.8 G 的 core dump）→ **他们的最后状态是崩的**

## 他们踩出来、且对我们也成立的坑（这才是价值）

| 现象 | 他们的结论 | 我们的对应经验 |
| --- | --- | --- |
| BW 镜像自带的 fatched RoPE 内核 **起 512 线程，而 kernel bound 是 256** | 改用已安装的 native Q/K norm + rotary 实现，不走 fused | 与 gotchas #15（aiter 残包）同类：**镜像里的"优化内核"未必能用** |
| 图像默认的 FlashAttention paged KV layout **按固定 64 token 索引**，与 Triton allocator 的 1-token page 不匹配 | 设 `SGLANG_KV_LAYOUT_HCU_FA=0` 关掉它 | 与 gotchas #2 同源（SGLang 默认后端不可用） |
| SGLang 的确定性 matmul 内核三阶段 BF16 要 **96 KiB shared memory，BW 只有 64 KiB** | `num_stages=1` | 新的硬约束：**片上 shared memory 上限会卡死默认 kernel 参数** |
| `torchaudio` 的 sinc 重采样需要与参考实现一致的 44.1→32 kHz 算法 | 直接从 pytorch/audio v2.10.0 拷函数（保留 BSD 声明），不装不兼容的 wheel | 与 gotchas #19 同类：**torchaudio 在 DCU 上是雷区** |
| 兼容服务跑起来了，但 `ar-probe.log` 有 HW queue VMFault + core dump | README 自述"adaptation in progress" | **"服务在监听" ≠ "能出声"**，必须验产物规格 |

## 对我们的用法

1. 先用他们的 `run.sh`（需 `MODEL_PATH` 指向 `~/models/MiniMax-Music3`）验证能否出声；
   他们环境变量里 `ACOUSTIC_PRECISION=fp32`、`MEM_FRAC=0.5`、`MAX_LEN=10240` 可直接沿用
2. 出声后再用我们的 `evaluate_official.py` 打官方 60 条条件（32 kHz 立体声、≤750 声学帧）
3. **如果他们那套本身就是崩的**，就把它记为 blocked + 证据（core dump / VMFault），
   而不是硬啃 —— 这与「TF03 不评分，只要真产物」的官方要求一致：宁可如实报 blocked，也不造假音频

## 注意

- 不要把 `core.*`（14 GB）拷进仓库或留在 `~/work`，已删除
- 镜像里这套是**镜像作者的个人工作区**，不是官方发布物；引用时注明来源与日期

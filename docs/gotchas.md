# DCU 适配踩坑清单

按「隐蔽程度」排序。每条都标注了现象、根因、解法，以及它值多少调试时间。

## 1. `/dev/mkfd` 必须传（最难定位）

**现象**：容器内 `torch.cuda.is_available()` 返回 `False`、`device_count()` 却是 1，底层报
`[ROCT-Thunk-Interface] Load devices failed` → `RuntimeError: No HIP GPUs are available`。
`hy-smi` 也报 `No device available`。

**根因**：BW1000 上存在**第二个 KFD 设备** `/dev/mkfd`（主机上 `crw-rw-rw- root render 507,0`）。
只传 `/dev/kfd` 时 HSA 能找到运行时但枚举不到设备。

**解法**：
```bash
podman run --device /dev/kfd --device /dev/mkfd --device /dev/dri ...
```

**排查过程**：换过 DTK 版本、完整 `LD_LIBRARY_PATH`、`--privileged`、挂 `/sys`、挂宿主 `/opt/hyhal`
全都没有用；最后是列出宿主 `/dev` 时发现 `mkfd` 才解开。

## 2. CUDA Graph 的假陷阱

**现象**：不加 `--disable-cuda-graph` 时进程在捕获阶段静默退出，`server.log` 里没有任何 traceback，
只有 Slurm 记录 `Killed`。

**被误导的结论**：以为国产卡不支持 CUDA Graph，于是首版提交统一加了 `--disable-cuda-graph`。

**真相**：这个现象只在 `--max-running-requests 8` 的配置下出现。**放宽到 64 后 CUDA Graph 可正常捕获**，
而且收益巨大：

| 框架 | 开启 CUDA Graph 后的提升 |
| --- | --- |
| SGLang 0.5.10 | 4K/c1 +84.5%、4K/c8 +100.6%、4K/c64 +466.4%、16K/c1 +73.1% |
| vLLM 0.20.0 | 4K/c1 +24.2%、4K/c8 +31.9%、4K/c64 +8.9% |

## 3. `/etc/subuid` 为空导致 rootless podman 无法解包

**现象**：`Error: writing blob: processing tar file(potentially insufficient UIDs or GIDs available in user namespace (requested 0:42 for /etc/gshadow): lchown /etc/gshadow: invalid argument)`

**解法**：`--storage-opt ignore_chown_errors=true`。**注意 `vfs` 驱动不支持该参数**（报
`vfs driver does not support ignore_chown_errors options`），必须用 `overlay`。

## 4. 容器存储不能放 NFS

**现象**：`podman load/pull` 报 `lsetxattr /boot: operation not supported`。

**根因**：overlayfs 需要的 xattr 在 NFS 上不可用。

**解法**：`--root /tmp/pr-$SLURM_JOB_ID --runroot /tmp/prr-$SLURM_JOB_ID`，放节点本地盘。
**每个作业用独立 `--root`**，否则同节点上一个作业残留的 podman pause 状态会让 `pull` 秒失败：
`invalid internal status, try resetting the pause process with "podman system migrate"`。

## 5. 私有 registry 的自签证书 + 只读 client key

**现象 1**：`x509: certificate signed by unknown authority` → 加 `--tls-verify=false`。
**现象 2**：`open /etc/docker/certs.d/image.ac.com:5000/image.ac.com.key: permission denied`
→ 集群在 certs.d 里放了 root-only 的 client key，podman 会去读。
**解法**：`--tls-verify=false --cert-dir=/tmp/emptycerts`（指向一个空目录，绕开 certs.d）。

## 6. rootless podman 缺 runtime 目录

**现象**：`invalid internal status` / `cannot re-exec process to join the existing user namespace`。
**解法**：`export XDG_RUNTIME_DIR=/tmp/xdg-$$ && chmod 700`。登录节点上 podman 本来就不可用，只能在计算节点跑。

## 7. 不要用宿主 DTK 覆盖容器内 `/opt/dtk`

**现象**：Triton JIT 报 `ROCm compiler not found under /opt/dtk/aillvm/bin/clang`。
**根因**：宿主 DTK 目录里没有镜像自带的 `aillvm`（clang-18）。
**解法**：保留镜像内 `/opt/dtk`，只挂宿主 `/opt/hyhal`（提供与内核驱动匹配的 `libhydmi.so`）。

## 8. `libmsgpackc.so.2` 藏在 lightop 里

**现象**：`import torch` 失败，`ImportError: libmsgpackc.so.2: cannot open shared object file`。
**解法**：`LD_LIBRARY_PATH` 加上 `/usr/local/lib/python3.10/dist-packages/lightop/lib`。

容器内完整的 `LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:\
/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:\
/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:\
/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:\
/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:$LD_LIBRARY_PATH
```

漏掉 `.hyhal/*/lib` 会分别报 `libhydmi.so`、`libhsa-runtime64.so`、`librocm_smi64.so.2` 找不到。

## 9. Notebook 镜像的 ENTRYPOINT 会吞掉命令

**现象**：`podman run <img> bash /script.sh` 报 `/usr/bin/bash: cannot execute binary file`。
**根因**：SothisAI Notebook 镜像的 `ENTRYPOINT=/bin/sh -c "<初始化>; /usr/sbin/sshd -D"`。
**解法**：`--entrypoint=/bin/bash`。

## 10. `/tokenize` 在大词表模型上会 500

**现象**：Kimi-Linear（词表 163840）上 `POST /tokenize` 返回 500，
日志 `TypeError: Integer exceeds 64-bit range`（sglang 用 orjson 序列化 token id）。
**影响**：任何依赖 `/tokenize` 做压测前置校验的脚本都会崩。
**解法**：前置校验改成「真实 chat 请求 + 长度阶梯探测」，见 `tools/preflight_lib.py`。

## 11. `--context-length` 必须留余量（会产出假数据）

**现象**：`--context-length 17408`（= 16384 + 1024）时 16K 档全部被服务端 400 拒绝，
但**压测客户端把 400 当"完成"**，于是打出 `output_throughput 9791 tok/s`、`TTFT 0.00 ms` 这种物理上不可能的数字。

**根因**：客户端 `--random-input-len` 生成的 token 数不含 chat template 包装开销，实际输入达 16740–18784。

**解法**：
1. `MAX_LEN` 给到 20480
2. 压测前做真实请求 preflight
3. 压测后校验 `mean_ttft_ms != 0` 且吞吐不超物理上限，否则判 INVALID 并拒绝写入报告

**这条最危险**：差一步就把假数据写进技术报告了，而"伪造日志"是取消资格项。

## 12. sbatch 下 `$0` 不是脚本路径

**现象**：`container/submit.slurm` 里 `REPO="$(cd "$(dirname "$0")/.." && pwd)"` 指向
`/opt/gridview/slurm/spool_slurmd/job.../slurm_script`，找不到 `env.sh`。
**解法**：用 `SLURM_SUBMIT_DIR`。

## 13. 节点根盘水位极不稳定

同一分区不同节点，根盘可用空间实测从 **0 GB 到 278 GB** 不等。镜像解压需要 30 GB 量级，
因此提交脚本必须前置 `df` 检查并主动 `exit 42` 让作业快速失败重投：

```bash
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE $FREE G"; exit 42; }
```

## 14. 多卡容器：`HIP error: invalid device ordinal`

**现象**：TP=4 起服务时报 `torch.AcceleratorError: HIP error: invalid device ordinal`，
`unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES` 无效。
**状态**：**未解决**。根因怀疑是 rootless 容器 + 多卡 cgroup 设备白名单映射不一致
（同样的 TP=4 在另一套启动路径下可以跑通）。单卡场景不受影响。

## 15. SGLang 默认选到的 `aiter` attention backend 残缺

**现象**：服务启动成功、`/health` 返回 200，但第一个 decode 请求就把整个进程带崩：

```
File ".../sglang/srt/layers/attention/aiter_backend.py", line 2535, in forward_decode
    paged_attention_ragged(
NameError: name 'paged_attention_ragged' is not defined
```

**根因**：镜像内 `aiter` 不完整（缺 `aiter.ops.triton.*`）。SGLang 在部分模型上会默认选
`aiter` 而不是 `triton` backend。K100-AI 的 `jupyter-sglang` 镜像更严重——`import` 阶段就崩
（`ModuleNotFoundError: No module named 'aiter.ops.triton.gemm'`）。

**解法**：**所有条目一律显式指定 `--attention-backend triton`**，不要依赖默认值。
MiMo / Kimi 两个条目一直带着这个参数所以没暴露；Recon2Reason 首次忘了加，就踩到了。

## 16. 容器保活后，容器状态不再能反映服务健康

**现象**：为解决「`run.sh` 后台起服务后自己退出 → 容器主进程退出 → 服务被杀」的问题，
把容器主进程改成 `exec tail -f /dev/null` 保活。但此后 `podman ps` 永远显示 `Up`，
即使里面 sglang 已经崩了，健康检查循环仍会一直等到 `LOAD_TIMEOUT`（实测白等 30 分钟）。

**解法**：健康检查里额外查服务进程：

```bash
$PD exec <container> bash -lc 'pgrep -f launch_server >/dev/null' \
  || { echo "SERVER_PROCESS_DIED"; grep -iE "error|NameError|Traceback" "$WORK/repo/logs/server.log" | tail -8; exit 1; }
```

## 17. `Qwen3VLForEmbedding` 两个框架都不支持

**现象**：

SGLang 0.5.10rc0：
```
ValueError: Cannot find model module. 'Qwen3VLForEmbedding' is not a registered model
in the Transformers library ... and 'AutoModel' is not pre...
```

vLLM 0.20.0：
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, Model architectures ['Qwen3VLForEmbedding'] are not supported for now.
```

**源码确认**（实测三个版本）：
- sglang `v0.5.10rc0` / `v0.5.18` / `main` 的 `qwen3_vl.py` 都只有
  `EntryClass = Qwen3VLForConditionalGeneration`，无 embedding 变体
- 三个版本都没有 `qwen3_vl_embedding.py` 文件（HTTP 404）

**影响范围**：`OpenBMB/Trident-Qwen3VL-2B` 这类嵌入模型无法用现成推理框架部署。

**可行路径**：用官方代码（OpenBMB/Trident 的 `TridentQwen3VLBackend`，Transformers 直载）自建 serving；
注意 query / doc 使用不同 instruction，属训练格式。

**反面参照**：`Qwen3-ASR-1.7B` 的 `Qwen3ASRForConditionalGeneration` 在 vLLM 0.20 里是支持的，
说明"非 CausalLM 架构"本身不是障碍，**关键是特定架构有没有被框架注册**。

## 18. 离线 wheel 是按 `cp312` 下的，容器却是 py3.10（最贵的一次误判）

**现象**：AuK 条目 120 条官方条件全部失败，`反馈表.csv` 里全是
`ModuleNotFoundError: No module named 'librosa'`；同一目录里 6 个 `[warn] fail:` 全是
`*-cp312-*.whl`。日志上游还写着「librosa 装不进 / 与已存在包冲突」，于是被误判成
「DCU 上 librosa 不可用」，白白多轮了 7 次作业。

**根因**：在**本机（py3.12）**上 `pip download librosa` 拿到的是 `cp312` 轮子，
上传到镜像（py3.10）后 pip 直接判定 tag 不兼容而拒装；`--no-deps --target` 又会让
`pip install` 一律静默不装任何依赖，两者叠加使「装不上」看起来没有原因。
`librosa-1.0.0-py3-none-any.whl` 本身也失败，因为它要求 `numpy>=2.1` 且依赖 numba 链，
而那批 numba/scipy/soxr 全是 cp312。

**解法**：下载时显式指定目标解释器，绝不依赖本机默认：
```bash
python3 -m pip download --no-deps --dest wheels \
  --only-binary=:all: --platform manylinux2014_x86_64 \
  --python-version 310 --implementation cp --abi cp310 \
  librosa numba llvmlite scipy soundfile soxr msgpack scikit-learn joblib pooch lazy_loader audioread
```
并把 `import librosa/scipy/numba/soundfile/soxr` 的版本打印写进 `inner.sh`，装不全就直接退出，
别让 5 小时的评测跑在一堆隐形缺失上。

**同类陷阱**：`pip install --no-deps` 会**完全不检查 `Requires-Dist`**。Qwen-Image 条目因此把
`huggingface_hub==0.36.2` 装给了要求 `hub>=1.32.0` 的 diffusers 0.41.0，报
`ImportError: cannot import name 'resolve_revision'`，被误当成「上游未来版本 API」。
判断「版本不存在」之前，先读被装包的 `requires_dist`。

## 19. `torchaudio.save()` 同样要走 torchcodec

**现象**：AuK 在 `eng.generate()` 之后写盘时抛
`RuntimeError: Could not load libtorchcodec. Likely causes:`，日志里只显示 120 条全失败，
看不出是「生成了但没存下来」。

**根因**：torchaudio 2.9 的 I/O **双向**都走 torchcodec（0.8 起 `load` 与 `save` 均如此），
DCU 版 torch 加载不了绑定 stock-torch ABI 的 `libtorchcodec*.so`。

**解法**：读用 `soundfile.read`，写用 `soundfile.write`，两边都别碰 `torchaudio`：
```python
_sf.write(out, arr.squeeze(), int(sr), subtype="PCM_16")   # arr: float32 numpy, (N,)
```
**判据提示**：如果 120 条里每条都耗时数秒而不是几十毫秒，说明推理**已经跑通**，
失败点在写盘/后处理，别再去调模型。

## 20. 官方条件清单里的 `params` 必须逐条读取，别信 pipeline 默认值

**现象**：Qwen-Image 条目「跑通」了 6 条探针，尺寸也对（2048×2048），但实际是 **40 步**——
官方条件是 **8 步**（Turbo + KV cache）。若不核对，180 条产物会全部偏离官方条件，
而且耗时约为应有值的 5 倍，造成卡时浪费。

**根因**：`evaluate_official.py` 把 `use_kv_cache=True` 写死，只从环境变量取 `WIDTH/HEIGHT`，
从未读 `conditions/*.jsonl` 里的 `num_inference_steps` / `guidance_scale` / `schedule`。

**解法**：条件清单是机器可读的**唯一权威**，逐条覆盖：
```python
pp = it.get("params") or {}
img = pipe(prompt=it["prompt"],
           width=int(pp.get("width") or 2048), height=int(pp.get("height") or 2048),
           num_inference_steps=int(pp.get("num_inference_steps") or 8),
           true_cfg_scale=float(pp.get("guidance_scale") or 1.0),
           use_kv_cache=bool(pp.get("use_kv_cache", True)),
           generator=g).images[0]
```
**顺带解决口径争议**：2048 vs 1024 的争论不用问人——清单里写的就是 2048，
以清单为准，并把差异记进 README。

## 21. vLLM 在 DCU 上的 16K decode 慢 5–7 倍，且**任何参数都改不动**

**现象**：同模型（MiMo-9B）同卡，16K 档 vLLM 8.19 tok/s vs SGLang 41.47（c1）、
44.83 vs 308.72（c8）。TPOT 从 4K 的 47.9 ms 涨到 119.3 ms（SGLang 约 24 ms）。

**启动日志唯一的线索**：
```
WARNING [chunked_prefill_paged_decode.py:400] Cannot use ROCm custom paged attention kernel,
        falling back to Triton implementation.
```

**逐条排除（三轮扫描，共 ~3 卡时，见
`models/XiaomiMiMo__*__vllm/sweep/results-2026-11-03.md`）**：

| 试过的旋钮 | 结果 |
| --- | --- |
| `--max-num-batched-tokens` 2048 / 8192 | 无变化（默认已开 chunked prefill） |
| `VLLM_ROCM_USE_AITER=1` | 无变化，警告照旧 |
| `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | 无变化 |
| `--kv-cache-dtype fp8` | **更差**（TPOT 119→127 ms） |
| `--tensor-parallel-size 2` | 启动失败（见下条坑） |
| **`VLLM_USE_V1=0`（退回 v0 引擎）** | **完全无效**：16K/c1 8.19 vs 8.19、16K/c8 44.86 vs 44.85（2026-11-05 实测） |

**结论**：这是**内核缺失**，不是配置问题、也不是 v1 引擎调度的问题（v0 实测同等）。
要补只能自建镜像（重编 aiter），或者干脆改用 SGLang。

**两条附带的新坑**：

1. **压测客户端不能放 vLLM 镜像**：`ModuleNotFoundError: No module named 'sglang'`。
   服务用 vLLM 镜像、压测必须用 SGLang 镜像（双容器，见条目 `container/submit.slurm`）。
2. **podman 不继承 Slurm 的 `HIP_VISIBLE_DEVICES`**：多卡 vLLM 报
   `local_world_size (2) must be less than or equal to the number of visible devices (1)`。
   必须显式 `-e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"`（**既不要 unset，也不要指望自动继承**）。

## 22. 量化在 DCU 上是「有路但没铺」：vLLM 在线 fp8 直接起不来

**现象**：`vllm serve ... --quantization fp8` 在本镜像（DCU 版 vLLM 0.20）上
**engine core 初始化即失败**，容器日志只剩
`RuntimeError: Engine core initialization failed. Failed core proc(s): {}`，
子进程真正的报错没有落盘。同一命令去掉 `--quantization fp8` 后 160 s 内正常起服务。

**后续（2026-11-06）**：`--quantization fp8` 确实不行，但 **AWQ（weight-only INT4）可以跑**：
`awq_marlin` 会退到 `TritonW4A16LinearKernel`，服务能起、生成正常。代价是慢——4K/c64 只有 185 tok/s
（bf16 的 MiMo-9B 是 998），TPOT 是 4–6 倍。详见 `sweeps/results-awq-2026-11-06.md`。

**影响**：本仓库除 AWQ 探针外**没有量化条目**（全 bf16）。这不是"没试"，是 fp8 这条路没有现成内核。

**要量化需要**（按成本排序）：
1. 权重侧量化（GPTQ/AWQ/INT8）—— 需要有对应模型的量化权重发布，或用 `llm-compressor` 自行量化；
2. vLLM 侧的量化 kernel 必须在 DCU 上可编译（参考 gotchas #21：paged-attention 内核同样缺失）；
3. 退路：用 CPU/其它框架做量化推理对照，但那已不是"国产卡上跑通"的证据。

**做量化实验时的对照基线**（同子集、同脚本，2026-11-04 实测）：
AISHELL-1 前 20 条 CER **0.40%**、LibriSpeech 前 40 条 CER **0.10%** / WER **0.53%**（bf16，60 条耗时 30 s）。
先有这条基线，量化后的劣化才有可比性 —— **不要拿全量 300 条的数字去跟子集比**。

## 23. 视频 pipeline 的"分辨率桶"会静默改尺寸（832×480 → 848×480）

**现象**：HY-OmniWeaving 按官方条件跑完 60 条，产物校验发现**全部是 848×480**，
而条件清单要的是 832×480（宽多 2%）。生成过程没有任何报错。

**根因**：上游 `generate.py` 传的是 `--aspect_ratio 16:9`，pipeline 内部按**比例**去
`generate_crop_size_list(base_size=640, patch_size=16)` 的 121 个桶里挑最接近的尺寸：
`16:9 = 0.5625` 最近的是 `(848, 480)`（0.5660），而目标 `832×480 = 0.5769` 不在 16:9 附近。

**解法**：直接把条件清单的宽高拼成比例字符串传进去（`--aspect_ratio 832:480`），
`(832,480)` 本身就在桶列表里，比例精确命中。

**教训**：**产物规格必须逐条测量，不能只看"跑完了"**。这一条是在跑完 60 条（26 卡时）后
用 `imageio` 读 header 才发现的 —— 如果只检查"文件存在且 >200 KB"，会一路绿到底。

## 24. 临时账号不是存储：分区的 `AllowAccounts` 会被移除

**现象**：2026-11-07 14:12，跑了 22 小时的作业被 `CANCELLED`；之后所有 `sbatch` 报
`Invalid account or account/partition combination specified`，连 5 分钟的最小作业也一样。
SSH 能登录、`/public` 能读写、GitHub/GitLab 正常，唯独不能提交。

**定位方法（五分钟内可完成）**：
```bash
# 1) 是不是集群没资源？
squeue -a -h | wc -l                      # 0 → 不是排队问题
# 2) 该账号是否在分区白名单？
scontrol -a show partition hx1hdnormal | tr ' ' '\n' | grep '^AllowAccounts=' \
  | cut -d= -f2 | tr ',' '\n' | grep -x "$USER"    # 无输出 → 就是它
# 3) 同批次其他账号在不在？（本例 scnethr2601a05 / a07 / scnethr202601 都在，只有 a10 没有）
```

**结论**：`hx1hdnormal` 的 `AllowAccounts` 共 10552 条，不含该账号。
这不是故障、不是脚本问题、不是 QOS 写法问题（`--qos=partition_hx1hdnormal` 只会得到
`Invalid qos specification`，因为该 assoc 挂的是 `user_<账号>`）。

**教训（最贵的一条）**：
1. **临时账号下的任何数据都会随账号一起消失**。所以镜像归档要么推自己的 Harbor 命名空间，
   要么第一时间搬到自己的账号目录 —— 本项目的 30 GB 镜像差点就这么没了。
2. **跨账号搬数据要先测通**：`/public/home/<别人>` 通常是 `700`，同机也搬不动；
   要么临时 `chmod o+rx` 走 `rsync`，要么用 OpenAPI 上传（>1 GB 会 `write operation timed out`，必须分块）。
3. **长作业要能在任意时刻中断而不丢结果**：本项目所有评测脚本都做了"产物存在则跳过"，
   所以 22 小时的作业被砍后，53/60 的产物仍可直接续跑，只损失 3.5 小时。

## 25. 换账号重启时的三个坑（多账号共享节点）

2026-11-11 把工作从 `scnethr2601a10` 切到 `luql` 时踩到的，都发生在**同一台计算节点**上：

| 现象 | 根因 | 修法 |
| --- | --- | --- |
| `bash: /tmp/load.log: Permission denied` | 计算节点的 `/tmp` 是**多账号共享**的，上一个账号（或别人）建的同名文件属主不同 | 所有临时文件带作业号：`/tmp/load-$SLURM_JOB_ID.log` |
| `podman load` 跑到一半 `no space left on device`（起始 84 G） | 解压 32 GB 镜像 + podman 自己的临时副本，实际需要 ≈2× | 门槛设 **100 G**；作业开头先 `rm -rf /tmp/pr-* /tmp/prr-*` 清理自己遗留的 podman 存储 |
| `ImportError: libomp.so: cannot open shared object file` | 容器里只传了部分 `LD_LIBRARY_PATH` | 必须带全 15 段（`/opt/dtk/llvm/lib` 等，见 HANDOFF §6.1）——`libomp.so` 在 llvm 目录下 |

**顺带验证到的一条好消息**：`gunzip -c 归档.tar.gz | podman load` **8 分钟**就能把 7.9 GB 归档还原成
32 GB 可用镜像，`torch.cuda.is_available()` 为 True、`Device: BW / 64 GB / gfx936`、bf16 matmul 正常。
说明"镜像归档 + 还原"这条路是真能救命的，不只是 sha256 对得上。

**跨账号搬数据的实测数字**：`/public` 同机直拷 **74 GB / 427 s ≈ 177 MB/s**，
比「拉到本机再上传」（~1.7 MB/s，30 GB 要 5 小时）快两个数量级 —— 条件是源目录临时 `chmod o+rx`。

## 26. 同一台机器上换账号：`WORK` 路径不改会写进别人家目录

**现象**：把工作从 `scnethr2601a10` 切到 `luql` 后，容器里的每个写操作都 `Permission denied`
（`反馈表.csv`、`stats.json`、`logs/shard/stderr_*.log`），但 `chmod 666/777`、`--userns=keep-id`、
改文件名、换 rsync 同步全都没用。

**根因**：`submit.slurm` 里 `WORK="${WORK:-/public/home/<旧账号>/work/...}"` —— 这个默认值没跟着换，
于是容器一直在往**另一个账号的家目录**写。权限模型本身没问题。

**定位方法（10 秒）**：在 submit.slurm 里加一行自检并打印出来
```bash
echo "[host] REPO=$REPO WORK=$WORK  run_missing.py=$(ls "$WORK/repo/run_missing.py" 2>/dev/null || echo 缺失)"
```
一眼就能看到路径对不对、文件有没有同步过去。**这行现在是所有 submit.slurm 的标配。**

**教训**：`Permission denied` 在跨账号场景下，先怀疑路径，再怀疑权限模型。
另外 `cp -rf` 在 NFS 上既不可靠覆盖、也可能整片漏拷（实测漏过 `.py`），同步一律用 `rsync -a`。

## 27. 同一模型「SGLang 能跑、vLLM 不能」常常是版本问题

**现象**：Recon2Reason / RoboBrain（Qwen3-VL 底座）在 vLLM 0.20 上，**纯文本正常、任何带图请求 500**：

```
ValueError: Requested more deepstack tokens than available in buffer: num_tokens=72 > 67
  File vllm/model_executor/models/qwen3_vl.py, line 1711, in _get_deepstack_input_embeds
```

**根因**：0.20 的实现里 deepstack 缓冲按**首个视觉请求的 token 数**分配，后续请求 token 更多就越界、
直接抛异常。**vLLM 0.21 已删除该断言**（只做切片），换镜像即可，单图/多图都正常。

**教训**：
1. 别急着判定"某引擎不支持某模型" —— 先在镜像里 `grep` 源码确认版本行为（`grep -n "<报错文本>" $SITE/vllm/...`）。
2. 换镜像通常比改模型/改源码便宜得多。
3. **收容器前先把 `podman logs` 落盘**：detached 容器的 stdout 不进 $WORK 日志文件，
   `podman rm` 后根因就没了（本项目为此白跑过两轮）。

## 28. `ModelRegistry.get_supported_archs()` 的 grep 别用下划线

**现象**：曾用 `[a for a in archs if "qwen3_vl" in a.lower()]` 判断支持性，结果返回空，
据此记下"vLLM 不支持 Qwen3-VL"——但同目录里明明有 `qwen3_vl.py`。

**根因**：架构类是 `Qwen3VLForConditionalGeneration`，小写后是 `qwen3vlfor...`，**不含下划线**。

**正确做法**：
```python
[k for k in ModelRegistry.get_supported_archs() if "qwen3vl" in k.lower()]   # 去掉下划线
# 或者直接看文件：ls $SITE/vllm/model_executor/models/ | grep -i qwen3
```
同样的坑也发生在 `KimiLinearForCausalLM` 上（曾被误记为"vLLM 不支持 Kimi-Linear"）。

## 29. Harbor 搜索：搜不到 ≠ 没有，关键词要试多个

**现象**：判定「HY-World-2.0 在 DCU 上不可行」，理由是"没有现成的 gsplat ROCm 构建"。
实际 Harbor 上有 **`dcu/acizv7iztz/base/hy-world2-bw-sglang`**，镜像里连 gsplat 的 HIP 补丁都编好了 ——
单场景 92–121 s，5 条官方条件一次跑通。

**实际检索过程**：`?q=gsplat`（空）、`?q=hyworld`（空）→ 曾据此判断"不存在"。
**实际仓库名**：`hy-world2-bw-sglang` —— 带连字符，搜 `world` 才命中。

**正确姿势**：

```bash
H=https://image.ac.com:5000
# 1) 用「模型名的一段」而不是整串，且不假设分隔符
for q in world hy-world2 hyworld2 gsplat 3dgs; do
  curl -sk "$H/api/v2.0/search?q=$q" | jq -r '.repository[].repository_name'
done
# 2) 按「同批用户已做过什么」找：社区作者会把整套 harness 留在镜像 /tmp 里
curl -sk "$H/api/v2.0/projects/dcu/repositories?page_size=100&q=name%3D~acizv7iztz"
```

**更强的判据**：先查**同批账号**（`acizv7iztz`、`wzx`、`linshp` 等）的仓库清单，
他们跑过什么、留了什么镜像，往往就意味着"这条路有人趟过了"。

**教训**：把"未检索到"写成"不可行"，是最容易犯、代价最大的一类错误 ——
它会直接让一个条目被标 blocked 并在报告里写成硬约束。

## 30. 申请 `--gres=dcu:4` 只分到 1 张卡：`-c` 太小，分配被静默缩水

**现象**：`#SBATCH --gres=dcu:4` 的作业，`scontrol show job` 的 `ReqTRES` 确实写着 `gres/dcu=4`，
但容器内 `torch.cuda.device_count()` 恒为 **1**；4 rank 的 `torchrun` 在 `set_device` 处报
`torch.AcceleratorError: HIP error: invalid device ordinal`（rank 1/2/3 同时挂，rank 0 正常）。

**排查弯路**：先怀疑设备可见性 —— 依次试过 `env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES
-u ROCR_VISIBLE_DEVICES`、容器内显式 `HIP_VISIBLE_DEVICES=0,1,2,3`、podman 级 `-e`、
逐个 `--device /dev/dri/renderD12x`、挂 `/sys/class/kfd` 与 `/sys/class/drm`、
`--group-add keep-groups`、`--userns=keep-id`、换镜像 —— **全部无效**（`hy-smi -L` 在容器里也只列 1 张卡）。

**根因**：不是容器、不是驱动、不是权限（`/dev/dri/renderD*` 是 `crw-rw-rw-`，全可读写）。
是 **Slurm 实际只分配了 1 张卡**：该分区 `Gres=dcu:Hygon:8(S:0-7)` 按 **socket 绑卡**，
`enforce-binding` 下 `-c` 不够时调度器把 GRES 请求降级，而且只体现在 `AllocTRES`，`ReqTRES` 仍是原值。

**判据**（一条命令定性，别再复跑探针）：
```bash
sacct -j <JOBID> -n -P -o JobID,ReqTRES,AllocTRES
# 缩水时：ReqTRES 有 gres/dcu=4，AllocTRES 只有 gres/dcu=1
```

**解法**：每张 DCU 至少给 **8 个 CPU**（`-c 32` 配 4 卡）。改完 `AllocTRES` 即为 `gres/dcu=4`，
容器内 `device_count()` 立刻变 4。

**代价**：这一条吃掉约 6 卡时的探针（5 次 25 min 作业）+ 大量推理时间，
且会把人误导到"平台不支持多卡""镜像驱动有问题"这类错误结论上。

## 31. 多卡起不来先看 `AllocTRES`，不要先怀疑容器

紧接第 30 条：`device_count()=1` 在 DCU 上有两个截然不同的原因，且现象几乎一样。

| 原因 | 判据 | 修法 |
| --- | --- | --- |
| 少传 `/dev/mkfd`（见第 1 条） | 容器内 `hy-smi` 报 `No device available`，日志有 `Load devices failed` | 加 `--device /dev/mkfd` |
| **GRES 分配缩水**（第 30 条） | `hy-smi -L` 正常列卡但只有 1 张；`AllocTRES` 的 `gres/dcu=1` | 抬高 `-c` 到每卡 ≥8 CPU |

**规则**：先查宿主的 `AllocTRES`（调度器说了算），再查容器内的 `hy-smi -L`（驱动说了算），
最后才查 torch 的 `device_count()`。顺序反了就会在容器层反复试参数。

## 32. 多 rank 全量加载编码器 → 宿主 OOM kill（exitcode -9，无 traceback）

**现象**：4 卡 SP 起来后，rank 2 以 `exitcode: -9` 退出，日志里**没有** traceback，
只有 `torch.distributed.elastic.multiprocessing.errors.ChildFailedError`；
宿主侧 `slurmstepd: error: Detected 2 oom_kill events`。

**根因**：HY-OmniWeaving 的序列并行只切 DiT 主干，**每个 rank 仍各持一份 VAE + 文本/视觉编码器**
并做 CPU offload；4 份加起来 ≈200 GB，而 `-c 32` 自动带出的 `--mem` 只有 121 GB。

**解法**：
```bash
#SBATCH -c 64
#SBATCH --mem=240G     # 受 DefMemPerCPU=3888M 约束：mem ≤ c × 3888M
```
`--mem` 不能脱离 `-c` 单独抬：超过 `DefMemPerCPU × c` 直接提交失败
（`too much memory was requested relative to the number of CPUs`）。

**教训**：`exitcode -9` + 无 traceback + `oom_kill events` 是**宿主内存**耗尽，
不是显存、也不是代码问题；Docker/podman 场景下容易被当成"容器被杀"去查驱动。

## 33. `WORK` 指到仓库目录：`cp` 把自己拷进自己，报错却是 `ModuleNotFoundError: dashscope`

**现象**：8 卡作业一切正常地起了容器、装了依赖、`wan import OK`，然后在生成阶段全部 rank 退出，
唯一可见的错误是：

```
File "/opt/wan-src/generate.py", line 19, in <module>
  from wan.utils.prompt_extend import DashScopePromptExpander
ModuleNotFoundError: No module named 'dashscope'
```

**根因**：与 dashscope 无关。提交脚本里有两行惯用写法：

```bash
WORK="${WORK:-/public/home/luql/work/zoo-wan}"     # ← 恰好等于 REPO
mkdir -p "$WORK/repo" && cp -rf "$REPO"/. "$WORK/repo"/ 2>/dev/null
```

把 `$REPO` 拷进它自己的子目录时，GNU `cp` **一开始就报错退出**（`cannot copy a directory into itself`），
而 `2>/dev/null` 把这个错误吞掉了。后果是 `/work/repo` 只有容器以前留下的旧内容、
**`patches/` 目录根本没拷进去**，于是"容器内拷贝源码 + 打补丁"那一步的 `for p in patches/*.patch`
匹配不到任何文件、静默跳过，运行时用的还是不支持惰性导入的原始源码。

**为什么错误信息会误导**：补丁没打上 → 官方 `generate.py` 在模块级 import `dashscope` →
而镜像里没有 `dashscope` → 报 `ModuleNotFoundError`。**真正缺的是补丁文件，不是 Python 包。**

**解法**：`WORK` 必须与 `REPO` 分开，并给 `cp` 加自保护（不要再 `2>/dev/null`）：

```bash
WORK="${WORK:-/public/home/luql/work/<条目>-run}"
mkdir -p "$WORK/repo"
if [ "$(cd "$REPO" && pwd -P)" = "$(cd "$WORK/repo" && pwd -P)" ]; then
  echo "[host] WORK/repo 与 REPO 相同，跳过拷贝"
else
  cp -rf "$REPO"/. "$WORK/repo"/ || { echo "[host] 拷贝仓库失败"; exit 1; }
fi
```

**代价**：两次 8 卡作业（各 ~30 min）白跑，且第一反应会是去装 `dashscope`。

**规则**：`2>/dev/null` 只允许出现在"失败也无所谓"的地方；凡是后续步骤依赖其产物的拷贝/安装，
一律保留 stderr 并在失败时 `exit`。

## 34. 数据并行分片模式里误用 `FSDP_NPROC`：每个分片起 N 个 rank 抢同一张卡

**现象**：`SHARD_TOTAL=8`（8 卡各跑各的）时，每个分片的日志里都出现
`torch.distributed.elastic...ChildFailedError`、`rank : 1 (local_rank: 1)`，
并且显存瞬间打满到 63.98 GiB 后 OOM。

**根因**：启动器把"进程数"和"卡数"混用了：

```python
nproc = int(os.environ.get("FSDP_NPROC", "8"))   # FSDP 下确实是 8
launcher = ["torchrun", f"--nproc_per_node={nproc}", ...]   # 但分片模式下应恒为 1
```

FSDP 是"N 个 rank 共同算一条"，分片是"1 个 rank 独占一张卡算自己那几条"。分片模式下
仍按 `FSDP_NPROC` 起进程，就是 8 个 rank 全落在 `HIP_VISIBLE_DEVICES=<i>` 指定的同一张卡上。

**解法**：

```python
nproc = int(os.environ.get("FSDP_NPROC", "8")) if fsdp else 1
```

**顺带确认的一条硬约束**（本次实测）：Wan2.1-I2V-14B 在 64 GB 单卡上**装不下**——
即使 `--offload_model True --t5_cpu` 也 OOM（`Tried to allocate 310 MiB, 0 bytes free`）。
所以这个模型**没有"分片"这个选项**，多卡只能走 FSDP。

## 35. 只测文本会误判"已跑通"：`qwen_vl_utils` 版本错，图像路径必崩

**现象**：给 Qwen3-VL 系模型搭自建 serving，文本路径完全正常（2048 维、L2 归一化），
一旦输入图片就 500：

```
RuntimeError: shape '[1, 1, 2, 3, 32, 2, 16, 32, 2, 16]' is invalid for input of size 6439776
```

**根因**：`qwen_vl_utils` 版本与 Qwen3-VL 的 grid/patch 约定不匹配（实测 0.0.8 崩、0.0.14 正常）。
官方 `trident_embedder.py` 的注释里其实写了 `pip install qwen-vl-utils==0.0.14`，
而集群 wheels 目录里只有早先为 OmniWeaving 下的 0.0.8 —— **版本号只差一位，症状完全是内核级的**。

**判据**（`diag_images.py` 的做法）：分别打印 processor 的 `pixel_values.shape` 与 `image_grid_thw`，
再对比模型期望的 `t×(h/merge)×(w/merge)×3×temporal×patch×patch`。
两者自洽 → 问题在"自建管线的前处理"（多半是 `qwen_vl_utils` 版本）；
两者矛盾 → 问题在 processor/model 版本不配套。

**解法**：`pip download qwen_vl_utils==0.0.14` 单独放一个 wheels 目录，容器里 `--no-deps --upgrade`
装进隔离目录。

**教训**：多模态模型的"跑通"必须包含**三类输入**（纯文本 / 纯图像 / 图文），
只测文本会把前处理问题全部漏掉；而这类问题的报错又长得像模型代码 bug，容易往错误方向查。

## 36. 压测客户端的 `ready-check-timeout` 默认 60 s：会把"服务在加载"记成"该档不支持"

**现象**：某个高负载档位（Kimi × vLLM 16K/c64）两轮都报失败，客户端日志只有

```
Waiting up to 60s for http://127.0.0.1:30000/v1/models to become ready...
Server at http://127.0.0.1:30000/v1/models is not ready. Exiting.
```

看起来像"这个档位不支持"，于是被写进报告当结论。

**根因**：SGLang/vLLM 的 `bench_serving` 客户端有个 `--ready-check-timeout-sec`，默认 **60 s**。
而 48B + TP=4 的冷启动（含 kernel 编译）实测要 **410 s**。于是客户端在服务还在加载时就退出，
真正的服务端错误被完全掩盖。

**解法**：把它显式设大（本仓库改为 `READY_TIMEOUT`，默认 900 s），并**务必保存服务端日志**
（`podman logs <name> > $WORK/vllm-logs-$JID.txt`，容器一删日志就没了）。

**改大之后才看到的真因**（同一档位）：

```
Invalid address access: 0x7f7e01015000, Error code: 1.
>>>>>>>> KERNEL VMFault !!!! <<<<<<
(EngineCore) ERROR [multiproc_executor.py:283] Worker proc VllmWorker-3 died unexpectedly, shutting down executor.
```

即 worker 被 **GPU 访存错误**打死（崩溃时 `num_running_reqs=33, num_waiting_reqs=31`），
而不是超时、不是 OOM、也不是"不支持"。

**教训**：凡是"客户端说服务没准备好"的失败，都要先把客户端的等待时间调到明显大于冷启动时间，
否则你记下的是自己设置的超时，而不是系统的行为。

## 37. 探测脚本漏传 `LD_LIBRARY_PATH`：`import torch` 报 `libgalaxyhip.so.5`

**现象**：新写的只读探测作业里，`python3 -c "import vllm"` / `import torch` / `import aiter` 全部报

```
ImportError: libgalaxyhip.so.5: cannot open shared object file: No such file or directory
```

同时 `python3 -c "import sys"` 正常、`ls site-packages/aiter` 也存在 —— 很像"镜像坏了"。

**根因**：探测脚本是我新写的，`podman run` 时只挂了 `/dev/kfd`，**没有**把仓库里统一的
`LIBPATH` 传进容器。`libgalaxyhip.so.5` 由 DTK/hyhal 提供，缺了它 `torch._C` 起不来；
而 `ls`/`grep` 这类不 import 的命令自然不受影响，于是同一份输出里"有的项有结果、有的项全挂"。

**判据**：凡是**同一份日志里** `ls`/`grep` 成功而 `import torch` 失败，先查环境变量，别怀疑镜像。
仓库里各条目的 `container/submit.slurm` 都有完整 `LIBPATH`，新脚本应当**照抄而不是重写**。

**解法**：`podman run` 显式传环境：

```bash
-e LD_LIBRARY_PATH="$LIBPATH" \
-e ROCM_PATH=/opt/dtk -e HIP_PATH=/opt/dtk/hip -e HYHAL_PATH=/opt/hyhal
```

**代价**：一轮 1 卡作业（~15 min）+ 一次重跑；结论上损失了整个 Q0/Q1/Q2/Q5。
与 gotcha #8（`libmsgpackc` 藏在 lightop 里）同类，区别是这次缺的库在 DTK 侧。

## 38. 换镜像不等于升级：`vllm_hcu` 用 `.t()` 加载 3D 权重，Kimi 在 0.21 上加载即崩

**现象**：Kimi-Linear-48B × vLLM **0.21**（`vllm_hcu` 的 `HCUPlatform`），`--max-model-len 17408`、
TP=4、`--max-num-seqs 64`，四个 rank 在**模型加载阶段**同时失败，服务从未起来：

```
File ".../vllm/model_executor/models/kimi_linear.py", line 565, in load_weights
    weight_loader(param, loaded_weight, **kwargs)
File ".../vllm_hcu/model_executor/layers/linear.py", line 629, in weight_loader_v2
    param.load_column_parallel_weight(loaded_weight=..., is_quantization=...)
File ".../vllm_hcu/model_executor/parameter.py", line 158, in load_column_parallel_weight
    loaded_weight = loaded_weight.t()
RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D
```

同一份权重在 0.20 上加载正常，0.20/0.21 的 `kimi_linear.py` 也几乎一致。

**根因**：`vllm_hcu` 是海光自研的并行线性层实现，`load_column_parallel_weight` **无条件**转置权重；
Kimi 的 `conv1d` 权重是 3D（`[dim, 1, kernel]`）。0.20 用的是 vLLM 上游 `parameter.py`，没有这个假设。

**判据**：换镜像/换版本后若**加载阶段**就崩、且栈里出现 `vllm_hcu/` 而不是 `vllm/`，
说明是海光私有实现与上游模型的兼容缺口，改配置无用（`--enforce-eager`、`--max-model-len` 都白搭）。

**影响**：闸门曾判断"0.21 自带 `causal_conv1d`，换镜像是跑通 16K/c64 最便宜的一枪" —— **该判断被实测推翻**。
0.21 对 Kimi 是**比 0.20 更差**。Kimi 的长上下文排查只能留在 0.20 内做。

**推广**：`vllm_hcu` 的平台名是 `HCUPlatform`、`get_device_capability()` 报 CUDA 9.3 等价物，
**不要用 `vllm/platforms/rocm.py` 的假设去推断 0.21 的行为**；私有层的路径是 `vllm_hcu/...`。

## 39. aiter 在 BW1000 上恒被关闭：先是缺 `rocminfo`，更根本是 MI300 白名单

**现象（表象）**：Kimi-Linear × vLLM 0.20 启动日志里，
`Using TRITON_MLA backend out of potential backends: ['TRITON_MLA']` —— 候选集只有一项，
看着像"vLLM 在 DCU 上就只能这样"。

**根因链（2026-11-15 两张只读探测拼出，作业 1121990 + 1122316）**：

```
容器无 rocminfo
  → import aiter 失败：Get GPU arch from rocminfo failed ... 'rocminfo'
  → is_aiter_found_and_supported() = False
  → rocm_aiter_ops.is_mla_enabled() 被 @if_aiter_supported 短路（返回 None）
  → _get_backend_priorities(use_mla=True) 走 else 分支，只返回 [TRITON_MLA]
  → Kimi 的 MLA 层只能用 Triton MLA kernel
```

`vllm/platforms/rocm.py` 的 `_get_backend_priorities` 原本会给
`[ROCM_AITER_MLA, TRITON_MLA, ROCM_AITER_TRITON_MLA]` 三项。

**判据**：看到"候选 backend 只有一项"时，**不要当成框架限制**。先查 `is_aiter_found_and_supported()`，
再看 `import aiter` 的报错。aiter 的所有能力开关（`is_mla_enabled` / `is_mha_enabled` /
`is_linear_fp8_enabled` …）都挂了 `@if_aiter_supported`，aiter 一挂**全部静默返回 None**，
不报错、不打日志。

**解法（第一步）**：把 `/opt/dtk/bin` 加进容器的 `PATH`，或让 aiter 的 arch 探测绕开它。
成本极低（一次镜像内 `cp` 或 `-v` 挂载），收益是可能换掉崩溃嫌疑最大的 Triton MLA kernel。

### 补一句：`rocminfo` **不是全部**（作业 1123060 实测）

上面这条链只对了一半。实测发现：

| 事实 | 证据 |
| --- | --- |
| `rocminfo` **本来就在镜像里**，只是不在 `PATH` | `FOUND /opt/dtk/bin/rocminfo`；`which rocminfo` 为空 |
| 把假 `rocminfo` 放进 PATH 后，`import aiter` **成功** | `aiter OK with fake rocminfo` |
| 但 `is_aiter_found_and_supported()` **仍然 False** | 同上作业直接调用 |
| 于是 `is_mla_enabled()` 仍为 `None`，候选集仍是 `['TRITON_MLA']` | `priorities(use_mla=True) = ['TRITON_MLA']` |

所以 `import aiter` 只是第一道门，`_aiter_ops.py` 里还有自己的判定条件。
**看到 `import aiter` 成功但开关仍为 None 时，不要以为已经修好了**——要直接读
`is_aiter_found_and_supported()` 的函数体，看它额外要求什么。

### 第二道门是**硬件型号白名单**，BW1000 永远过不了（作业 1123513 + 1123605）

```python
# vllm/_aiter_ops.py
def is_aiter_found_and_supported() -> bool:
    if current_platform.is_rocm() and IS_AITER_FOUND:
        from vllm.platforms.rocm import on_mi3xx
        return on_mi3xx()

# vllm/platforms/rocm.py
_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])
def on_mi3xx() -> bool: return _ON_MI3XX
```

本机实测（作业 1123605）：

```
_CudaDeviceProperties(name='BW', major=9, minor=3,
                      gcnArchName='gfx936:sramecc+:xnack-', ...)
```

**gfx936 不在 `["gfx942","gfx950"]` 里 → `on_mi3xx()` 恒为 False。**
这条与 `rocminfo`、`PATH`、`VLLM_ROCM_USE_AITER_*` 环境变量**全都无关**。

结论修正：**"缺 rocminfo"只是链条最外面的一层，真正把 aiter MLA 关掉的是这条
硬编码的 MI300 白名单。** 在 BW1000 上，无论补什么库、设什么环境变量，
`_get_backend_priorities(use_mla=True)` 都不会返回 `ROCM_AITER_MLA`。
唯一剩下的可能是显式 `--attention-backend ROCM_AITER_MLA` 绕过候选集
（该函数 docstring 写着 "Can explicitly request aiter regardless of env var"），
但其能否绕过 `@if_aiter_supported` 装饰器尚待实测。

**同类教训**：`@if_aiter_supported` 这种"装饰器短路"是本仓库里第二次踩到
（第一次是 gotcha #37 的 `LD_LIBRARY_PATH`）——**能力探测返回 None 时，先怀疑环境，再怀疑框架**。

### ✅ 显式 `--attention-backend` **确实能绕过白名单**（作业 1123927 + 1124018）

上面最后一句话"尚待实测"已经有答案了，而且和直觉相反：

```python
# vllm/platforms/rocm.py:468-497
def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None) -> str:
    device_capability = cls.get_device_capability()
    # First try checking just the selected backend, if there is one.
    if selected_backend is not None:                    # ← 478: 显式分支
        try:
            backend_class = selected_backend.get_class()          # 直接拿类
            invalid_reasons = backend_class.validate_configuration(  # 只做这一道检查
                device_capability=device_capability, **attn_selector_config._asdict())
        except ImportError:
            invalid_reasons = ["ImportError"]
        ...
        return selected_backend.get_path()               # ← 完全不碰 _get_backend_priorities
    # 只有走到这里才会查候选集（进而查 on_mi3xx）
    valid_backends_priorities, invalid_reasons = cls.get_valid_backends(...)
```

实测（作业 1124018，`PATH` 里补了 `/opt/dtk/bin`）：

```
ON_MI3XX = False                     is_mla_enabled() = None      ← 白名单依旧是关的
get_class() -> AiterMLABackend       ← ✅ 成功
validate_configuration() -> OK (empty)  ← ✅ 通过
ROCM_AITER_TRITON_MLA 也 get_class OK / validate OK
```

**结论：`--attention-backend ROCM_AITER_MLA` 是 BW1000 上唯一能让 aiter kernel 上场的入口。**
两条路径的差别：

| 调用方式 | 经过 `_get_backend_priorities` | 受 `on_mi3xx()` 限制 |
| --- | --- | --- |
| 默认（不传 `--attention-backend`） | 是 | **是** |
| 显式 `--attention-backend X` | **否** | **否** |

**但要注意适用范围**：这条只对 **attention backend** 有效。
GEMM 没有等价的"显式指定"旋钮，所以 fp8 依然被挡住（见 #46）。
**看到"某个能力被硬件白名单关掉"时，去读分配该类的那段代码——
很可能存在一条"用户显式指定就跳过检查"的旁路。**

## 40. 作业不做 podman 清理会把节点 `/tmp` 写满，且事后**无法**删除

**现象**：一个正常跑完/被取消的作业之后，**下一个**作业在 `podman load` 阶段失败：

```
write /usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda...so: no space left on device
Error: creating runtime temporary files directory: mkdir /tmp/podman-run-2525/libpod/tmp: no space left on device
```

**根因**：仓库里所有编排脚本都用 `podman --root /tmp/pr-$JID`，作业结束即退出，
但**存储目录留在节点本地盘**。镜像解包单个就 8–31 GB。实测某节点上累积了 8 个历史作业的目录，
`/tmp/pr-1122012` 一个就 31 GB。

**为什么很难事后补救**：Slurm 在作业结束后立即 `pam_slurm_adopt` 切断节点访问
（`Access denied by pam_slurm_adopt: you have no active jobs on this node`），
所以**登录节点上没有权限去删**自己留下的目录。只能等节点自动回收，或另起一个作业去清理。

**解法**：每个编排脚本都必须有

```bash
cleanup_podman() {
  $PD rmi -af >/dev/null 2>&1
  $PD system prune -af >/dev/null 2>&1
  rm -rf "/tmp/pr-$JID" "/tmp/prr-$JID" "$XDG_RUNTIME_DIR" 2>/dev/null
}
trap cleanup_podman EXIT
```

并且**按需加载镜像**：服务镜像先加载，压测客户端的镜像等真正要用时再加载
（一次作业同时展开两个镜像，磁盘峰值翻倍，几百 G 的节点也会被打满）。

**判据**：作业启动时先打印 `df -Pm /tmp`，低于 45 GB 直接退出并报 `NOT_ENOUGH_SPACE`，
不要让它跑到一半才发现。本仓库早期的 `container/submit.slurm` 只检查一次 68G，
没有退出清理，属同一类问题。

## 41. 同一作业展开两个 9 GB 镜像 + 容器层，79 GB 的 `/tmp` 也会被打满

**现象**（matrix 作业两次失败，1122422 / 1122564）：

```
write /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_hip.so: no space left on device
```

作业启动时 `df` 显示 `tmp_free=95G` / `79G`，预检都通过了，跑到 `podman load` 才爆。

**根因**：解包一个 9.3 GB 的 tar.gz 镜像，落盘后的 overlay 层远超 9 GB
（torch + DTK 库展开后几十 GB），**而且 `podman load` 期间同时存在压缩流缓冲与解包中间层**。
再叠加第二个镜像和容器可写层，79 GB 不够。

**判据**：预检阈值不能按"镜像 tar 大小"设。经验值：**单个镜像至少留 45 GB**，
两个镜像同作业至少 90–100 GB；`tmp_free` 低于 60 GB 时不要在同一作业里加载第二个镜像。

**补充（同日又踩一次，job 1123259）**：预检通过也不算数——探测作业启动时 `tmp_free=84G`，
仍报 `write .../libmagma.so: no space left on device`。原因是**同一节点上还有别的作业在写**，
空闲空间在解包期间被别人吃掉。所以：

- 预检只是"入场券"，不是保证；判断依据要看**节点磁盘类型**——`/tmp` 在 NVMe 上的节点
  （如 `f11r3n17`，437G 盘）比 SATA 节点（876G 总盘但已用 96%）稳得多；
- 探测类 1 卡作业失败成本低，**直接换节点重提**比在原地调参数快。

**解法**：
1. 一个作业只加载一个镜像；压测客户端镜像**用到时再加载**（`load_cli()` 里 Check-then-load）；
2. 启动预检阈值提到 45 GB 并打印实际值；
3. 末了 `trap` 里清理（gotcha #40）。

## 42. `validate_configuration(dtype=...)` 要 `torch.dtype` 对象；传字符串会"全体不支持"

**现象**：逐个 MLA backend 调 `validate_configuration()` 判断可用性，结果**每一个**都返回

```
['dtype not supported']
```

看起来像"DCU 上没有任何 MLA backend 可用"。

**根因**：探测脚本里传的是字符串

```python
cfg = AttentionSelectorConfig(dtype='bfloat16', ...)   # ❌ str
```

而 `supports_dtype()` 是拿它跟 `cls.supported_dtypes` 比，那里装的是
`[torch.float16, torch.bfloat16]` 这样的 `torch.dtype` **对象**。`'bfloat16' != torch.bfloat16`，
于是无条件为假。正确写法：

```python
cfg = AttentionSelectorConfig(dtype=torch.bfloat16, ...)   # ✅
```

**判据（值得当成通则）**：**当一整张表的所有实现都返回同一个失败理由时，先怀疑自己的入参，
而不是相信"它们都不支持"**。真实世界里各实现的支持集通常不一样；完全一致的结果多半是
筛选条件本身恒假。

**代价**：这条错误结论一度被写进 README 和实验记录，并在 40 分钟内被当成"换 backend 这条路不存在"
的依据。同一天里，这是第二个"自己造的假象"（前一个是 gotcha #37 的 `LD_LIBRARY_PATH`）。
共同点：**探测脚本的输出看起来像框架的行为，其实是我们没把环境/类型配对。**

## 43. `--max-model-len` 按"输入+输出"精确算会差一个 token，压测全 400

**现象**（作业 1123145 的 B 档）：设 `--max-model-len 17408`（= 16384 输入 + 1024 输出），
压测客户端 17 秒"跑完" 128 个请求，然后崩在

```
IndexError: index -1 is out of bounds for axis 0 with size 0
```

`e2e_latencies` 是空数组——**所有请求都失败了**，只是失败得比超时快得多。

**根因**：服务端逐条返回 400：

```
VLLMValidationError: This model's maximum context length is 17408 tokens.
However, you requested 1024 output tokens and your prompt contains at least 16385 input tokens,
for a total of at least 17409 tokens.
```

实际 prompt 是 **16385** 而不是 16384（多 1 个 token：BOS / chat template）。
`17408 - 1024 = 16384` 恰好卡死，所以每条请求都超 1 个 token。

**判据**：压测"跑得异常快"（17 秒 vs 正常 547 秒）+ `completed` 计数满 + 结果表
`no data` → **先怀疑 400，不要怀疑服务端崩了**。`bench.sh` 的 `preflight_lib.py`
用长度阶梯探测能过，因为它测的是"服务端接不接受这个长度"，而 16384 确实能接受；
被拒的是 16385。

**解法**：留余量。16K 输入 + 1K 输出直接取 `20480`（默认值就是它），不要精确减。
确需压到最小，取 `16384 + 1024 + 512 = 17920` 这类有余量的值。

## 44. 单轮成功不能作为"问题已解决"的证据

**事故**（2026-11-15，作业 1123145）：`--enforce-eager` 下第 1 轮 16K/c64
完整跑通 128 个请求（239.32 tok/s、峰值并发 65），我据此在 README 和结果文档里
写了"P0-A 已达成、cudagraph 是元凶"。**随后第 2 轮在同一配置下 `KERNEL VMFault` 崩溃**
（`Running=62, Waiting=2`），全部推翻。

**根因**：这是一个**概率性越界**，不是确定性阈值。round1 峰值并发 65 侥幸通过，
round2 在 62 崩了。`--enforce-eager` 只是把崩溃概率往下压，不是消除。

**为什么 `bench.sh` 要跑两轮**：第 1 轮是预热，取第 2 轮。这个设计**正是为了防这个**——
它假定第 1 轮可能受冷启动影响。反过来也成立：**第 1 轮成功、第 2 轮失败**同样是信号，
说明结果不稳定。

**判据**：任何"修好了"的结论，至少要两轮都出数；概率性故障还要再多跑几次。
写下结论时明确写"几轮里几轮成功"，不要写"已修复"。

**同一天的相关纠正**：这是本日第三次推翻自己的结论（前两次是 gotchas #42 的 dtype 字符串、
`rocminfo` 的因果链只对了一半）。共同点是**证据只覆盖了一个样本就下结论**。

## 45. `bench.sh` 的有效性校验自己从未执行过（`os` 未导入）

**事故**（2026-11-15，job 1123145 后复盘）：16K/c64 的 B 档（`--max-model-len 17408`）
全部请求 400，脚本没有按设计报 `INVALID ... TTFT=0`，而是以
`np.percentile` 的 `IndexError` 收场，"配置错误"被表现成了"压测脚本崩了"。

**根因**：`bench.sh` 末尾的有效性校验段写的是

```python
import json, sys                                 # ← 漏了 os
ROUND_FILE = "bench/round2.jsonl" if os.path.exists("bench/round2.jsonl") else ...
```

`os` 未导入 → 第 3 行就 `NameError` → 整个校验段被杀掉。**它从来没跑过一次**，
所以"TTFT=0 / 吞吐超物理上限"这两类无效数据一直都能混进 `round2.jsonl`。

**影响面**：这段模板被复制到 `models/*/bench.sh` 全部 10 个文件，全部失效。
2026-11-15 统一改为 `import json, os, sys`。

**修复后仍复发一次（16:53 发现）**：同一个 heredoc 里还有第二行坏代码
`EXPECT = len(ILENS.split()) * len(CONCS.split())`——`ILENS`/`CONCS` 是 **bash** 变量，
在 Python heredoc 里不存在，`os` 修好后紧接着就是 `NameError: name 'ILENS' is not defined`。
两道 bug 串联，只修第一道时校验段仍然必死。

更隐蔽的是**远程副本没同步**：本地 `models/*/bench.sh` 已改，但
`luql:/public/home/luql/work/zoo-sync/` 下只有 Kimi 一个条目存在，
且它仍是旧版；作业启动时会 `rsync` 到自己的 `WORK/repo`，于是**作业里跑的仍是坏脚本**。
判据：作业 stdout 里出现

```
[bench] 有效性校验
Traceback (most recent call last):
  NameError: name 'ILENS' is not defined
```

就说明这一轮的"有效数据"没有经过闸门。**改完 bench.sh 必须 rsync 到远程再提交作业**，
或直接在作业脚本里就地 `sed` 掉那两行。

**判据**：`bench.sh` 跑完必须看到 `[bench] 有效性校验` 后面跟着 `  all valid`。
只看到 `[bench] done` 而没有 `all valid`，说明校验段自己失败了（或根本没跑到）。
**"校验没报错"不等于"校验通过了"**——先确认它真的执行了。

**同类风险**：任何"闸门/校验"代码被放进 `try/except` 或独立 heredoc 时，
失败可能只表现为"没有输出"。写校验的人要顺手验证一次**它能否正确报错**
（故意喂一条坏数据看它是否触发）。

## 46. DCU 上 fp8 没有 GEMM kernel：`torch.addmm` 直接 `NotImplementedError`

**现象**（2026-11-15，job 1123947，vLLM 0.20 + BW1000）：

```python
torch.float8_e4m3fn  torch.mm(a, b)  →  NotImplementedError: "addmm_cuda" not implemented for 'Float8_e4m3fn'
torch.float8_e5m2    torch.mm(a, b)  →  NotImplementedError: "addmm_cuda" not implemented for 'Float8_e5m2'
```

**这条容易被误判成配置问题**，因为看起来相关的三样东西都在：

| 看起来该有的 | 实际状态 |
| --- | --- |
| `--quantization fp8` 在白名单里 | ✅ `RocmPlatform.supported_quantization` 含 `fp8` |
| `Fp8Config` 能解析 | ✅ `get_quantization_config("fp8") -> Fp8Config` |
| `fp8_dtype` 有定义 | ✅ `torch.float8_e4m3fn` |
| hipBLASLt 认 fp8 | ✅ `libhipblaslt.so.0.10` 含 `R_8F_E4M3` / `R_8F_E5M2` |
| **torch 能把 fp8 GEMM 分发下去** | ❌ **没有** |

**根因**：PyTorch 的 DCU/HIP 后端没有为 fp8 dtype 注册 `addmm` 实现。
hipBLASLt 库里的 fp8 符号没有被接进来，所以**库有 = 用不上**。
任何上层（vLLM、PEFT）都无法在 `addmm` 之上凭空造出 GEMM。

**为什么表现为"engine core 初始化失败"**：`get_device_capability()` 返回
`DeviceCapability(major=9, minor=3)`——gfx936 被硬套成 CUDA 9.3，于是 vLLM 认为
"fp8 应该支持"，一路放行到第一次预热 GEMM 才炸。报错位置离真因很远。

**解法**：没有。需要 torch 侧补 fp8 GEMM 绑定，或 vLLM 走绕过 torch 的专用 kernel
（aiter 的 fp8 kernel 就是这种，但被 `on_mi3xx()` 挡在 gfx936 之外，见 #39）。

**判据**：想验证任何"DCU 上 fp8 能不能用"的话，**先跑这一行**：

```bash
python3 -c "import torch; a=torch.zeros(16,16,device='cuda',dtype=torch.float8_e4m3fn); print(torch.mm(a,a).dtype)"
```

它失败就不用往下查了——查白名单、查 hipBLASLt 版本都是白费。

完整记录见 `sweeps/results-kimi-quantigate-2026-11-15.md`。

## 47. `bench/*.log` 不会被清空，读之前先核对 mtime

**事故**（2026-11-15，job 1123874）：检查 `--max-num-seqs 24` 的结果时，
读到 `bench/round2_16384_1024_c64.log` 末尾是
`IndexError: index -1 is out of bounds for axis 0 with size 0` +
`All requests failed. This is likely due to a misconfiguration`，
一度判定"降到 24 也崩"。核对时间戳后才发现该文件是 **15:13** 写的，
而当时作业 1123874 的服务 **15:46** 才就绪、正在加载 CLI 镜像——**根本没开始压测**。

**根因**：`bench.sh:8` 只清理结果文件：

```bash
mkdir -p bench && rm -f bench/round*.jsonl     # ← 只清 jsonl
```

`bench/${TAG}.log` 由 `> "bench/${TAG}.log"` 覆盖写，但在**该档真正跑到那一步之前**，
文件还是上一轮的内容。而 `$WORK/repo` 是 `rsync -a --exclude bench` 同步的，
`bench/` 目录本身在 `$WORK` 里长期存在，跨作业残留。

**判据 / 解法**：看任何 `bench/*.log` 之前，先确认

```bash
stat -c %y bench/round2_*.log      # 或 ls -la --time-style=+%H:%M:%S
```

文件时间必须晚于**本次作业的服务就绪时间**。否则你读的是别人的结论。
更稳的做法是给每个作业独立 `$WORK`（本仓库新脚本已这么做，如
`kimi-16k-020-aitermla.slurm` 用 `/public/home/luql/work/kimi-16k-020-aitermla`）。

**同类**：这与 gotcha #44"单轮成功不等于修复"是一对——
一个防"读早了"，一个防"读少了"。共同点是**结论必须先确认证据真的来自当前实验**。

## 48. 用 `--max-num-seqs` 换稳定，代价全在 TTFT 上（不是吞吐）

**实测**（2026-11-15，Kimi 16K/c64，job 1123874 vs 1123145）：

| 配置 | 成功 | Output tok/s | **Mean TTFT** | 崩溃 |
| --- | --- | ---: | ---: | --- |
| `--enforce-eager`（并发自然到 62） | 128/128 | 239.32 | **43.6 s** | round2 崩 |
| `--max-num-seqs 24` | 128/128 | 216.66 | **156.2 s** | round1 无 |

**吞吐只降 9.5%，但 TTFT 恶化 3.6×。** 这和直觉相反——
一般认为"限批 → GPU 利用率下降 → 吞吐大降"，实际吞吐几乎没变，因为 16K 长上下文时
瓶颈在单请求的 KV 读写而非 batch 并行度。真正被牺牲的是**排队时间**：

- 客户端仍打 64 并发，服务端每批只跑 24 个；
- 其余 40 个请求要等前一批生成完 1024 个 token 才被调度；
- 16K 输入 × 1024 输出的单请求耗时约 100 s，所以后 40 个请求的 TTFT ≈ 2 个整批 ≈ 150–200 s。

**判据**：调 `--max-num-seqs` 时，**不要只看吞吐和"有没有崩"**，必须同时看
`mean_ttft_ms` / `p99_ttft_ms` / `Waiting` 队列长度。服务端日志里
`Running: 24 reqs, Waiting: 36 reqs` 这一行就是"TTFT 会很难看"的直接证据。

**适用结论**：`--max-num-seqs` 适合"离线批处理、只关心总吞吐和成功率"的场景；
不适合任何交互式用途。报告里必须同时给出两个数，不能只写吞吐。

## 49. backend 类能 `get_class()` 成功，不等于它的模块能 import 成功

**现象**（2026-11-15，作业 1124165，vLLM 0.20 / BW1000）：

```
(Worker_TP0 pid=228) ERROR [multiproc_executor.py:870]
  ImportError: cannot import name 'flash_attn_varlen_func' from 'aiter'
  (/usr/local/lib/python3.10/dist-packages/aiter/__init__.py)
(APIServer pid=1) RuntimeError: Engine core initialization failed.
```

四个 TP rank 同时死，`--enforce-eager` 一样死。

**判据**：`validate_configuration()` 只检查配置合法性，**不导入 backend 模块**。
`get_class()` 拿到的是一条 `"module:Class"` 字符串，真正的 import 发生在引擎初始化时。
所以在 1 卡探测作业里看到 "OK" 只是证明了"这个名字存在"，没证明"这个实现能用"。

**根因**：vLLM 0.20 的 `rocm_aiter_mla.py` 需要 aiter 导出 `flash_attn_varlen_func`，
而镜像自带的 aiter 版本没有这个符号。这是**vLLM 与 aiter 的版本错配**，与 gfx936、
`on_mi3xx()`、`PATH` 都无关——即使白名单放行也会死在这里。

**解法 / 代价**：没有配置层面的解法。要么改镜像换 aiter 版本（牵动整个运行时），
要么改 vLLM 源码。按本仓库原则（不改上游、不换镜像）→ **关闭这条路线**。

**教训**：任何"绕过某道闸门"的探测，必须做到"真实起一次引擎"，而不是"拿到类名 + 校验通过"。
在 gfx936 这种上游未覆盖的架构上，后者遗漏的概率很高。

## 50. sglang 压测进度条到 100% ≠ 请求成功；要看 `Successful requests`

**现象**（2026-11-15，job 1123874 `mns32eager` 的 round2）：服务端崩了之后压测客户端继续跑，
进度条一路走到 `100%|`，看起来"这轮终于没崩"。实际日志里：

```
Successful requests:                     31
Benchmark duration (s):                  209.51
```

而正常一轮是 `Successful requests: 128` / `~550–690 s`。

**根因**：`sglang.bench_serving` 的进度条按"已发出并收到响应（含连接失败）的请求数"推进，
服务端 `Connection refused` 时请求**立刻**返回错误，于是进度条飞速到顶。
崩溃后的几十个请求都在几百毫秒内失败，整轮反而比正常轮**快得多**。

**判据**（三个一起看，缺一不可）：
1. `Successful requests` 必须等于 `--num-prompts`（本仓库是 128）；
2. `Benchmark duration` 落在该档位的正常区间（16K/c64 约 550–700 s）；
3. 结果 JSON 里 `mean_ttft_ms > 0` 且吞吐不超过物理上限（`bench.sh` 的校验段）。

**另外**：进度条文件是 `bench/round2_16384_1024_c64.log`，若这一轮的 bench 没跑起来，
读到的可能是**上一轮残留的同名日志**（gotchas #47）。先核对 mtime 再看内容。

**适用结论**：任何"跑完了""100%"的结论，都要用 `Successful requests` 复核一遍。

## 51. Kimi-Linear 16K/c64 的崩溃是**累计状态**问题：每轮重启服务即可稳定出数

**现象**（2026-11-15，vLLM 0.20 + BW1000 ×4 TP）：16K/c64 压测**第一轮永远通过、
第二轮永远 `KERNEL VMFault`**。四个 TP rank 同时崩、地址各不相同。

**排除过的解释**（每一条都有作业号）：
| 假设 | 反证 |
| --- | --- |
| cudagraph 桶边界 | `--enforce-eager`（无 cudagraph）照样崩，job 1123145 |
| 单批规模太大 | `--max-num-seqs` 降到 24 / 16 仍崩，job 1123874 |
| 显存不够 | 崩溃时 KV cache 只用 10–25%，无 OOM |
| 某次 kernel 越界 | 崩点随配置乱跳（29 / 62 / 23 / 16），与 `Running` 无关 |
| aiter MLA kernel 更好 | `ROCM_AITER_MLA` 服务都起不来（`flash_attn_varlen_func` 缺失） |

**真因（重读 job 1120618 的 3 MB 日志得出）**：那次作业用**一个服务进程连跑 6 档序列**，
10:30 启动、**11:02:58** 才崩，期间跑完 10 个档位、约 1200 个请求。
而后续作业只跑 16K/c64 一档（128 请求 ≈ 10 分钟），于是表现为"第二轮必崩"。

**是请求数而不是时钟时长**：job 1124366 的 R1 进程活了 **27 分钟**、只处理 128 个请求，
照样干净通过。所以危险量级在"上千请求"，不在分钟数。
→ **变量是服务进程的累计运行时长/处理量，不是第几轮、也不是并发数。**

**解法（实测有效，job 1124366）**：**每轮压测前 `podman rm -f` 并重启服务**。
**连续四轮各 128/128 全过、0 处 VMFault**，且吞吐比 `--enforce-eager` 更高：

| 轮 | 吞吐 | TTFT | 与 eager 对比 |
| --- | ---: | ---: | --- |
| R1 | 235.14 tok/s | 47.5 s | 页缓存未暖，**不作稳态引用** |
| R2 | 283.53 tok/s | 31.5 s | 吞吐 **+18%**，TTFT **−28%** |
| R3 | 284.62 tok/s | 30.8 s | 同上 |
| R4 | **286.84 tok/s** | **30.6 s** | 同上 |

R2–R4 极差仅 **1.2%**，这才是可引用的稳态区间（283.53–286.84）。

**代价**：每轮多 3–4 分钟冷启动（模型加载 + preflight），换来可重复的有效数据。

**判据**：比较"崩/不崩"时必须记录**该进程已运行分钟数**和**已成功请求总数**，
只写 `Running` 数会误判（本仓库在此浪费了 5 个作业）。

## 52. 「官方未下发」是**假设**，不是事实——T01/T02 基线本来就是公开的

**现象**：11 个 T01–T02 条目的 §6 全部挂着 `[BLOCKED]`，22 个文件里写了 44 处
「官方未下发 `baselines/`」「评测器尚未下发」。整个仓库把这条当成不可解的外部阻塞，
在 `HANDOFF.md` §6 里记为「已决定不管」，直接放弃了 4 个文本类条目的精度验收。

**实际情况（2026-11-15 官方澄清）**：T01–T02 的验收标准**直接采用官方公开基线，
不再单独下发** `baselines/<模型>.json` 与评测器。

**代价**：这不是配置问题，是**把"我没拿到"写成了"官方没发"**。
两者对下一个人意味着完全不同的行动：前者要派人去取，后者是死路。

**判据（以后写阻塞前先问一遍）**：
1. 「官方未下发」有没有**原文引用**？没有就只是推测，措辞必须是「**我们尚未取得**」。
2. 该材料是**公开数据集/公开榜单**吗？GSM8K、MMLU、OCRBench、MMMU 都是公开的——
   就算官方不给配置文件，基线数值本身也查得到。
3. 阻塞描述必须写**解除动作**（谁去做、去哪里取），只写「等下发」等于没有下一步。

**影响**：`docs/entry-spec.md` §6 红线 1 已补注这条政策；各条目 §6/MISSING 已改为
「待取得」而非「等下发」。`verify.py` 的读取路径不变，取到官方数值按原格式落盘即可跑通。

## 53. 「官方有公开分数」≠「可以拿来当验收基线」——三方口径不同，相减就是伪造

**现象**：官方澄清「T01–T02 用官方公开基线即可，不再单独下发配置文件」后，
本仓库第一反应是"去查分数填进 `baselines/` 就能跑通"。查证后发现这条路**走不通**。

**三个口径互不相同**：

| | shot | 模型 | 指标 |
| --- | --- | --- | --- |
| 我们 `verify.py` | **0-shot** greedy | Instruct（我们部署的这个） | GSM8K / plain MMLU |
| Kimi 官方 Table 9 | 未标 shot | Instruct | **只有** MMLU-Redux 86.9 / MMLU-Pro 72.7 |
| Kimi 官方 Table 8 | **8-shot** | **Base** | GSM8K 86.3 |
| MiMo 官方 | avg@1/avg@3 | Distill | **完全没有 MMLU/GSM8K**，全是 SWE/Terminal/Tool |

**为什么必须停下**：0-shot instruct 的成绩去比 8-shot base 的分数，差出来的每一分
都可能是 shot 数或模型版本造成的。减出来的"劣化"不是硬件/框架的锅，
**但它会以"劣化 ≤2pt → 通过"的形式写进 README**——这正是红线 1 要防的
"看起来通过了验收、实际不可比"。**这比不填数字更糟。**

**判据（以后遇到"用公开基线"的说法，先问三句）**：
1. 官方这个分数是**哪个 checkpoint**（base / instruct / 某个中间版）？
2. **几 shot、什么解码**？和我们的评测脚本一致吗？
3. 官方榜单里有**我们要评的那个指标**吗？（Kimi Instruct 表里根本没有 GSM8K，
   MiMo 一个 MMLU/GSM8K 都没有——不是"没找到"，是**没发布**。）

**附带发现**：`verify.py` 里"从 `baselines/<模型>.json` 读取子集/题数/max_tokens"这套设计，
是 2026-11-01 建库时在**假设官方会下发该文件**的前提下写的（commit `f7e1806`），
**该文件从未存在**。所以"取到数值按原格式落盘即可跑通"这句话本身也不成立——
`n` 填多少、`subset` 指哪个文件，仍无官方依据。

**数据通道也不通**：本集群 `huggingface.co` / `github.com` 均 `Connection timed out`；
`modelscope/gsm8k` 仓库只有 `gsm8k.py` 加载脚本，数据要回源到 OpenAI CDN，同样不可达。

→ 结论：§6 保持 `[BLOCKED]`，理由从"等官方发"改为"口径不可比 + 数据不可得"。
详见 `docs/t01-t02-baseline-investigation.md`。

## 54. 启动参数有两份来源时，`submit.slurm` 里的内联命令会盖掉 `env.sh` 声明的配置

**现象**（2026-11-15 排查发现）：`XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__vllm` 条目里，

- `env.sh` 写着 `EXTRA_ARGS="...--max-num-seqs 128"`，注释称「CUDA Graph 保持开启（比 `--enforce-eager` 快 3%~32%）」；
- `run.sh` 用的是 `${CMD[@]} $EXTRA_ARGS` —— 正确，无 `--enforce-eager`；
- **`container/submit.slurm` 却内联了一整条 `vllm ... --enforce-eager` 命令，且根本不调用 `run.sh`。**

也就是说：**README、`env.sh`、`run.sh`、bench 数据全都是"graph 开启"口径，
唯独"一键复现"入口 `submit.slurm` 实际跑的是关闭 graph 的首版配置。**
复现者按 README 说的 `sbatch container/submit.slurm` 跑，会拿到比文档低 3%~32% 的数字，
而且会以为是自己环境的问题。

**判据**：`submit.slurm` 里只要出现 `exec python3 -m ...api_server` 或 `sglang.launch_server`
这类**内联启动命令**，它就和 `run.sh` 构成了两份真值来源，必然漂移。二选一：

1. **推荐**：`submit.slurm` 只负责起容器，服务由 `bash run.sh` 拉起（Kimi SGLang 条目就是这么做的，没有这个坑）；
2. 保留内联命令，但**必须与 `env.sh` 的 `EXTRA_ARGS` 手工保持一致**，且改一处就要改另一处。

**自查命令**（一眼看出有没有漂移）：

```bash
grep -lE "exec python3 -m (vllm\.entrypoints|sglang\.launch_server)" models/*/container/submit.slurm
# 命中的条目逐个核对：内联命令的参数是否等于 env.sh 的 CMD+EXTRA_ARGS
```

**同类隐患**：`run.sh` 的注释和 CMD 也可能是从别的条目复制来的残留——
本仓库多个 `__vllm` 条目的 `run.sh` 头部注释写的是"拉起 SGLang"、CMD 是 `sglang.launch_server`，
真正生效的却是 `submit.slurm` 的内联 vLLM 命令。**注释不可信，只信实际执行路径。**

## 55. `env.sh` 里 export 了参数，但没人读它 —— 三种"看起来配了、其实没生效"

2026-11-15 对 20 个条目做全量参数审计，发现三类**静默失效**，共同点是**没有任何报错**：

### 甲：`EXTRA_ARGS` 从头到尾没人引用

`BAAI__Recon2Reason-*__vllm` 与 `BAAI__RoboBrain2.5-*__vllm`：

```
env.sh:24  export EXTRA_ARGS="--attention-backend triton --trust-remote-code --enable-multimodal"
env.sh:33  export EXTRA_ARGS="--trust-remote-code --dtype bfloat16 --max-num-seqs $MAX_NUM_SEQS"
container/submit.slurm:  grep -c EXTRA_ARGS → 0
```

三重失效叠在一起：

1. **同一变量 export 两次**，前一行被后一行整个覆盖；
2. **`submit.slurm` 的内联启动命令压根不引用 `$EXTRA_ARGS`**（它是把参数一条条写死的）；
3. 于是 `env.sh:25` 那句"必须显式 `--attention-backend triton`，否则首个 decode 请求崩溃"
   **从未被执行过**——而服务照样正常跑，因为那本来是 0.20 镜像的限制，这两个条目后来换成了 0.21。

**为什么没人发现**：参数失效不会报错，只会让"我以为设了"和"实际没设"长期分叉。
直到某天要调一个旋钮，才发现调了没反应。

### 乙：`submit.slurm` 内联命令与 `env.sh` 漂移（见 #54）

`MiMo × vLLM` 的 `submit.slurm` 留着 `--enforce-eager`，而 README/env.sh/run.sh/bench 都是"graph 开启"口径。

### 丙：日志写到了仓库外，无法核对生效值

`Recon2Reason/RoboBrain × vLLM` 的 `submit.slurm` 把服务端日志写到 `$WORK/server-$JID.log`（仓库外），
仓库内 `logs/server.log` 只有一个占位 JSON。结果：**想知道 `max_num_seqs` 实际是多少，没有证据**。
（SGLang 那两个条目反而有完整 `server.log`，里面能直接读到 `max_running_requests=64`。）

## 判据

**改任何启动参数前，先证明它真的到了命令行上**：

```bash
# 1) env.sh 里定义了但没人用？
for f in models/*/env.sh; do
  n=$(grep -c EXTRA_ARGS "${f%env.sh}container/submit.slurm" 2>/dev/null || echo 0)
  [ "$n" = 0 ] && echo "未被引用: $f"
done

# 2) 同一变量是不是 export 了两次？（后者覆盖前者）
grep -c "^export EXTRA_ARGS" models/*/env.sh | grep -v ':1$'

# 3) 实际生效值从哪看？——只在 logs/server.log 或 $WORK/server-$JID.log 里
grep -oE "max_num_seqs=[0-9]+|max_running_requests=[0-9]+|enforce_eager=[A-Za-z]+" logs/server.log*
```

**规则**：`env.sh` 只应作为**单一真值来源**，`submit.slurm` 必须读它（`$EXTRA_ARGS` 或 `bash run.sh`），
不要内联复制一份参数。内联就必然漂移，这是 #54 的同一个根因。

**附带记录（不是缺陷，是物理上限）**：Recon2Reason/RoboBrain 的 16K/c64 只有 149/141 tok/s，
是因为单卡 KV cache 只有 331145 / 272807 tokens，而 64×17408 ≈ 1.11M，相差 3.4–4.1 倍，
单卡最多同时容纳 15–19 条。提高 `MEM_FRAC` 只多约 12%，救不回来。这是硬件限制，别当配置错误修。

## 56. `scancel -u $USER` 会杀掉自己的**全部**作业——包括正在跑的

**事故**（2026-11-15）：发现一个来历不明的 PENDING 作业（名字是 `df`，`scontrol` 显示
`Command=df`，疑似误把 shell 命令提交成了作业），想清掉它。随手用了
`scancel -u luql`——**按用户取消**，结果：

| 作业 | 提交时状态 | 被杀时的状态 | 损失 |
| --- | --- | --- | --- |
| 1125845 kimi-chunk | RUNNING 26 min | **B 变体刚启动**（A 变体两档已跑完） | B 变体 20 min 白跑 |
| 1126031 t02-chunk | PENDING | 还没开始 | 无损失，重新提交 |
| 1125815 df | PENDING | 目标 | 已达成 |

两个刚提交的验证作业一起被杀，B 变体 20 分钟卡时作废。

**为什么这是高危命令**：`scancel` 不带参数 = 当前 shell 的默认作业（通常为空）；
带 `-u $USER` = **该用户在此调度器上的所有作业，跨所有节点、所有分区、所有状态**。
多个人共用账号、或同一个账号开多个会话时，还会误杀**别人的**会话提交的作业。

**判据**：

```bash
scancel <JOBID>          # 永远带作业号；不知道作业号就先 squeue 查
squeue -u $USER          # 提交前/取消前都看一眼，确认没有不认识的作业
scancel --state=PENDING -u $USER   # 如果确实要清排队中的，限制状态（也不推荐）
```

**另外**：`sbatch` 的 **`--export` 才是传变量的正确方式**。本次重提交用
`sbatch --export=ALL,SKIP_A=1 xxx.slurm`，比在脚本里 hard-code 或改文件头更干净——
skip 类开关应该做成环境变量，被取消的作业重跑时可以**从断点续**（本次 A 变体的
两份 jsonl 留在了 WORK 里，重跑只补 B 变体，省 20 min）。

## 57. Triton 在 DCU 上 JIT 编出的 decode-attention kernel 会 VMFault（Recon/RoboBrain 复现阻塞）

**现象**（2026-11-15）：`BAAI__Recon2Reason-Reasoning-4B` 与 `BAAI__RoboBrain2.5-8B-NV`
的 SGLang 条目**服务能启动、`/health` 却永远 503**，约 20 秒后进程被 abort。
三个不同节点（f11r1n15 / f11r2n02 / f11r2n03）表现完全一致。

**崩溃点（traceback 逐帧）**：

```
radix_attention.py:127          forward
 → triton_backend.py:1199      forward_decode
  → decode_attention.py:771    decode_attention_fwd
   → decode_attention.py:712   decode_attention_fwd_grouped
    → decode_attention.py:612  _decode_softmax_reducev_fwd   ← 在此 JIT 编译
     → triton/runtime/jit.py:861  _do_compile
      → HSA_STATUS_ERROR_EXCEPTION → KERNEL VMFault
```

崩掉的 kernel 是 `_fwd_grouped_kernel_stage1`，grid `x:16384, y:8, z:16`
（16K 序列的分组 decode softmax reducev）。

**已排除**（都有对照）：

| 假设 | 反证 |
| --- | --- |
| cuda graph 捕获 | `--disable-cuda-graph` **同样崩** |
| 启动参数 | 与 2026-11-02 成功日志逐行对齐：`#tokens=331145`、`avail mem=9.44/9.30 GB`、捕获 bs 列表全部一致 |
| 节点故障 | 三个节点表现相同 |
| 显存不足 | KV cache 分配成功，剩余 9.44 GB，与成功时相同 |
| 磁盘 | 另一个失败作业（1126552）确实是 `/` 只剩 27G 导致 `no space left`，但与本次崩溃不同 |

**关键对比**：2026-11-02 那次成功日志里有这一行，本次没有：

```
Failed to load JIT KV-Cache kernel with row_bytes=2048:
  Could not find CUDA installation. Please set CUDA_HOME environment variable.
```

**注意这句报错在 DCU 上是误报** —— DTK 是 HIP/ROCm 体系，根本没有 CUDA。
Triton 没找到工具链就放弃了 JIT，回落到一条不编译的路径，于是能跑通出数据。
现在 JIT **编译成功了**，但编出来的 kernel 在 gfx936 上访存越界。

→ 所以**当年那批 6 档数据是在「JIT 编译失败」的降级状态下产出的**。
复现它要么恢复同样的降级，要么让 Triton 正确指向 DTK 的编译器（`hipcc`）。

**未决**：归档镜像实测 manifest digest `sha256:9d16c7f3…`
与 `image.lock` 记录的 `sha256:db981c46…` 不一致。但 podman `load` 后计算的 digest
与 registry 的 digest 算法可能不同，**尚不能断定镜像被替换**，需进一步核对。

**影响**：这两个条目当前**无法产出任何新数据**，包括本想做的 chunked-prefill 扫描。
历史数据与产物仍然有效（它们是当年在可用状态下产出的）。

## 58. Recon/RoboBrain × SGLang 封存结论：环境回归，非硬件极限，不再投卡时

**处置**（2026-11-15 用户决策）：**封存**。这两个条目的 SGLang 侧不再排查。

## 封存前已确证的排除项

| 假设 | 反证 | 证据来源 |
| --- | --- | --- |
| 节点故障 | 三个不同节点表现一致（f11r1n15 / f11r2n02 / f11r2n03） | 作业 1126056 / 1126225 |
| cuda graph 捕获 | `--disable-cuda-graph` **同样崩** | 作业 1126456 探针 |
| 启动参数 | 与 2026-11-02 成功日志逐行一致：KV tokens 331145、avail mem 9.44/9.30 GB、捕获 bs 列表全同 | `models/BAAI__*/logs/server.log:43-52` |
| 显存不足 | KV cache 分配成功，剩余 9.44 GB | 同上 |
| 磁盘 | 该失败（`no space left`）属另一条独立问题，与 VMFault 无关 | 作业 1126552 |
| Triton 后端选错 | 走的是 **amd** 后端，缓存里有 `.amdgcn`/`.hsaco`，目标架构 **gfx936** 正确 | 作业 1127698 / 1127816 |
| wave 尺寸不匹配 | `HIPOptions` 从 gfx936 推出 warp_size=64，与运行时一致 | Triton 源码 `backends/amd/compiler_hpy` |
| 镜像整体损坏 | **Kimi-Linear-48B 用同一镜像当日跑通**（16K/c64 228.40 tok/s，128/128） | 作业 1126055 |
| Kimi 可作为对照组 | ❌ **不成立**：Kimi 走 `TRITON_MLA` + fla KDA，**从不调用** `decode_attention_fwd_grouped` | `logs/server.log.gz` 有 `MLA optimization is turned on` |

## 最可能的根因（未最终验证）

`tvm-ffi` 的 JIT KV-cache 路径（`store_kvcache`）：

- **11-02**：`_find_rocm_home()` 未识别到 HIP 工具链 → 抛 `Could not find CUDA installation`（日志里逐字存在）→ `can_use_store_cache=False` → **回落** `k_cache[indices] = k` → 正常
- **现在**：`ROCM_PATH=/opt/dtk` 被新版 tvm-ffi 识别 → JIT 编成 → 走进新路径 → 产出在 gfx936 上越界的 kernel

崩溃点固定在 `Capture cuda graph begin` 之后、捕获第一个 batch（bs=64）时，
崩掉的 kernel 为 `_fwd_grouped_kernel_stage1`，grid `x:16384, y:8, z:16`。

**注意 x:16384 这个值可疑**：y=8 对应 head_tiles、z=16 对应 `triton_attention_num_kv_splits` 都对得上，
但 x 是 decode token 数，本配置下应 ≤256；16384 只与 `max_prefill_tokens` 吻合。
→ 若属实，崩的是 **prefill 尺寸的 batch**，指向 SGLang 版本/形状漂移而非 Triton 码选。此项未验证。

## 为什么不再投入

1. **交付不依赖这条路**：两个条目的 vLLM 兄弟条目已在 2026-11-12 产出完整 6 档数据
2. **成本失控**：每轮诊断需重新加载 8.5 GB 镜像，而节点根盘水位在 0~278 GB 间波动，多次因 `no space left` 失败
3. **收益为零**：本想做的 chunked-prefill 扫描预期只有几个百分点，且这两个条目受单卡 KV 上限压制（33 万 tokens vs 64×17K 需 111 万）

## 若将来要重新打开

优先试这两个（agent 分析给出，未验证）：
- `export TVM_FFI_GPU_BACKEND=cuda`（务必无 `CUDA_HOME`）→ 强制复刻 11-02 的"JIT 失败、回落"状态，成功率估 60-70%
- `rm -rf ~/.triton/cache` + `TRITON_CACHE_DIR` 指向本地 + `TRITON_ALWAYS_COMPILE=1` → 排除陈旧 hsaco，估 35-45%

历史数据与产物仍然有效（当年在可用状态下产出）。

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
而且耗时多 5 倍，白烧卡时。

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
# 2) 是不是我们不在分区白名单？
scontrol -a show partition hx1hdnormal | tr ' ' '\n' | grep '^AllowAccounts=' \
  | cut -d= -f2 | tr ',' '\n' | grep -x "$USER"    # 无输出 → 就是它
# 3) 同批次其他账号在不在？（本例 scnethr2601a05 / a07 / scnethr202601 都在，只有 a10 没有）
```

**结论**：`hx1hdnormal` 的 `AllowAccounts` 有 10552 条，我们被移除了。
这不是故障、不是脚本问题、不是 QOS 写法问题（`--qos=partition_hx1hdnormal` 只会得到
`Invalid qos specification`，因为我们 assoc 上挂的是 `user_<账号>`）。

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

**我当初怎么搜的**：`?q=gsplat`（空）、`?q=hyworld`（空）→ 结论"没有"。
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

**教训**：把"我没找到"写成"不可行"，是最容易犯、代价最大的一类错误 ——
它会直接让一个条目被标 blocked 并在报告里写成硬约束。

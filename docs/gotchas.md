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

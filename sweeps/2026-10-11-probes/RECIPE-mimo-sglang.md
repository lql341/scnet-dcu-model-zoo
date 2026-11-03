# MiMo-V2.6-Distill-Qwen-9B on BW1000 + SGLang —— 实测可用配方

状态：**已跑通**（2026-10-29 13:07，job 995884，node f11r1n19）
卡：`BW` / 64.0 GB / `gfx936:sramecc+:xnack-` ／ DTK 26.04 ／ HIP 6.3.26113

## 1. 已验证事实

| 项 | 实测值 |
|---|---|
| SGLang | `0.5.10rc0+das.opt2.alpha.dtk2604` |
| torch | `2.9.0` + hip `6.3.26113` |
| triton | `3.5.1` |
| 模型加载 | 81% VRAM（~52 GB / 64 GB），`--mem-fraction-static 0.8` |
| 冷启动 | **410 s**（含 Triton kernel JIT），`LOAD_TIMEOUT` 建议 1800 s |
| 单流生成 | ~24 tok/s |
| GSM8K 抽查 | ✅ `\boxed{10}` 正确，`finish_reason=stop` |
| 卡功耗 | 541 W / 1000 W |

## 2. 镜像

```
image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
```
- Harbor 匿名可拉，**不需要账号**（8 GiB 压缩 / 32.2 GB 解压 / 35 层）
- 本地已缓存到 `/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar`（31 GB），用 `podman load` 约 70–290 s
- 镜像里自带一份**别人的**同题提交物：`/root/hackathon-mimo/`（env.sh / run.sh / smoke.sh / env_check.sh / env.log / smoke.log / README.md / test_acc.py），可作为报告撰写参考

## 3. 拉镜像 / 载入镜像 必须的参数

```bash
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr --runroot /tmp/prr \
    --storage-driver overlay --storage-opt ignore_chown_errors=true"
```

| 坑 | 现象 | 原因 / 解法 |
|---|---|---|
| 无 `XDG_RUNTIME_DIR` | `invalid internal status` / `cannot re-exec` | 自建目录并 chmod 700 |
| `/etc/subuid` 为空 | `lchown /etc/gshadow: invalid argument` | `--storage-opt ignore_chown_errors=true`（**vfs 不支持该参数**，必须 overlay） |
| certs.d 有 root-only key | `open /etc/docker/certs.d/.../image.ac.com.key: permission denied` | `--tls-verify=false --cert-dir=/tmp/emptycerts` |
| graphroot 放 /public | `lsetxattr /boot: operation not supported` | NFS 不支持 overlay 所需 xattr → **graphroot 必须放节点本地 /tmp** |
| 节点根盘经常 100% | `No space left` | 提交前 `df` 检查，`<45G` 就 `exit 42` 重投 |
| `fuse-overlayfs` | `driver not supported` | 不可用，用 overlay |

## 4. 跑容器看到 GPU 的正确组合（关键）

```bash
podman run --rm --ipc=host \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  --privileged --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v <model_dir>:/models/MiMo:ro \
  <image> bash /inner.sh
```

四条硬性要求，缺一不可：

1. **必须传 `/dev/mkfd`** —— BW1000 上是第二个 KFD 设备，只传 `/dev/kfd` 时 HSA 会 "No HIP GPUs available"
2. **必须挂 host `/opt/hyhal`** —— 提供与内核驱动匹配的 `libhydmi.so` / `libhsa-runtime64.so`
3. **不要覆盖 `/opt/dtk`** —— 用镜像自带的（里面有 `/opt/dtk/aillvm/bin/clang`）；挂 host DTK 会导致
   `ROCm compiler not found under /opt/dtk/aillvm/bin/clang`
4. 容器内 `LD_LIBRARY_PATH` 需补齐：

```bash
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:\
/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:\
/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:\
/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:\
/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:$LD_LIBRARY_PATH
```
（`lightop/lib` 是 `libmsgpackc.so.2` 所在处，漏了会 `ImportError`）

## 5. 启动服务

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_USE_AITER_FP8_ASM_MOE=1
export SGLANG_USE_TRITON_EXTEND_FROM_AITER=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0

python3 -m sglang.launch_server \
  --model-path /models/MiMo --served-model-name mimo-v2.6-9b \
  --host 127.0.0.1 --port 30000 --trust-remote-code \
  --attention-backend triton --mem-fraction-static 0.8 \
  --max-running-requests 16 --context-length 4096 --disable-cuda-graph
```

- `--disable-cuda-graph` **必须**（国产卡上 CUDA Graph 捕获会静默退出）
- `--trust-remote-code` **必须**
- 日志里 `Could not find CUDA installation. Please set CUDA_HOME` 是 JIT KV-cache kernel 的可忽略告警

## 6. 模型

```
/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B   (18 GB, 18 files)
```
从 ModelScope 直连下载，**24 MB/s，3 分钟**（`snapshot_download`，不要走 dl_model.sh 的内置代理，它慢一倍）

## 7. 还差什么才能提交

| 项 | 状态 |
|---|---|
| `run.sh` / `smoke.sh` 产出 | ✅ 可复现 |
| `env_check.sh` → `env.log` | ⬜ 待补（把 §1 的版本表格式化输出） |
| `verify.py` + `baselines/<模型>.json` + `data/` | ⬜ **官方未下发**，需群里要；上一份提交物同样卡在这里 |
| `bench.sh` 6 档 × 2 次 | ⬜ 可跑，但 `--context-length 4096` 装不下 16K 输入档，需调到 ≥ 17K（显存够，64 GB） |
| INT8/INT4 单件 | ⬜ 未做 |
| 镜像推到自己命名空间（push 需 Harbor 凭据） | ⬜ 匿名只能 pull |

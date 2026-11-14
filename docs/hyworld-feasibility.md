# HY-World-2.0 可行性结论（2026-11-14 修订）

## 结论

HY-World-2.0（WorldMirror 2.0 重建部分）在 BW1000 上**可运行**，官方 5 条 DTU 场景全部完成。
此前将其列为「需自编 ROCm 版 gsplat、8–16 卡时、可能失败」的判断不成立，原因是未检索到
Harbor 上已有的社区适配镜像。

## 依据

Harbor 存在 `dcu/acizv7iztz/base/hy-world2-bw-sglang:20261103-rc1`
（digest `sha256:3d436d42be4784a9ffe4b7f0690f16d2657ce2e1408408259196cdb593704aa0`），
镜像 `/tmp/` 下包含完整适配工程：

```
/tmp/hyworld2-bw-source-update.tgz
  ├── upstream/                        官方 hyworld2（含 vendored gsplat 1.5.3 变体）
  ├── patches/0001-bw-compatibility.patch  gsplat 的 HIP 改动
  ├── build_gsplat.sh                  PYTORCH_ROCM_ARCH=gfx936 构建脚本
  ├── sglang_pipelines/hy_world.py     SGLang ComposedPipelineBase 子类
  ├── run_conditions.py                逐条执行官方条件（含素材 sha256 校验）
  ├── wheelhouse/ + fetch_wheels.py    离线依赖
  └── run.sh / smoke.sh / env_check.sh
/tmp/hyworld-gsplat-build/             已编译的 gsplat（build/ + gsplat/）
/opt/hyworld2-bw/deps/                 33 个离线依赖
```

其中 `patches/0001-bw-compatibility.patch` 覆盖了 DCU 适配所需的改动：完整 GLM 头文件、
HIP 设备声明、逻辑线程组归约、hipCUB 别名、设备安全的固定数组索引、内核函数指针转换。
即 gsplat 的 ROCm 构建问题已由社区解决，无需自行编译。

## 实测结果

```
status: completed
  dtu_scan1   completed 121 s
  dtu_scan4   completed  92 s
  dtu_scan9   completed  97 s
  dtu_scan10  completed  97 s
  dtu_scan11  completed 104 s
```

每场景 50 文件 / 74–79 MB，官方 `required_roles` 三类产物齐全：
`points.ply` + `gaussians.ply`（scene）、`camera_params.json`（cameras）、
`renders/view_000..009.png` + `render_mosaic.png`（render），另有 `depth/`、`normal/`。

## 落地时的四个必要条件

| # | 条件 | 说明 |
| --- | --- | --- |
| 1 | `source env.sh` | 该脚本设置 `PYTHONPATH`（upstream + `/opt/hyworld2-bw/deps` + vendored gsplat），缺失时 `import hyworld2` 失败 |
| 2 | `skyseg.onnx`（176 MB） | 镜像内只有 `assets/skyseg-provenance.json`（记录 URL 与 sha256），需从 `hf-mirror.com/JianyuanWang/skyseg` 下载，并在 cwd 建立 `skyseg.onnx` 链接 |
| 3 | 输出目录必须全新 | `run_conditions.py` 对已存在的输出目录直接抛 `FileExistsError` |
| 4 | 素材挂载须为真实目录树 | 条件文件中路径为 `../../data/dtu/...`，若挂载点为符号链接会解析到链接外侧 |

## 资源口径修正

| 早期判断 | 实际 |
| --- | --- |
| 权重 174.7 GB，超出配额 | 174.7 GB 中 169.4 GB 为 HY-Pano（全景生成），与本题无关；本题仅需 WorldMirror 的 5.1 GB |
| 需自编 ROCm 版 gsplat | 社区镜像已包含 HIP 补丁与已编译产物 |
| 需 Python 3.11 + FA2/FA3 | 镜像为 Python 3.10 + 厂商 FlashAttention，实测可用 |

## 未覆盖范围

镜像作者的 README 明确说明：该工程只覆盖 **WorldMirror 2.0 重建**，不含 HY-Pano / WorldStereo
世界生成链路，且原作者的 SGLang Diffusion 端到端验收尚未完成。本次实测的 5 条官方条件
（DTU 多视图重建）在 `run_conditions.py` 的官方 PyTorch 管线上完成，与 SGLang Diffusion
接入路径无关。

## 关联资产

| 资产 | 位置 | 校验 |
| --- | --- | --- |
| 镜像归档 | `luql:archive/zoo-dcu-2026-11/images/hy-world2-bw-sglang_20261103-rc1.tar.gz` | sha256 `3c7ffee6…` |
| 适配工程 | `assets/hyworld2-bw-harness.tar.gz`（198 MB，不含 skyseg.onnx 与 upstream 权重） | sha256 `0b55cfca…` |
| 官方产物 | `products/hyworld2/outputs/` | 20 个渲染图 + 点云 + 相机参数 |
| 条目 | `models/Tencent-Hunyuan__HY-World-2.0__bw1000__sglang/` | lint 通过 |

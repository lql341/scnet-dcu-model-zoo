# HY-World-2.0（WorldMirror 2.0）× BW1000 —— ✅ 官方 5/5 场景完成

> 这条曾经被我判成"跑不动"，**是错的**。2026-11-13 找到 Harbor 上社区作者的现成镜像，
> 内含 gsplat 的 HIP 补丁与完整 harness，1 晚跑通。

## 1. 简介

| 项 | 值 |
| --- | --- |
| 任务 | T03 多视图重建：5 场景 × 10 视图 → 重建文件 + 预测相机 + 展示图 |
| 模型 | **WorldMirror 2.0**（约 1.263B 参数，FP32 safetensors 5.054 GB） |
| 卡型 | BW1000 × 1（bf16，保留官方关键层 FP32 路径） |
| 框架 | 官方 WorldMirror PyTorch 管线，经 SGLang Diffusion 的 `ComposedPipelineBase` 接入 |

**与官方部署方式的差异**：
1. 官方要求 Python 3.11 + FlashAttention-2/3；本条目用 **Python 3.10 + 厂商 FlashAttention**（镜像自带）。
2. `gsplat` 用官方仓库内 **1.5.3 变体** + HIP 补丁（`PYTORCH_ROCM_ARCH=gfx936`），**未改权重与模型结构**。
3. 官方全集 174.7 GB 里 **169.4 GB 是 HY-Pano**（全景生成），与本题无关；本题只需 WorldMirror 的 **5.1 GB**。

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `dcu/acizv7iztz/base/hy-world2-bw-sglang:20261103-rc1` → `20261103-rc1`（见 `image.lock`） |
| Python / DTK | 3.10.12 / DTK 26.04 |
| gsplat | 1.5.3 变体（镜像内已编译 `csrc.so`，sha256 `bacab01e…`） |
| 离线依赖 | 33 个，安装到 `/opt/hyworld2-bw/deps`（不改厂商 torch） |
| 单场景耗时 | **92–121 s**（10 视图，含 gsplat 光栅化渲染） |

## 3. 适配优化方案

### 3.1 基础镜像

`image.ac.com:5000/dcu/acizv7iztz/base/hy-world2-bw-sglang:20261103-rc1` —— **不自建**。
镜像 `/tmp/` 下留着作者的完整适配工程（`hyworld2-bw-source-update.tgz`），已解包复用：
`upstream/`（官方代码 + vendored gsplat）、`patches/0001-bw-compatibility.patch`、
`build_gsplat.sh`、`run_conditions.py`、`wheelhouse/`。

### 3.2 软件包升级

无需额外安装：依赖已在镜像的 `/opt/hyworld2-bw/deps`（33 个，含 onnxruntime / jaxtyping / matplotlib）。
唯一需要外挂的是 **`skyseg.onnx` 176 MB**（`JianyuanWang/skyseg`，sha256 `ab9c34c6…`），
`run_conditions.py` 以相对 cwd 的 `skyseg.onnx` 引用 —— 必须 `ln -sf assets/skyseg.onnx skyseg.onnx`。

### 3.3 代码改动或参数

不改模型代码。踩到并绕过的四点：

| 问题 | 现象 | 解法 |
| --- | --- | --- |
| `hyworld2` 模块找不到 | `ModuleNotFoundError` | 跑之前必须 `source ./env.sh`（它设 PYTHONPATH=upstream + deps + vendored gsplat） |
| `skyseg.onnx` 缺失 | `onnxruntime ... NO_SUCHFILE` | 从 `hf-mirror.com/JianyuanWang/skyseg` 下载 176 MB 放到 `assets/`，并在 cwd 建软链 |
| 输出目录必须全新 | `FileExistsError: Use a fresh output directory` | 每次用 `out-$(date +%H%M%S)` |
| 素材路径 | 条件里是 `../../data/dtu/...` | 挂载点必须是**自包含真实目录**（不能用符号链接，容器内解析会断） |

## 4. 服务启动脚本

无常驻服务（一次性批处理）。`container/submit.slurm`：
拉镜像 → 拷 harness 到工作区 → 容器内 `build_gsplat.sh`（有缓存，约 7 min）→ `env_check.sh`
→ `run_conditions.py` 跑 5 条官方条件 → 打印每个场景的 status 与耗时。

## 5. Smoke 测试

| 检查项 | 结果 |
| --- | --- |
| gsplat 编译 | ✅ 复用镜像内 `csrc.so`（sha256 `bacab01e…`），无需重编 |
| 模型加载 | ✅ 1545 个检查点张量全部匹配；缺的 48 项是按公式生成的 RoPE 周期缓冲区 |
| 单场景 | ✅ `dtu_scan1` **110.8 s** 完成，产出 51 个文件 |
| 3DGS 渲染 | ✅ 每场景 10 个视图 PNG + mosaic |

## 6. 精度测评（官方条件）

官方条件：`conditions/hy_world.jsonl`，**5 条**（DTU scan1/4/9/10/11，每场景 10 视图）。
`expected.required_roles = ['scene', 'cameras', 'render']`。

### 结果：**5 / 5 完成**

| 场景 | 状态 | 耗时 | 产物 |
| --- | --- | ---: | --- |
| dtu_scan1 | ✅ | 120.8 s | 50 文件 / 77.4 MB |
| dtu_scan4 | ✅ | 92.5 s | 50 文件 / 79.0 MB |
| dtu_scan9 | ✅ | 96.7 s | 50 文件 / 73.7 MB |
| dtu_scan10 | ✅ | 97.0 s | 50 文件 / 73.8 MB |
| dtu_scan11 | ✅ | 104.4 s | 50 文件 / 74.3 MB |

每个场景的产物：`points.ply` / `gaussians.ply`（scene）、`camera_params.json`（cameras）、
`renders/view_000..009.png` + `render_mosaic.png`（render）、`depth/`、`normal/`。

**官方明示不评分** → 不计算 DTU Chamfer / PSNR 等指标。入库只存统计与 mosaic 缩略图（378 MB 产物不入库）。

## 7. 性能参考

| 场景 | 耗时 | 备注 |
| --- | ---: | --- |
| 单场景（10 视图，含 gsplat 渲染） | **92–121 s** | 不含 7 min 的 gsplat 编译与模型加载 |
| 5 场景全量 | ~8.5 min（不含构建） | 单卡 |

## 8. 注意事项

- **`source env.sh` 是必须的**：`run_conditions.py` 依赖它设的 PYTHONPATH，否则 `import hyworld2` 直接失败。
- **`skyseg.onnx` 不在镜像里**：只有 `assets/skyseg-provenance.json`（记录 URL 与 sha256），需自己下 176 MB。
- **输出目录不能复用**：脚本对已存在的 out 目录直接 `FileExistsError`。
- **素材挂载不能用符号链接**：条件里 `../../data/dtu/...` 会解析到链接外侧，必须挂真实目录树。
- 这条例目最值得记的教训：**不要把我"没找到镜像"写成"不可行"**（详见 `docs/hyworld-feasibility.md`）。

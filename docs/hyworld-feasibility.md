# HY-World-2.0 在 BW1000 上能不能跑？—— **能，我之前的判断是错的**

> 2026-11-13 更正。此前 `docs/REPORT.md` §4 把 HY-World 列为"硬约束 5：需 ROCm 版 gsplat，
> 8–16 卡时，可能失败"。**这个结论不成立** —— 原因是漏查了 Harbor 上的现成镜像。

## 决定性证据

Harbor 上存在 **`dcu/acizv7iztz/base/hy-world2-bw-sglang:20261103-rc1`**
（与 Music3 同一个社区作者）。镜像里留了他的**完整适配工程**：

```
/tmp/hyworld2-bw-source-update.tgz     ← 源码 + harness（已解包核对）
  ├── upstream/                        官方 hyworld2（含 vendored gsplat 1.5.3 变体）
  ├── patches/0001-bw-compatibility.patch   ← gsplat 的 HIP 改动
  ├── build_gsplat.sh                   PYTORCH_ROCM_ARCH=gfx936，setup.py build_ext --inplace
  ├── sglang_pipelines/hy_world.py      SGLang ComposedPipelineBase 子类（注册表可发现）
  ├── run_conditions.py                 逐条跑官方条件（含 sha256 校验视图、配置校验）
  ├── wheelhouse/ + fetch_wheels.py     离线依赖
  └── run.sh / smoke.sh / env_check.sh
/tmp/hyworld-gsplat-build/             ← 已编译的 gsplat（build/ + gsplat/）
/opt/hyworld2-bw/deps/                 ← 33 个离线依赖（含 jaxtyping、matplotlib、onnxruntime…）
/tmp/hyworld2-bw-results-20261103.tgz  ← 他的构建记录（gsplat-build-v1…v11）
```

他的 README 关键内容：

- **WorldMirror 2.0 重建模型**：约 1.263B 参数，FP32 safetensors **5.054 GB**（单卡 BW 足够）
- gsplat 用官方仓库内 1.5.3 变体，HIP 改动写在 `patches/0001-bw-compatibility.patch`：
  完整 GLM 头文件、HIP 设备声明、逻辑线程组归约、hipCUB 别名、设备安全的固定数组索引、内核函数指针转换
- 检查点 1545 个张量全部存在且形状匹配（缺的 48 项是按公式生成的 RoPE 周期缓冲区）
- **保留厂商 Torch/DTK/FlashAttention，未改权重、模型结构或评测输入**

## 但也没到我原以为的地步

他自己的 README 写的是：

> 当前正在验证 SGLang Diffusion 接入和 3DGS 渲染；**尚不声明赛事验收通过**。
> 这里没有完整 HY-Pano / WorldStereo 世界生成链路。

即：**重建（WorldMirror）+ gsplat 编译这一段是通的，端到端 5 条官方条件他也没跑完**。

## 所以正确的结论

| 我原来的说法 | 实际 |
| --- | --- |
| 「需自编 ROCm 版 gsplat，8–16 卡时，可能失败」 | ❌ 错：gsplat 的 HIP 补丁已有人做好，镜像是现成的 |
| 「建议记为受阻」 | ❌ 过早：应改为「可复用社区镜像，待端到端验证」 |
| 「权重 174.7 GB 放不下」 | ⚠️ 半对：174.7 GB 里 **169.4 GB 是 HY-Pano**，本次只需 WorldMirror 的 **5.1 GB** |

## ✅ 2026-11-14：已经跑通了

```
真总状态: completed
  dtu_scan1  completed 121 s
  dtu_scan4  completed  92 s
  dtu_scan9  completed  97 s
  dtu_scan10 completed  97 s
  dtu_scan11 completed 104 s
每场景 50 文件 / 74–79 MB：points.ply · gaussians.ply · camera_params.json
  · renders/view_000..009.png · render_mosaic.png · depth/ · normal/
```

踩到并解决的四个点（都已写进条目 README §3.3）：
1. 必须 `source env.sh`（PYTHONPATH 指向 upstream + deps + vendored gsplat）
2. `skyseg.onnx` 176 MB 不在镜像里（只有 provenance），要从 hf-mirror 下并软链到 cwd
3. 输出目录必须全新（`FileExistsError: Use a fresh output directory`）
4. 素材挂载必须是真实目录树，不能用符号链接（`../../data/dtu/...` 会解析到链接外侧）

镜像 digest：`sha256:3d436d42…`；harness 已归档 `assets/hyworld2-bw-harness.tar.gz`（198 MB）。

## 当初的判断错在哪

1. **权重** 5.1 GB（WorldMirror 子集；Mac 上正在下，集群登录节点 `import torch` 缺 `libMIOpen` 下不了）
2. **官方素材** `data/dtu`（123 个文件，30 MB，已从赛事 zip 解出）
3. **复用他的 harness**（已解包到 `luql:~/work/hyw-extract/harness/`）+ 该镜像
4. GPU 侧：`bash build_gsplat.sh`（镜像内有编译缓存，预计 10–20 min）→ `run_conditions.py`
5. 预计 1–2 卡时；风险是他自述的「3DGS 渲染尚未验证」

## 教训

**别把"我没找到"写成"不可行"**。这次是搜 Harbor 时只搜了 `gsplat`/`hyworld`（空），
而仓库名是 `hy-world2-bw-sglang`（搜 `world` 才命中）。查现成镜像要用**多个关键词**，
并且优先按"同批用户（acizv7iztz 等）已做了什么"去搜。

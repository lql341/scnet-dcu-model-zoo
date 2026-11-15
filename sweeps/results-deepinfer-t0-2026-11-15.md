# DeepInfer T0 冒烟（BW1000 / gfx936）—— 两次运行的事实与判据

> 背景：这是 2026-11-15 上午开始、交接文档尚未收录的一条支线（`luql:~/deepinfer`，
> DeepSeek 开源的推理框架）。T0 的目标只是**"接得上"**：不编译 vendor 算子、不跑模型。
> 作业：`di-t0` 1121697（12:14）与 1121816（12:28），各 1 卡 46 s，产物在
> `luql:~/work/deepinfer-t0/`。

## 关键前提（已确认）

| 项 | 值 | 来源 |
| --- | --- | --- |
| torch | `2.13.0a0+gitcf30153`，HIP 5.3.26113 | `env.log` |
| 设备 | `gfx936:sramecc+:xnack-`，`is_available=True` | `env.log` |
| DTK | `26.04`（`ROCM_PATH=/public/software/compiler/dtk-26.04`） | `env.log` |
| 节点 gcc 默认 | gcc-9.3.0，其 `libstdc++` **缺** `GLIBCXX_3.4.30` | 第 1 轮报错 |
| gcc-12.2 | 存在，`libstdc++.so.6.0.30` 齐全 | `ls` |

## 第 1 轮（1121697）：环境变量假象

```
ImportError: /public/software/compiler/gcc-9.3.0/lib64/libstdc++.so.6:
  version `GLIBCXX_3.4.30' not found (required by .../libtorch_python.so)
```

第 1 轮脚本往 `LD_LIBRARY_PATH` 里前置了 gcc-12.2 的 `lib64`，但**没能生效**——
报错里仍是 gcc-9.3.0 的路径，说明前置被后续 `module load compiler/dtk/26.04` 覆盖了。
第 2 轮把 `module load` 移到前面，torch 就正常了。

**这条值得记**：`module load` 会重写 `LD_LIBRARY_PATH`，任何"前置 libstdc++"的操作必须放在
`module load` **之后**，否则静默失效（不是报错，而是错误地回落到系统 gcc 的旧 libstdc++）。

## 第 2 轮（1121816）：torch 通了，剩下三处断点

### ① tilelang / tile_kernels 没装上（可修）

```
ERROR: Could not find a version that satisfies the requirement z3-solver<4.15.5,>=4.13.0
       (from tilelang) (from versions: none)
```

`--no-index --find-links ~/softwares/wheels/t0` 报 `from versions: none`，但目录里**当时还没有**
`z3_solver-4.15.4.0` 的轮子（12:30 才补上）。补上后离线解析已通，但暴露出第二层依赖冲突：

```
tilelang 0.1.15  → ml-dtypes  → numpy>=2.0
本地 numpy 是 1.26.4
```

实测 `pip install --no-index --find-links ~/softwares/wheels --find-links ~/softwares/wheels/t0
tilelang==0.1.15 ml-dtypes==0.6.0 "numpy>=2.3.0" --dry-run` **可以解析**（会装 numpy 2.5.2）。
`tile_kernels==2.1.0` 必须加 `--no-deps`：它声明 `torch>=2.13`，而本地 torch 的版本号是
`2.13.0a0` 预发布版，pip 认为不满足。该包是纯 python 轮子，`--no-deps` 安装即可。

### ② `import deepinfer` 成功，但子模块全不到（要查）

```
"import deepinfer": "OK",
"deepinfer.environ": "ModuleNotFoundError: No module named 'deepinfer.environ'",
"deepinfer.platform": "ModuleNotFoundError: No module named 'deepinfer.platform'",
…（model_executor / engine / worker 同样）
```

而仓库里 `ls ~/deepinfer/deepinfer/` **明明有** `environ.py` / `platform.py` / `engine/`。
说明 `import deepinfer` 命中的不是 `/public/home/luql/deepinfer`，而是**别处的同名包**
（venv 里已装过一份，或 `deepinfer/` 同名子目录遮蔽）。探测脚本用了
`sys.path.insert(0, "/public/home/luql/deepinfer")`，但 python 仍可能因为
**已缓存的 `deepinfer` 或其父路径**而解析到别处。

**下一步判据**：打印 `deepinfer.__file__` 与 `deepinfer.__path__`。若指向 venv
`site-packages`，则 T0 测的根本不是待评估的源码树 —— 这比任何 kernel 问题都优先。

### ③ `deepinfer._C` 构建失败：缺 `VIRTUAL_ENV`

```
setup.py:57 RuntimeError: Neither CONDA_PREFIX nor VIRTUAL_ENV is set;
                  a conda env or virtualenv is required to build.
```

第 1 轮卡在 `import torch`，第 2 轮 torch 通了就走到这条。作业里用的是
`$VENV/bin/python setup.py build`，但 `VIRTUAL_ENV` **不会被 `$VENV/bin/python` 自动设置**
（它只在 `source $VENV/bin/activate` 时设置）。

**解法**：`export VIRTUAL_ENV=/public/home/luql/softwares/vllmenv`（或 `source activate`）。
这条是纯环境问题，不是 DeepInfer 的问题。

## 其他已记录事实

- `flash_mla` 导入失败：`undefined symbol: _ZN3c103hip28c10_hip_check_implementationEiPKcS2_ib`
  —— 该 `.so` 是为另一个 torch ABI 编译的（镜像里是 torch 2.9/2.10，venv 是 2.13）。
  与 gotcha #18（wheel 的 cp312 vs 容器 py3.10）同类，属**ABI 不匹配**，不是缺库。
- `deep_gemm` / `deep_ep` / `deep_select` 三个包根本未安装。
- 平台 hack 设了 `DEEPINFER_ACCELERATOR=nvidia`（DeepInfer 的 `__init__` 读 `envs.DEEPINFER_ACCELERATOR`，
  只接受 `nvidia|ascend`，无 DCU 分支）。DCU 走的是 DTK 的 CUDA 兼容层，因此这条路只能"骗过"平台判定，
  `_C` 扩展的 `cuda_utils.cpp` 也要靠 hipcc 的 CUDA 兼容编译 —— **可行性仍未验证**。

## 下一步（按代价排序）

1. `export VIRTUAL_ENV=...` 后重跑 `setup.py build`，看 hipcc 能否编过 `cuda_utils.cpp`；
2. 装 `tilelang==0.1.15 --no-index` + `tile-kernels==2.1.0 --no-deps`，跑 gfx936 的 tilelang JIT 冒烟；
3. 打印 `deepinfer.__file__`，确认测的是源码树；
4. 只有 1–3 全绿，才谈得上跑 DeepSeek-V4.1-Flash（本机 `~/models` 下**没有**该权重）。

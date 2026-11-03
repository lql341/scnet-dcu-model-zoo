# 2026-10 ~ 11 调优过程原件（48 个 slurm + 1 份配方）

这些是**调优过程的原始脚本**，原先只存在于本机 `~/Downloads/pi-test/`，不在版本库里。
按「镜像与经验 > 比赛提交」的第一原则，过程与结论同等重要：**结论在 `docs/gotchas.md`，
过程在这些脚本里**——下次遇到新卡/新框架，照着某一族脚本改两行就能重跑同一条实验链路。

> 定位说明：`models/` 下的是**条目**（已定型、可复现、有报告）；
> 这里是**实验**（未定型、可能互相矛盾、保留失败）。两者不要混。

## 按族索引

| 族 | 文件 | 在干什么 | 结论落在 |
| --- | --- | --- | --- |
| 裸机探测 | `zz_probe{,2,3,4}.slurm`、`p4c/p4d/p5/p6/p7.slurm`、`reg.slurm` | 最早的 DCU 可用性探测：`torch.cuda.is_available()`、`/dev/mkfd`、HSA、`hy-smi` | gotchas #1（`/dev/mkfd`）、`docs/environment.md` |
| 镜像探测 | `img1`–`img13.slurm` | 逐个试 Harbor 镜像：DTK/torch/sglang/vllm 版本、aiter 是否残包、能否起服务 | gotchas #15、`models/*/image.lock` |
| 本地镜像化 | `dev1/dev2/dev3.slurm` | 用已 load 的本地 tar 起服务（省 4–6 min pull） | `docs/image-archive.md` |
| MiMo 引入 | `mimo1`–`mimo4.slurm` | MiMo-9B 在 SGLang 上的最小可用启动与排错 | `models/XiaomiMiMo__*__sglang` |
| 压测链路 | `bench1.slurm`、`bench2.slurm` | 6 档基准脚本的成型过程（含 preflight、有效性校验） | `tools/preflight_lib.py`、gotchas #11 |
| SGLang 调参 | `sweep_mimo.slurm`、`opt_mimo.slurm` | 找出 `--max-running-requests 8→64` + CUDA Graph 的最优组合（+73%～+466%） | `docs/gotchas.md` #2、`HANDOFF.md` §7 |
| Kimi 引入与调参 | `kimi_try1.slurm`、`kimi_bench.slurm`、`sweep_kimi.slurm`、`sweep_kimi2.slurm` | TP=4 起 Kimi-48B；`HIP error: invalid device ordinal` 的定位；CUDA Graph 开/关对比 | gotchas #14、`models/moonshotai__Kimi-*` |
| vLLM 调参 | `vllm_bw.slurm`、`vllm_bench{,2}.slurm`、`sweep_vllm{,2}.slurm`、`vllm_final.slurm` | 去 `--enforce-eager` + `max-num-seqs 128`；双容器压测（vLLM 服务 + SGLang 客户端） | `models/XiaomiMiMo__*__vllm` |
| 架构支持探测 | `probe_w2.slurm` | W2 三个候选模型的 `architectures` 与框架注册情况（Trident 之类） | gotchas #17 |
| 归档 | `archive_images.slurm`、`archive_vllm.slurm`、`verify_archive.slurm` | `podman save | gzip`、MANIFEST、load 回验 | `docs/image-archive.md` |
| 配方 | `RECIPE-mimo-sglang.md` | 2026-10-29 版 MiMo×SGLang 全量配方（现已被条目 README 取代，保留作时间戳） | `models/XiaomiMiMo__*__sglang/README.md` |

## 用的时候

```bash
# 换模型/换镜像只改 env 段，逻辑段（podman 参数、磁盘守门、preflight）不要动
grep -n "IMAGE=\|MODEL\|PORT=" sweeps/2026-10-11-probes/sweep_mimo.slurm
```

四条必须保留的（详见 `docs/gotchas.md`）：

1. `podman run --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro`
2. `--storage-driver overlay --storage-opt ignore_chown_errors=true` + 每作业独立 `--root /tmp/pr-$JOBID`
3. 起服务前 `df` 守门（`NOT_ENOUGH_SPACE` 直接 `exit 42`，交给外层重投循环）
4. 压测前 `MAX_LEN ≥ 20480`，并用 `preflight_lib.py` 挡掉「400 当成功」的假数据

## 已知矛盾（保留不删）

- `sweep_kimi.slurm` 里 unset 了 `HIP_VISIBLE_DEVICES`，导致 `invalid device ordinal`；
  `sweep_kimi2.slurm` 是修正版（保留 Slurm 设的可见性）。**照抄时用 2 不用 1。**
- 各脚本的 `podman run -c "..."` 内联逻辑后来都改成了独立 `container/inner.sh`
  （gotchas 里「$ 被外层展开」那一条）。这里的旧写法保留作为反例。

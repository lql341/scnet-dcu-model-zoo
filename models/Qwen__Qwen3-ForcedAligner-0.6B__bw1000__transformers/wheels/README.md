# 离线 wheel（不入库，仅记录清单）

计算节点无公网，`qwen-asr` 及其依赖需在登录节点预先下载后带进容器。
**wheel 本身不入库**（最大 12 MB，超仓库单文件上限），只保留 `stub/` 与本文档。

## 清单（实测可用的版本与平台）

| 包 | 版本 | 说明 |
| --- | --- | --- |
| `transformers` | **4.57.6** | `qwen-asr` 钉死 `==4.57.6`；镜像内是 5.8.1，`check_model_inputs` 签名已变，必须隔离 |
| `huggingface_hub` | **0.36.2** | transformers 4.57.6 要求 `<1.0`；镜像内是 1.11.0，必须隔离 |
| `tokenizers` | **0.22.2** | transformers 4.57.6 要求 `>=0.21,<0.23` |
| `qwen_asr` | 0.0.6 | 提供 `Qwen3ForcedAligner.from_pretrained` 与 `.align()` |
| `soynlp` | 0.0.493 | 韩语分词（同 nagisa，本条目英文评测不触达） |
| `qwen_omni_utils` | 0.0.9 | 音频预处理工具 |

## 关键：平台必须匹配容器

容器是 **Python 3.10 / linux x86_64**，登录节点是 3.12。
直接 `pip download` 会拿到 `cp312` 的 wheel（例如 nagisa），装进容器会失败。
必须显式指定目标平台：

```bash
python3 -m pip download "nagisa==0.2.11" --no-deps -d . \
  --python-version 310 --only-binary=:all: --platform manylinux2014_x86_64
```

## 隔离安装（不污染镜像）

镜像里的 `vllm 0.20.0` 依赖 `transformers 5.8.1`。因此不能用普通 `pip install` 装 4.57.6，
否则会连带影响 vllm。做法是装到独立目录并用 `PYTHONPATH` 前置：

```bash
pip install -q --no-deps --target /opt/fa-libs \
  transformers-4.57.6-py3-none-any.whl huggingface_hub-*.whl tokenizers-*.whl \
  qwen_asr-*.whl soynlp-*.whl qwen_omni_utils-*.whl
cp stub/nagisa.py /opt/fa-libs/nagisa.py     # 见 stub/ 说明
export PYTHONPATH=/opt/fa-libs:$PYTHONPATH
```

## stub/nagisa.py

真实 `nagisa` 依赖 **dyNET**（重 C++ 库），而 `qwen_asr` 里 nagisa 只被
`tokenize_japanese()` 使用（源码里唯一一处 `nagisa.tagging(text).words`）。
本条目评测集是 30 条英文，不触达该路径，故用 stub 替换；若真跑日语会**显式抛错**，不会静默出错。

## 放置位置

集群上放在 `/public/home/scnethr2601a10/wheels/fa310/`，由 `env.sh` 的 `WHEELS_HOST` 指定，
容器内挂载为 `/wheels`。

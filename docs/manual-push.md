# 手动推送镜像到 SCNet 镜像库

本文件给出把现有镜像归到**自有命名空间**、并进 SCNet 镜像库的完整手动流程。

## 为什么需要这一步

| 现状 | 问题 |
| --- | --- |
| 本项目 19 个条目引用的 5 个镜像均属**第三方账号**（`dcu/linshp`、`dcu/wqxyyt06`、`dcu/acizv7iztz`） | 上游可随时删仓库，届时 `podman pull` 失败，提交不可复现 |
| 赛题要求「报告中用到的镜像**提交到** SCNet 镜像库」 | 报告里写的是别人的镜像地址 |

已做的缓解：`image.lock` 锁定 digest，能防"tag 被重指"，**防不住"仓库被删"**。
彻底解法是把镜像归档到自有命名空间。

## 地址与命名（2026-11-11 实测）

| 项 | 值 |
| --- | --- |
| Harbor 地址 | **`image.ac.com:5000`**（内网 IP `10.13.17.207`，仅 HTTPS 5000 端口） |
| 登录节点可达性 | ✅ 域名解析正常、5000 端口通、`/v2/` 返回 401（需认证） |
| 匿名可读 | ✅ 可列项目/仓库/拉镜像；**推送需登录** |
| 个人命名空间 | `image.ac.com:5000/dcu/luql/base/<name>:<tag>` |
| **平台镜像库（比赛用这个）** | `image.ac.com:5000/aihub/dcu/luql/<name>:<tag>` |
| luql 现有仓库数 | **0**（从未推送过，首次 push 会自动建仓库） |

命名是照同批账号的实测惯例（`aihub/dcu/<账号>/<名字>`，例如
`aihub/dcu/zhanghengyu3766/mimo-v2.6-distill-qwen-9b-vllm-bw`）。

**一键脚本**：`tools/push_harbor.sh`（3 个镜像 × 2 个目标，含 digest 校验与本地存储释放）

```bash
# 只需先准备好凭据文件（见下节），然后
ACCOUNT=luql sbatch tools/push_harbor.sh          # 全推
ONLY=m3 ACCOUNT=luql sbatch tools/push_harbor.sh  # 只推 Music3
DRY_RUN=1 bash tools/push_harbor.sh               # 只看命令不执行
```

## 前置：拿 Harbor 凭据

按实测，凭据**不在集群上、也不在 OpenAPI 里**（OpenAPI 的"容器"章节只有 Notebook 实例，无镜像推送 API）。
只能从网页端取，三条路径按成功率排：

1. **个人中心 → 访问控制**（拿 AK/SK 的那一页）—— 找「镜像仓库」/「容器镜像」/「Harbor」/「仓库密码」
2. **镜像详情页** `scnet.cn/ui/aihub/image` → 任意镜像详情 → 「使用此镜像」区块常带 `podman login` 提示
3. **直接用平台账号密码试** —— Harbor 常接 LDAP/SSO：`podman login image.ac.com:5000 -u <平台账号>`

旁证：同批账号 `scnethr2601c20` / `c15` / `c10` / `c28` 在 Harbor 里共有 **46 个仓库**，
说明凭据是每个账号自行获取的。

把密码写进本机文件（**不要贴进聊天**）：

```bash
printf '%s' '你的仓库密码' > ~/.harbor-pass && chmod 600 ~/.harbor-pass
```

## 命名规范（按同批账号的实际做法）

```
image.ac.com:5000/dcu/<账号>/base/<name>:<tag>        # 个人命名空间
image.ac.com:5000/aihub/dcu/<账号>/<name>:<tag>       # aihub 项目 = 平台镜像库可见
```

实测示例（别人的）：`dcu/scnethr2601c20/base/vllm_0.21_notebook`，同时存在于
`aihub/dcu/scnethr2601c20/vllm_0.21_notebook` 与 `aihubprivate/...`。

## 路径 A：在线推送（镜像还在 Harbor 上时，最省事）

在**计算节点**执行（登录节点的 podman 不可用）：

```bash
export XDG_RUNTIME_DIR=/tmp/xdg-$$ && mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
JID=${SLURM_JOB_ID:-$$}
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID \
    --storage-driver overlay --storage-opt ignore_chown_errors=true"

# 1) 登录
$PD login image.ac.com:5000 -u <账号> --password-stdin < ~/.harbor-pass --tls-verify=false

# 2) 拉原镜像
SRC=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
$PD pull --tls-verify=false --cert-dir=/tmp/emptycerts $SRC

# 3) 打自有标签（个人空间 + aihub）
$PD tag $SRC image.ac.com:5000/dcu/<账号>/base/mimo-sglang-bw10000:1.1.1
$PD tag $SRC image.ac.com:5000/aihub/dcu/<账号>/mimo-sglang-bw10000:1.1.1

# 4) 推送
$PD push --tls-verify=false image.ac.com:5000/dcu/<账号>/base/mimo-sglang-bw10000:1.1.1
$PD push --tls-verify=false image.ac.com:5000/aihub/dcu/<账号>/mimo-sglang-bw10000:1.1.1

# 5) 校验：digest 必须与原镜像一致（证明内容未改动）
$PD image inspect image.ac.com:5000/aihub/dcu/<账号>/mimo-sglang-bw10000:1.1.1 --format '{{.Digest}}'
```

> 上传量约 8 GiB/镜像，内网，单镜像 6–10 分钟。

## 路径 B：离线 tar 推送（Harbor 上已没有，但本地有归档）

归档位置：`/public/home/luql/archive/zoo-dcu-2026-11/images/`（luql 账户，项目结项后唯一副本；临时账号 `scnethr2601a10` 已于 2026-11-07 被回收）

```bash
# 1) 还原
gunzip -c /public/home/luql/archive/zoo-dcu-2026-11/images/mimo-sglang-bw10000_1.1.1.tar.gz | $PD load
gunzip -c /public/home/luql/archive/zoo-dcu-2026-11/images/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw_1.0.0.tar.gz | $PD load

# 2) 之后的 tag / login / push / 校验与路径 A 的第 3–5 步相同
```

**注意**：`podman load` 会恢复**原镜像名**（`dcu/linshp/...`），所以必须 `podman tag` 成自己的
路径再 push。

## 路径 C：用 skopeo 直接跨仓库复制（若装了 skopeo）

```bash
skopeo copy --src-tls-verify=false --dest-tls-verify=false \
  --dest-creds <账号>:<密码> \
  docker://image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1 \
  docker://image.ac.com:5000/aihub/dcu/<账号>/mimo-sglang-bw10000:1.1.1
```

省掉本地存储，直接仓库到仓库。

## 推送后要同步改的地方

1. 各条目 `env.sh` 的 `IMAGE=` → 新地址
2. 各条目 `image.lock` 的 `repository` / `tag` / `digest`（digest 不变）
3. 各条目 `README.md` §2 镜像行、§3.1 基础镜像段落 —— **必须写明来源**，参考措辞：

   > 本镜像为 `dcu/linshp/base/mimo-sglang-bw10000:1.1.1`（digest `sha256:db98…`）的归档副本，
   > 内容未做任何修改，仅迁至自有命名空间以避免上游删除导致不可复现。

4. `docs/environment.md` 的镜像表
5. 跑一遍 `./tools/lint_entry.sh` + `python3 tools/gen_index.py`

## ✅ 结论（2026-11-11）：本路径关闭

平台**没有自助获取 Harbor 凭据的接口**（网页端也没有可用的「镜像仓库密码」入口），
用户侧无法取得推送权限。**本项目不再尝试推送到自有命名空间**，改走下面的「保留现状 + 诚实标注」。

- 已写好的一键脚本 `tools/push_harbor.sh` 保留（含 `DRY_RUN=1`），**若将来凭据可得，直接跑即可**
- 已实测：`image.ac.com:5000` 可达、匿名可 pull、推送需登录、luql 在 Harbor 里 0 个仓库
- 风险状态：镜像仍指向第三方账号（`dcu/linshp`、`dcu/acizv7iztz`），**已用 3 份 tar.gz 归档兜底**
  （`/public/home/luql/archive/zoo-dcu-2026-11/images/`，sha256 全核对 + 功能性还原验证通过）

### 外部托管平台评估（2026-11-11 实测连通性，结论：暂不采用）

| 平台 | 集群登录节点 | 本机 | 能否当**运行**镜像源 | 说明 |
| --- | --- | --- | --- | --- |
| HuggingFace 直连 | ❌ 不通 | ✅ 通 | ❌ | 集群被墙；`hf-mirror.com` 虽通但**上传端点 404**（只读） |
| ghcr.io / Docker Hub | ❌ 不通 | ✅ 通 | ❌ | 同上 |
| quay.io | ⚠️ 首页通、`/v2/` 不通 | — | ❌ | 半通，不可用 |
| 阿里云 ACR / 腾讯云 CCR / 华为云 SWR | ✅ **401（通，需登录）** | — | ❌ | 真 OCI registry，但**计算节点无公网**，跑作业时拉不到 |
| SCNet Harbor | ✅ 内网 | — | ✅ **唯一** | 需凭据，平台不提供 → 本路径已关闭 |

**根本约束**：计算节点无公网 → 任何外部平台都只能当"异地备份"，不能当运行镜像源。
因此外部托管**不能解决比赛"镜像提交到镜像库"的要求**，只能降低"两台服务器都被清"的极小概率风险。

**当前决定（2026-11-11）**：备份暂放在郑州 `luql:/public/home/luql/archive/zoo-dcu-2026-11/`
（与 `scnethr` 是不同账号），外部托管待定。若将来要做，最省事的是 HF 私有 dataset（本机上传，需免费 token）。

## 如果最终拿不到凭据

退路是保留现状 + 诚实标注。在 `MISSING.md` 写明：

- 镜像来源为第三方账号 `dcu/linshp`
- 已用 `image.lock` 锁 digest
- 本地有 tar 归档（`/public/home/luql/archive/zoo-dcu-2026-11/images/`，含 sha256）
- 风险：上游删除仓库时不可复现，需从归档恢复

**不要**把别人的镜像写成自己构建的。

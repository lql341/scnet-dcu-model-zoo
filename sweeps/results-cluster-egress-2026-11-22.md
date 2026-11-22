# 集群出网能力实测：无外网（2026-11-22）

## 结论

**`hx1hdnormal` 分区的登录节点与计算节点都没有出网能力，且不是 DNS 故障。**

不是"域名解析失败"而是"根本没有到公网的路径"——用 IP 直连绕过 DNS 同样超时。
因此**不能**在集群上 `snapshot_download` 下载权重，硬编码 IP 也无效。

## 证据（作业 1191852 / 1191861 / 1191888 / 1191934，节点 f11r3n20 / f11r3n19 / f13r4n02）

### 登录节点

| 探测 | 结果 |
| --- | --- |
| `getent hosts www.modelscope.cn` | FAIL |
| `curl https://www.modelscope.cn/` | `Could not resolve host` |
| `curl https://huggingface.co/` | `Could not resolve host` |
| `python3 -c "import modelscope"` | `ImportError: libgalaxyhip.so.5`（容器外无 DTK 库，属正常） |

### 计算节点（`/etc/resolv.conf` 为平台 DNS `11.13.20.240`）

| 探测 | 结果 | 判读 |
| --- | --- | --- |
| TCP 443 → 223.5.5.5 / 114.114.114.114 / 47.96.8.1 / 8.8.8.8 / 202.96.128.86 | **全部超时** | 公网无路由，与 DNS 无关 |
| TCP 53 → 11.13.20.240（平台 DNS） | **OPEN** | 内网通，DNS 端口可达 |
| TCP 53 → 223.5.5.5 / 114.114.114.114 | 超时 | 外部 DNS 不可达 |
| `dig @223.5.5.5` / `@114.114.114.114` / `@8.8.8.8` | `connection timed out` | 同上 |
| `getent hosts www.modelscope.cn` | FAIL | 平台 DNS 不解析外网域名 |
| 网关代理口 10.6.1.254:{3128,8080,1080}、11.6.1.254:{3128,8080} | 全部 closed | 无可用代理 |
| 内网 pip 源 10.16.1.201:{9929,80} | 全部 closed | 与 HANDOFF P1-6（xfuser 装不上）的结论一致 |

## 这是近期变化，不是一直如此

| 时间 | 事件 | 证据 |
| --- | --- | --- |
| 2026-11-05 | Kimi-48B 权重 ModelScope 直连下载成功（17 分 43 秒 / 39 文件） | `models/moonshotai__Kimi-*__sglang/README.md` |
| 2026-11-15 22:48 | DeepSeek-V4.1-Flash 权重下载完成，**85 个文件逐条 OK** | `luql:~/models/DeepSeek-V4.1-Flash-1017/.dl.progress` |
| 2026-11-22 | 上述全部渠道失效 | 本文件 |

`dl_curl.sh`（当时可用的脚本）用 `curl -L -C -` 直连 `https://modelscope.cn/models/.../resolve/master`，
今天用同样命令全部超时。**历史条目的权重获取流程不可照抄。**

> 注意 `dl.log` 里还有一个独立的老问题：登录节点的 `modelscope` CLI 一 import 就崩
> （`ImportError: libmsgpackc.so.2` / `libgalaxyhip.so.5`）。这是**另一件事**，
> 2026-11-15 当时是靠 `dl_curl.sh` 绕开 Python 包、直接 curl 才成功的。现在连 curl 也不通了。

## 现行可行路径（已实测速率）

1. **本机**下载：`curl` 直连 `modelscope.cn`，实测 **16 MB/s**（1.7 GB 文件 110 s）；
2. **OpenAPI** 上传：`scnet.py --region 20091 upload <file> <dir>`，实测 **10 MB/s**（100 MB / 10 s）；
3. 传输量 16.34 GiB → 合计约 **50 分钟**。

局限：OpenAPI `upload` **只接受单个文件**，目录会报
`error: local file is not a file`（实测）。所以要么逐文件传，要么本地打 tar 传一个包再在作业里解包。

## 若将来要重新打开"集群侧下载"

按优先级：

1. 问平台是否有**离线传输/摆渡**通道（对象存储、平台附件、Notebook 侧上传）；
2. 确认是否有其它分区（如历史上用过的 `hx1hcnormal`）能出网；
3. 确认是否能给计算节点加代理（当前网关 5 个常见代理口全 closed）；
4. 以上都不行，就沿用本机下载 + OpenAPI 上传。

## 复现命令

探针脚本原文：`luql:/public/home/luql/work/qwenvl8b/egress_probe.sh`，
输出：同目录 `egress_probe.out`（作业 1191934）。

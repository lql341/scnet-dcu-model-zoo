# 材料状态

## ✅ 已到（2026-11-02）

官方发布 `T03_T04模型可行性方案_框架数据接入评测.zip`，本条目所需材料**齐备**：

| 材料 | 位置 | 说明 |
| --- | --- | --- |
| 条件清单 300 条 | `official/t03-t04/conditions/qwen3_asr.jsonl` | AISHELL-1 test 100(zh) + LibriSpeech test-clean 100(en) + test-other 100(en) |
| 音频素材 | 需向官方原件领取 | 47 MB（aishell 14 M + LibriSpeech 26 M + 其他）；sha256 见 `official/t03-t04/file-manifest.json` |
| 反馈表模板 | `official/t03-t04/反馈表.csv` | |
| 使用说明 | `official/t03-t04/T03_T04模型可行性方案_框架数据接入评测.md` | |

## 官方对本任务的规定（直接影响本条目）

1. **只传 `assets.audio` 与 `params`**，不传参考 `text`，不传 `language` 标签
2. `max_new_tokens = 1024`，`temperature = 0`（greedy），`language = null`（自动识别）
3. **不要求提供服务接口**，不限定框架与调用方式
4. **不进行统一评分和性能比较** → 本条目 `meta.bench = not-required`，不需要 6 档基准
5. 输出：`<id>.txt`（UTF-8），如 `BAC009S0764W0121.txt`
6. 提交：产物 + `反馈表.csv` + 环境说明（模型/权重版本、框架版本、设备型号与数量）
7. **不得把数据集里的参考答案当作测试结果**

## ⬜ 未完成

| # | 事项 | 说明 |
| --- | --- | --- |
| ① | 官方 300 条正式评测 | 当前 `smoke.log` 是自造 TTS 音频的单样本链路验证，**不能替代官方评测** |
| ② | 输出 `<id>.txt` 与反馈表 | 待评测脚本落地 |
| ③ | 环境说明文档 | 待补 |

## 说明：现有 smoke 数据的定位

`assets/asr_test.wav`（macOS TTS 合成，参考文本「甚至出现交易几乎停滞的情况」）只用于验证
音频解码 → 特征 → 编码器 → 解码器 → 文本后处理链路可用，单样本 CER 1/12。

巧合：官方清单第一条 `BAC009S0764W0121` 的参考文本正是这句话（因为二者都出自官方模型卡示例），
但我们的音频是 TTS 合成、**不是同一条录音**，不能当作官方数据使用。

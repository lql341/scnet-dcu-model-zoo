# 官方材料索引

## T03 / T04 模型适配测试数据（2026-10-30 交付）

来源：赛事官方发布的 `T03_T04模型可行性方案_框架数据接入评测.zip`（99 MB）

**入库策略**：条件清单（764 KB）与文件清单/来源记录入库；**音频/图像等素材不入库**（99 MB，
按 `file-manifest.json` 的 sha256 校验后可重新领取）。

```
official/t03-t04/
├── T03_T04模型可行性方案_框架数据接入评测.md   使用说明（主文档）
├── README.md                                   官方原文
├── 反馈表.csv                                  空白模板
├── conditions/                                 9 份条件清单 JSONL（入库）
│   ├── qwen_image.jsonl        180 条（60 输入 × 3 seed）
│   ├── omniweaving.jsonl        60 条
│   ├── wan_i2v.jsonl            60 条
│   ├── music3.jsonl             60 条
│   ├── hy_world.jsonl            5 条
│   ├── auk.jsonl               120 条（40 输入 × 3 seed）
│   ├── auk_flash.jsonl         120 条  ← 主文档表格未列，但目录里有
│   ├── qwen3_asr.jsonl         300 条
│   ├── forced_aligner.jsonl     30 条
│   └── 来源记录.json
├── file-manifest.json                          463 个素材的 sha256
└── provenance/                                 来源与选取规则
```

## 八条关键约束（直接影响条目规范）

1. **不要求提供服务接口** —— 「不限定框架或调用方式，不要求提供服务接口」
2. **不做统一评分和性能比较** —— 原文明确。因此 T03/T04 条目**不需要 6 档基准**
3. **必须返回真实产物** —— 「不能直接把数据集里的参考答案当作测试结果」
4. **产物按 `id` 命名** —— 如 `geneval_00000_s42.png`；多文件则用该 id 建目录
5. **反馈表按 seed 分行** —— 同一条件不同 seed 各占一行
6. **必须附环境说明** —— 模型/权重版本、框架版本、设备型号与数量；「无法取得确切版本时如实注明，不编造」
7. **不得编造版本、结果或运行数据**
8. **参考文本与语言标签不得作为输入** —— ASR 条目明确「仅将 `assets.audio` 输入模型」

## 各模型输出规格（官方固定条件）

| 模型 | 输入数 | 条目数 | 固定输出规格 |
| --- | ---: | ---: | --- |
| Qwen-Image-2.1-Turbo | 60 | 180 | 图像 **2048×2048** |
| HY-OmniWeaving | 20 | 60 | 视频 832×480、81 帧、16 fps |
| Wan2.1-I2V-14B-480P | 20 | 60 | 视频 832×480、81 帧、16 fps |
| MiniMax-Music3 | 20 | 60 | 最多 750 声学帧、32 kHz 立体声 |
| HY-World-2.0 | 5 场景×10 视图 | 5 | 重建文件 + 相机信息 + 展示图 |
| AuK | 40 | 120 | 24 kHz 单声道 |
| Qwen3-ASR-1.7B | 300 | 300 | UTF-8 文本；greedy、max_new_tokens=1024、语言自动识别 |
| Qwen3-ForcedAligner-0.6B | 30 | 30 | 词级起止时间（目前 `BOUNDARIES_PENDING`，不计算准确率） |

生成类 seed：`42 / 2026 / 1234`；识别、对齐、重建：`42`。

## ASR 清单构成（我们已有该条目）

| 数据源 | 语言 | 条数 |
| --- | --- | ---: |
| AISHELL-1 test | zh | 100 |
| LibriSpeech test-clean | en | 100 |
| LibriSpeech test-other | en | 100 |

输入约束：**只传 `assets.audio` 与 `params`，不传 `text` 与 `language` 标签**。

## 校对结果（2026-11-02 入库时实测）

- 9 份清单行数与主文档一致（`auk_flash.jsonl` 为主文档未列但目录存在的第 9 份）
- 抽查 ASR 3 条素材 sha256：全部匹配
- `qwen3_asr` / `forced_aligner` / `auk` 三份清单的素材路径：**零缺失**

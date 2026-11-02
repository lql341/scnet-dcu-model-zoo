# 材料状态

## ✅ 已到（2026-11-02），已完成

| 材料 | 位置 |
| --- | --- |
| 条件清单 30 条 | `official/t03-t04/conditions/forced_aligner.jsonl` |
| 音频（复用 LibriSpeech test-clean） | 官方原件 `data/LibriSpeech/test-clean/`，sha256 见 `file-manifest.json` |
| 反馈表模板 | `official/t03-t04/反馈表.csv` |

## 官方对本任务的规定

1. 输入：`assets.audio` + **正确 text** + `language`
2. 输出：每个单词的开始与结束时间，**注明单位**
3. **`BOUNDARIES_PENDING`**：没有人工词边界，官方明示「**仅收集词级起止时间预测，不计算边界准确率**」，
   也不要求用户补标
4. 不进行统一评分和性能比较 → `meta.bench = not-required`

**因此本条目不存在"精度是否达标"这个判断**——官方当前不评分。这与 T01 的 2% 门槛是两套口径。

## 已完成

- ✅ 30/30 条成功，输出词级时间戳（`official-eval/outputs/<id>.json` 与 `.tsv`）
- ✅ 预测词数 572 = 参考词数 572（说明分词切分与参考文本完全对齐，但**这不等于边界准确率**）
- ✅ 反馈表、环境说明、`eval_result.json` 齐全

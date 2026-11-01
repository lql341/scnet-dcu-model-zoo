# 测试音频

`asr_test.wav`：macOS `say` 合成（语音 Flo，中文），16 kHz / 16-bit / 单声道，103 KB。

- 参考文本：`甚至出现交易几乎停滞的情况`
- 生成命令：
  ```bash
  say -v "Flo (Chinese (China mainland))" -o a.aiff "甚至出现交易几乎停滞的情况"
  afconvert -f WAVE -d LEI16@16000 -c 1 a.aiff asr_test.wav
  ```

选这句话是因为它出现在 Qwen3-ASR 官方模型卡的用法示例里，便于横向对照。
放进仓库是为了让 `smoke.sh` 不依赖外部下载 —— 计算节点无公网。

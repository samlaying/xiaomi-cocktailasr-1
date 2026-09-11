# **Xiaomi-CocktailASR-1—— 目标说话人语音识别模型**

中文 | [English](README_en.md)

---

## 简介

Xiaomi-CocktailASR-1是小米推出的目标说话人语音识别大模型。模型基于大规模多说话人数据训练，在多个多说话人测试集上达到 SOTA 水平，同时在单说话人场景下性能可与主流单人 ASR 模型相媲美。Xiaomi-CocktailASR-1 还具备负样本拒识能力和思维链（Chain\-of\-Thought）推理能力，能够对识别过程进行可解释的推理。

---

## 亮点 🎯

- **🏆 多说话人 SOTA**：在多个主流多说话人测试集（AliMeeting、AMI、LibriMix 等）上达到最优性能。

- **🎯 单说话人兼容**：单说话人测试集上性能与普通 ASR 模型可比。只需同一模型，便能同时兼顾单人与多人场景，无需切换普通 ASR 模型。

- **🚫 负样本拒识**：具备对全非目标说话人音频的拒识能力，有效降低误触发。

- **🧠 思维链推理**：支持 Chain\-of\-Thought（CoT）模式，对识别结果提供可解释的推理过程。

---

## 性能表现

### 多说话人测试集\-模拟（WER% ↓）

|模型|LibriMix 2mix|LibriMix 3mix|LibriSpeechMix 2mix|LibriSpeechMix 3mix|
|---|---|---|---|---|
|**Xiaomi-CocktailASR-1**|**4\.11**|**12\.29**|**2\.90**|**4\.91**|
|Qwen3\-ASR|68\.75|106\.04|92\.17|160\.82|
|Gemini|48\.41|76\.10|30\.69|52\.34|
|StepAudio|71\.23|121\.08|92\.72|164\.66|
|previous SOTA works|4\.84 \[1\]|12\.23 \[1\]|5\.4 \[2\]|7\.6 \[2\]|

> \[1\] Thinking in Cocktail Party: Chain\-of\-Thought and Reinforcement Learning for Target Speaker Automatic Speech Recognition
> 
> \[2\] Conformer\-based target\-speaker automatic speech recognition for single\-channel audio
> 
> 

### 多说话人测试集\-真实（WER% ↓）

|模型|AMI SDM|AliMeeting Far|
|---|---|---|
|**Xiaomi-CocktailASR-1**|**21\.81**|**20\.63**|
|Qwen3\-ASR|38\.18|39\.64|
|Gemini|52\.95|56\.75|
|StepAudio|110\.50|76\.82|
|Whisper Large\-v2|36\.4|\-|
|previous SOTA works|22\.0 \[3\]|27\.5 \[4\] |

> \[3\] SQ\-Whisper: Speaker\-Querying based Whisper Model for Target\-Speaker ASR
> 
> \[4\] A Comparative Study on Speaker\-attributed Automatic Speech Recognition in Multi\-party Meetings
> 
> 

### 单说话人测试集（Non-empty WER / FRR ↓）

- 单说话人数据集测试，参考音频为同一人的样本。下表先报告去除空结果后的 WER：

|模型|LibriSpeech|AliMeeting-near|AMI-ihm|WenetSpeech(meeting)|CommonVoice(zh)|
|---|---|---|---|---|---|
|**Xiaomi-CocktailASR-1**|1.73|6.57|8.89|5.81|4.95|
|Qwen3-ASR-1.7b|1.87|6.39|10.56|5.84|5.39|
|StepAudio2|1.58|6.82|37.54|5.46|5.07|
|Whisper Large-v2|2.70|-|16.90|-|26.8|
|Gemini-2.5-pro|6.77|16.10|21.02|27.58|14.01|

- 由于 Xiaomi-CocktailASR-1 模型有拒识能力，因此正样本有小概率误识别为空，导致 WER 有波动；下表报告误拒率（FRR）以说明这一点：

|模型|LibriSpeech|AliMeeting-near|AMI-ihm|WenetSpeech(meeting)|CommonVoice(zh)|
|---|---|---|---|---|---|
|**Xiaomi-CocktailASR-1**|0.36|0.38|0.01|0|0.73|
|Qwen3-ASR-1.7b|0|0|0|0|0|
|StepAudio2|0|0|0|0|0|
|Whisper Large-v2|0|-|0|-|0|
|Gemini-2.5-pro|21.31|14.95|24.30|0|0.003|

### 负样本拒识（拒识率% ↑）

- 负样本集的生成方式为：随机选取与输入音频无关的说话人音频作为参考音频。

|模型|LibriSpeech neg|Aishell neg|Chinese in house neg|
|---|---|---|---|
|**Xiaomi-CocktailASR-1**|79\.59|75\.35|68\.54|
|Qwen3\-ASR|0|0|0|
|Gemini|81\.79|64\.2|54\.7|
|StepAudio|0|0|0|

### 思维链（CoT）效果

|测试集|non\-CoT WER ↓|CoT WER ↓|Δ|
|---|---|---|---|
|LibriMix 2mix|4\.11|3\.87|\-0\.24|
|LibriMix 3mix|12\.287|12\.285|\-0\.001|
|LibriSpeechMix 2mix|2\.90|2\.88|\-0\.02|
|LibriSpeechMix 3mix|4\.91|4\.81|\-0\.1|

## 快速开始

### 环境安装

```Bash
pip install torch torchaudio transformers soundfile
```

### 模型文件结构

从 HuggingFace 下载后，模型目录包含：

```Plaintext
Xiaomi-CocktailASR-1/                           # HuggingFace 实际下载路径
├── config.json                    # MicAsrConfig（含 auto_map、text_config、d2v2_config）
├── configuration_mic_asr.py       # 自定义配置类
├── modeling_mic_asr.py            # 自定义模型类（内联 D2V2 音频编码器）
├── feature_extraction_mic_asr.py  # 音频拼接/特征处理
├── d2v2_config.json               # D2V2 音频编码器配置
├── pytorch_model.bin              # 所有权重（D2V2 + Adapter + LLM）
├── tokenizer.json                 # Tokenizer
└── tokenizer_config.json          # Tokenizer 配置
```

### 单条推理

给定一段参考说话人音频（`ref.wav`）和待识别的单人或多人混合音频（`target.wav`），模型只转录目标说话人的语音。内部会自动完成「参考音频 + 1s 静音 + 目标音频」的拼接。

```Python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Ease3/Xiaomi-CocktailASR-1", trust_remote_code=True, torch_dtype="bfloat16"
).cuda().eval()

# 标准模式：直接返回目标说话人的转录文本
# prompt：Based on the reference speech at the start, only transcribe the target speaker's speech into text.
text = model("target.wav", "ref_speaker.wav")
print(text)
```

`model(target, ref)` 接受音频文件路径、`numpy.ndarray` 或 `torch.Tensor`（16k 单声道，非 16k 会自动重采样）。

### 思维链推理

CoT 模式下，模型会在 `<think>` 标签中输出推理过程，在 `<answer>` 标签中输出最终结果：

```Python
# CoT 模式
# prompt：Based on the reference speech at the start, only transcribe the target speaker's speech into text. Please think step by step and provide a detailed reasoning process in <think> </think>.Please output the final answer in <answer> </answer>.
text = model("target.wav", "ref_speaker.wav", cot=True)
print(text)
```

### 负样本拒识

当参考说话人不在混合音频中时，模型应输出空文本（拒识）。使用方式与标准推理相同，不需要特殊参数：

```Python
# 参考音频对应的说话人不在 target 中 → 模型输出为空
text = model("target.wav", "ref_speaker.wav")
print(text)  # → ""
```

### 批量推理

直接对 5 列 TSV（`utt_id, wav, text, ref_wav, ref_id`）批量转录，无需预拼接——脚本内部逐行调用 `model(wav, ref)`（在模型内部完成 ref+静音+target 拼接），并严格按文件顺序处理：

```Bash
CUDA_VISIBLE_DEVICES=0 python tools/test_batch_scp.py \
    --hf_model_dir Ease3/Xiaomi-CocktailASR-1 \
    --input_scp scp.tsv \
    --output_file out/result.txt
# CoT 模式追加 --cot
```

输出 `out/result.txt`（预测文本）以及同目录 `out/text`（参考文本），便于后续打分。

---

## 模型下载

|模型|下载链接|
|---|---|
|Xiaomi-CocktailASR-1|https://huggingface.co/Ease3/Xiaomi-CocktailASR-1|

---

## 项目结构

```Plaintext
├── README.md
├── README_en.md
├── LICENSE
├── .gitignore
├── tools/
│   └── test_batch_scp.py          # 批量推理：直接读tsv文件，内部调 model(wav, ref)
└── demo/                          # 示例音频（正/负样本）
```

> 模型定义代码与权重发布在 HuggingFace（`Ease3/Xiaomi-CocktailASR-1`），经 `trust_remote_code` 加载，故本仓库不含模型 `.py` 与权重文件。

---

## Demo

### 1. 正样本

| 参考语音 | 待识别语音 |
|:---:|:---:|
| ![ref-1547-1](demo/ref-1547-1.mp4) | ![1552-1](demo/1552-1.mp4) |

**识别结果**：人们设计AI配方的过程本质上主要还是一个不断试错的过程

### 2. 负样本

| 参考语音 | 待识别语音 |
|:---:|:---:|
| ![neg-ref](demo/neg-ref.mp4) | ![neg-asr](demo/neg-asr.mp4) |

**识别结果**：【空】

---

## Citation

```Plaintext
@misc{zhang2026xiaomicocktailasr1technicalreport,
      title={Xiaomi-CocktailASR-1 Technical Report}, 
      author={Yiru Zhang and Hang Su and Lichun Fan and Ying Zeng and Chang Liu and Yifeng Wang and Yuquan Liang and Tao Li and Lian Li and Wenhao Yang and Jian Luan and Cong Zou and Heng Qu},
      year={2026},
      eprint={2609.11274},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2609.11274}, 
}
```

---

## 许可证

本项目基于 Apache License 2\.0 开源。

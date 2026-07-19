# text2semantic

基于 Qwen3.5-2B 的 12Hz 自回归 text-to-semantic 训练代码。TTS 部分同步自
[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 的 `main`
分支，基准提交为 `022e286`。

本目录只保留训练所需内容：

- 12Hz 音频 tokenizer 和训练数据预处理；
- Qwen3.5 talker 主干，以及 Qwen3-TTS code predictor、speaker encoder 的模型与配置；
- 单说话人 SFT 数据集、训练脚本和上游训练说明。

未同步 25Hz tokenizer、CLI/WebUI/API、示例、CI、模型权重和其他推理入口。

## 环境

```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

训练脚本固定使用 BF16 和 FlashAttention 2，需要支持 BF16 的 CUDA GPU。
模型和 tokenizer 可以使用 Hugging Face 仓库名，也可以传入本地目录。训练时会从
`Qwen/Qwen3.5-2B-Base` 初始化 talker 主干，并从 Qwen3-TTS 1.7B 初始化
codec embedding、codec head、code predictor 和 speaker encoder。

## 数据预处理

原始 JSONL 每行需要包含 `audio`、`text` 和 `ref_audio`。`ref_audio`
必须为 24 kHz；单说话人训练建议所有样本使用同一段 `ref_audio`。

```bash
uv run python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

## 训练

```bash
uv run accelerate launch finetuning/sft_12hz.py \
  --base_model_path Qwen/Qwen3.5-2B-Base \
  --tts_init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name speaker_1
```

Qwen3.5 的视觉编码器和语言模型 LM head 不会加载；只加载并训练 24 层
Qwen3.5 text backbone。该 SFT 同时优化第 0 个 codec codebook 的自回归
talker loss，以及其余
15 个 codebook 的 code predictor loss。更完整的数据格式和参数说明见
`finetuning/README.md`。

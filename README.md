# text2semantic

Qwen3-TTS 12Hz 自回归 text-to-semantic 训练代码。代码同步自
[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 的 `main`
分支，基准提交为 `022e286`。

本目录只保留训练所需内容：

- 12Hz 音频 tokenizer 和训练数据预处理；
- Qwen3-TTS talker、code predictor、speaker encoder 的模型与配置；
- 单说话人 SFT 数据集、训练脚本和上游训练说明。

未同步 25Hz tokenizer、CLI/WebUI/API、示例、CI、模型权重和其他推理入口。

## 环境

```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

训练脚本固定使用 BF16 和 FlashAttention 2，需要支持 BF16 的 CUDA GPU。
模型和 tokenizer 可以使用 Hugging Face 仓库名，也可以传入本地目录。

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
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name speaker_1
```

该 SFT 同时优化第 0 个 codec codebook 的自回归 talker loss，以及其余
15 个 codebook 的 code predictor loss。更完整的数据格式和参数说明见
`finetuning/README.md`。

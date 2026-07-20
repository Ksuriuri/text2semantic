# text2semantic

基于预训练 Qwen3.5 的单码本自回归 text-to-semantic 模型。模型只预测
MaskGCT RepCodec 的离散 semantic index，不预测后续 acoustic codebook，也不在本项目
中把 token 解码成波形。

核心约束：

- Qwen3.5 text backbone 从预训练权重加载并全参数训练；
- speech embedding 和 speech output head 独立随机初始化；
- semantic codec 只用于离线生成 `[0, 8191]` 的整数标签；
- speech 词表为 `0..8191`、`BOS=8192`、`EOS/PAD=8193`；
- checkpoint 不包含 codec codebook、speaker encoder 或 acoustic predictor。

## 安装

```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

默认训练使用 BF16 和 FlashAttention 2。没有 FlashAttention 时可传
`--attn_implementation sdpa`。

## 数据预处理

原始 JSONL 每行只需要音频和对应文本：

```json
{"audio":"./data/utt0001.wav","text":"这是一条训练文本。"}
```

使用与 IndexTTS2 一致的 W2V-BERT layer 17 + RepCodec 单码本 pipeline。
W2V-BERT、归一化和量化固定使用 FP32，并按 feature attention mask 去除尾部 padding：

```bash
uv run python finetuning/prepare_data.py \
  --device cuda:0 \
  --w2v_bert_path /path/to/w2v-bert-2.0 \
  --stats_path /path/to/wav2vec2bert_stats.pt \
  --repcodec_config_path /path/to/config.yaml \
  --repcodec_checkpoint_path /path/to/semantic_codec/model.safetensors \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_semantic.jsonl
```

输出会增加一维 `semantic_codes`，并移除旧格式中的 `audio_codes/ref_audio`。

## 全参数训练

先把预处理结果划分为互不重叠的训练集与验证集，并通过环境变量提供 W&B key：

```bash
export WANDB_API_KEY="<your-wandb-api-key>"

uv run accelerate launch finetuning/train.py \
  --base_model_path Qwen/Qwen3.5-2B-Base \
  --train_jsonl train_semantic.jsonl \
  --eval_jsonl eval_semantic.jsonl \
  --output_model_path output \
  --batch_size 2 \
  --lr 2e-6 \
  --num_epochs 3 \
  --gradient_accumulation_steps 4
```

损失只覆盖 semantic codes 和 EOS。文本仅作为 causal prefix，不计算 text LM loss。
训练指标、验证 loss、token accuracy 和 EOS accuracy 写入
`haoyuanhuang22-jcxy/text2semantic` W&B project。API key 不应写进脚本或提交到仓库。

每 500 个 optimizer step 和每个 epoch 保存可恢复 checkpoint。断点续训：

```bash
uv run accelerate launch finetuning/train.py \
  --base_model_path Qwen/Qwen3.5-2B-Base \
  --train_jsonl train_semantic.jsonl \
  --eval_jsonl eval_semantic.jsonl \
  --output_model_path output \
  --resume_from_checkpoint output/checkpoint-step-500
```

## 推理

```python
import torch
from qwen_tts import Text2SemanticModel

model = Text2SemanticModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device="cuda:0",
    dtype=torch.bfloat16,
)
semantic_tokens = model.generate(
    "She said she would be here by noon.",
    max_new_tokens=1000,
    temperature=0.8,
    top_k=30,
)
```

返回值是每条文本对应的一维 token tensor，不包含 BOS/EOS，也不返回音频。

## 测试

```bash
uv run --extra test pytest -q
```

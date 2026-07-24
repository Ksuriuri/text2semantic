# text2semantic

基于预训练 Qwen3.5 的单码本自回归 text-to-semantic 模型。模型只预测
MaskGCT RepCodec 的离散 semantic index，不预测后续 acoustic codebook，也不在本项目
中把 token 解码成波形。

核心约束：

- Qwen3.5 text backbone 从预训练权重加载并全参数训练；
- speech embedding 和 speech output head 独立随机初始化；
- semantic codec 只用于离线生成 `[0, 8191]` 的整数标签；
- 参考音频经冻结 W2V-BERT layer 17 和可训练 Conformer + Perceiver
  压缩为固定 `[32, 1280]` speaker latent；
- speech 词表为 `0..8191`、`BOS=8192`、`EOS=8193`、`PAD=8194`；
- checkpoint 包含 speaker encoder，不包含 W2V-BERT、codec codebook 或 acoustic predictor。

## 安装

```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

默认训练使用 BF16 和 FlashAttention 2。没有 FlashAttention 时可传
`--attn_implementation sdpa`。

## 数据预处理

原始 JSONL 每行需要目标音频和对应文本，并且必须能提供独立于目标音频的
speaker reference。可以显式提供 `ref_audio`，或提供 `speaker_id` 让训练集从同一
speaker 的其他音频中选择参考音频；无法找到独立参考音频的样本会被过滤：

```json
{"audio":"./data/utt0001.wav","ref_audio":"./refs/spk1.wav","text":"这是一条训练文本。"}
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

输出会增加一维 `semantic_codes`，移除旧格式中的 `audio_codes`，并保留
`audio/ref_audio` 供训练时在线提取 speaker 特征。

## 全参数训练

先把预处理结果划分为互不重叠的训练集与验证集。训练脚本会自动加载项目根目录下
被 Git 忽略的 `.env`，也可以通过环境变量覆盖其中的 W&B key：

```bash
export WANDB_API_KEY="<your-wandb-api-key>"

uv run accelerate launch finetuning/train.py \
  --base_model_path Qwen/Qwen3.5-2B-Base \
  --w2v_bert_path /path/to/w2v-bert-2.0 \
  --stats_path /path/to/wav2vec2bert_stats.pt \
  --train_jsonl train_semantic.jsonl \
  --eval_jsonl eval_semantic.jsonl \
  --output_model_path output \
  --batch_size 2 \
  --lr 4e-5 \
  --new_module_lr 2e-4 \
  --max_train_steps 100000 \
  --gradient_accumulation_steps 4
```

损失只覆盖 semantic codes 和 EOS。文本仅作为 causal prefix，不计算 text LM loss。
训练时冻结的 W2V-BERT 在线产生可变长 `[T,1024]` 特征；随机初始化且可训练的
Conformer + Perceiver 将其压缩为 32 个固定 speaker token。文本采用 Fish Speech
风格的 ChatML：system 指令为 `Speak out the provided text.`，随后是 user 文本和
assistant generation prompt。模板显式构造，不调用 Qwen3.5 默认 chat template，
因此不会插入 `<think></think>`；speech BOS 对应 voice 模态起点。

每条训练序列按
`[speaker_bos][speaker×32][speaker_eos][ChatML text][speech_bos/codes]`
组织，再在整个序列右侧 padding。批量推理的完整 prompt 改用左 padding，使最后
一个有效 token 对齐。所有 padding 都由 attention mask 排除，label padding 为
`-100`。
Qwen backbone 默认学习率为 `4e-5`；随机初始化的 speech embedding/head、speaker
encoder、speaker projection 和 speaker boundary embedding 默认使用 `2e-4`。同一个
cosine warmup scheduler 会分别衰减各参数组的学习率，并保持二者比例。

训练指标、验证 loss、token accuracy 和 EOS accuracy 写入
`haoyuanhuang22-jcxy/text2semantic` W&B project。API key 不应写进脚本或提交到仓库。

默认每 1000 个 optimizer step 保存一个可恢复 checkpoint，只保留最新 2 个普通
`checkpoint-step-*`；每 10000 step 额外保存一个 `checkpoint-keep-step-*`，不会自动删除。
断点续训：

```bash
uv run accelerate launch finetuning/train.py \
  --base_model_path Qwen/Qwen3.5-2B-Base \
  --w2v_bert_path /path/to/w2v-bert-2.0 \
  --stats_path /path/to/wav2vec2bert_stats.pt \
  --train_jsonl train_semantic.jsonl \
  --eval_jsonl eval_semantic.jsonl \
  --output_model_path output \
  --resume_from_checkpoint output/checkpoint-step-1000
```

## 推理

```python
import torch
from qwen_tts import Text2SemanticModel

model = Text2SemanticModel.from_pretrained(
    "output/checkpoint-epoch-2",
    w2v_bert_path="/path/to/w2v-bert-2.0",
    stats_path="/path/to/wav2vec2bert_stats.pt",
    device="cuda:0",
    dtype=torch.bfloat16,
)
semantic_tokens = model.generate(
    "She said she would be here by noon.",
    ref_audio="./refs/spk1.wav",
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

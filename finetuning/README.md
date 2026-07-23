## Qwen3.5 single-codebook text-to-semantic training

### Data contract

Raw JSONL:

```jsonl
{"audio":"./data/utt0001.wav","ref_audio":"./refs/spk1.wav","text":"其实我很善于观察别人的情绪。"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon."}
```

Prepared JSONL:

```jsonl
{"audio":"./data/utt0001.wav","ref_audio":"./refs/spk1.wav","text":"其实我很善于观察别人的情绪。","semantic_codes":[52,481,709]}
```

`semantic_codes` must be a non-empty one-dimensional list with values in
`[0, 8191]`. `ref_audio` is the preferred speaker reference and falls back to
`audio` when omitted. Legacy 16-layer `audio_codes` are not used.

### Extract semantic labels

The extractor exactly follows the semantic side of the IndexTTS2 MaskGCT
pipeline:

1. load mono audio at 16 kHz;
2. compute SeamlessM4T features;
3. take W2V-BERT hidden layer 17;
4. normalize with the supplied training statistics;
5. call the frozen, single-codebook RepCodec quantizer.

All semantic extraction stages run in FP32. Codes corresponding to padded
feature frames are removed using the feature attention mask.

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

The codec is a label generator only. Its vector-quantizer embeddings are never
copied into or shared with the autoregressive model.

### Train

Split the prepared data into disjoint train and evaluation JSONL files. The
training script automatically loads the Git-ignored project-root `.env`;
an exported environment variable can override it:

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
  --lr 2e-6 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --gradient_accumulation_steps 4
```

The Qwen3.5 backbone is loaded from pretrained weights. The independent
8194-entry speech embedding, 8194-class output head, and IndexTTS2-style
Conformer + Perceiver speaker encoder are random and trainable. A frozen
W2V-BERT layer 17 front end runs online in FP32 and the speaker encoder maps
its variable `[B,T,1024]` output to fixed `[B,32,1280]` latents. After
projection, the text prefix uses the Fish Speech-style ChatML system prompt
`Speak out the provided text.`, followed by the user text and assistant
generation prompt. It is rendered explicitly instead of using Qwen3.5's
default chat template, so no `<think></think>` block is inserted. The full
training sequence is:

```text
[speaker_bos][speaker x32][speaker_eos][ChatML text][speech_bos/codes][right padding]
```

Batch inference left-pads the complete prompt so every sample's final valid
token is aligned for autoregressive generation. Speech BOS is the local
equivalent of Fish Speech's `<|voice|>` boundary.
Teacher forcing uses:

```text
speech input:  [BOS, code_0, ..., code_n]
speech target: [code_0, ..., code_n, EOS]
```

Padded target positions use `-100`; no loss is applied to the text prefix.
Gradient checkpointing is enabled by default. Use
`--no-gradient_checkpointing` only when memory allows it. If FlashAttention 2
is unavailable, pass `--attn_implementation sdpa`.

Training loss/LR and validation loss, semantic-token accuracy, and EOS accuracy
are logged to the `text2semantic` project under the
`haoyuanhuang22-jcxy` W&B entity.

Checkpoints contain the complete Qwen3.5 backbone, trained speech parameters,
speaker encoder, tokenizer, model config, and generation defaults. They do not
contain the frozen W2V-BERT front end, code predictor, acoustic codebook, or
waveform decoder.
Every checkpoint also contains an `accelerator_state/` directory with model,
optimizer, scheduler, scaler, and RNG state. Resume without resetting the
epoch or dataloader position:

```bash
uv run accelerate launch finetuning/train.py \
  --base_model_path Qwen/Qwen3.5-2B-Base \
  --w2v_bert_path /path/to/w2v-bert-2.0 \
  --stats_path /path/to/wav2vec2bert_stats.pt \
  --train_jsonl train_semantic.jsonl \
  --eval_jsonl eval_semantic.jsonl \
  --output_model_path output \
  --resume_from_checkpoint output/checkpoint-step-500
```

## Qwen3.5 single-codebook text-to-semantic training

### Data contract

Raw JSONL:

```jsonl
{"audio":"./data/utt0001.wav","text":"其实我很善于观察别人的情绪。"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon."}
```

Prepared JSONL:

```jsonl
{"audio":"./data/utt0001.wav","text":"其实我很善于观察别人的情绪。","semantic_codes":[52,481,709]}
```

`semantic_codes` must be a non-empty one-dimensional list with values in
`[0, 8191]`. `ref_audio` and 16-layer `audio_codes` are not used.

### Extract semantic labels

The extractor exactly follows the semantic side of the IndexTTS2 MaskGCT
pipeline:

1. load mono audio at 16 kHz;
2. compute SeamlessM4T features;
3. take W2V-BERT hidden layer 17;
4. normalize with the supplied training statistics;
5. call the frozen, single-codebook RepCodec quantizer.

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

```bash
uv run accelerate launch finetuning/sft_12hz.py \
  --base_model_path Qwen/Qwen3.5-2B-Base \
  --train_jsonl train_semantic.jsonl \
  --output_model_path output \
  --batch_size 2 \
  --lr 2e-6 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --gradient_accumulation_steps 4
```

The Qwen3.5 backbone is loaded from pretrained weights. The independent
8194-entry speech embedding and 8194-class output head are random. All three
parts are trainable. Teacher forcing uses:

```text
speech input:  [BOS, code_0, ..., code_n]
speech target: [code_0, ..., code_n, EOS]
```

Padded target positions use `-100`; no loss is applied to the text prefix.
Gradient checkpointing is enabled by default. Use
`--no-gradient_checkpointing` only when memory allows it. If FlashAttention 2
is unavailable, pass `--attn_implementation sdpa`.

Checkpoints contain the complete Qwen3.5 backbone, random speech parameters
after training, tokenizer, model config, and generation defaults. They contain
no speaker encoder, code predictor, acoustic codebook, or waveform decoder.

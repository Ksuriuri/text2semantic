## Qwen3.5 single-codebook text-to-semantic training

### Data contract

Raw JSONL:

```jsonl
{"audio":"./data/utt0001.wav","ref_audio":"./refs/spk1.wav","text":"其实我很善于观察别人的情绪。"}
{"audio":"./data/utt0002.wav","speaker_id":"spk1","text":"She said she would be here by noon."}
```

Prepared JSONL:

```jsonl
{"audio":"./data/utt0001.wav","ref_audio":"./refs/spk1.wav","text":"其实我很善于观察别人的情绪。","semantic_codes":[52,481,709]}
```

`semantic_codes` must be a non-empty one-dimensional list with values in
`[0, 8191]`. `ref_audio` is the preferred speaker reference. When it is
omitted, training chooses a different utterance from the same `speaker_id`.
Samples without an independent reference audio are filtered. Legacy 16-layer
`audio_codes` are not used.

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
  --lr 4e-5 \
  --new_module_lr 2e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --num_epochs 1 \
  --gradient_accumulation_steps 4
```

To start a fresh finetune from a complete Text2Semantic checkpoint, pass
`--init_model_path /path/to/checkpoint`. This restores the trained model and its
tokenizer but intentionally starts a new optimizer, learning-rate schedule,
dataloader position, and W&B run. Use `--resume_from_checkpoint` only to resume
an interrupted run with its accelerator state.

Language and affect controls are enabled at loader time, so the original
manifest text stays unchanged:

```bash
uv run accelerate launch finetuning/train.py \
  --init_model_path /path/to/text2semantic-checkpoint \
  --base_model_path Qwen/Qwen3.5-2B-Base \
  --w2v_bert_path /path/to/w2v-bert-2.0 \
  --stats_path /path/to/wav2vec2bert_stats.pt \
  --train_jsonl /path/to/manifests/train.jsonl \
  --eval_jsonl /path/to/manifests/eval.jsonl \
  --output_model_path output-affect \
  --language_tag_prob 0.6 \
  --emotion_conditioning \
  --emotion_synonyms /path/to/emotion_synonyms.v3.json \
  --emotion_synonym_prob 0.7 \
  --num_epochs 1
```

This adds one atomic token per supported language plus the two emotion boundary
tokens. Training redraws the optional language tag and emotion synonyms on each
read. Evaluation keeps a stable language-tag draw and disables synonym
augmentation. Checkpoints save the expanded tokenizer alongside the resized
text embedding, and inference accepts explicit `language` and `emotion`
controls.

Run length comes from exactly one of `--num_epochs` or `--max_train_steps`;
passing neither is an error. Warmup and the cosine decay are sized off whichever
is given, so `--num_epochs 1` keeps the schedule matched to the manifest as the
trainset grows, while `--max_train_steps` is for deliberately training a fixed
fraction of an epoch. Setting one to a number the other contradicts is what used
to decay the learning rate to zero partway through a run.

Both numbers count **optimizer steps of the sharded run**, which is not what
`accelerator.prepare` assumes: a prepared scheduler is advanced once per process
per optimizer step, so the cosine is built for `steps x num_processes` scheduler
steps (`schedule_steps_for_accelerate`). The startup banner prints both counts —
check them, because getting this wrong scales with the cluster: on 32 processes
the LR used to reach zero about 6,200 steps into a 199,723-step epoch and stay
there. A checkpoint written before that fix carries scheduler state in the old,
unmultiplied space, so resuming it on the fixed code makes the LR jump; start
such a run fresh instead.

`--lr_schedule` picks the shape after warmup: `cosine` (default) decays to zero
over the run length, `constant` holds the peak LR forever. Use `constant` when
the run length is a budget decision rather than a convergence plan -- stopping
early or adding steps then does not retroactively change the LR of the steps
already taken, which a cosine sized off the old length does. Warmup is scaled
for the prepared scheduler either way.

The Qwen3.5 backbone is loaded from pretrained weights. The independent
8195-entry speech embedding, 8195-class output head, and IndexTTS2-style
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

The Qwen backbone uses `--lr` and the randomly initialized speech/speaker
modules use `--new_module_lr`. The cosine warmup scheduler decays each
parameter group's own learning rate, so the default `4e-5` and `2e-4` rates
decay together while keeping their ratio.

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
  --resume_from_checkpoint output/checkpoint-step-1000
```

A rolling `checkpoint-step-*` is saved when two conditions hold at once: the
global step is a multiple of `--checkpointing_steps` (default 50) *and* at least
`--checkpointing_min_interval_minutes` (default 30) have passed since the last
save. The step multiple sets the granularity, the interval sets the rate; only
the latest `--checkpoint_total_limit` (default 2) are kept. Every
`--keep_checkpointing_steps` (default 5000) steps a `checkpoint-keep-step-*` is
saved instead, and those are never rotated. A `SIGTERM` or `SIGUSR1` forces one
last checkpoint off schedule and then stops the run cleanly. A normally
completed run also saves its exact terminal step even when it falls between
rolling intervals; this matters for short finetunes shorter than the time gate.

On preemptible hardware add `--checkpoint_remote_dir gs://bucket/prefix`. Each
save is then mirrored to object storage, every node uploading the files it
wrote, and a `_UPLOAD_COMPLETE` marker is written last so a resume can never
pick up an upload that a preemption cut short. With a remote prefix configured,
`--resume_from_checkpoint auto` reads the newest complete *remote* checkpoint
and ignores local directories: a node that survived a preemption may still hold
a checkpoint whose upload never finished, and two nodes resuming from different
weights is not something DDP detects.

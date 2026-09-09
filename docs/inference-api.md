# Asynchronous TTS inference API

The API entrypoint is `scripts/api_server.py`. The existing WebUI remains the
HF baseline. Training checkpoints are unchanged; vLLM uses a separate export.

## Install and export

Use a separate environment with `pip install -e '.[infer,vllm]'`.
The adapter targets **vLLM 0.21.0** and registers a text-only speech Qwen3.5
class through its plugin entrypoint. Use one visible GPU per process initially;
select it with `CUDA_VISIBLE_DEVICES`, and keep `--device cuda:0`.

```bash
python -m qwen_tts.inference.vllm_export /models/training-checkpoint /models/vllm-speech
python scripts/api_server.py \
  --checkpoint /models/training-checkpoint \
  --ar-backend vllm --vllm-model /models/vllm-speech \
  --indextts-root /models/indextts-2.5 --codec-dir /models/indextts-2.5/checkpoints \
  --bigvgan-dir /models/indextts-2.5/checkpoints/bigvgan \
  --w2v-bert-path /models/indextts-2.5/checkpoints/w2v-bert-2.0 \
  --stats-path /models/indextts-2.5/checkpoints/wav2vec2bert_stats.pt \
  --semantic2any-root /models/semantic2any --s2vae-config /models/s2vae/config.yaml \
  --s2vae-checkpoint /models/s2vae/s2mel.pth --dots-tts-dir /models/dots-tts \
  --acoustic-batch-size 8 --batch-wait-ms 5 --max-concurrent 16 \
  --vllm-memory-fraction 0.35 --warmup-ref /models/example.wav --port 8081
```

`--ar-backend hf` uses the existing AR path for comparison. It serializes HF
sampling to protect process-global RNG; acoustic requests still use the batch
queue. Start one uvicorn worker: extra workers duplicate all model allocations.
Tune memory fraction and batch size together. CFG doubles estimator batch
capacity; it does not represent an extra independent request.

## Requests

Set `TTS_API_KEY` to require `Authorization: Bearer ...` on inference and deep
health endpoints. Never put real keys in examples or source files.

```bash
curl http://localhost:8081/tts \
  -F 'synthesis_text=你好，欢迎使用语音合成。' \
  -F 'wav=@reference.wav' -F vocoder_backend=s2vae --output speech.wav
```

`/api/tts` and `/api/generate` accept the same request; multipart `text` and
`ref_audio` are accepted aliases. JSON uses `synthesis_text`, `wav_base64`,
`vocoder_backend`, optional `language`, `emotion`, `temperature`, `top_k`,
`max_new_tokens`, `repetition_penalty`, and `seed`. Single responses are WAV,
with `X-Sample-Rate`, `X-Inference-Ms` and `X-Request-ID` headers. This returns a
completed waveform, not incremental streaming audio.

`POST /tts_batch` accepts either a single synthesis payload plus `repeat_num`
(1–16; seed increments per repetition), or `{"items": [payload, ...]}` for
independent texts/references. Modes cannot be mixed. Returns ordered `results`
with `index`, `audio_base64`, sample rate, timing and per-segment measurements;
failed items contain `error` and `status_code`. Do not assume this schema is
byte-for-byte compatible with the reference project's `audio_base64_list`.
IndexTTS2-only fields such as `duration`, `spk_id`, `emo` and `quality_preset`
are rejected rather than silently changing synthesis semantics.

`GET /health` (alias `/api/health`) checks model worker state without inference.
`GET /health/deep` performs a short synthesis with `--warmup-ref`, caching the
result for 30 seconds. Without a probe reference it returns 503. Engine failure
or acoustic device failure makes health non-green. A process supervisor should
restart an unhealthy worker; this service does not create/delete cloud resources.

## Batch behavior and limits

The AR engine dynamically schedules independent requests. Acoustic microbatches
combine different references and lengths with per-item prompt/valid masks and
per-request seeded noise. Requests are limited by queue capacity, code count and
estimator sequence capacity. Long input is split before AR; each generated
segment is decoded and reassembled in source order.

CFM is truly batched. Codec/reference preprocessing remains individual to preserve
context and normalization. BigVGAN/AudioVAE decoding batches equal-length outputs;
other lengths form separate groups because padding can change convolutional
boundaries. It is not claimed that every decoder invocation contains every item.
Flow/BigVGAN remain FP32. No quantization is introduced. Cancellation stops AR;
an already-running acoustic/CPU worker finishes before request files are removed.
Thus cleanup can outlast the configured request deadline.

The weight export preserves speech IDs, output head, and speaker/text prefill.
Random sampling across HF/vLLM is not promised bit-identical despite equal seeds.
Performance and waveform parity require real checkpoint tests on the selected
hardware; CPU unit tests alone are insufficient evidence of speedup.

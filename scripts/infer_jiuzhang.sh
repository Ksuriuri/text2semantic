#!/usr/bin/env bash
# Ready-to-run wrapper for Jiuzhang. Does not launch training GPUs itself;
# pass --device cuda:N to pick a card with spare VRAM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${TEXT2SEMANTIC_PYTHON:-/hunshan/hhy/envs/text2semantic-py310/bin/python}"
CKPT="${T2S_CHECKPOINT:-/hunshan/hhy/workspace/text2semantic/output_v1/checkpoint-keep-step-10000}"
if [[ ! -d "$CKPT" ]]; then
  CKPT="/hunshan/hhy/workspace/text2semantic/output_v1/checkpoint-step-10000"
fi
exec "$PY" "$ROOT/scripts/infer.py" \
  --checkpoint "$CKPT" \
  --indextts-root /hunshan/hhy/workspace/indextts-new/index-tts \
  --codec-dir /hunshan/hhy/models/IndexTTS-2.5 \
  --bigvgan-dir /hunshan/hhy/models/IndexTTS-2.5/bigvgan \
  --w2v-bert-path /hunshan/hhy/models/IndexTTS-2.5/w2v-bert-2.0 \
  --stats-path /hunshan/hhy/models/IndexTTS-2.5/wav2vec2bert_stats.pt \
  "$@"

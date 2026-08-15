#!/usr/bin/env bash
# Portable WebUI launcher for text2semantic + IndexTTS-2.5.
# Paths come from the environment so the same script works on any machine.
#
# Required:
#   T2S_CHECKPOINT   HF dir with model.safetensors + tokenizer (infer-only is enough)
#   INDEXTTS_ROOT    IndexTTS-2.5 Python checkout (contains indextts/)
#   CODEC_DIR        weights dir: codec.pth, s2mel.pth, config.yaml, campplus,
#                    wav2vec2bert_stats.pt, w2v-bert-2.0/, bigvgan/
#
# Optional:
#   TEXT2SEMANTIC_PYTHON   python binary (default: python)
#   T2S_DEVICE             cuda:0 / cpu (default: cuda:0)
#   WEBUI_HOST             default 0.0.0.0
#   WEBUI_PORT             default 7860
#   WEBUI_OUT_DIR          default ./webui_outputs
#   BIGVGAN_DIR W2V_BERT_PATH STATS_PATH
#     override if those files are not inside CODEC_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${TEXT2SEMANTIC_PYTHON:-python}"

: "${T2S_CHECKPOINT:?set T2S_CHECKPOINT to the HF checkpoint directory}"
: "${INDEXTTS_ROOT:?set INDEXTTS_ROOT to the IndexTTS-2.5 code checkout}"
: "${CODEC_DIR:?set CODEC_DIR to the IndexTTS-2.5 weights directory}"

BIGVGAN_DIR="${BIGVGAN_DIR:-$CODEC_DIR/bigvgan}"
W2V_BERT_PATH="${W2V_BERT_PATH:-$CODEC_DIR/w2v-bert-2.0}"
STATS_PATH="${STATS_PATH:-$CODEC_DIR/wav2vec2bert_stats.pt}"
DEVICE="${T2S_DEVICE:-cuda:0}"

exec "$PY" "$ROOT/scripts/webui.py" \
  --checkpoint "$T2S_CHECKPOINT" \
  --indextts-root "$INDEXTTS_ROOT" \
  --codec-dir "$CODEC_DIR" \
  --bigvgan-dir "$BIGVGAN_DIR" \
  --w2v-bert-path "$W2V_BERT_PATH" \
  --stats-path "$STATS_PATH" \
  --device "$DEVICE" \
  "$@"

# 完整推理与 WebUI 部署

本文说明如何在**另一台机器**上跑当前 text2semantic 的完整波形推理：命令行和 WebUI 走同一条链路。
GCP L4 dest 听音盒的实测路径在文末。

## 链路

1. 参考音频 + 目标文本 → 本仓库训练的 text2semantic 自回归模型 → **25 Hz** semantic codes（IndexTTS-2.5 EnhancedCodec，单码本 `[0, 8191]`）
2. `EnhancedCodec.decode(codes)`：decoder + 2× nearest upsample，得到 **50 Hz** 连续特征  
   **不要**把 25 Hz id 或 quantize 的预解码 embedding 直接喂给 s2mel
3. s2mel 的 prompt 用**原始** w2v-bert layer-17 特征（不经过 codec）；target 用 decode 后的 50 Hz 特征；`target_lengths = decoded_T * 1.72`
4. CAMPPlus style + CFM（25 step，cfg 0.7）+ BigVGAN → **22.05 kHz** wav

仓库里的 `Text2SemanticModel.generate()` 只出 codes。旧脚本 `t2s_infer.py`（MaskGCT 50 Hz + IndexTTS-2 s2mel）和当前 25 Hz 训练不兼容，不要用。

## 需要拷到新机器的东西

推理 checkpoint **不需要** optimizer / accelerator_state。一份可用的目录大约 **4.4G**：

```
checkpoint-step-21000/
  model.safetensors
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
```

trainer 前缀在 S3 上是 18.53 GB / 75 个对象。听音盒磁盘紧的话只拉上面这些。

另外两份外部依赖（本仓库不包含）：

| 路径 | 内容 |
|---|---|
| `INDEXTTS_ROOT` | IndexTTS-2.5 代码，需能 `import indextts`（含 `EnhancedCodec`、s2mel、CAMPPlus、BigVGAN） |
| `CODEC_DIR` | `codec.pth`、`s2mel.pth`、`config.yaml`、`campplus_cn_common.bin`、`wav2vec2bert_stats.pt`、`w2v-bert-2.0/`、`bigvgan/{config.json,bigvgan_generator.pt}` |

`bigvgan/`、`w2v-bert-2.0/`、`wav2vec2bert_stats.pt` 可以和 IndexTTS-2 共用；`codec.pth` / `s2mel.pth` 必须是 2.5 的。

建议磁盘布局：

```
/opt/t2s/
  text2semantic/              # 本仓库
  index-tts/                  # INDEXTTS_ROOT
  IndexTTS-2.5/               # CODEC_DIR
  checkpoint-step-21000/      # T2S_CHECKPOINT
```

显存：2B AR + w2v-bert + codec + s2mel + BigVGAN，建议 **≥ 24 GB**。两份 AR replica（默认）在 L4 24G 上大约 16 / 23 GiB。没有 GPU 可设 `T2S_DEVICE=cpu`，会很慢。

## Python 环境

训练用的 venv 可以复用。新机器最少需要：

```bash
# 本仓库
pip install -e ".[infer]"
# 或 uv pip install -e ".[infer]"

# IndexTTS-2.5 代码里的额外依赖（按那边 README 装）
# 常见缺包：omegaconf einops munch librosa
```

`[infer]` 会装 `gradio`、`fastapi`、`uvicorn`、`python-multipart`、`omegaconf`、`torchaudio`、`munch`。
BigVGAN 从本地 `bigvgan_generator.pt` 加载，wav 用标准库 `wave` 写 PCM16，不依赖 `torchcodec`，也不走 Hugging Face Hub。

## 命令行

```bash
python scripts/infer.py \
  --checkpoint /opt/t2s/checkpoint-step-21000 \
  --indextts-root /opt/t2s/index-tts \
  --codec-dir /opt/t2s/IndexTTS-2.5 \
  --ref-audio /path/to/ref.wav \
  --text "这是一句测试文本。" \
  --out /tmp/out.wav \
  --device cuda:0
```

九章上的包装脚本是 `scripts/infer_jiuzhang.sh`（路径写死在那台机器）。换机器请用上面的参数或下面的 WebUI 脚本。

## WebUI 与 REST

```bash
export T2S_CHECKPOINT=/opt/t2s/checkpoint-step-21000
export INDEXTTS_ROOT=/opt/t2s/index-tts
export CODEC_DIR=/opt/t2s/IndexTTS-2.5
export T2S_DEVICE=cuda:0
export WEBUI_HOST=0.0.0.0
export WEBUI_PORT=7860

# 可选
export TEXT2SEMANTIC_PYTHON=/path/to/venv/bin/python
export WEBUI_OUT_DIR=/opt/t2s/webui_outputs
export T2S_REPLICAS=2
export WEBUI_EXAMPLES_DIR=/opt/t2s/webui_examples

bash scripts/launch_webui.sh
# 或带 pidfile 重启：bash scripts/deploy/start_webui.sh
```

浏览器打开 `http://<机器IP>:<port>/`。上传参考音频、输入文本、点生成。模型在启动时加载一次，之后请求复用。

等价的直接调用：

```bash
python scripts/webui.py \
  --checkpoint "$T2S_CHECKPOINT" \
  --indextts-root "$INDEXTTS_ROOT" \
  --codec-dir "$CODEC_DIR" \
  --host 0.0.0.0 --port 7860 \
  --t2s-replicas 2
```

### 并行

默认 `--t2s-replicas 2`：两份 AR 各走自己的 CUDA stream，vocoder 共享一把锁。
s2mel 的 `setup_caches(max_batch_size=1)`，vocoder **必须**串行。
`POST /api/tts` 用 `asyncio.to_thread`，输出文件名带 thread id，避免并发互相覆盖。

L4 24G 上实测：串行 39.7 s / 并发 2 路 29.4 s，大约 **1.35×**，不是 2×（kernel 争用）。
VRAM 大约 16 / 23 GiB。再加 replica 会 OOM。

### REST

这是 **speaker clone**：只要目标文本 + 参考音频，没有 prompt 文本（不是 paper 那种 continuation）。

```
GET  /api/health
POST /api/tts
POST /api/generate    # 别名
```

`POST` 用 `multipart/form-data`：

| 字段 | 必填 | 默认 |
|---|---|---|
| `text` | 是 | |
| `ref_audio` | 是 | 音频文件 |
| `temperature` | | 0.5 |
| `top_k` | | 8 |
| `max_new_tokens` | | 1500 |
| `repetition_penalty` | | 10.0 |
| `seed` | | -1（随机） |

长文本会先切段再拼接：中文/日韩非空白字 >50 按句末标点切，单段 >100 强制切；英语类词数 >30 按句末标点切，单段 >50 词强制切。

`speaker_sim_boost` 默认关。打开后 Whisper 切参考（>10s 取前 10 秒内最后一个停顿 + 约 120ms 静音；≤10s 整段），切片文本拼在目标前面，切片 codes 作为 AR prefix，只保留新生成的 codes。speaker 特征仍用整段参考。

成功返回 `audio/wav`，状态在 `X-TTS-Status`。

## 环境变量

| 变量 | 必填 | 含义 |
|---|---|---|
| `T2S_CHECKPOINT` | 是 | HF checkpoint 目录 |
| `INDEXTTS_ROOT` | 是 | IndexTTS-2.5 代码根目录 |
| `CODEC_DIR` | 是 | 2.5 权重目录 |
| `TEXT2SEMANTIC_PYTHON` | 否 | Python 可执行文件 |
| `T2S_DEVICE` | 否 | 默认 `cuda:0` |
| `WEBUI_HOST` / `WEBUI_PORT` | 否 | 默认 `0.0.0.0:7860` |
| `WEBUI_OUT_DIR` | 否 | 生成 wav 落盘目录 |
| `T2S_REPLICAS` | 否 | AR 副本数，默认 2 |
| `WEBUI_EXAMPLES_DIR` | 否 | Gradio 参考音频示例目录 |
| `BIGVGAN_DIR` | 否 | 默认 `$CODEC_DIR/bigvgan` |
| `W2V_BERT_PATH` | 否 | 默认 `$CODEC_DIR/w2v-bert-2.0` |
| `STATS_PATH` | 否 | 默认 `$CODEC_DIR/wav2vec2bert_stats.pt` |

## 自检

启动后先用一条短中文和 3–8 秒参考音频跑通。成功时状态栏会打印 code 数和时长。常见失败：

- `checkpoint not found`：目录里没有 `model.safetensors` / `config.json`
- `BigVGAN local files missing`：`bigvgan/config.json` 或 `bigvgan_generator.pt` 不在
- `No module named indextts`：`INDEXTTS_ROOT` 不对，或没装那边的依赖
- `gradio is not installed`：`pip install 'gradio>=4.0'` 或 `pip install -e ".[infer]"`

## GCP L4 dest（asia-northeast3-a）

听音盒，不是 prod SpeechService。

| | |
|---|---|
| 实例 | `dev-noiz-speech-01`，`34.22.68.219:8080`，zone `asia-northeast3-a` |
| SSH | `gcloud compute ssh dev-noiz-speech-01 --zone=asia-northeast3-a`（用户 `babysor00`） |
| 仓库 | `/home/babysor00/t2s/text2semantic` |
| 当前权重 | `/home/babysor00/t2s/checkpoint-step-32765`（另留 17450 / 21000 回滚） |
| IndexTTS | `/home/babysor00/indextts-2.5/indextts-2.5` |
| 环境 | `scripts/deploy/gcp-l4.webui.env.example` |
| 启动 | `scripts/deploy/start_webui.sh`（pidfile；不要 `pkill -f webui.py`） |

换权重：只拉推理文件到新目录，改 `T2S_CHECKPOINT`，再跑 `start_webui.sh`。
这台机器没有 AWS CLI。从有凭证的盒子 presign 再 curl，或本机拉下来之后 `gcloud compute scp`。

机器上现在的 `start_webui.sh` 仍用 `pkill -f`。仓库里的脚本改成了 pidfile，
因为 `pkill -f` 的 pattern 如果也出现在 ssh `--command` 里会杀掉自己的远程壳。

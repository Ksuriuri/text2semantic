#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Gradio WebUI & REST API for the text2semantic + IndexTTS-2.5 wav pipeline.

Loads the AR model, EnhancedCodec, s2mel and BigVGAN once at startup, then
serves text + reference-audio -> 22.05 kHz wav.
Provides:
  - Gradio WebUI at /
  - Native REST API at POST /api/tts (multipart/form-data)
  - Native REST API at POST /api/generate (alias)
  - Health check at GET /api/health

Two AR replicas share one vocoder. s2mel caches are built with
``max_batch_size=1``, so the vocoder stays serial under a lock; only the
autoregressive half overlaps. On an L4 that is about 1.35x, not 2x.

Example::

    python scripts/webui.py \\
        --checkpoint /path/to/checkpoint-step-21000 \\
        --indextts-root /path/to/index-tts \\
        --codec-dir /path/to/IndexTTS-2.5 \\
        --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import argparse
import asyncio
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
import uvicorn

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import infer as t2s_infer  # noqa: E402


class InferenceApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        n = max(1, int(getattr(args, "t2s_replicas", 2)))
        self.t2s_pool: queue.Queue = queue.Queue()
        self.vocoder_lock = threading.Lock()
        self.vocoder_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        for i in range(n):
            print(f"loading text2semantic replica {i + 1}/{n} from {args.checkpoint}", flush=True)
            t2s = t2s_infer.load_t2s(
                args.checkpoint,
                w2v_bert_path=args.w2v_bert_path,
                stats_path=args.stats_path,
                device=str(self.device),
                dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
            )
            stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
            self.t2s_pool.put((t2s, stream))
        print("loading IndexTTS-2.5 codec + s2mel + BigVGAN", flush=True)
        self.vocoder = t2s_infer.IndexTTS25Vocoder(
            indextts_root=args.indextts_root,
            codec_dir=args.codec_dir,
            bigvgan_dir=args.bigvgan_dir,
            device=self.device,
            diffusion_steps=args.diffusion_steps,
            cfg_rate=args.cfg_rate,
            duration_factor=args.duration_factor,
        )
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"ready on {self.device}", flush=True)

    def generate(
        self,
        text: str,
        ref_audio,
        temperature: float = 0.5,
        top_k: int = 8,
        max_new_tokens: int = 1500,
        repetition_penalty: float = 10.0,
        seed: int = -1,
    ):
        text = (text or "").strip()
        if not text:
            raise ValueError("请输入要合成的文本")
        ref_path = _as_audio_path(ref_audio)
        if not ref_path:
            raise ValueError("请上传参考音频")

        t2s, stream = self.t2s_pool.get()
        try:
            if seed is not None and int(seed) >= 0:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
            else:
                rnd = int(time.time() * 1_000_000) % (2**31)
                torch.manual_seed(rnd)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(rnd)

            t0 = time.time()
            if stream is not None:
                with torch.cuda.stream(stream):
                    codes = t2s_infer.generate_codes(
                        t2s,
                        text,
                        ref_path,
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_k=int(top_k),
                        repetition_penalty=float(repetition_penalty),
                    )
                stream.synchronize()
            else:
                codes = t2s_infer.generate_codes(
                    t2s,
                    text,
                    ref_path,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                )
        finally:
            self.t2s_pool.put((t2s, stream))

        with self.vocoder_lock:
            if self.vocoder_stream is not None:
                with torch.cuda.stream(self.vocoder_stream):
                    wav, info = self.vocoder.vocode(codes, ref_path)
                self.vocoder_stream.synchronize()
            else:
                wav, info = self.vocoder.vocode(codes, ref_path)
            elapsed = time.time() - t0
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = self.out_dir / f"webui_{stamp}_{threading.get_ident()}.wav"
            t2s_infer.save_wav(str(out_path), wav, info["sample_rate"])

        status = (
            f"ok  {info['n_codes']} codes ({info['code_seconds']:.2f}s @ 25Hz)  "
            f"wav {info['wav_seconds']:.2f}s  wall {elapsed:.1f}s\n"
            f"{out_path}"
        )
        return str(out_path), status


def _as_audio_path(ref_audio) -> str | None:
    if ref_audio is None:
        return None
    if isinstance(ref_audio, str) and os.path.isfile(ref_audio):
        return ref_audio
    if isinstance(ref_audio, dict):
        path = ref_audio.get("path") or ref_audio.get("name")
        if path and os.path.isfile(path):
            return path
    if isinstance(ref_audio, (tuple, list)) and len(ref_audio) == 2:
        sr, data = ref_audio
        data = np.asarray(data)
        if data.ndim == 2:
            data = data.mean(axis=1)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        t2s_infer.save_wav(
            tmp.name,
            torch.from_numpy(np.asarray(data, dtype=np.float32)),
            int(sr),
        )
        return tmp.name
    return None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="HF checkpoint dir (model.safetensors + tokenizer)")
    p.add_argument("--indextts-root", default=t2s_infer.DEFAULT_INDEXTTS_ROOT)
    p.add_argument("--codec-dir", default=t2s_infer.DEFAULT_CODEC_DIR)
    p.add_argument("--bigvgan-dir", default=t2s_infer.DEFAULT_BIGVGAN_DIR)
    p.add_argument("--w2v-bert-path", default=t2s_infer.DEFAULT_W2V_BERT)
    p.add_argument("--stats-path", default=t2s_infer.DEFAULT_STATS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--diffusion-steps", type=int, default=t2s_infer.DIFFUSION_STEPS)
    p.add_argument("--cfg-rate", type=float, default=t2s_infer.INFERENCE_CFG_RATE)
    p.add_argument("--duration-factor", type=float, default=1.0)
    p.add_argument("--host", default=os.environ.get("WEBUI_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("WEBUI_PORT", "7860")))
    p.add_argument("--t2s-replicas", type=int, default=int(os.environ.get("T2S_REPLICAS", "2")),
                   help="parallel AR copies; vocoder stays shared")
    p.add_argument(
        "--examples-dir",
        default=os.environ.get("WEBUI_EXAMPLES_DIR", ""),
        help="optional directory of reference wavs shown in the Gradio examples",
    )
    p.add_argument("--share", action="store_true", help="unused; kept for CLI compatibility")
    p.add_argument(
        "--out-dir",
        default=os.environ.get("WEBUI_OUT_DIR", str(Path.cwd() / "webui_outputs")),
    )
    return p.parse_args(argv)


def build_ui(app: InferenceApp):
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(
            "gradio is not installed. On the deploy machine run:\n"
            "  pip install 'gradio>=4.0'"
        ) from exc

    example_dir = Path(app.args.examples_dir) if app.args.examples_dir else None
    example_wavs = sorted(str(p) for p in example_dir.glob("*.wav")) if example_dir and example_dir.is_dir() else []
    text_examples = [
        ["今天天气真不错，我们一起去公园走走吧。"],
        ["The weather is lovely today. Shall we take a walk in the park?"],
        ["今日はいい天気ですね。一緒に公園を散歩しませんか。"],
        ["오늘 날씨가 정말 좋네요. 공원에 같이 산책하러 갈까요?"],
        ["Hoy hace un día precioso. ¿Damos un paseo por el parque?"],
        ["O tempo está ótimo hoje. Vamos dar um passeio no parque?"],
        ["Il fait très beau aujourd'hui. On se promène au parc ?"],
        ["Сегодня прекрасная погода. Давайте прогуляемся в парке."],
        ["Heute ist wirklich schönes Wetter. Gehen wir im Park spazieren?"],
    ]
    with gr.Blocks(title="Noiz TTS v0.1") as demo:
        gr.Markdown(
            "## Noiz TTS v0.1\n"
            "小规模数据测试版 · 约 1 万小时训练（合计 9258h）。中文仅 332h，占比很低。\n\n"
            "| | EN | JA | PT | KO | ES | FR | RU | DE | ZH | AR |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            "| Hours | 2523 | 1135 | 1087 | 1060 | 1015 | 731 | 677 | 556 | 332 | 142 |"
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="文本", lines=4, placeholder="输入要合成的句子")
                ref = gr.Audio(label="参考音频", type="filepath")
                with gr.Accordion("生成参数", open=False):
                    temperature = gr.Slider(0.1, 1.5, value=0.5, step=0.05, label="temperature")
                    top_k = gr.Slider(1, 200, value=8, step=1, label="top_k")
                    max_new_tokens = gr.Slider(64, 2000, value=1500, step=16, label="max_new_tokens")
                    repetition_penalty = gr.Slider(1.0, 10.0, value=10.0, step=0.05, label="repetition_penalty")
                    seed = gr.Number(value=-1, precision=0, label="seed（-1 为随机）")
                btn = gr.Button("生成", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="合成结果", type="filepath")
                status = gr.Textbox(label="状态", lines=3)
        btn.click(
            fn=app.generate,
            inputs=[text, ref, temperature, top_k, max_new_tokens, repetition_penalty, seed],
            outputs=[audio_out, status],
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 文本示例（中英日韩西葡法俄德）")
                gr.Examples(
                    examples=text_examples,
                    inputs=[text],
                    label="点一下填入对应语言文本",
                    examples_per_page=9,
                )
            with gr.Column():
                gr.Markdown("### 参考音频示例")
                if example_wavs:
                    gr.Examples(
                        examples=[[w] for w in example_wavs],
                        inputs=[ref],
                        label="点一下选用参考音频",
                    )
                else:
                    gr.Markdown("_未配置参考音频示例目录（`--examples-dir` / `WEBUI_EXAMPLES_DIR`）_")
        gr.Markdown(
            f"checkpoint: `{app.args.checkpoint}`  \n"
            f"device: `{app.device}`  \n"
            f"codec: `{app.args.codec_dir}`  \n"
            f"t2s replicas: `{getattr(app.args, 't2s_replicas', 2)}`"
        )
    return demo


def create_app(app: InferenceApp) -> FastAPI:
    import gradio as gr

    fastapi_app = FastAPI(title="Noiz TTS API", version="0.1.0")

    @fastapi_app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "device": str(app.device),
            "model": "Noiz TTS v0.1",
            "t2s_replicas": int(getattr(app.args, "t2s_replicas", 2)),
            "default_params": {
                "temperature": 0.5,
                "top_k": 8,
                "max_new_tokens": 1500,
                "repetition_penalty": 10.0,
                "seed": -1,
            },
        }

    @fastapi_app.post("/api/tts")
    @fastapi_app.post("/api/generate")
    async def api_tts(
        text: str = Form(..., description="要合成的目标文本"),
        ref_audio: UploadFile = File(..., description="参考音频文件（WAV/MP3等）"),
        temperature: float = Form(0.5, description="采样温度，默认 0.5"),
        top_k: int = Form(8, description="top_k，默认 8"),
        max_new_tokens: int = Form(1500, description="最大 token 数，默认 1500"),
        repetition_penalty: float = Form(10.0, description="重复惩罚，默认 10.0"),
        seed: int = Form(-1, description="随机种子，默认 -1 随机"),
    ):
        text = (text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text cannot be empty")

        suffix = Path(ref_audio.filename or "ref.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_f:
            tmp_path = tmp_f.name
            content = await ref_audio.read()
            tmp_f.write(content)

        try:
            out_path, status = await asyncio.to_thread(
                app.generate,
                text,
                tmp_path,
                temperature,
                top_k,
                max_new_tokens,
                repetition_penalty,
                seed,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        out_file = Path(out_path)
        return FileResponse(
            path=str(out_file),
            media_type="audio/wav",
            filename=out_file.name,
            headers={
                "X-TTS-Status": status.replace("\n", " "),
            },
        )

    demo = build_ui(app)
    demo.queue(max_size=8)
    allowed = [str(app.out_dir)]
    if app.args.examples_dir:
        allowed.append(str(Path(app.args.examples_dir)))
    fastapi_app = gr.mount_gradio_app(
        fastapi_app,
        demo,
        path="/",
        allowed_paths=allowed,
    )
    return fastapi_app


def main(argv=None) -> int:
    args = parse_args(argv)
    for label, path in (
        ("checkpoint", args.checkpoint),
        ("indextts-root", args.indextts_root),
        ("codec-dir", args.codec_dir),
    ):
        if not path or not os.path.exists(path):
            raise SystemExit(f"{label} not found: {path}")
    app = InferenceApp(args)
    fastapi_app = create_app(app)
    uvicorn.run(fastapi_app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

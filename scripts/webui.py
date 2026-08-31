#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Gradio WebUI & REST API for the text2semantic + IndexTTS-2.5 wav pipeline.

Loads the AR model and the configured s2vae/s2mel acoustic backends once at
startup. The default is s2vae (48 kHz); s2mel remains selectable (22.05 kHz).
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
        requested = [item.strip() for item in args.vocoder_backends.split(",") if item.strip()]
        unknown = sorted(set(requested) - {"s2vae", "s2mel"})
        if unknown:
            raise ValueError(f"unknown vocoder backends: {unknown}")
        if not requested:
            raise ValueError("at least one vocoder backend must be configured")
        if args.default_vocoder not in requested:
            raise ValueError("--default-vocoder must be included in --vocoder-backends")
        self.default_vocoder = args.default_vocoder
        self.vocoders = {}
        if "s2vae" in requested:
            print("loading s2vae step checkpoint + dots.tts AudioVAE", flush=True)
            self.vocoders["s2vae"] = t2s_infer.S2VAEVocoder(
                semantic2any_root=args.semantic2any_root,
                config_path=args.s2vae_config,
                checkpoint_path=args.s2vae_checkpoint,
                dots_tts_dir=args.dots_tts_dir,
                indextts_root=args.indextts_root,
                codec_dir=args.codec_dir,
                device=self.device,
                diffusion_steps=args.diffusion_steps,
                cfg_rate=args.cfg_rate,
                temperature=args.s2vae_temperature,
                prompt_min_seconds=args.s2vae_prompt_min_seconds,
            )
        if "s2mel" in requested:
            print("loading IndexTTS-2.5 codec + s2mel + BigVGAN", flush=True)
            self.vocoders["s2mel"] = t2s_infer.IndexTTS25Vocoder(
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

    def _validate_request(self, text: str, ref_audio) -> tuple[str, str]:
        text = (text or "").strip()
        if not text:
            raise ValueError("请输入要合成的文本")
        ref_path = _as_audio_path(ref_audio)
        if not ref_path:
            raise ValueError("请上传参考音频")
        return text, ref_path

    def _generate_semantics(
        self,
        *,
        text: str,
        ref_path: str,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
        repetition_penalty: float,
        seed: int,
        language: str | None,
        emotion: str | None,
        need_prompt_features: bool,
    ):
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

            started = time.time()
            if stream is not None:
                with torch.cuda.stream(stream):
                    generated = t2s_infer.generate_codes(
                        t2s,
                        text,
                        ref_path,
                        return_prompt_features=need_prompt_features,
                        language=language or None,
                        emotion=(emotion or "").strip() or None,
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_k=int(top_k),
                        repetition_penalty=float(repetition_penalty),
                    )
                stream.synchronize()
            else:
                generated = t2s_infer.generate_codes(
                    t2s,
                    text,
                    ref_path,
                    return_prompt_features=need_prompt_features,
                    language=language or None,
                    emotion=(emotion or "").strip() or None,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                )
        finally:
            self.t2s_pool.put((t2s, stream))
        elapsed = time.time() - started
        if need_prompt_features:
            codes, prompt_features, prompt_feature_length = generated
            return codes, prompt_features, prompt_feature_length, elapsed
        return generated, None, None, elapsed

    def _vocode(
        self,
        *,
        backend: str,
        codes: torch.Tensor,
        ref_path: str,
        prompt_features: torch.Tensor | None,
        prompt_feature_length: int | None,
        t2s_elapsed: float,
    ) -> tuple[str, str]:
        started = time.time()
        with self.vocoder_lock:
            if self.vocoder_stream is not None:
                with torch.cuda.stream(self.vocoder_stream):
                    if backend == "s2vae":
                        wav, info = self.vocoders[backend].vocode(
                            codes,
                            ref_path,
                            prompt_features=prompt_features,
                            prompt_feature_length=int(prompt_feature_length),
                        )
                    else:
                        wav, info = self.vocoders[backend].vocode(codes, ref_path)
                self.vocoder_stream.synchronize()
            else:
                if backend == "s2vae":
                    wav, info = self.vocoders[backend].vocode(
                        codes,
                        ref_path,
                        prompt_features=prompt_features,
                        prompt_feature_length=int(prompt_feature_length),
                    )
                else:
                    wav, info = self.vocoders[backend].vocode(codes, ref_path)
            info.setdefault("backend", backend)
            backend_elapsed = time.time() - started
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = self.out_dir / (
                f"webui_{stamp}_{threading.get_ident()}_{backend}.wav"
            )
            t2s_infer.save_wav(str(out_path), wav, info["sample_rate"])

        status = (
            f"ok [{backend}]  {info['n_codes']} codes ({info['code_seconds']:.2f}s @ 25Hz)  "
            f"wav {info['wav_seconds']:.2f}s  "
            f"t2s {t2s_elapsed:.1f}s + backend {backend_elapsed:.1f}s\n"
            f"{out_path}"
        )
        return str(out_path), status

    def generate(
        self,
        text: str,
        ref_audio,
        temperature: float = 0.5,
        top_k: int = 8,
        max_new_tokens: int = 1500,
        repetition_penalty: float = 10.0,
        seed: int = -1,
        language: str | None = None,
        vocoder_backend: str | None = None,
        emotion: str | None = None,
    ):
        text, ref_path = self._validate_request(text, ref_audio)
        backend = (vocoder_backend or self.default_vocoder).strip().lower()
        if backend not in self.vocoders:
            raise ValueError(
                f"vocoder backend {backend!r} is unavailable; "
                f"choose one of {sorted(self.vocoders)}"
            )
        codes, prompt_features, prompt_feature_length, t2s_elapsed = (
            self._generate_semantics(
                text=text,
                ref_path=ref_path,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                seed=seed,
                language=language,
                emotion=emotion,
                need_prompt_features=backend == "s2vae",
            )
        )
        return self._vocode(
            backend=backend,
            codes=codes,
            ref_path=ref_path,
            prompt_features=prompt_features,
            prompt_feature_length=prompt_feature_length,
            t2s_elapsed=t2s_elapsed,
        )

    def generate_comparison(
        self,
        text: str,
        ref_audio,
        temperature: float = 0.5,
        top_k: int = 8,
        max_new_tokens: int = 1500,
        repetition_penalty: float = 10.0,
        seed: int = -1,
        language: str | None = None,
    ):
        missing = [name for name in ("s2vae", "s2mel") if name not in self.vocoders]
        if missing:
            raise ValueError(f"comparison requires both backends; missing {missing}")
        text, ref_path = self._validate_request(text, ref_audio)
        codes, prompt_features, prompt_feature_length, t2s_elapsed = (
            self._generate_semantics(
                text=text,
                ref_path=ref_path,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                seed=seed,
                language=language,
                emotion=None,
                need_prompt_features=True,
            )
        )
        s2vae = self._vocode(
            backend="s2vae",
            codes=codes,
            ref_path=ref_path,
            prompt_features=prompt_features,
            prompt_feature_length=prompt_feature_length,
            t2s_elapsed=t2s_elapsed,
        )
        s2mel = self._vocode(
            backend="s2mel",
            codes=codes,
            ref_path=ref_path,
            prompt_features=None,
            prompt_feature_length=None,
            t2s_elapsed=t2s_elapsed,
        )
        return s2vae[0], s2vae[1], s2mel[0], s2mel[1]


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
    p.add_argument(
        "--vocoder-backends",
        default=os.environ.get("VOCODER_BACKENDS", "s2vae,s2mel"),
        help="comma-separated backends to load: s2vae,s2mel",
    )
    p.add_argument(
        "--default-vocoder",
        choices=("s2vae", "s2mel"),
        default=os.environ.get("DEFAULT_VOCODER", "s2vae"),
    )
    p.add_argument("--semantic2any-root", default=t2s_infer.DEFAULT_SEMANTIC2ANY_ROOT)
    p.add_argument("--s2vae-config", default=t2s_infer.DEFAULT_S2VAE_CONFIG)
    p.add_argument("--s2vae-checkpoint", default=t2s_infer.DEFAULT_S2VAE_CHECKPOINT)
    p.add_argument("--dots-tts-dir", default=t2s_infer.DEFAULT_DOTS_TTS_DIR)
    p.add_argument("--s2vae-temperature", type=float, default=0.7)
    p.add_argument(
        "--s2vae-prompt-min-seconds",
        type=float,
        default=t2s_infer.S2VAE_PROMPT_MIN_SECONDS,
    )
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
        ["The weather is lovely today. Shall we take a walk in the park?"],
        ["今日はいい天気ですね。一緒に公園を散歩しませんか。"],
        ["오늘 날씨가 정말 좋네요. 공원에 같이 산책하러 갈까요?"],
        ["Hoy hace un día precioso. ¿Damos un paseo por el parque?"],
        ["O tempo está ótimo hoje. Vamos dar um passeio no parque?"],
        ["Il fait très beau aujourd'hui. On se promène au parc ?"],
        ["Сегодня прекрасная погода. Давайте прогуляемся в парке."],
        ["Heute ist wirklich schönes Wetter. Gehen wir im Park spazieren?"],
        ["今天天气真不错，我们一起去公园走走吧。"],
    ]
    with gr.Blocks(title="Noiz TTS v0.x") as demo:
        gr.Markdown("## Noiz TTS v0.x")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="文本",
                    lines=4,
                    placeholder="输入句子；用 [sighing] 这类方括号插入情绪/气声控制",
                )
                ref = gr.Audio(label="参考音频", type="filepath")
                language = gr.Dropdown(
                    choices=["auto", "ar", "de", "en", "es", "fr", "ja", "ko", "pt", "ru", "zh"],
                    value="auto",
                    label="语言控制（auto = 不加标签）",
                )
                with gr.Accordion("生成参数", open=False):
                    temperature = gr.Slider(0.1, 1.5, value=0.5, step=0.05, label="temperature")
                    top_k = gr.Slider(1, 200, value=8, step=1, label="top_k")
                    max_new_tokens = gr.Slider(64, 2000, value=1500, step=16, label="max_new_tokens")
                    repetition_penalty = gr.Slider(1.0, 10.0, value=10.0, step=0.05, label="repetition_penalty")
                    seed = gr.Number(value=-1, precision=0, label="seed（-1 为随机）")
                btn = gr.Button("生成 S2VAE + S2Mel 对比", variant="primary")
            with gr.Column():
                audio_s2vae = gr.Audio(label="S2VAE（48kHz）", type="filepath")
                status_s2vae = gr.Textbox(label="S2VAE 状态", lines=3)
            with gr.Column():
                audio_s2mel = gr.Audio(label="S2Mel（22.05kHz）", type="filepath")
                status_s2mel = gr.Textbox(label="S2Mel 状态", lines=3)
        btn.click(
            fn=app.generate_comparison,
            inputs=[
                text, ref, temperature, top_k, max_new_tokens,
                repetition_penalty, seed, language,
            ],
            outputs=[audio_s2vae, status_s2vae, audio_s2mel, status_s2mel],
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 文本示例（英日韩西葡法俄德中）")
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
    return demo


def create_app(app: InferenceApp) -> FastAPI:
    import gradio as gr

    fastapi_app = FastAPI(title="Noiz TTS API", version="0.1.0")

    @fastapi_app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "device": str(app.device),
            "model": "Noiz TTS v0.x",
            "t2s_replicas": int(getattr(app.args, "t2s_replicas", 2)),
            "vocoder_default": app.default_vocoder,
            "vocoder_backends": list(app.vocoders),
            "checkpoint": app.args.checkpoint,
            "s2vae_prompt_min_seconds": app.args.s2vae_prompt_min_seconds,
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
        language: str | None = Form(None, description="可选语言控制码"),
        emotion: str | None = Form(None, description="可选情绪/气声描述"),
        vocoder_backend: str | None = Form(None, description="s2vae 或 s2mel"),
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
                language,
                vocoder_backend,
                emotion,
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
    requested = {item.strip() for item in args.vocoder_backends.split(",") if item.strip()}
    if "s2vae" in requested:
        for label, path in (
            ("semantic2any-root", args.semantic2any_root),
            ("s2vae-config", args.s2vae_config),
            ("s2vae-checkpoint", args.s2vae_checkpoint),
            ("dots-tts-dir", args.dots_tts_dir),
        ):
            if not path or not os.path.exists(path):
                raise SystemExit(f"{label} not found: {path}")
    app = InferenceApp(args)
    fastapi_app = create_app(app)
    uvicorn.run(fastapi_app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

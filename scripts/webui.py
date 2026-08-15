#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Gradio WebUI for the text2semantic + IndexTTS-2.5 wav pipeline.

Loads the AR model, EnhancedCodec, s2mel and BigVGAN once at startup, then
serves text + reference-audio → 22.05 kHz wav. Same code path as
``scripts/infer.py``.

Example::

    python scripts/webui.py \\
        --checkpoint /path/to/checkpoint-step-21000 \\
        --indextts-root /path/to/index-tts \\
        --codec-dir /path/to/IndexTTS-2.5 \\
        --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import infer as t2s_infer  # noqa: E402


class InferenceApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        print(f"loading text2semantic from {args.checkpoint}", flush=True)
        self.t2s = t2s_infer.load_t2s(
            args.checkpoint,
            w2v_bert_path=args.w2v_bert_path,
            stats_path=args.stats_path,
            device=str(self.device),
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        )
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
        temperature: float,
        top_k: int,
        max_new_tokens: int,
        seed: int,
    ):
        text = (text or "").strip()
        if not text:
            raise ValueError("请输入要合成的文本")
        ref_path = _as_audio_path(ref_audio)
        if not ref_path:
            raise ValueError("请上传参考音频")

        if seed is not None and int(seed) >= 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        t0 = time.time()
        codes = t2s_infer.generate_codes(
            self.t2s,
            text,
            ref_path,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
        )
        wav, info = self.vocoder.vocode(codes, ref_path)
        elapsed = time.time() - t0

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = self.out_dir / f"webui_{stamp}.wav"
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
    p.add_argument("--share", action="store_true", help="create a Gradio public link")
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

    with gr.Blocks(title="text2semantic + IndexTTS-2.5") as demo:
        gr.Markdown(
            "## text2semantic 推理\n"
            "参考音频 + 文本 → 25Hz AR codes → EnhancedCodec.decode (50Hz) "
            "→ s2mel → BigVGAN（22.05 kHz）"
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="文本", lines=4, placeholder="输入要合成的句子")
                ref = gr.Audio(label="参考音频", type="filepath")
                with gr.Accordion("生成参数", open=False):
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature")
                    top_k = gr.Slider(1, 200, value=50, step=1, label="top_k")
                    max_new_tokens = gr.Slider(64, 2000, value=1500, step=16, label="max_new_tokens")
                    seed = gr.Number(value=0, precision=0, label="seed")
                btn = gr.Button("生成", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="合成结果", type="filepath")
                status = gr.Textbox(label="状态", lines=3)
        btn.click(
            fn=app.generate,
            inputs=[text, ref, temperature, top_k, max_new_tokens, seed],
            outputs=[audio_out, status],
        )
        gr.Markdown(
            f"checkpoint: `{app.args.checkpoint}`  \n"
            f"device: `{app.device}`  \n"
            f"codec: `{app.args.codec_dir}`"
        )
    return demo


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
    demo = build_ui(app)
    demo.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

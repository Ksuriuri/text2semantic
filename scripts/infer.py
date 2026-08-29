#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""End-to-end inference for the current text2semantic + IndexTTS-2.5 stack.

Pipeline (must stay bit-for-bit with ``indextts/infer_v2_5.py`` after the
text2semantic AR step):

1. text + reference audio → this project's Text2Semantic AR model
   → discrete semantic codes at **25 Hz** (IndexTTS-2.5 EnhancedCodec ids).
2. ``EnhancedCodec.decode(codes)`` → continuous features at **50 Hz**
   (decoder ConvNeXt + nearest upsample ×2 + ``up`` conv). Do **not** feed
   raw 25 Hz ids or ``quantize()``'s pre-decoder embedding into s2mel.
3. s2mel prompt / reference = raw w2v-bert layer-17 features (never the
   codec). Target = the decoded 50 Hz features. ``target_lengths`` uses the
   official ``1.72`` ratio on the **decoded** time axis.
4. CAMPPlus style + CFM (25 steps, cfg 0.7) + BigVGAN → 22.05 kHz wav.

This project previously only had ``Text2SemanticModel.generate()`` (codes
only) and a leftover MaskGCT-50 Hz vocoder script. That old path is **not**
compatible with the current 25 Hz training codes.

Example::

    python scripts/infer.py \\
        --checkpoint /path/to/checkpoint-step-5000 \\
        --ref-audio /path/to/ref.wav \\
        --text "这是一句测试文本。" \\
        --out /tmp/out.wav
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _first_existing(*paths: str) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0]


DEFAULT_INDEXTTS_ROOT = _first_existing(
    "/hunshan/hhy/workspace/indextts-new/index-tts",
    "/mnt/data_sdd/hhy/indextts-new/index-tts",
)
DEFAULT_CODEC_DIR = _first_existing(
    "/hunshan/hhy/models/IndexTTS-2.5",
    "/mnt/data_sdd/hhy/indextts-new/index-tts/checkpoints/indextts25_codec",
)
DEFAULT_BIGVGAN_DIR = _first_existing(
    os.path.join(DEFAULT_CODEC_DIR, "bigvgan"),
    "/hunshan/hhy/models/IndexTTS-2-vLLM/bigvgan",
    "/mnt/data_sdd/hhy/indextts-2.5/checkpoints/hf_cache/bigvgan",
)
DEFAULT_W2V_BERT = _first_existing(
    os.path.join(DEFAULT_CODEC_DIR, "w2v-bert-2.0"),
    "/hunshan/hhy/models/IndexTTS-2-vLLM/w2v-bert-2.0",
)
DEFAULT_STATS = _first_existing(
    os.path.join(DEFAULT_CODEC_DIR, "wav2vec2bert_stats.pt"),
    "/hunshan/hhy/models/IndexTTS-2-vLLM/wav2vec2bert_stats.pt",
)

# Official infer_v2_5.py constants. length_regulator sees *decoded* 50 Hz
# features, so this is the same 1.72 used by the old 50 Hz codec — decode()
# has already undone the 25 Hz downsample.
TARGET_LENGTH_RATIO = 1.72
DIFFUSION_STEPS = 25
INFERENCE_CFG_RATE = 0.7
PROMPT_MAX_SECONDS = 15.0
CODE_FPS = 25.0
FEATURE_FPS = 50.0
CODEBOOK_SIZE = 8192


def _load_bigvgan_local(bigvgan_mod, bigvgan_dir: str):
    """Load BigVGAN from a local dir. Avoids huggingface_hub mixin kwargs
    that newer hub versions no longer pass through to `_from_pretrained`."""
    config_path = os.path.join(bigvgan_dir, "config.json")
    weight_path = os.path.join(bigvgan_dir, "bigvgan_generator.pt")
    if not os.path.isfile(config_path) or not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"BigVGAN local files missing under {bigvgan_dir}"
        )
    h = bigvgan_mod.load_hparams_from_json(config_path)
    voc = bigvgan_mod.BigVGAN(h, use_cuda_kernel=False)
    ckpt = torch.load(weight_path, map_location="cpu")
    state = ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt
    voc.load_state_dict(state)
    return voc


def _import_indextts(indextts_root: str):
    root = os.path.abspath(indextts_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from indextts.codec.models import EnhancedCodec
    from indextts.s2mel.modules.audio import mel_spectrogram
    from indextts.s2mel.modules.bigvgan import bigvgan
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    from omegaconf import OmegaConf
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

    return {
        "EnhancedCodec": EnhancedCodec,
        "mel_spectrogram": mel_spectrogram,
        "bigvgan": bigvgan,
        "CAMPPlus": CAMPPlus,
        "MyModel": MyModel,
        "load_checkpoint2": load_checkpoint2,
        "OmegaConf": OmegaConf,
        "SeamlessM4TFeatureExtractor": SeamlessM4TFeatureExtractor,
        "Wav2Vec2BertModel": Wav2Vec2BertModel,
    }


def _strip_special_codes(codes: torch.Tensor) -> torch.Tensor:
    """Keep only valid codebook ids. generate_semantic already drops BOS/EOS
    but a truncated run can still emit them."""
    codes = codes.long().reshape(-1)
    keep = (codes >= 0) & (codes < CODEBOOK_SIZE)
    if not bool(keep.any()):
        raise RuntimeError("text2semantic produced no valid semantic codes")
    return codes[keep]


class IndexTTS25Vocoder:
    """IndexTTS-2.5 s2mel + BigVGAN, wired exactly like infer_v2_5.py.

    ``use_gpt_latent`` is forced off: this project's AR model does not emit
    the GPT latent that official IndexTTS-2.5 optionally adds onto decode().
    """

    def __init__(
        self,
        *,
        indextts_root: str,
        codec_dir: str,
        bigvgan_dir: str,
        device: torch.device,
        diffusion_steps: int = DIFFUSION_STEPS,
        cfg_rate: float = INFERENCE_CFG_RATE,
        duration_factor: float = 1.0,
        prompt_max_seconds: float = PROMPT_MAX_SECONDS,
    ):
        mods = _import_indextts(indextts_root)
        self.device = device
        self.diffusion_steps = int(diffusion_steps)
        self.cfg_rate = float(cfg_rate)
        self.duration_factor = float(duration_factor)
        self.prompt_max_seconds = float(prompt_max_seconds)

        cfg = mods["OmegaConf"].load(os.path.join(codec_dir, "config.yaml"))
        self.cfg = cfg

        w2v_dir = os.path.join(codec_dir, "w2v-bert-2.0")
        if not os.path.isdir(w2v_dir):
            w2v_dir = DEFAULT_W2V_BERT
        stats_path = os.path.join(codec_dir, cfg.w2v_stat)

        self.extract_features = mods["SeamlessM4TFeatureExtractor"].from_pretrained(
            w2v_dir, local_files_only=True
        )
        self.semantic_model = (
            mods["Wav2Vec2BertModel"]
            .from_pretrained(w2v_dir, local_files_only=True)
            .to(device)
            .eval()
        )
        self.semantic_model.requires_grad_(False)
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        self.semantic_mean = stats["mean"].to(device)
        self.semantic_std = torch.sqrt(stats["var"]).to(device)

        codec = mods["EnhancedCodec"](
            **cfg.semantic_codec, cfg=cfg.semantic_codec
        )
        codec_ckpt = os.path.join(codec_dir, "codec.pth")
        codec.load_checkpoint(codec_ckpt)
        self.semantic_codec = codec.to(device).eval()
        self.semantic_codec.requires_grad_(False)

        s2mel_path = os.path.join(codec_dir, cfg.s2mel_checkpoint)
        s2mel = mods["MyModel"](cfg.s2mel, use_gpt_latent=False)
        s2mel, _, _, _ = mods["load_checkpoint2"](
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(device).eval()
        self.s2mel.models["cfm"].estimator.setup_caches(
            max_batch_size=1, max_seq_length=8192
        )

        campplus_ckpt = os.path.join(codec_dir, "campplus_cn_common.bin")
        campplus = mods["CAMPPlus"](feat_dim=80, embedding_size=192)
        campplus.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
        self.campplus = campplus.to(device).eval()

        voc = _load_bigvgan_local(mods["bigvgan"], bigvgan_dir)
        voc = voc.to(device)
        voc.remove_weight_norm()
        self.bigvgan = voc.eval()

        spect = cfg.s2mel.preprocess_params.spect_params
        self.mel_sr = int(cfg.s2mel.preprocess_params.sr)
        self.mel_hop = int(spect.hop_length)
        fmax = spect.get("fmax", "None")
        self.mel_fn = lambda x: mods["mel_spectrogram"](
            x,
            n_fft=spect.n_fft,
            win_size=spect.win_length,
            hop_size=spect.hop_length,
            num_mels=spect.n_mels,
            sampling_rate=self.mel_sr,
            fmin=spect.get("fmin", 0),
            fmax=None if str(fmax) == "None" else fmax,
            center=False,
        )

    @torch.no_grad()
    def get_emb(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Raw continuous w2v-bert feature. Prompt/reference never touches the codec."""
        inputs = self.extract_features(
            audio_16k.detach().cpu(), sampling_rate=16000, return_tensors="pt"
        )
        feat = self.semantic_model(
            input_features=inputs["input_features"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            output_hidden_states=True,
        ).hidden_states[17]
        return (feat - self.semantic_mean) / self.semantic_std

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """25 Hz ids → 50 Hz features via official EnhancedCodec.decode()."""
        codes = _strip_special_codes(codes).unsqueeze(0).to(self.device)
        return self.semantic_codec.decode(codes)

    def _load_ref(self, ref_audio: str):
        import librosa

        audio, sr = librosa.load(ref_audio, sr=None, mono=True)
        audio = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        max_n = int(self.prompt_max_seconds * sr)
        if audio.shape[1] > max_n:
            audio = audio[:, :max_n]
        audio_22k = torchaudio.functional.resample(audio, sr, self.mel_sr)
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return audio_22k, audio_16k

    @torch.no_grad()
    def vocode(self, codes: torch.Tensor, ref_audio: str) -> tuple[torch.Tensor, dict]:
        audio_22k, audio_16k = self._load_ref(ref_audio)
        audio_22k = audio_22k.to(self.device)
        audio_16k = audio_16k.to(self.device)

        # Official path: s2mel / codec stay in FP32 (infer_v2_5 sets dtype=None).
        spk_cond_emb = self.get_emb(audio_16k)
        ref_mel = self.mel_fn(audio_22k.float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(self.device)
        prompt_condition = self.s2mel.models["length_regulator"](
            spk_cond_emb,
            ylens=ref_target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]

        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus(feat.unsqueeze(0))

        S_infer = self.decode_codes(codes)
        target_lengths = torch.LongTensor(
            [int(S_infer.shape[1] * TARGET_LENGTH_RATIO * self.duration_factor)]
        ).to(self.device)
        cond = self.s2mel.models["length_regulator"](
            S_infer,
            ylens=target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]

        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        vc_target = self.s2mel.models["cfm"].inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(self.device),
            ref_mel,
            style,
            None,
            self.diffusion_steps,
            inference_cfg_rate=self.cfg_rate,
        )
        vc_target = vc_target[:, :, ref_mel.size(-1) :]
        wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
        wav = torch.clamp(wav, -1.0, 1.0).cpu()

        info = {
            "n_codes": int(_strip_special_codes(codes).numel()),
            "code_seconds": float(_strip_special_codes(codes).numel() / CODE_FPS),
            "decoded_frames": int(S_infer.shape[1]),
            "decoded_seconds": float(S_infer.shape[1] / FEATURE_FPS),
            "target_mel_frames": int(target_lengths.item()),
            "wav_seconds": float(wav.shape[-1] / self.mel_sr),
            "sample_rate": self.mel_sr,
        }
        return wav, info


def load_t2s(
    checkpoint: str,
    *,
    w2v_bert_path: str,
    stats_path: str,
    device: str,
    dtype: torch.dtype,
    attn_implementation: str,
):
    from qwen_tts.inference.text2semantic_model import Text2SemanticModel

    return Text2SemanticModel.from_pretrained(
        checkpoint,
        w2v_bert_path=w2v_bert_path,
        stats_path=stats_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )


def generate_codes(t2s, text: str, ref_audio: str, **gen_kwargs) -> torch.Tensor:
    codes_list = t2s.generate(text, ref_audio, **gen_kwargs)
    return _strip_special_codes(codes_list[0].detach().cpu())


def save_wav(path: str, wav: torch.Tensor, sr: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    wav = wav.detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    pcm = (wav.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy()
    # torchaudio 2.11+ save() requires torchcodec; write PCM16 directly.
    import wave

    with wave.open(path, "wb") as handle:
        handle.setnchannels(int(pcm.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(int(sr))
        handle.writeframes(pcm.T.tobytes() if pcm.shape[0] > 1 else pcm.tobytes())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="text2semantic HF checkpoint dir (not needed with --codes-npy)")
    p.add_argument("--ref-audio", required=True, help="reference wav used as speaker prompt")
    p.add_argument("--text", help="target text (required unless --codes-npy)")
    p.add_argument(
        "--language",
        choices=("ar", "de", "en", "es", "fr", "ja", "ko", "pt", "ru", "zh"),
        help="Optional atomic language control placed before the text.",
    )
    p.add_argument(
        "--emotion",
        help="Optional freeform affect/event control wrapped in emo boundary tokens.",
    )
    p.add_argument("--out", required=True, help="output wav path")
    p.add_argument("--codes-npy", help="skip AR generation and vocode these 25 Hz ids")
    p.add_argument("--save-codes", help="optional .npy path for the generated 25 Hz ids")
    p.add_argument("--indextts-root", default=DEFAULT_INDEXTTS_ROOT)
    p.add_argument("--codec-dir", default=DEFAULT_CODEC_DIR)
    p.add_argument("--bigvgan-dir", default=DEFAULT_BIGVGAN_DIR)
    p.add_argument("--w2v-bert-path", default=DEFAULT_W2V_BERT)
    p.add_argument("--stats-path", default=DEFAULT_STATS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--max-new-tokens", type=int, default=1500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--diffusion-steps", type=int, default=DIFFUSION_STEPS)
    p.add_argument("--cfg-rate", type=float, default=INFERENCE_CFG_RATE)
    p.add_argument("--duration-factor", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.codes_npy and (not args.checkpoint or not args.text):
        raise SystemExit("--checkpoint and --text are required unless --codes-npy is set")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    if args.codes_npy:
        codes = _strip_special_codes(torch.from_numpy(np.load(args.codes_npy)))
        print(f"loaded {codes.numel()} codes from {args.codes_npy} ({codes.numel()/CODE_FPS:.2f}s @ 25Hz)")
    else:
        print(f"loading text2semantic from {args.checkpoint}")
        t2s = load_t2s(
            args.checkpoint,
            w2v_bert_path=args.w2v_bert_path,
            stats_path=args.stats_path,
            device=str(device),
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        )
        print(f"generating 25Hz codes for: {args.text}")
        codes = generate_codes(
            t2s,
            args.text,
            args.ref_audio,
            language=args.language,
            emotion=args.emotion,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"  {codes.numel()} codes ({codes.numel()/CODE_FPS:.2f}s @ 25Hz)")
        del t2s
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.save_codes:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_codes)) or ".", exist_ok=True)
        np.save(args.save_codes, codes.cpu().numpy().astype(np.int32))
        print(f"saved codes → {args.save_codes}")

    print("loading IndexTTS-2.5 codec + s2mel + BigVGAN")
    vocoder = IndexTTS25Vocoder(
        indextts_root=args.indextts_root,
        codec_dir=args.codec_dir,
        bigvgan_dir=args.bigvgan_dir,
        device=device,
        diffusion_steps=args.diffusion_steps,
        cfg_rate=args.cfg_rate,
        duration_factor=args.duration_factor,
    )
    wav, info = vocoder.vocode(codes, args.ref_audio)
    save_wav(args.out, wav, info["sample_rate"])
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"saved wav → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

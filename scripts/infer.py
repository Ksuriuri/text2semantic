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
4. Selectable acoustic backend:
   - s2mel: CAMPPlus + CFM + BigVGAN → 22.05 kHz wav.
   - s2vae: CFM predicts frozen dots.tts AudioVAE latents → 48 kHz wav.

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
DEFAULT_SEMANTIC2ANY_ROOT = _first_existing(
    "/mnt/data_sdd/hhy/noiz-tts/semantic2any",
    "/home/babysor00/t2s/semantic2any",
)
DEFAULT_S2VAE_CONFIG = os.path.join(
    DEFAULT_SEMANTIC2ANY_ROOT, "configs/s2vae_dit_indextts25_hf_equalized5k.yaml"
)
DEFAULT_S2VAE_CHECKPOINT = os.path.join(
    DEFAULT_SEMANTIC2ANY_ROOT,
    "exp/s2vae_dit_indextts25_hf_equalized5k/s2mel_step311673.pth",
)
DEFAULT_DOTS_TTS_DIR = os.path.join(DEFAULT_SEMANTIC2ANY_ROOT, "checkpoints/dots-tts")

# Official infer_v2_5.py constants. length_regulator sees *decoded* 50 Hz
# features, so this is the same 1.72 used by the old 50 Hz codec — decode()
# has already undone the 25 Hz downsample.
TARGET_LENGTH_RATIO = 1.72
DIFFUSION_STEPS = 25
INFERENCE_CFG_RATE = 0.7
PROMPT_MAX_SECONDS = 15.0
S2VAE_PROMPT_MIN_SECONDS = 2.0
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


class S2VAEVocoder:
    """IndexTTS-2.5 codes to dots.tts AudioVAE latents and 48 kHz audio.

    Prompt and target codes are decoded independently because EnhancedCodec's
    decoder is context dependent. The flow model must stay in float32; fp16
    Euler sampling collapses the VAE latents to near-silence.
    """

    def __init__(
        self,
        *,
        semantic2any_root: str,
        config_path: str,
        checkpoint_path: str,
        dots_tts_dir: str,
        indextts_root: str,
        codec_dir: str,
        device: torch.device,
        diffusion_steps: int = DIFFUSION_STEPS,
        cfg_rate: float = INFERENCE_CFG_RATE,
        temperature: float = 0.7,
        prompt_min_seconds: float = S2VAE_PROMPT_MIN_SECONDS,
        prompt_max_seconds: float = PROMPT_MAX_SECONDS,
    ):
        root = os.path.abspath(semantic2any_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from omegaconf import OmegaConf
        from semantic2any.models import Semantic2MelModel
        from semantic2any.utils.checkpoint import load_compatible_checkpoint
        from semantic2any.utils.dots_audiovae import DotsAudioVAE

        self.device = device
        self.diffusion_steps = int(diffusion_steps)
        self.cfg_rate = float(cfg_rate)
        self.temperature = float(temperature)
        self.prompt_min_seconds = float(prompt_min_seconds)
        self.prompt_max_seconds = min(float(prompt_max_seconds), 30.0)
        if self.prompt_min_seconds <= 0 or self.prompt_min_seconds > self.prompt_max_seconds:
            raise ValueError(
                "s2vae prompt_min_seconds must be positive and no greater than "
                f"prompt_max_seconds ({self.prompt_min_seconds} vs {self.prompt_max_seconds})"
            )

        cfg = OmegaConf.load(config_path)
        model = Semantic2MelModel(cfg.s2mel)
        load_compatible_checkpoint(model, checkpoint_path, strict=True)
        self.model = model.to(device=device, dtype=torch.float32).eval()
        self.model.requires_grad_(False)
        self.model.models["cfm"].setup_estimator_caches(
            max_batch_size=2 if self.cfg_rate > 0 else 1,
            max_seq_length=int(cfg.s2mel.DiT.block_size),
        )
        self.audio_vae = DotsAudioVAE.from_pretrained(dots_tts_dir).to(device).eval()

        mods = _import_indextts(indextts_root)
        codec_cfg = mods["OmegaConf"].load(os.path.join(codec_dir, "config.yaml"))
        codec = mods["EnhancedCodec"](
            **codec_cfg.semantic_codec, cfg=codec_cfg.semantic_codec
        )
        codec.load_checkpoint(os.path.join(codec_dir, "codec.pth"))
        self.semantic_codec = codec.to(device).eval()
        self.semantic_codec.requires_grad_(False)

    def _load_prompt_wav(self, ref_audio: str) -> tuple[torch.Tensor, int]:
        import librosa

        audio, sr = librosa.load(ref_audio, sr=None, mono=True)
        audio = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        max_n = int(self.prompt_max_seconds * sr)
        audio = audio[:, :max_n]
        audio = torchaudio.functional.resample(audio, sr, int(self.audio_vae.sample_rate))
        sample_count = int(audio.shape[-1])
        hop = int(self.audio_vae.hop_size)
        min_frames = int(np.ceil(self.prompt_min_seconds * CODE_FPS))
        if sample_count // hop < min_frames:
            raise ValueError(
                "s2vae reference audio must be at least "
                f"{self.prompt_min_seconds:g} seconds"
            )
        return audio.to(self.device), sample_count

    @torch.no_grad()
    def _prompt_codes(self, prompt_features: torch.Tensor, prompt_feature_length: int):
        feature = prompt_features[:, : int(prompt_feature_length)].to(
            self.device, dtype=torch.float32
        )
        codes, _ = self.semantic_codec.quantize(feature)
        return _strip_special_codes(codes)

    @torch.no_grad()
    def vocode(
        self,
        codes: torch.Tensor,
        ref_audio: str,
        *,
        prompt_features: torch.Tensor,
        prompt_feature_length: int,
    ) -> tuple[torch.Tensor, dict]:
        target_codes = _strip_special_codes(codes).to(self.device)
        if target_codes.numel() < 8:
            raise ValueError("s2vae target must contain at least 8 codes")
        if target_codes.numel() > 750:
            raise ValueError("s2vae target exceeds the trained 30-second limit (750 codes)")

        prompt_wav, prompt_samples = self._load_prompt_wav(ref_audio)
        prompt_latent = self.audio_vae.encode_mean(
            prompt_wav.unsqueeze(0),
            sample_lengths=[prompt_samples],
            normalize=True,
        )
        prompt_codes = self._prompt_codes(prompt_features, prompt_feature_length)
        prompt_frames = min(int(prompt_latent.shape[-1]), int(prompt_codes.numel()))
        min_frames = int(np.ceil(self.prompt_min_seconds * CODE_FPS))
        if prompt_frames < min_frames:
            raise ValueError(
                "s2vae aligned reference prompt is shorter than "
                f"{self.prompt_min_seconds:g} seconds"
            )
        prompt_latent = prompt_latent[:, :, :prompt_frames].float()
        prompt_codes = prompt_codes[:prompt_frames]

        # Never decode across the prompt/target seam: EnhancedCodec mixes context.
        prompt_sem = self.semantic_codec.decode(prompt_codes.unsqueeze(0)).float()
        target_sem = self.semantic_codec.decode(target_codes.unsqueeze(0)).float()
        semantic = torch.cat([prompt_sem, target_sem], dim=1)
        target_frames = int(target_codes.numel())
        total_frames = prompt_frames + target_frames
        x_lens = torch.tensor([total_frames], device=self.device, dtype=torch.long)
        semantic_lens = torch.tensor(
            [semantic.shape[1]], device=self.device, dtype=torch.long
        )
        mu = self.model.build_condition(
            semantic, x_lens, semantic_lens=semantic_lens
        )
        latent = self.model.models["cfm"].inference(
            mu=mu,
            x_lens=x_lens,
            prompt=prompt_latent,
            style=torch.zeros(1, 192, device=self.device),
            f0=None,
            n_timesteps=self.diffusion_steps,
            temperature=self.temperature,
            inference_cfg_rate=self.cfg_rate,
            drop_style=True,
        )
        target_latent = latent[:, :, prompt_frames:total_frames]
        wav = self.audio_vae.decode(target_latent.float(), normalized=True)[0]
        wav = torch.clamp(wav, -1.0, 1.0).cpu()
        sample_rate = int(self.audio_vae.sample_rate)
        info = {
            "backend": "s2vae",
            "n_codes": target_frames,
            "code_seconds": float(target_frames / CODE_FPS),
            "prompt_codes": prompt_frames,
            "target_latent_frames": target_frames,
            "wav_seconds": float(wav.shape[-1] / sample_rate),
            "sample_rate": sample_rate,
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


def generate_codes(
    t2s,
    text: str,
    ref_audio: str,
    *,
    return_prompt_features: bool = False,
    **gen_kwargs,
):
    result = t2s.generate(
        text,
        ref_audio,
        return_speaker_features=return_prompt_features,
        **gen_kwargs,
    )
    if not return_prompt_features:
        return _strip_special_codes(result[0].detach().cpu())
    codes_list, features, lengths = result
    codes = _strip_special_codes(codes_list[0].detach().cpu())
    return codes, features[0:1].detach(), int(lengths[0].item())


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
        choices=("auto", "ar", "de", "en", "es", "fr", "ja", "ko", "pt", "ru", "zh"),
        default="auto",
        help="Language control placed before the text; auto adds no language tag.",
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
    p.add_argument("--vocoder-backend", choices=("s2vae", "s2mel"), default="s2vae")
    p.add_argument("--semantic2any-root", default=DEFAULT_SEMANTIC2ANY_ROOT)
    p.add_argument("--s2vae-config", default=DEFAULT_S2VAE_CONFIG)
    p.add_argument("--s2vae-checkpoint", default=DEFAULT_S2VAE_CHECKPOINT)
    p.add_argument("--dots-tts-dir", default=DEFAULT_DOTS_TTS_DIR)
    p.add_argument("--s2vae-temperature", type=float, default=0.7)
    p.add_argument(
        "--s2vae-prompt-min-seconds",
        type=float,
        default=S2VAE_PROMPT_MIN_SECONDS,
    )
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
        generated = generate_codes(
            t2s,
            args.text,
            args.ref_audio,
            return_prompt_features=args.vocoder_backend == "s2vae",
            language=args.language,
            emotion=args.emotion,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        if args.vocoder_backend == "s2vae":
            codes, prompt_features, prompt_feature_length = generated
        else:
            codes = generated
        print(f"  {codes.numel()} codes ({codes.numel()/CODE_FPS:.2f}s @ 25Hz)")
        del t2s
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.save_codes:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_codes)) or ".", exist_ok=True)
        np.save(args.save_codes, codes.cpu().numpy().astype(np.int32))
        print(f"saved codes → {args.save_codes}")

    if args.vocoder_backend == "s2vae":
        if args.codes_npy:
            raise SystemExit("--codes-npy with s2vae is unsupported: prompt features are required")
        print("loading S2VAE step checkpoint + dots.tts AudioVAE")
        vocoder = S2VAEVocoder(
            semantic2any_root=args.semantic2any_root,
            config_path=args.s2vae_config,
            checkpoint_path=args.s2vae_checkpoint,
            dots_tts_dir=args.dots_tts_dir,
            indextts_root=args.indextts_root,
            codec_dir=args.codec_dir,
            device=device,
            diffusion_steps=args.diffusion_steps,
            cfg_rate=args.cfg_rate,
            temperature=args.s2vae_temperature,
            prompt_min_seconds=args.s2vae_prompt_min_seconds,
        )
        wav, info = vocoder.vocode(
            codes,
            args.ref_audio,
            prompt_features=prompt_features,
            prompt_feature_length=prompt_feature_length,
        )
    else:
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
        info["backend"] = "s2mel"
    save_wav(args.out, wav, info["sample_rate"])
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"saved wav → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

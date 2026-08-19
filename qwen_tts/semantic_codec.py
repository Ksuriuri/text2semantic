# Copyright (c) 2024 Amphion.
# Copyright 2026
# SPDX-License-Identifier: MIT
"""Minimal RepCodec inference stack used to create semantic labels.

Two codecs share this stack: IndexTTS-2.5's EnhancedCodec (the default, 25 Hz
codes) and MaskGCT's original semantic codec (50 Hz codes).  They differ only in
``downsample_scale``; see :data:`SEMANTIC_CODECS`.
"""

from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from safetensors.torch import load_model
from torch import nn
from torch.nn.utils import weight_norm
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel


#: The semantic codecs this module can run.  ``indextts25`` is IndexTTS-2.5's
#: EnhancedCodec, the default: one code per two w2v-bert frames, so 25 Hz codes.
#: ``maskgct`` is the original 50 Hz codec, kept selectable because the existing
#: code shards in the bucket were produced with it and its ids are not
#: interchangeable with the new ones.
SEMANTIC_CODECS = {
    "indextts25": {
        "downsample_scale": 2,
        "frames_per_code": 2,
        "code_fps": 25.0,
        "feature_fps": 50.0,
        "asset_dir": "enhanced_codec",
        "config_name": "config.yaml",
        "checkpoint_name": "codec.pth",
        "source_model": "IndexTTS-2.5/EnhancedCodec",
    },
    "maskgct": {
        "downsample_scale": 1,
        "frames_per_code": 1,
        "code_fps": 50.0,
        "feature_fps": 50.0,
        "asset_dir": "semantic_codec",
        "config_name": "config.yaml",
        "checkpoint_name": "model.safetensors",
        "source_model": "amphion/MaskGCT",
    },
}

DEFAULT_SEMANTIC_CODEC = "indextts25"

#: Manifest spellings that mean the same codec.  The code-generation workers
#: stamp "indextts2.5"; the selector here stays a plain identifier because it
#: also names files and directories.
SEMANTIC_CODEC_ALIASES = {
    "indextts2.5": "indextts25",
    "indextts-2.5": "indextts25",
    "indextts_2.5": "indextts25",
    "enhancedcodec": "indextts25",
}


#: Floating point dtypes selectable from the command line.
FLOAT_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_float_dtype(dtype):
    """Accept ``None``, a ``torch.dtype``, or one of :data:`FLOAT_DTYPES`."""
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        if not dtype.is_floating_point:
            raise ValueError(f"{dtype} is not a floating point dtype.")
        return dtype
    key = str(dtype).strip().lower()
    if key not in FLOAT_DTYPES:
        raise ValueError(
            f"Unknown dtype {dtype!r}; choose from {sorted(FLOAT_DTYPES)}"
        )
    return FLOAT_DTYPES[key]


def canonical_semantic_codec(name):
    key = str(name).strip().lower()
    return SEMANTIC_CODEC_ALIASES.get(key, key)


def semantic_codec_spec(name):
    """Rates and asset names for one codec; raises on an unknown selector."""
    key = canonical_semantic_codec(name)
    if key not in SEMANTIC_CODECS:
        raise ValueError(
            f"Unknown semantic codec {name!r}; choose from "
            f"{sorted(SEMANTIC_CODECS)}"
        )
    return dict(SEMANTIC_CODECS[key], name=key)


def resolve_codec_assets(model_dir, codec_type=DEFAULT_SEMANTIC_CODEC):
    """(config, checkpoint) for a codec inside a scripts/download_models.py dir."""
    spec = semantic_codec_spec(codec_type)
    root = Path(model_dir) / spec["asset_dir"]
    return root / spec["config_name"], root / spec["checkpoint_name"]


def load_codec_config(path, codec_type=DEFAULT_SEMANTIC_CODEC):
    """Read a RepCodec architecture block and reconcile it with the codec type.

    ``downsample_scale`` is what separates the two codecs, and the published
    IndexTTS-2.5 config omits it (upstream hard-codes it), so fill it in from the
    codec type -- but never override a value the file states, since a silent
    override would produce ids that decode to noise.
    """
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if isinstance(config, dict) and "model" in config:
        config = config["model"]
    config = config.get("semantic_codec", config)
    config = {key: value for key, value in config.items()}
    spec = semantic_codec_spec(codec_type)
    stated = config.get("downsample_scale")
    if stated is None:
        config["downsample_scale"] = spec["downsample_scale"]
    elif int(stated) != int(spec["downsample_scale"]):
        raise ValueError(
            f"{path} declares downsample_scale={stated}, but codec "
            f"{spec['name']!r} needs {spec['downsample_scale']}"
        )
    return config


def load_codec_checkpoint(codec, checkpoint_path):
    """Load ``.safetensors`` (MaskGCT) or ``.pth`` (IndexTTS-2.5) weights."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        load_model(codec, str(checkpoint_path), strict=True)
        return
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("model", "state_dict", "codec", "semantic_codec"):
            inner = state.get(key)
            if isinstance(inner, dict) and any(
                isinstance(value, torch.Tensor) for value in inner.values()
            ):
                state = inner
                break
    codec.load_state_dict(state, strict=True)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, layer_scale_init_value):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        return residual + (self.gamma * x).transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(self, input_channels, dim, intermediate_dim, num_layers):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = False
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(dim, intermediate_dim, 1 / num_layers)
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        commitment=0.15,
        codebook_loss_weight=1.0,
        use_l2_normlize=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.use_l2_normlize = use_l2_normlize
        self.in_project = weight_norm(
            nn.Conv1d(input_dim, codebook_dim, kernel_size=1)
        )
        self.out_project = weight_norm(
            nn.Conv1d(codebook_dim, input_dim, kernel_size=1)
        )
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        z_e = self.in_project(z)
        encodings = rearrange(z_e, "b d t -> (b t) d")
        codebook = self.codebook.weight
        if self.use_l2_normlize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)
        distances = (
            encodings.square().sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.square().sum(1, keepdim=True).t()
        )
        indices = rearrange(
            distances.argmin(1), "(b t) -> b t", b=z.size(0)
        )
        z_q = F.embedding(indices, self.codebook.weight).transpose(1, 2)
        commit_loss = torch.zeros(z.size(0), device=z.device)
        codebook_loss = torch.zeros(z.size(0), device=z.device)
        return self.out_project(z_q), commit_loss, codebook_loss, indices, z_e

    def decode_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight).transpose(1, 2)

    def vq2emb(self, vq, out_proj=True):
        embedding = self.decode_code(vq)
        return self.out_project(embedding) if out_proj else embedding


class ResidualVQ(nn.Module):
    def __init__(self, input_dim, codebook_size, codebook_dim):
        super().__init__()
        self.input_dim = input_dim
        self.num_quantizers = 1
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_type = "fvq"
        self.quantizer_dropout = 0.0
        self.quantizers = nn.ModuleList(
            [
                FactorizedVectorQuantize(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                )
            ]
        )

    def forward(self, z):
        z_q, commit, codebook, indices, encoded = self.quantizers[0](z)
        return (
            z_q,
            indices.unsqueeze(0),
            commit.unsqueeze(0),
            codebook.unsqueeze(0),
            encoded.unsqueeze(0),
        )

    def vq2emb(self, vq, n_quantizers=None):
        quantized_out = 0.0
        n_quantizers = self.num_quantizers if n_quantizers is None else n_quantizers
        for index, quantizer in enumerate(self.quantizers):
            if index >= n_quantizers:
                break
            quantized_out += quantizer.vq2emb(vq[index])
        return quantized_out


class RepCodec(nn.Module):
    def __init__(
        self,
        codebook_size=8192,
        hidden_size=1024,
        codebook_dim=8,
        vocos_dim=384,
        vocos_intermediate_dim=2048,
        vocos_num_layers=12,
        num_quantizers=1,
        downsample_scale=1,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.num_quantizers = num_quantizers
        self.downsample_scale = downsample_scale
        if self.downsample_scale is not None and self.downsample_scale > 1:
            # IndexTTS-2.5's EnhancedCodec: the stride-2 conv is what puts the
            # codes at half the feature rate, and `up` undoes it on decode.
            self.down = nn.Conv1d(
                hidden_size, hidden_size, kernel_size=3, stride=2, padding=1
            )
            self.up = nn.Conv1d(
                hidden_size, hidden_size, kernel_size=3, stride=1, padding=1
            )
        self.encoder = nn.Sequential(
            VocosBackbone(
                hidden_size,
                vocos_dim,
                vocos_intermediate_dim,
                vocos_num_layers,
            ),
            nn.Linear(vocos_dim, hidden_size),
        )
        # Label extraction only calls the encoder and quantizer, but the decoder
        # is needed for checkpoint compatibility and for decode().
        self.decoder = nn.Sequential(
            VocosBackbone(
                hidden_size,
                vocos_dim,
                vocos_intermediate_dim,
                vocos_num_layers,
            ),
            nn.Linear(vocos_dim, hidden_size),
        )
        self.quantizer = ResidualVQ(hidden_size, codebook_size, codebook_dim)

    def quantize(self, x):
        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = F.gelu(self.down(x.transpose(1, 2))).transpose(1, 2)
        encoded = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        quantized, indices, _, _, _ = self.quantizer(encoded)
        return indices.squeeze(0), quantized.transpose(1, 2)

    def decode(self, codes):
        """Reconstruct w2v-bert features from code ids.

        With ``downsample_scale == 2`` the result is twice as long as ``codes``,
        back at the 50 Hz feature rate.  This is not a per-code lookup: the
        decoder is a ConvNeXt stack, so frame t depends on its neighbours and
        each row must be one unpadded utterance -- padding a batch changes the
        values near every tail.
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        if codes.dim() != 3:
            raise ValueError(
                f"codes must be [B,T] or [N,B,T], got {tuple(codes.shape)}"
            )
        x = self.decoder(self.quantizer.vq2emb(codes.long()))
        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = F.interpolate(
                x.transpose(1, 2),
                scale_factor=int(self.downsample_scale),
                mode="nearest",
            )
            x = self.up(x).transpose(1, 2)
        return x


class SpeakerMelExtractor:
    """The CPU half of :class:`MaskGCTFeatureExtractor`: the 80-bin log-mel.

    Split out so a DataLoader worker can own it. In the training loop this ran
    inline on the main process, single-threaded, for batch_size ref clips per
    rank per step -- 912 ms of a 3.955 s B200 step at batch_size 32 and the 20 s
    padded ref length, with the GPU idle throughout. It holds no model weights,
    so a copy per worker costs nothing.
    """

    def __init__(self, w2v_bert_path):
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            w2v_bert_path
        )

    def __call__(self, audios, max_audio_seconds=15.0):
        """(input_features, attention_mask) on the CPU, padded to the batch."""
        if not audios:
            raise ValueError("audios must not be empty.")
        if max_audio_seconds is not None and max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive or None.")
        if max_audio_seconds is not None:
            limit = int(16000 * max_audio_seconds)
            audios = [audio[:limit] for audio in audios]
        inputs = self.feature_extractor(
            audios,
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return inputs.input_features, inputs.get("attention_mask")


class MaskGCTFeatureExtractor:
    """Frozen W2V-BERT layer-17 feature extractor used by MaskGCT.

    ``dtype`` selects the precision of the frozen forward.  It stays fp32 by
    default because this class also feeds the RepCodec tokenizer, whose ids must
    be bit-reproducible against the shards already in the bucket.  Speaker
    conditioning has no such constraint: the layer-17 output is mean/std
    normalised and consumed by a trainable encoder, and bf16 measured 3.2x
    faster on B200 for a 0.031 relative difference.
    """

    # Off by default, and deliberately so: MaskGCTSemanticTokenizer builds one of
    # these, and its RepCodec ids have to stay bit-reproducible against the shards
    # already in the bucket. Chunking is exact in exact arithmetic -- rows do not
    # attend to each other -- but cuBLAS picks different kernels for a batch of 8
    # than for 48, and in bf16 that measured a 4.3e-2 relative difference. Only
    # the trainer, whose features are mean/std normalised and fed to a trainable
    # encoder, opts in. Also a class attribute so an instance built with __new__ --
    # which is how the tests avoid downloading W2V-BERT -- still has a chunk size.
    chunk_size = 0

    def __init__(
        self,
        *,
        w2v_bert_path,
        stats_path,
        device="cuda:0",
        dtype=torch.float32,
        chunk_size=None,
    ):
        self.device = torch.device(device)
        self.dtype = resolve_float_dtype(dtype)
        # W2V-BERT uses relative_key attention, which materialises a
        # batch x L x L x head_dim tensor. Measured on one H100 at batch 48 and
        # 20 s refs, the frozen forward peaks at 11.15 GiB whole and 3.24 GiB in
        # chunks of 8, at ~1.3% of a step either way. Worth knowing, but it did
        # not move the training step's peak (75.8 GiB at batch 48 either way):
        # that peak belongs to a different moment in the step, and what 20 s refs
        # really cost is the *trainable* speaker encoder's retained activations
        # over twice as many frames.
        if chunk_size is not None:
            self.chunk_size = chunk_size
        self.mel_extractor = SpeakerMelExtractor(w2v_bert_path)
        self.feature_extractor = self.mel_extractor.feature_extractor
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            w2v_bert_path, torch_dtype=self.dtype
        ).to(self.device, dtype=self.dtype)
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        self.mean = stats["mean"].to(self.device, dtype=torch.float32)
        self.std = stats["var"].sqrt().to(self.device, dtype=torch.float32)
        self.semantic_model.eval()
        self.semantic_model.requires_grad_(False)

    @torch.inference_mode()
    def encode_files(self, audio_paths, max_audio_seconds=15.0):
        if not audio_paths:
            raise ValueError("audio_paths must not be empty.")
        if max_audio_seconds is not None and max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive or None.")

        audios = [
            librosa.load(
                Path(audio_path),
                sr=16000,
                mono=True,
                duration=max_audio_seconds,
            )[0]
            for audio_path in audio_paths
        ]
        return self.encode_audios(audios, max_audio_seconds=max_audio_seconds)

    @torch.inference_mode()
    def encode_audios(self, audios, max_audio_seconds=15.0):
        """Same as :meth:`encode_files`, for waveforms that are already decoded.

        Packed refs are read out of their shard and decoded in the DataLoader
        workers, so the training loop hands the waveforms straight over instead
        of writing them out and reading them back.
        """
        if not audios:
            raise ValueError("audios must not be empty.")
        if max_audio_seconds is not None and max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive or None.")

        return self.encode_features(
            *self.mel_extractor(audios, max_audio_seconds=max_audio_seconds)
        )

    def _chunks(self, batch_size):
        step = self.chunk_size if self.chunk_size and self.chunk_size > 0 else batch_size
        return [
            (start, min(start + step, batch_size))
            for start in range(0, batch_size, step)
        ]

    @torch.inference_mode()
    def encode_features(self, input_features, attention_mask=None):
        """The GPU half: layer-17 features from an already computed log-mel.

        Callable directly so the mel can be produced in a DataLoader worker
        rather than inline in the training step.

        With ``chunk_size`` set, the rows go through the frozen model in groups.
        Every row attends only to itself, so the result is the same in exact
        arithmetic; what changes is that the transient relative_key attention
        buffer is sized by the chunk instead of by the whole batch, and that the
        matmuls are shaped differently enough for cuBLAS to round differently.
        """
        input_features = input_features.to(self.device, self.dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            features = torch.cat(
                [
                    (
                        self.semantic_model(
                            input_features=input_features[start:stop],
                            attention_mask=(
                                None if attention_mask is None
                                else attention_mask[start:stop]
                            ),
                            output_hidden_states=True,
                        ).hidden_states[17].float()
                        - self.mean
                    ) / self.std
                    for start, stop in self._chunks(input_features.size(0))
                ]
            )

        feature_length = features.size(1)
        if attention_mask is None:
            lengths = torch.full(
                (features.size(0),),
                feature_length,
                dtype=torch.long,
                device=self.device,
            )
        else:
            if attention_mask.size(1) < feature_length:
                attention_mask = F.pad(
                    attention_mask,
                    (0, feature_length - attention_mask.size(1)),
                )
            attention_mask = attention_mask[:, :feature_length]
            lengths = attention_mask.ne(0).sum(dim=1).clamp(
                min=0, max=feature_length
            )
            features = features.masked_fill(
                ~attention_mask.ne(0).unsqueeze(-1), 0.0
            )
        return features.float(), lengths.long()


class MaskGCTSemanticTokenizer:
    """Frozen W2V-BERT layer-17 plus single-codebook RepCodec tokenizer.

    ``codec_type`` selects between IndexTTS-2.5's EnhancedCodec (default) and
    MaskGCT's codec.  It is not cosmetic: the two emit ids at different rates
    from different codebooks, so codes from one cannot be decoded by the other.
    """

    def __init__(
        self,
        *,
        w2v_bert_path,
        stats_path,
        repcodec_config_path,
        repcodec_checkpoint_path,
        codec_type=DEFAULT_SEMANTIC_CODEC,
        device="cuda:0",
    ):
        self.device = torch.device(device)
        self.feature_extractor = MaskGCTFeatureExtractor(
            w2v_bert_path=w2v_bert_path,
            stats_path=stats_path,
            device=self.device,
        )

        spec = semantic_codec_spec(codec_type)
        self.codec_type = spec["name"]
        self.frames_per_code = int(spec["frames_per_code"])
        self.code_fps = float(spec["code_fps"])
        config = load_codec_config(repcodec_config_path, self.codec_type)
        self.codebook_size = int(config.get("codebook_size", 8192))
        self.codec = RepCodec(**config).to(self.device)
        load_codec_checkpoint(self.codec, repcodec_checkpoint_path)
        self.codec.eval()
        self.codec.requires_grad_(False)

    @classmethod
    def from_model_dir(
        cls,
        model_dir,
        *,
        codec_type=DEFAULT_SEMANTIC_CODEC,
        device="cuda:0",
    ):
        """Build from a scripts/download_models.py directory."""
        model_dir = Path(model_dir)
        config_path, checkpoint_path = resolve_codec_assets(model_dir, codec_type)
        return cls(
            w2v_bert_path=model_dir / "w2v-bert-2.0",
            stats_path=model_dir / "wav2vec2bert_stats.pt",
            repcodec_config_path=config_path,
            repcodec_checkpoint_path=checkpoint_path,
            codec_type=codec_type,
            device=device,
        )

    @torch.inference_mode()
    def decode_codes(self, codes):
        """Ids -> 50 Hz features, for verifying stored codes against the audio.

        One unpadded utterance per call: the decode is context dependent.
        """
        codes = torch.as_tensor(codes, device=self.device).long().reshape(1, -1)
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            return self.codec.decode(codes)[0].float().cpu()

    @torch.inference_mode()
    def encode_file(self, audio_path):
        if hasattr(self.feature_extractor, "encode_files"):
            features, lengths = self.feature_extractor.encode_files(
                [audio_path],
                max_audio_seconds=None,
            )
        else:
            # Keep the original components independently replaceable for
            # lightweight callers and tests.
            audio, _ = librosa.load(Path(audio_path), sr=16000, mono=True)
            inputs = self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(
                self.device, torch.float32
            )
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            with torch.amp.autocast(
                device_type=self.device.type, enabled=False
            ):
                outputs = self.semantic_model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                features = (
                    outputs.hidden_states[17].float() - self.mean
                ) / self.std
            feature_length = features.size(1)
            if attention_mask is None:
                lengths = torch.tensor(
                    [feature_length],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                lengths = attention_mask[:, :feature_length].ne(0).sum(dim=1)
                lengths = lengths.clamp(min=0, max=feature_length)
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            codes, _ = self.codec.quantize(features.float())
        codes = codes.squeeze(0).long().cpu()
        # lengths counts 50 Hz feature frames; the stride-2 conv rounds up, so
        # ceil() here rather than a floor that would drop a real final code.
        frames_per_code = getattr(self, "frames_per_code", 1)
        valid_codes = -(-int(lengths[0].item()) // frames_per_code)
        valid_length = min(valid_codes, codes.numel())
        codes = codes[:valid_length]
        if (
            codes.numel() == 0
            or codes.min() < 0
            or codes.max() >= self.codebook_size
        ):
            raise RuntimeError(f"Invalid semantic codes produced for {audio_path}.")
        return codes


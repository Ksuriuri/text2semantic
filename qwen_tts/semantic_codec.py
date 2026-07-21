# Copyright (c) 2024 Amphion.
# Copyright 2026
# SPDX-License-Identifier: MIT
"""Minimal MaskGCT RepCodec inference stack used to create semantic labels."""

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


class RepCodec(nn.Module):
    def __init__(
        self,
        codebook_size=8192,
        hidden_size=1024,
        codebook_dim=8,
        vocos_dim=384,
        vocos_intermediate_dim=2048,
        vocos_num_layers=12,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.num_quantizers = 1
        self.downsample_scale = 1
        self.encoder = nn.Sequential(
            VocosBackbone(
                hidden_size,
                vocos_dim,
                vocos_intermediate_dim,
                vocos_num_layers,
            ),
            nn.Linear(vocos_dim, hidden_size),
        )
        # The decoder is needed for checkpoint compatibility, although label
        # extraction only calls the encoder and quantizer.
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
        encoded = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        quantized, indices, _, _, _ = self.quantizer(encoded)
        return indices.squeeze(0), quantized.transpose(1, 2)


class MaskGCTFeatureExtractor:
    """Frozen W2V-BERT layer-17 feature extractor used by MaskGCT."""

    def __init__(
        self,
        *,
        w2v_bert_path,
        stats_path,
        device="cuda:0",
    ):
        self.device = torch.device(device)
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            w2v_bert_path
        )
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            w2v_bert_path, torch_dtype=torch.float32
        ).to(self.device, dtype=torch.float32)
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

        max_audio_samples = (
            None
            if max_audio_seconds is None
            else int(16000 * max_audio_seconds)
        )
        audios = []
        for audio_path in audio_paths:
            audio, _ = librosa.load(Path(audio_path), sr=16000, mono=True)
            if max_audio_samples is not None:
                audio = audio[:max_audio_samples]
            audios.append(audio)

        inputs = self.feature_extractor(
            audios,
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, torch.float32)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=False):
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
    """Frozen W2V-BERT layer-17 plus single-codebook RepCodec tokenizer."""

    def __init__(
        self,
        *,
        w2v_bert_path,
        stats_path,
        repcodec_config_path,
        repcodec_checkpoint_path,
        device="cuda:0",
    ):
        self.device = torch.device(device)
        self.feature_extractor = MaskGCTFeatureExtractor(
            w2v_bert_path=w2v_bert_path,
            stats_path=stats_path,
            device=self.device,
        )

        with open(repcodec_config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config = config.get("semantic_codec", config)
        self.codebook_size = int(config.get("codebook_size", 8192))
        self.codec = RepCodec(**config).to(self.device)
        load_model(self.codec, str(repcodec_checkpoint_path), strict=True)
        self.codec.eval()
        self.codec.requires_grad_(False)

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
        valid_length = min(int(lengths[0].item()), codes.numel())
        codes = codes[:valid_length]
        if (
            codes.numel() == 0
            or codes.min() < 0
            or codes.max() >= self.codebook_size
        ):
            raise RuntimeError(f"Invalid semantic codes produced for {audio_path}.")
        return codes


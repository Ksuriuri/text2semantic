# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-length speaker conditioning encoder."""

import torch
from torch import nn

from .conformer_encoder import ConformerEncoder
from .perceiver import PerceiverResampler


class SpeakerConditioningEncoder(nn.Module):
    """Encode variable-length speaker features into fixed conditioning tokens.

    Args:
        input_dim: Feature dimension of ``speaker_features``.
        conformer_output_dim: Hidden dimension produced by the Conformer.
        conformer_num_blocks: Number of Conformer blocks.
        conformer_attention_heads: Number of Conformer attention heads.
        conformer_linear_units: Hidden dimension of Conformer feed-forward layers.
        conformer_input_layer: Input projection/subsampling type.
        perceiver_num_latents: Number of fixed output tokens.
        perceiver_latent_dim: Feature dimension of fixed output tokens.
        perceiver_depth: Number of Perceiver cross-attention layers.
        perceiver_heads: Number of Perceiver attention heads.
        perceiver_ff_mult: Perceiver feed-forward multiplier.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        conformer_output_dim: int = 512,
        conformer_num_blocks: int = 6,
        conformer_attention_heads: int = 8,
        conformer_linear_units: int = 2048,
        conformer_input_layer: str = "conv2d2",
        perceiver_num_latents: int = 32,
        perceiver_latent_dim: int = 1280,
        perceiver_depth: int = 2,
        perceiver_heads: int = 8,
        perceiver_ff_mult: float = 2,
        dropout_rate: float = 0.0,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_latents = perceiver_num_latents
        self.latent_dim = perceiver_latent_dim

        self.conformer = ConformerEncoder(
            input_size=input_dim,
            output_size=conformer_output_dim,
            attention_heads=conformer_attention_heads,
            linear_units=conformer_linear_units,
            num_blocks=conformer_num_blocks,
            dropout_rate=dropout_rate,
            input_layer=conformer_input_layer,
        )
        self.perceiver = PerceiverResampler(
            dim=perceiver_latent_dim,
            depth=perceiver_depth,
            dim_context=conformer_output_dim,
            num_latents=perceiver_num_latents,
            heads=perceiver_heads,
            ff_mult=perceiver_ff_mult,
            use_flash_attn=use_flash_attn,
        )

    def forward(
        self,
        speaker_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return speaker conditioning with shape ``[B, num_latents, latent_dim]``.

        ``lengths`` describes valid frames before Conformer subsampling. The
        Conformer returns the corresponding post-subsampling validity mask.
        Because each Perceiver cross-attention layer prepends its latent queries
        to the key/value context, an all-valid latent mask is prepended as well.
        """
        if speaker_features.ndim != 3:
            raise ValueError(
                "speaker_features must have shape [B, T, input_dim], "
                f"got {tuple(speaker_features.shape)}"
            )
        batch, time, feature_dim = speaker_features.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"expected speaker feature dimension {self.input_dim}, "
                f"got {feature_dim}"
            )
        if lengths.ndim != 1 or lengths.shape[0] != batch:
            raise ValueError(
                f"lengths must have shape [{batch}], got {tuple(lengths.shape)}"
            )
        if torch.any(lengths < 0) or torch.any(lengths > time):
            raise ValueError(f"lengths must be in the range [0, {time}]")

        lengths = lengths.to(device=speaker_features.device, dtype=torch.long)
        encoded, valid_mask = self.conformer(speaker_features, lengths)
        context_mask = valid_mask.squeeze(1).bool()
        latent_mask = torch.ones(
            (batch, self.num_latents),
            dtype=torch.bool,
            device=context_mask.device,
        )
        perceiver_mask = torch.cat((latent_mask, context_mask), dim=1)
        return self.perceiver(encoded, mask=perceiver_mask)

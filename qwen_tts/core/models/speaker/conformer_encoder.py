# Speaker Conformer adapted from IndexTTS2.
# Conformer components retain their original Apache-2.0 notices in
# speaker/conformer/{attention,embedding,subsampling}.py.

"""IndexTTS2-style Conformer encoder for speaker features."""

from typing import Optional, Tuple

import torch
from torch import nn

from .conformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from .conformer.embedding import (
    NoPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
)
from .conformer.subsampling import (
    Conv2dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    LinearNoSubsampling,
)
from .utils import make_pad_mask


class PositionwiseFeedForward(nn.Module):
    """Position-wise two-layer feed-forward network."""

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class ConvolutionModule(nn.Module):
    """Conformer convolution module."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        bias: bool = True,
    ):
        super().__init__()
        if (kernel_size - 1) % 2 != 0:
            raise ValueError("kernel_size must be odd for non-causal convolution")

        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1, bias=bias)
        self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.use_layer_norm = True
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1, bias=bias)
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        if mask_pad is not None and mask_pad.size(-1) > 0:
            x = x.masked_fill(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache is None or cache.size(2) == 0:
                x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                x = torch.cat((cache, x), dim=2)
            new_cache = x[:, :, -self.lorder :]
        else:
            new_cache = x.new_zeros((0, 0, 0))

        x = nn.functional.glu(self.pointwise_conv1(x), dim=1)
        x = self.depthwise_conv(x).transpose(1, 2)
        x = self.activation(self.norm(x)).transpose(1, 2)
        x = self.pointwise_conv2(x)
        if mask_pad is not None and mask_pad.size(-1) > 0:
            x = x.masked_fill(~mask_pad, 0.0)
        return x.transpose(1, 2), new_cache


class ConformerEncoderLayer(nn.Module):
    """One IndexTTS2-style Conformer block."""

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()
        if feed_forward is None:
            raise ValueError("feed_forward is required")
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-5)
            self.norm_final = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = (
            nn.Linear(size + size, size) if concat_after else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        att_cache: Optional[torch.Tensor] = None,
        cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, pos_emb, att_cache
        )
        if self.concat_after:
            x = residual + self.concat_linear(torch.cat((x, x_att), dim=-1))
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        new_cnn_cache = x.new_zeros((0, 0, 0))
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)
        if self.conv_module is not None:
            x = self.norm_final(x)
        return x, mask, new_att_cache, new_cnn_cache


class BaseEncoder(nn.Module):
    """Shared input embedding and forward path for Conformer encoders."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()
        del attention_heads, linear_units, num_blocks, concat_after
        self._output_size = output_size

        positional_encodings = {
            "abs_pos": PositionalEncoding,
            "rel_pos": RelPositionalEncoding,
            "no_pos": NoPositionalEncoding,
        }
        subsampling_layers = {
            "linear": LinearNoSubsampling,
            "conv2d2": Conv2dSubsampling2,
            "conv2d": Conv2dSubsampling4,
            "conv2d6": Conv2dSubsampling6,
            "conv2d8": Conv2dSubsampling8,
        }
        if pos_enc_layer_type not in positional_encodings:
            raise ValueError(f"unknown pos_enc_layer: {pos_enc_layer_type}")
        if input_layer not in subsampling_layers:
            raise ValueError(f"unknown input_layer: {input_layer}")

        pos_enc_class = positional_encodings[pos_enc_layer_type]
        subsampling_class = subsampling_layers[input_layer]
        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, dropout_rate),
        )
        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_mask = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb, valid_mask = self.embed(xs, valid_mask)
        mask_pad = valid_mask
        chunk_mask = valid_mask
        for layer in self.encoders:
            xs, chunk_mask, _, _ = layer(
                xs, chunk_mask, pos_emb, mask_pad
            )
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, valid_mask


class ConformerEncoder(BaseEncoder):
    """Conformer encoder matching the IndexTTS2 parameterization."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        macaron_style: bool = False,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
    ):
        super().__init__(
            input_size,
            output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            input_layer,
            pos_enc_layer_type,
            normalize_before,
            concat_after,
        )

        activation = nn.SiLU()
        attention_class = (
            RelPositionMultiHeadedAttention
            if pos_enc_layer_type == "rel_pos"
            else MultiHeadedAttention
        )
        attention_args = (attention_heads, output_size, dropout_rate)
        feed_forward_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        convolution_args = (output_size, cnn_module_kernel, activation)

        self.encoders = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    attention_class(*attention_args),
                    PositionwiseFeedForward(*feed_forward_args),
                    (
                        PositionwiseFeedForward(*feed_forward_args)
                        if macaron_style
                        else None
                    ),
                    (
                        ConvolutionModule(*convolution_args)
                        if use_cnn_module
                        else None
                    ),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                )
                for _ in range(num_blocks)
            ]
        )

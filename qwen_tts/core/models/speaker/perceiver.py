# Adapted from:
# https://github.com/lucidrains/naturalspeech2-pytorch/blob/659bec7f7543e7747e809e950cc2f84242fbeec7/naturalspeech2_pytorch/naturalspeech2_pytorch.py#L532

"""Perceiver resampler adapted from the IndexTTS2 implementation."""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn


class Attend(nn.Module):
    """Attention operation with optional PyTorch SDPA."""

    def __init__(
        self,
        dropout: float = 0.0,
        causal: bool = False,
        use_flash: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.use_flash = use_flash
        self.register_buffer("mask", None, persistent=False)

    def get_mask(self, length: int, device: torch.device) -> torch.Tensor:
        if self.mask is not None and self.mask.shape[-1] >= length:
            return self.mask[:length, :length]
        mask = torch.ones((length, length), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_length = q.shape[-2]
        if self.use_flash:
            if k.ndim == 3:
                k = rearrange(k, "b ... -> b 1 ...").expand_as(q)
            if v.ndim == 3:
                v = rearrange(v, "b ... -> b 1 ...").expand_as(q)
            attention_mask = (
                rearrange(mask, "b j -> b 1 1 j") if mask is not None else None
            )
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )

        key_einsum = "b j d" if k.ndim == 3 else "b h j d"
        scores = (
            einsum(f"b h i d, {key_einsum} -> b h i j", q, k)
            * q.shape[-1] ** -0.5
        )
        if mask is not None:
            scores = scores.masked_fill(
                ~rearrange(mask, "b j -> b 1 1 j"),
                -torch.finfo(scores.dtype).max,
            )
        if self.causal:
            scores = scores.masked_fill(
                self.get_mask(query_length, q.device),
                -torch.finfo(scores.dtype).max,
            )
        attention = self.attn_dropout(scores.softmax(dim=-1))
        return einsum(
            f"b h i j, {key_einsum} -> b h i d",
            attention,
            v,
        )


class RMSNorm(nn.Module):
    """RMS normalization matching the IndexTTS2 Perceiver."""

    def __init__(
        self,
        dim: int,
        scale: bool = True,
        dim_cond: Optional[int] = None,
    ):
        super().__init__()
        self.cond = dim_cond is not None
        self.to_gamma_beta = (
            nn.Linear(dim_cond, dim * 2) if self.cond else None
        )
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gamma = self.gamma if self.gamma is not None else 1
        out = F.normalize(x, dim=-1) * self.scale * gamma
        if not self.cond:
            return out
        if cond is None:
            raise ValueError("cond is required for conditional RMSNorm")
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma, beta = (
            rearrange(value, "b d -> b 1 d") for value in (gamma, beta)
        )
        return out * gamma + beta


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation
        (stride,) = self.stride
        if stride != 1:
            raise ValueError("CausalConv1d only supports stride=1")
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.causal_padding, 0), value=0.0))


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(
    dim: int,
    mult: float = 4,
    causal_conv: bool = False,
) -> nn.Sequential:
    dim_inner = int(dim * mult * 2 / 3)
    modules = [nn.Linear(dim, dim_inner * 2), GEGLU()]
    if causal_conv:
        modules.append(
            nn.Sequential(
                Rearrange("b n d -> b d n"),
                CausalConv1d(dim_inner, dim_inner, 3),
                Rearrange("b d n -> b n d"),
            )
        )
    modules.append(nn.Linear(dim_inner, dim))
    return nn.Sequential(*modules)


class Attention(nn.Module):
    """Cross-attention used by each Perceiver layer."""

    def __init__(
        self,
        dim: int,
        *,
        dim_context: Optional[int] = None,
        causal: bool = False,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        use_flash: bool = False,
        cross_attn_include_queries: bool = False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries
        dim_inner = dim_head * heads
        dim_context = dim if dim_context is None else dim_context
        self.attend = Attend(causal=causal, dropout=dropout, use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        has_context = context is not None
        context = x if context is None else context
        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)
        if mask is not None and mask.shape != context.shape[:2]:
            raise ValueError(
                "attention mask must cover the complete key sequence; "
                f"expected {tuple(context.shape[:2])}, got {tuple(mask.shape)}"
            )

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = (
            rearrange(value, "b n (h d) -> b h n d", h=self.heads)
            for value in (q, k, v)
        )
        out = self.attend(q, k, v, mask=mask)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class PerceiverResampler(nn.Module):
    """Resample variable-length context into fixed learned latents."""

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        dim_context: Optional[int] = None,
        num_latents: int = 32,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: float = 4,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        dim_context = dim if dim_context is None else dim_context
        self.proj_context = (
            nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        )
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            use_flash=use_flash_attn,
                            cross_attn_include_queries=True,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.proj_context(x)
        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])
        for attention, feed_forward in self.layers:
            latents = attention(latents, x, mask=mask) + latents
            latents = feed_forward(latents) + latents
        return self.norm(latents)

# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Tensor helpers for speaker encoders."""

import torch


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Return a boolean mask whose true entries are padding positions."""
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1-D, got shape {tuple(lengths.shape)}")
    if lengths.numel() == 0:
        return torch.empty((0, max_len), dtype=torch.bool, device=lengths.device)
    if torch.any(lengths < 0):
        raise ValueError("lengths must be non-negative")

    resolved_max_len = max_len if max_len > 0 else int(lengths.max().item())
    sequence = torch.arange(resolved_max_len, device=lengths.device)
    return sequence.unsqueeze(0) >= lengths.unsqueeze(1)

# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""Positional encoding modules."""

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):
    """Absolute sinusoidal positional encoding."""

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        max_len: int = 5000,
        reverse: bool = False,
    ):
        super().__init__()
        del reverse
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_emb = self.position_encoding(offset, x.size(1), apply_dropout=False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(
        self,
        offset: Union[int, torch.Tensor],
        size: int,
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        if isinstance(offset, int) or (
            isinstance(offset, torch.Tensor) and offset.dim() == 0
        ):
            if int(offset) + size > self.max_len:
                raise ValueError(
                    f"position range {int(offset) + size} exceeds max_len={self.max_len}"
                )
            pos_emb = self.pe[:, offset : offset + size]
        else:
            if int(torch.max(offset).item()) + size > self.max_len:
                raise ValueError("batched position range exceeds max_len")
            index = offset.unsqueeze(1) + torch.arange(size, device=offset.device)
            index = index * (index > 0)
            pos_emb = F.embedding(index, self.pe[0])
        return self.dropout(pos_emb) if apply_dropout else pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(
        self,
        x: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), apply_dropout=False)
        return self.dropout(x), self.dropout(pos_emb)


class NoPositionalEncoding(torch.nn.Module):
    """No-op positional encoding for interface compatibility."""

    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del offset
        pos_emb = x.new_zeros((1, x.size(1), self.d_model))
        return self.dropout(x), pos_emb

    def position_encoding(
        self,
        offset: Union[int, torch.Tensor],
        size: int,
    ) -> torch.Tensor:
        del offset
        return torch.zeros(1, size, self.d_model)

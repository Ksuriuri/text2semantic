# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""Convolutional subsampling layers."""

from typing import Tuple, Union

import torch


class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(
        self,
        offset: Union[int, torch.Tensor],
        size: int,
    ) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """Linear projection without temporal subsampling."""

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: torch.nn.Module,
    ):
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, pos_emb = self.pos_enc(self.out(x), offset)
        return x, pos_emb, x_mask


class Conv2dSubsampling2(BaseSubsampling):
    """2-D convolutional subsampling to half the temporal length."""

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: torch.nn.Module,
    ):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((idim - 1) // 2), odim)
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 2
        self.right_context = 2

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x.unsqueeze(1))
        batch, channels, time, features = x.size()
        x = self.out(
            x.transpose(1, 2).contiguous().view(batch, time, channels * features)
        )
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2]


class Conv2dSubsampling4(BaseSubsampling):
    """2-D convolutional subsampling to one quarter temporal length."""

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: torch.nn.Module,
    ):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        self.right_context = 6

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x.unsqueeze(1))
        batch, channels, time, features = x.size()
        x = self.out(
            x.transpose(1, 2).contiguous().view(batch, time, channels * features)
        )
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]


class Conv2dSubsampling6(BaseSubsampling):
    """2-D convolutional subsampling to one sixth temporal length."""

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: torch.nn.Module,
    ):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(
            odim * (((idim - 1) // 2 - 2) // 3), odim
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x.unsqueeze(1))
        batch, channels, time, features = x.size()
        x = self.linear(
            x.transpose(1, 2).contiguous().view(batch, time, channels * features)
        )
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 4::3]


class Conv2dSubsampling8(BaseSubsampling):
    """2-D convolutional subsampling to one eighth temporal length."""

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: torch.nn.Module,
    ):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        self.right_context = 14

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x.unsqueeze(1))
        batch, channels, time, features = x.size()
        x = self.linear(
            x.transpose(1, 2).contiguous().view(batch, time, channels * features)
        )
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]

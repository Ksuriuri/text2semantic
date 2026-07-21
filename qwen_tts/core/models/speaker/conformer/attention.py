# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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

"""Multi-head attention layers used by the speaker Conformer."""

import math
from typing import Optional, Tuple

import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        super().__init__()
        if n_feat % n_head != 0:
            raise ValueError(f"n_feat ({n_feat}) must be divisible by n_head ({n_head})")
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(dropout_rate)

    def forward_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = query.size(0)
        q = self.linear_q(query).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch = value.size(0)
        if mask is not None and mask.size(-1) > 0:
            invalid = ~mask.unsqueeze(1).bool()
            invalid = invalid[..., : scores.size(-1)]
            scores = scores.masked_fill(invalid, -torch.finfo(scores.dtype).max)
            attn = torch.softmax(scores, dim=-1).masked_fill(invalid, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(self.dropout(attn), value)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del pos_emb
        q, k, v = self.forward_qkv(query, key, value)
        if cache is not None and cache.size(0) > 0:
            key_cache, value_cache = torch.chunk(cache, 2, dim=-1)
            k = torch.cat((key_cache, k), dim=2)
            v = torch.cat((value_cache, v), dim=2)
        new_cache = torch.cat((k, v), dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-head attention with relative positional encoding."""

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        super().__init__(n_head, n_feat, dropout_rate)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pos_emb is None:
            raise ValueError("pos_emb is required for relative-position attention")

        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)
        if cache is not None and cache.size(0) > 0:
            key_cache, value_cache = torch.chunk(cache, 2, dim=-1)
            k = torch.cat((key_cache, k), dim=2)
            v = torch.cat((value_cache, v), dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache

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

"""Conformer building blocks adapted from IndexTTS2."""

from .attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from .embedding import NoPositionalEncoding, PositionalEncoding, RelPositionalEncoding
from .subsampling import (
    Conv2dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    LinearNoSubsampling,
)

__all__ = [
    "Conv2dSubsampling2",
    "Conv2dSubsampling4",
    "Conv2dSubsampling6",
    "Conv2dSubsampling8",
    "LinearNoSubsampling",
    "MultiHeadedAttention",
    "NoPositionalEncoding",
    "PositionalEncoding",
    "RelPositionalEncoding",
    "RelPositionMultiHeadedAttention",
]

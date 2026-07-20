# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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

import argparse
import json

import torch

from qwen_tts.semantic_codec import MaskGCTSemanticTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--w2v_bert_path", type=str, required=True)
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--repcodec_config_path", type=str, required=True)
    parser.add_argument("--repcodec_checkpoint_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    args = parser.parse_args()

    tokenizer = MaskGCTSemanticTokenizer(
        w2v_bert_path=args.w2v_bert_path,
        stats_path=args.stats_path,
        repcodec_config_path=args.repcodec_config_path,
        repcodec_checkpoint_path=args.repcodec_checkpoint_path,
        device=args.device,
        dtype=torch.float32 if args.dtype == "float32" else torch.bfloat16,
    )
    with (
        open(args.input_jsonl, encoding="utf-8") as source,
        open(args.output_jsonl, "w", encoding="utf-8") as destination,
    ):
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            if "audio" not in item or "text" not in item:
                raise ValueError(
                    f"Line {line_number} must contain 'audio' and 'text'."
                )
            item["semantic_codes"] = tokenizer.encode_file(
                item["audio"]
            ).tolist()
            item.pop("audio_codes", None)
            item.pop("ref_audio", None)
            destination.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoTokenizer

from ..core.models.modeling_text2semantic import Text2SemanticForCausalLM


class Text2SemanticModel:
    """Small inference wrapper returning semantic-token tensors only."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        *,
        device="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ):
        model = Text2SemanticForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, tokenizer)

    @torch.inference_mode()
    def generate(self, text, **generation_kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": value}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for value in texts
        ]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_length = max(len(ids) for ids in encoded)
        input_ids = torch.full(
            (len(encoded), max_length),
            pad_id,
            dtype=torch.long,
            device=self.model.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(encoded):
            start = max_length - len(ids)
            input_ids[row, start:] = torch.tensor(
                ids, device=input_ids.device
            )
            attention_mask[row, start:] = 1
        return self.model.generate_semantic(
            input_ids,
            text_attention_mask=attention_mask,
            **generation_kwargs,
        )


# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.utils.data import Dataset


class Text2SemanticDataset(Dataset):
    """Text prefix and MaskGCT semantic-code teacher-forcing dataset."""

    def __init__(
        self,
        data,
        tokenizer,
        *,
        speech_bos_token_id=8192,
        speech_eos_token_id=8193,
        max_text_tokens=None,
        max_semantic_tokens=None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.speech_bos_token_id = speech_bos_token_id
        self.speech_eos_token_id = speech_eos_token_id
        self.max_text_tokens = max_text_tokens
        self.max_semantic_tokens = max_semantic_tokens

    def __len__(self):
        return len(self.data)

    def _tokenize_text(self, text):
        messages = [{"role": "user", "content": text}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            ids = self.tokenizer(text, add_special_tokens=True)["input_ids"]
        if self.max_text_tokens is not None:
            ids = ids[: self.max_text_tokens]
        if not ids:
            raise ValueError("Tokenized text must not be empty.")
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, index):
        item = self.data[index]
        if "text" not in item or "semantic_codes" not in item:
            raise ValueError("Each sample needs 'text' and 'semantic_codes'.")
        codes = item["semantic_codes"]
        if self.max_semantic_tokens is not None:
            codes = codes[: self.max_semantic_tokens]
        codes = torch.tensor(codes, dtype=torch.long)
        if codes.numel() == 0:
            raise ValueError("semantic_codes must not be empty.")
        if int(codes.min()) < 0 or int(codes.max()) >= 8192:
            raise ValueError("semantic_codes must be in [0, 8191].")
        return {
            "text_input_ids": self._tokenize_text(item["text"]),
            "speech_input_ids": torch.cat(
                (torch.tensor([self.speech_bos_token_id]), codes)
            ),
            "labels": torch.cat(
                (codes, torch.tensor([self.speech_eos_token_id]))
            ),
        }

    def collate_fn(self, samples):
        batch_size = len(samples)
        max_text = max(x["text_input_ids"].numel() for x in samples)
        max_speech = max(x["speech_input_ids"].numel() for x in samples)
        text_pad = self.tokenizer.pad_token_id
        if text_pad is None:
            text_pad = self.tokenizer.eos_token_id
        if text_pad is None:
            raise ValueError("The Qwen tokenizer must define pad_token_id or eos_token_id.")

        text_ids = torch.full((batch_size, max_text), text_pad, dtype=torch.long)
        text_mask = torch.zeros((batch_size, max_text), dtype=torch.long)
        speech_ids = torch.full(
            (batch_size, max_speech),
            self.speech_eos_token_id,
            dtype=torch.long,
        )
        speech_mask = torch.zeros((batch_size, max_speech), dtype=torch.long)
        labels = torch.full((batch_size, max_speech), -100, dtype=torch.long)

        for row, sample in enumerate(samples):
            text_length = sample["text_input_ids"].numel()
            speech_length = sample["speech_input_ids"].numel()
            text_start = max_text - text_length
            text_ids[row, text_start:] = sample["text_input_ids"]
            text_mask[row, text_start:] = 1
            speech_ids[row, :speech_length] = sample["speech_input_ids"]
            speech_mask[row, :speech_length] = 1
            labels[row, :speech_length] = sample["labels"]

        return {
            "text_input_ids": text_ids,
            "text_attention_mask": text_mask,
            "speech_input_ids": speech_ids,
            "speech_attention_mask": speech_mask,
            "labels": labels,
        }

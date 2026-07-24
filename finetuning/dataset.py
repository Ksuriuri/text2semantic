# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from torch.utils.data import Dataset

from qwen_tts.text_template import tokenize_tts_prompt


class Text2SemanticDataset(Dataset):
    """Text prefix and MaskGCT semantic-code teacher-forcing dataset."""

    def __init__(
        self,
        data,
        tokenizer,
        *,
        semantic_vocab_size=8192,
        speech_bos_token_id=8192,
        speech_eos_token_id=8193,
        speech_pad_token_id=8194,
        max_text_tokens=None,
        max_semantic_tokens=None,
        speaker_counts=None,
        speaker_audio_paths_by_id=None,
        min_speaker_records=2,
        max_target_seconds=30.0,
    ):
        self.raw_size = len(data)
        self.tokenizer = tokenizer
        self.semantic_vocab_size = semantic_vocab_size
        self.speech_bos_token_id = speech_bos_token_id
        self.speech_eos_token_id = speech_eos_token_id
        self.speech_pad_token_id = speech_pad_token_id
        self.max_text_tokens = max_text_tokens
        self.max_semantic_tokens = max_semantic_tokens
        self.speaker_counts = speaker_counts or self._count_speakers(data)
        self.speaker_audio_paths_by_id = (
            speaker_audio_paths_by_id
            or self._collect_speaker_audio_paths(data)
        )
        self.min_speaker_records = min_speaker_records
        self.max_target_seconds = max_target_seconds
        self._semantic_code_cache = {}
        self.data = [item for item in data if self._is_usable(item)]
        if not self.data:
            raise ValueError("No usable samples remain after dataset filtering.")
        self.filtered_size = self.raw_size - len(self.data)

    @staticmethod
    def _target_audio_path(item):
        return item.get("audio") or item.get("audio_path")

    @staticmethod
    def _speaker_key(item):
        speaker_id = item.get("speaker_id")
        if speaker_id is None:
            return None
        language = item.get("language") or item.get("lang")
        return language, speaker_id

    @staticmethod
    def _count_speakers(data):
        counts = {}
        for item in data:
            speaker_key = Text2SemanticDataset._speaker_key(item)
            if speaker_key is not None:
                counts[speaker_key] = counts.get(speaker_key, 0) + 1
        return counts

    @classmethod
    def _collect_speaker_audio_paths(cls, data):
        paths_by_id = {}
        for item in data:
            speaker_key = cls._speaker_key(item)
            audio_path = cls._target_audio_path(item)
            if speaker_key is None or audio_path is None:
                continue
            paths_by_id.setdefault(speaker_key, [])
            if audio_path not in paths_by_id[speaker_key]:
                paths_by_id[speaker_key].append(audio_path)
        return paths_by_id

    def _is_usable(self, item):
        if self._speaker_audio_path(item) is None:
            return False
        speaker_key = self._speaker_key(item)
        if (
            speaker_key is not None
            and self.speaker_counts.get(speaker_key, 0) < self.min_speaker_records
        ):
            return False
        duration = item.get("duration")
        if (
            duration is not None
            and self.max_target_seconds is not None
            and float(duration) > self.max_target_seconds
        ):
            return False
        semantic_length = self._semantic_length(item)
        if (
            semantic_length is not None
            and self.max_semantic_tokens is not None
            and semantic_length > self.max_semantic_tokens
        ):
            return False
        return True

    @staticmethod
    def _semantic_length(item):
        if "semantic_codes" in item:
            return len(item["semantic_codes"])
        if "semantic_code_length" in item:
            return int(item["semantic_code_length"])
        return None

    def __len__(self):
        return len(self.data)

    def _tokenize_text(self, text):
        ids = tokenize_tts_prompt(self.tokenizer, text)
        if self.max_text_tokens is not None:
            ids = ids[: self.max_text_tokens]
        if not ids:
            raise ValueError("Tokenized text must not be empty.")
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, index):
        item = self.data[index]
        if "text" not in item:
            raise ValueError("Each sample needs 'text'.")
        speaker_audio_path = self._speaker_audio_path(item)
        if speaker_audio_path is None:
            raise ValueError(
                "Each sample needs 'ref_audio', 'audio', or 'audio_path'."
            )
        codes = self._semantic_codes(item)
        if codes.numel() == 0:
            raise ValueError("semantic_codes must not be empty.")
        if int(codes.min()) < 0 or int(codes.max()) >= self.semantic_vocab_size:
            raise ValueError(
                f"semantic_codes must be in [0, {self.semantic_vocab_size - 1}]."
            )
        return {
            "text_input_ids": self._tokenize_text(item["text"]),
            "speech_input_ids": torch.cat(
                (torch.tensor([self.speech_bos_token_id]), codes)
            ),
            "labels": torch.cat(
                (codes, torch.tensor([self.speech_eos_token_id]))
            ),
            "speaker_audio_path": speaker_audio_path,
        }

    def _speaker_audio_path(self, item):
        explicit = item.get("ref_audio") or item.get("ref_audio_path")
        target_audio_path = self._target_audio_path(item)
        if explicit is not None:
            return explicit if explicit != target_audio_path else None
        speaker_key = self._speaker_key(item)
        if speaker_key is None:
            return None
        for audio_path in self.speaker_audio_paths_by_id.get(speaker_key, ()):
            if audio_path != target_audio_path:
                return audio_path
        return None

    def _semantic_codes(self, item):
        if "semantic_codes" in item:
            return torch.tensor(item["semantic_codes"], dtype=torch.long)
        required = (
            "semantic_code_path",
            "semantic_code_offset",
            "semantic_code_length",
        )
        if not all(key in item for key in required):
            raise ValueError(
                "Each sample needs 'semantic_codes' or compact semantic code fields."
            )
        path = item["semantic_code_path"]
        codes = self._semantic_code_cache.get(path)
        if codes is None:
            codes = np.memmap(path, dtype="<u2", mode="r")
            self._semantic_code_cache[path] = codes
        offset = int(item["semantic_code_offset"])
        length = int(item["semantic_code_length"])
        if offset < 0 or length <= 0 or offset + length > len(codes):
            raise ValueError(
                "Compact semantic code range is out of bounds: "
                f"path={path}, offset={offset}, length={length}, "
                f"available={len(codes)}."
            )
        return torch.tensor(codes[offset : offset + length], dtype=torch.long)

    def collate_fn(self, samples):
        batch_size = len(samples)
        max_text = max(x["text_input_ids"].numel() for x in samples)
        max_speech = max(x["speech_input_ids"].numel() for x in samples)
        text_pad = self.tokenizer.pad_token_id
        if text_pad is None:
            text_pad = self.tokenizer.eos_token_id
        if text_pad is None:
            raise ValueError(
                "The Qwen tokenizer must define pad_token_id or eos_token_id."
            )

        text_ids = torch.full((batch_size, max_text), text_pad, dtype=torch.long)
        text_mask = torch.zeros((batch_size, max_text), dtype=torch.long)
        speech_ids = torch.full(
            (batch_size, max_speech),
            self.speech_pad_token_id,
            dtype=torch.long,
        )
        speech_mask = torch.zeros((batch_size, max_speech), dtype=torch.long)
        labels = torch.full((batch_size, max_speech), -100, dtype=torch.long)

        for row, sample in enumerate(samples):
            text_length = sample["text_input_ids"].numel()
            speech_length = sample["speech_input_ids"].numel()
            text_ids[row, :text_length] = sample["text_input_ids"]
            text_mask[row, :text_length] = 1
            speech_ids[row, :speech_length] = sample["speech_input_ids"]
            speech_mask[row, :speech_length] = 1
            labels[row, :speech_length] = sample["labels"]

        return {
            "text_input_ids": text_ids,
            "text_attention_mask": text_mask,
            "speech_input_ids": speech_ids,
            "speech_attention_mask": speech_mask,
            "labels": labels,
            "speaker_audio_paths": [
                sample["speaker_audio_path"] for sample in samples
            ],
        }

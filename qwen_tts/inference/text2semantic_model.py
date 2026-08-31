# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..core.models.modeling_text2semantic import Text2SemanticForCausalLM
from ..semantic_codec import MaskGCTFeatureExtractor
from ..text_conditioning import (
    CONDITIONING_SPECIAL_TOKENS,
    condition_inference_text,
    validate_conditioning_tokens,
)
from ..text_template import tokenize_tts_prompt


class Text2SemanticModel:
    """Small inference wrapper returning semantic-token tensors only."""

    def __init__(
        self,
        model,
        tokenizer,
        speaker_feature_extractor,
        *,
        max_ref_seconds=15.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.speaker_feature_extractor = speaker_feature_extractor
        self.max_ref_seconds = max_ref_seconds

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        *,
        w2v_bert_path,
        stats_path,
        device="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        max_ref_seconds=15.0,
    ):
        model = Text2SemanticForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        speaker_feature_extractor = MaskGCTFeatureExtractor(
            w2v_bert_path=w2v_bert_path,
            stats_path=stats_path,
            device=device,
        )
        return cls(
            model,
            tokenizer,
            speaker_feature_extractor,
            max_ref_seconds=max_ref_seconds,
        )

    @torch.inference_mode()
    def generate(
        self,
        text,
        ref_audio,
        *,
        language=None,
        emotion=None,
        return_speaker_features=False,
        **generation_kwargs,
    ):
        texts = [text] if isinstance(text, str) else list(text)
        def controls(value, name):
            if value is None or isinstance(value, str):
                return [value] * len(texts)
            values = list(value)
            if len(values) != len(texts):
                raise ValueError(f"{name} must be scalar or align with text batch")
            return values

        languages = controls(language, "language")
        emotions = controls(emotion, "emotion")
        texts = [
            condition_inference_text(value, language=lang, emotion=emo)
            for value, lang, emo in zip(texts, languages, emotions)
        ]
        if any(
            token in value
            for value in texts
            for token in CONDITIONING_SPECIAL_TOKENS
        ):
            validate_conditioning_tokens(self.tokenizer)
        if isinstance(ref_audio, (str, Path)):
            ref_audios = [str(ref_audio)] * len(texts)
        else:
            ref_audios = [str(path) for path in ref_audio]
            if len(ref_audios) == 1 and len(texts) > 1:
                ref_audios *= len(texts)
            elif len(ref_audios) != len(texts):
                raise ValueError(
                    "ref_audio must be a single path or align with the text batch."
                )
        encoded = [
            tokenize_tts_prompt(self.tokenizer, value)
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
        speaker_features, speaker_feature_lengths = (
            self.speaker_feature_extractor.encode_files(
                ref_audios,
                max_audio_seconds=self.max_ref_seconds,
            )
        )
        generated = self.model.generate_semantic(
            input_ids,
            text_attention_mask=attention_mask,
            speaker_features=speaker_features,
            speaker_feature_lengths=speaker_feature_lengths,
            **generation_kwargs,
        )
        if return_speaker_features:
            return generated, speaker_features, speaker_feature_lengths
        return generated

# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

from transformers import PretrainedConfig


class Text2SemanticConfig(PretrainedConfig):
    """Configuration for a Qwen3.5 text-to-semantic language model."""

    model_type = "qwen3_5_text2semantic"

    def __init__(
        self,
        qwen_config=None,
        semantic_vocab_size=8192,
        speech_bos_token_id=8192,
        speech_eos_token_id=8193,
        speech_pad_token_id=8193,
        initializer_range=0.02,
        codec_name="maskgct_repcodec",
        codec_frame_rate=50,
        speaker_input_dim=1024,
        speaker_conformer_output_size=512,
        speaker_conformer_linear_units=2048,
        speaker_conformer_attention_heads=8,
        speaker_conformer_num_blocks=6,
        speaker_conformer_input_layer="conv2d2",
        speaker_num_latents=32,
        speaker_latent_dim=1280,
        speaker_perceiver_depth=2,
        speaker_perceiver_ff_mult=2,
        **kwargs,
    ):
        kwargs.pop("bos_token_id", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)
        super().__init__(
            bos_token_id=speech_bos_token_id,
            eos_token_id=speech_eos_token_id,
            pad_token_id=speech_pad_token_id,
            **kwargs,
        )
        self.qwen_config = qwen_config or {}
        self.semantic_vocab_size = semantic_vocab_size
        self.speech_vocab_size = semantic_vocab_size + 2
        self.speech_bos_token_id = speech_bos_token_id
        self.speech_eos_token_id = speech_eos_token_id
        self.speech_pad_token_id = speech_pad_token_id
        self.initializer_range = initializer_range
        self.codec_name = codec_name
        self.codec_frame_rate = codec_frame_rate
        self.speaker_input_dim = speaker_input_dim
        self.speaker_conformer_output_size = speaker_conformer_output_size
        self.speaker_conformer_linear_units = speaker_conformer_linear_units
        self.speaker_conformer_attention_heads = speaker_conformer_attention_heads
        self.speaker_conformer_num_blocks = speaker_conformer_num_blocks
        self.speaker_conformer_input_layer = speaker_conformer_input_layer
        self.speaker_num_latents = speaker_num_latents
        self.speaker_latent_dim = speaker_latent_dim
        self.speaker_perceiver_depth = speaker_perceiver_depth
        self.speaker_perceiver_ff_mult = speaker_perceiver_ff_mult


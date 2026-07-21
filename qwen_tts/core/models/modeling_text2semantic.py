# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    PreTrainedModel,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
)
from transformers.utils import ModelOutput

from .configuration_text2semantic import Text2SemanticConfig
from .speaker import SpeakerConditioningEncoder


@dataclass
class Text2SemanticOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[object] = None


class Text2SemanticForCausalLM(PreTrainedModel):
    """Qwen3.5 conditioned autoregressive model over MaskGCT semantic indices."""

    config_class = Text2SemanticConfig
    base_model_prefix = "text2semantic"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: Text2SemanticConfig):
        super().__init__(config)
        qwen_config = Qwen3_5TextConfig.from_dict(config.qwen_config)
        self.backbone = Qwen3_5TextModel(qwen_config)
        self.speech_embedding = nn.Embedding(
            config.speech_vocab_size,
            qwen_config.hidden_size,
            padding_idx=config.speech_pad_token_id,
        )
        self.speech_head = nn.Linear(
            qwen_config.hidden_size,
            config.speech_vocab_size,
            bias=False,
        )
        self.speaker_encoder = SpeakerConditioningEncoder(
            input_dim=config.speaker_input_dim,
            conformer_output_dim=config.speaker_conformer_output_size,
            conformer_linear_units=config.speaker_conformer_linear_units,
            conformer_attention_heads=config.speaker_conformer_attention_heads,
            conformer_num_blocks=config.speaker_conformer_num_blocks,
            conformer_input_layer=config.speaker_conformer_input_layer,
            perceiver_num_latents=config.speaker_num_latents,
            perceiver_latent_dim=config.speaker_latent_dim,
            perceiver_depth=config.speaker_perceiver_depth,
            perceiver_ff_mult=config.speaker_perceiver_ff_mult,
        )
        if config.speaker_latent_dim == qwen_config.hidden_size:
            self.speaker_projection = nn.Identity()
        else:
            self.speaker_projection = nn.Linear(
                config.speaker_latent_dim,
                qwen_config.hidden_size,
                bias=False,
            )
        self.post_init()
        self._init_speech_parameters()

    def _init_speech_parameters(self):
        nn.init.normal_(
            self.speech_embedding.weight,
            mean=0.0,
            std=self.config.initializer_range,
        )
        nn.init.normal_(
            self.speech_head.weight,
            mean=0.0,
            std=self.config.initializer_range,
        )
        with torch.no_grad():
            self.speech_embedding.weight[self.config.speech_pad_token_id].zero_()

    @classmethod
    def from_qwen_pretrained(
        cls,
        model_name_or_path,
        *,
        semantic_vocab_size=8192,
        codec_name="maskgct_repcodec",
        codec_frame_rate=50,
        **kwargs,
    ):
        """Load only the pretrained Qwen3.5 backbone; speech parameters stay random."""
        causal_lm = Qwen3_5ForCausalLM.from_pretrained(
            model_name_or_path, **kwargs
        )
        backbone = causal_lm.model
        del causal_lm
        qwen_config = backbone.config
        initializer_range = getattr(qwen_config, "initializer_range", 0.02)
        config = Text2SemanticConfig(
            qwen_config=qwen_config.to_dict(),
            semantic_vocab_size=semantic_vocab_size,
            speech_bos_token_id=semantic_vocab_size,
            speech_eos_token_id=semantic_vocab_size + 1,
            speech_pad_token_id=semantic_vocab_size + 1,
            initializer_range=initializer_range,
            codec_name=codec_name,
            codec_frame_rate=codec_frame_rate,
        )
        model = cls(config)
        model.backbone = backbone
        return model

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def _validate_speech_ids(self, speech_ids):
        if speech_ids.numel() == 0:
            raise ValueError("speech_input_ids must not be empty.")
        minimum = int(speech_ids.min())
        maximum = int(speech_ids.max())
        if minimum < 0 or maximum >= self.config.speech_vocab_size:
            raise ValueError(
                f"Speech token IDs must be in [0, {self.config.speech_vocab_size - 1}], "
                f"got [{minimum}, {maximum}]."
            )

    def _build_conditioned_prefix(
        self,
        text_input_ids,
        text_attention_mask,
        speaker_features,
        speaker_feature_lengths,
    ):
        if speaker_features is None or speaker_feature_lengths is None:
            raise ValueError(
                "speaker_features and speaker_feature_lengths are required."
            )
        if speaker_features.ndim != 3:
            raise ValueError("speaker_features must have shape [batch, time, dim].")
        if speaker_features.size(0) != text_input_ids.size(0):
            raise ValueError("speaker_features and text_input_ids batch sizes differ.")
        if speaker_features.size(2) != self.config.speaker_input_dim:
            raise ValueError(
                f"Expected speaker feature dim {self.config.speaker_input_dim}, "
                f"got {speaker_features.size(2)}."
            )
        if speaker_feature_lengths.shape != (speaker_features.size(0),):
            raise ValueError("speaker_feature_lengths must have shape [batch].")
        speaker_parameter = next(self.speaker_encoder.parameters())
        speaker_features = speaker_features.to(
            device=speaker_parameter.device,
            dtype=speaker_parameter.dtype,
        )
        speaker_feature_lengths = speaker_feature_lengths.to(
            device=speaker_parameter.device,
            dtype=torch.long,
        )
        if (
            bool((speaker_feature_lengths <= 0).any())
            or bool((speaker_feature_lengths > speaker_features.size(1)).any())
        ):
            raise ValueError(
                "speaker_feature_lengths must be in [1, speaker feature time]."
            )
        speaker_latents = self.speaker_encoder(
            speaker_features,
            speaker_feature_lengths,
        )
        speaker_embeds = self.speaker_projection(speaker_latents)
        text_embeds = self.get_input_embeddings()(text_input_ids)
        speaker_embeds = speaker_embeds.to(dtype=text_embeds.dtype)

        batch_size, padded_text_length = text_input_ids.shape
        speaker_length = speaker_embeds.size(1)
        prefix_embeds = text_embeds.new_zeros(
            batch_size,
            padded_text_length + speaker_length,
            text_embeds.size(-1),
        )
        prefix_mask = text_attention_mask.new_zeros(
            batch_size,
            padded_text_length + speaker_length,
        )
        for row in range(batch_size):
            valid_text = text_embeds[row][text_attention_mask[row].bool()]
            padding = padded_text_length - valid_text.size(0)
            prefix_embeds[row, padding : padding + speaker_length] = speaker_embeds[row]
            prefix_embeds[row, padding + speaker_length :] = valid_text
            prefix_mask[row, padding:] = 1
        return prefix_embeds, prefix_mask

    def forward(
        self,
        text_input_ids,
        speech_input_ids,
        text_attention_mask=None,
        speech_attention_mask=None,
        speaker_features=None,
        speaker_feature_lengths=None,
        labels=None,
        use_cache=None,
        **kwargs,
    ):
        if use_cache is None:
            use_cache = False
        self._validate_speech_ids(speech_input_ids)
        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(text_input_ids)
        if speech_attention_mask is None:
            speech_attention_mask = torch.ones_like(speech_input_ids)

        prefix_embeds, prefix_mask = self._build_conditioned_prefix(
            text_input_ids,
            text_attention_mask,
            speaker_features,
            speaker_feature_lengths,
        )
        speech_embeds = self.speech_embedding(speech_input_ids)
        inputs_embeds = torch.cat((prefix_embeds, speech_embeds), dim=1)
        attention_mask = torch.cat(
            (prefix_mask, speech_attention_mask), dim=1
        )

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        speech_hidden = outputs.last_hidden_state[:, -speech_input_ids.size(1) :]
        logits = self.speech_head(speech_hidden)
        loss = None
        if labels is not None:
            if labels.shape != speech_input_ids.shape:
                raise ValueError(
                    "labels and speech_input_ids must have identical shapes."
                )
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.config.speech_vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return Text2SemanticOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    @torch.inference_mode()
    def generate_semantic(
        self,
        text_input_ids,
        text_attention_mask=None,
        speaker_features=None,
        speaker_feature_lengths=None,
        max_new_tokens=1500,
        temperature=1.0,
        top_k=0,
        do_sample=True,
    ):
        """Generate semantic codec indices, excluding BOS and EOS."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(text_input_ids)

        batch_size = text_input_ids.size(0)
        generated = torch.full(
            (batch_size, 1),
            self.config.speech_bos_token_id,
            dtype=torch.long,
            device=text_input_ids.device,
        )
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=text_input_ids.device
        )

        prefix_embeds, prefix_mask = self._build_conditioned_prefix(
            text_input_ids,
            text_attention_mask,
            speaker_features,
            speaker_feature_lengths,
        )
        speech_embeds = self.speech_embedding(generated)
        attention_mask = torch.cat(
            (
                prefix_mask,
                torch.ones_like(generated),
            ),
            dim=1,
        )
        output = self.backbone(
            inputs_embeds=torch.cat((prefix_embeds, speech_embeds), dim=1),
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        next_logits = self.speech_head(output.last_hidden_state[:, -1]).float()

        for _ in range(max_new_tokens):
            next_logits = next_logits / temperature
            # BOS is an input-only control token and must never be emitted.
            next_logits[:, self.config.speech_bos_token_id] = -torch.inf
            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                threshold = torch.topk(next_logits, k, dim=-1).values[:, -1:]
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, -torch.inf
                )
            if do_sample:
                next_token = torch.multinomial(
                    torch.softmax(next_logits, dim=-1), num_samples=1
                )
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, self.config.speech_eos_token_id),
                next_token,
            )
            generated = torch.cat((generated, next_token), dim=1)
            finished |= next_token.squeeze(1).eq(
                self.config.speech_eos_token_id
            )
            if bool(finished.all()):
                break
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token)), dim=1
            )
            output = self.backbone(
                inputs_embeds=self.speech_embedding(next_token),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            next_logits = self.speech_head(
                output.last_hidden_state[:, -1]
            ).float()

        results = []
        for sequence in generated[:, 1:]:
            eos = (sequence == self.config.speech_eos_token_id).nonzero(
                as_tuple=False
            )
            end = int(eos[0]) if eos.numel() else sequence.numel()
            results.append(sequence[:end])
        return results


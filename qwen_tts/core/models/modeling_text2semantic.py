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

    def forward(
        self,
        text_input_ids,
        speech_input_ids,
        text_attention_mask=None,
        speech_attention_mask=None,
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

        text_embeds = self.get_input_embeddings()(text_input_ids)
        speech_embeds = self.speech_embedding(speech_input_ids)
        inputs_embeds = torch.cat((text_embeds, speech_embeds), dim=1)
        attention_mask = torch.cat(
            (text_attention_mask, speech_attention_mask), dim=1
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

        text_embeds = self.get_input_embeddings()(text_input_ids)
        speech_embeds = self.speech_embedding(generated)
        attention_mask = torch.cat(
            (
                text_attention_mask,
                torch.ones_like(generated),
            ),
            dim=1,
        )
        output = self.backbone(
            inputs_embeds=torch.cat((text_embeds, speech_embeds), dim=1),
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


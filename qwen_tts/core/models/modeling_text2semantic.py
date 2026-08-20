# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
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

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:  # optional: the Triton kernels are CUDA-only
    LigerFusedLinearCrossEntropyLoss = None

# Escape hatches for A/B-ing the two changes below against the old behaviour.
_FUSED_CE_ENABLED = os.environ.get("T2S_FUSED_CE", "1") != "0"
_ALWAYS_VALIDATE_SPEECH_IDS = (
    os.environ.get("T2S_VALIDATE_SPEECH_IDS", "once") == "always"
)


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
        # Fused linear + cross entropy over speech_head. The unfused path holds
        # three (B, L, speech_vocab_size) tensors alive through backward -- the
        # bf16 logits, the fp32 copy from .float(), and cross_entropy's saved
        # fp32 log_softmax -- which is ~32 GiB at B=48, L=8000, V=8195. Liger
        # chunks the head matmul and the log_softmax together so none of them is
        # ever materialised. The training loop only reads output.loss; evaluate()
        # needs output.logits and runs under model.eval(), so it keeps the
        # unfused path and its metrics are unchanged.
        self._fused_ce = None
        if LigerFusedLinearCrossEntropyLoss is not None and _FUSED_CE_ENABLED:
            self._fused_ce = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                reduction="mean",
                accum_dtype=torch.float32,
            )
        # The speech id range is a property of the manifest, not of a batch, so
        # it is checked once rather than on every forward. See
        # _validate_speech_ids.
        self._speech_ids_validated = False
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
        self.speaker_boundary_embedding = nn.Embedding(2, qwen_config.hidden_size)
        self.speaker_gradient_checkpointing = False
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
            speech_pad_token_id=semantic_vocab_size + 2,
            initializer_range=initializer_range,
            codec_name=codec_name,
            codec_frame_rate=codec_frame_rate,
        )
        model = cls(config)
        model.backbone = backbone
        return model

    @staticmethod
    def _position_ids_from_attention_mask(attention_mask):
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        return position_ids.masked_fill(attention_mask.eq(0), 0)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Checkpoint the backbone, and the speaker encoder along with it.

        The speaker encoder is a plain nn.Module, so transformers' own
        `gradient_checkpointing_enable` walks straight past it -- which is easy to
        miss, because the backbone is where the parameters are. The activations
        are somewhere else: at batch 48 the training step peaked at 75.8 GiB with
        20 s references against 62.3 GiB with 10 s ones, and that 13.5 GiB is this
        module holding twice as many frames, not the frozen W2V-BERT (measured:
        11.15 GiB whole-batch, and chunking it did not move the step's peak).

        Recomputing it is cheap in the only currency that matters here. Six
        Conformer blocks at width 512 over subsampled frames plus a two-layer
        Perceiver is on the order of 1% of the step's arithmetic next to a 2B
        backbone, and buying ~13 GiB with that is what makes a larger batch fit.
        """
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        self.speaker_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()
        self.speaker_gradient_checkpointing = False

    def _validate_speech_ids(self, speech_ids):
        if speech_ids.numel() == 0:
            raise ValueError("speech_input_ids must not be empty.")
        # int() on a CUDA tensor blocks until the device drains its queue and
        # copies the scalar back. Sitting at the top of forward, two of them per
        # micro-batch also destroy the overlap with the previous step. The bound
        # being checked comes from the manifest and cannot change between
        # batches, so the first one is enough to catch a bad manifest.
        if self._speech_ids_validated and not _ALWAYS_VALIDATE_SPEECH_IDS:
            return
        minimum = int(speech_ids.min())
        maximum = int(speech_ids.max())
        if minimum < 0 or maximum >= self.config.speech_vocab_size:
            raise ValueError(
                f"Speech token IDs must be in [0, {self.config.speech_vocab_size - 1}], "
                f"got [{minimum}, {maximum}]."
            )
        self._speech_ids_validated = True

    def _encode_speaker_prefix(
        self,
        speaker_features,
        speaker_feature_lengths,
    ):
        if speaker_features is None or speaker_feature_lengths is None:
            raise ValueError(
                "speaker_features and speaker_feature_lengths are required."
            )
        if speaker_features.ndim != 3:
            raise ValueError("speaker_features must have shape [batch, time, dim].")
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
        if (
            self.speaker_gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            # Both arguments come out of the frozen W2V-BERT, whose encode_*
            # methods are @torch.inference_mode(), and checkpointing has to save
            # its inputs to replay the forward: "Inference tensors cannot be saved
            # for backward". Autograd tolerates them on the ordinary path, so this
            # is specific to checkpointing -- and cheap, about 100 MB at batch 48
            # with 20 s references. Both, because the conversions above only clear
            # the inference flag when they actually copy: features change dtype
            # and so come back normal, while `lengths.to(dtype=torch.long)` on an
            # already-long tensor returns the same inference tensor, and that is
            # the argument that failed on 8 GPUs.
            if torch.is_inference(speaker_features):
                speaker_features = speaker_features.clone()
            if torch.is_inference(speaker_feature_lengths):
                speaker_feature_lengths = speaker_feature_lengths.clone()
            # use_reentrant=False so the recomputation sees the same RNG state and
            # so this composes with the backbone's own checkpointing. The encoder
            # is built with dropout 0 anyway, which is why the recomputed forward
            # is not merely statistically equivalent but identical.
            speaker_latents = torch.utils.checkpoint.checkpoint(
                self.speaker_encoder,
                speaker_features,
                speaker_feature_lengths,
                use_reentrant=False,
            )
        else:
            speaker_latents = self.speaker_encoder(
                speaker_features,
                speaker_feature_lengths,
            )
        speaker_embeds = self.speaker_projection(speaker_latents)
        boundary_ids = torch.arange(
            2,
            device=speaker_embeds.device,
            dtype=torch.long,
        )
        boundaries = self.speaker_boundary_embedding(boundary_ids)
        boundaries = boundaries.to(dtype=speaker_embeds.dtype)
        return torch.cat(
            (
                boundaries[0].view(1, 1, -1).expand(speaker_embeds.size(0), -1, -1),
                speaker_embeds,
                boundaries[1].view(1, 1, -1).expand(speaker_embeds.size(0), -1, -1),
            ),
            dim=1,
        )

    def _build_training_inputs(
        self,
        text_input_ids,
        text_attention_mask,
        speech_input_ids,
        speech_attention_mask,
        speaker_features,
        speaker_feature_lengths,
    ):
        if speaker_features is not None and speaker_features.size(0) != text_input_ids.size(0):
            raise ValueError("speaker_features and text_input_ids batch sizes differ.")
        speaker_embeds = self._encode_speaker_prefix(
            speaker_features,
            speaker_feature_lengths,
        )
        text_embeds = self.get_input_embeddings()(text_input_ids)
        speech_embeds = self.speech_embedding(speech_input_ids)
        speaker_embeds = speaker_embeds.to(dtype=text_embeds.dtype)
        speech_embeds = speech_embeds.to(dtype=text_embeds.dtype)

        batch_size = text_input_ids.size(0)
        prefix_length = speaker_embeds.size(1)
        hidden_size = text_embeds.size(-1)
        device = text_embeds.device

        text_lengths = text_attention_mask.sum(dim=1).long()
        speech_lengths = speech_attention_mask.sum(dim=1).long()
        speech_starts = prefix_length + text_lengths
        total_lengths = speech_starts + speech_lengths
        # The one device sync left: the padded width has to reach the host to
        # size the buffer. Everything below stays on the device, because a
        # Python loop over the batch cost 2 * batch_size syncs per step here and
        # the GPU drains on every one of them.
        max_total_length = int(total_lengths.max().item())

        # Rows are assembled with two scatters. One spare trailing column
        # absorbs the padded source positions and is sliced off afterwards, so
        # no index has to be computed on the host.
        dump_column = max_total_length
        scratch = text_embeds.new_zeros(
            batch_size,
            max_total_length + 1,
            hidden_size,
        )

        text_positions = torch.arange(text_embeds.size(1), device=device)
        text_dest = torch.where(
            text_positions.unsqueeze(0) < text_lengths.unsqueeze(1),
            prefix_length + text_positions.unsqueeze(0).expand(batch_size, -1),
            dump_column,
        )
        scratch = scratch.scatter(
            1,
            text_dest.unsqueeze(-1).expand(-1, -1, hidden_size),
            text_embeds,
        )

        speech_positions = torch.arange(speech_embeds.size(1), device=device)
        speech_dest = torch.where(
            speech_positions.unsqueeze(0) < speech_lengths.unsqueeze(1),
            speech_starts.unsqueeze(1) + speech_positions.unsqueeze(0),
            dump_column,
        )
        scratch = scratch.scatter(
            1,
            speech_dest.unsqueeze(-1).expand(-1, -1, hidden_size),
            speech_embeds,
        )

        # Neither scatter targets [0, prefix_length), so the speaker prefix can
        # simply be prepended.
        inputs_embeds = torch.cat(
            (speaker_embeds, scratch[:, prefix_length:max_total_length]),
            dim=1,
        )
        attention_mask = (
            torch.arange(max_total_length, device=device).unsqueeze(0)
            < total_lengths.unsqueeze(1)
        ).to(dtype=text_attention_mask.dtype)
        return (
            inputs_embeds,
            attention_mask,
            speech_starts,
            speech_lengths,
        )

    @staticmethod
    def _gather_speech_hidden(
        hidden_states,
        speech_starts,
        speech_lengths,
        speech_width,
    ):
        """Pull each row's speech span out of the packed sequence.

        One gather rather than a row loop, for the same reason as
        :meth:`_build_training_inputs`: the loop's int(speech_lengths[row]) was
        a device sync per row, and this one sits between the backbone forward
        and the loss, so it stalls the step twice over (once again in backward,
        as batch_size separate slice-assign gradients).
        """
        device = hidden_states.device
        positions = torch.arange(speech_width, device=device)
        keep = positions.unsqueeze(0) < speech_lengths.unsqueeze(1)
        # Padded rows read a clamped index and are then zeroed, which is what
        # the row loop left behind by never writing them.
        index = (speech_starts.unsqueeze(1) + positions.unsqueeze(0)).clamp(
            max=hidden_states.size(1) - 1
        )
        gathered = hidden_states.gather(
            1,
            index.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)),
        )
        return gathered * keep.unsqueeze(-1).to(dtype=gathered.dtype)

    def _build_generation_prompt(
        self,
        text_input_ids,
        text_attention_mask,
        speaker_features,
        speaker_feature_lengths,
        speech_bos_ids,
    ):
        if speaker_features is not None and speaker_features.size(0) != text_input_ids.size(0):
            raise ValueError("speaker_features and text_input_ids batch sizes differ.")
        speaker_embeds = self._encode_speaker_prefix(
            speaker_features,
            speaker_feature_lengths,
        )
        text_embeds = self.get_input_embeddings()(text_input_ids)
        speech_bos_embeds = self.speech_embedding(speech_bos_ids)
        speaker_embeds = speaker_embeds.to(dtype=text_embeds.dtype)
        speech_bos_embeds = speech_bos_embeds.to(dtype=text_embeds.dtype)

        text_lengths = text_attention_mask.sum(dim=1).long()
        prompt_lengths = text_lengths + speaker_embeds.size(1) + 1
        max_prompt_length = int(prompt_lengths.max().item())
        prompt_embeds = text_embeds.new_zeros(
            text_input_ids.size(0),
            max_prompt_length,
            text_embeds.size(-1),
        )
        prompt_mask = text_attention_mask.new_zeros(
            text_input_ids.size(0),
            max_prompt_length,
        )
        for row in range(text_input_ids.size(0)):
            valid_text = text_embeds[row][text_attention_mask[row].bool()]
            sequence = torch.cat(
                (
                    speaker_embeds[row],
                    valid_text,
                    speech_bos_embeds[row],
                ),
                dim=0,
            )
            padding = max_prompt_length - sequence.size(0)
            prompt_embeds[row, padding:] = sequence
            prompt_mask[row, padding:] = 1
        position_ids = self._position_ids_from_attention_mask(prompt_mask)
        return prompt_embeds, prompt_mask, position_ids

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

        (
            inputs_embeds,
            attention_mask,
            speech_starts,
            speech_lengths,
        ) = self._build_training_inputs(
            text_input_ids,
            text_attention_mask,
            speech_input_ids,
            speech_attention_mask,
            speaker_features,
            speaker_feature_lengths,
        )

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        speech_hidden = self._gather_speech_hidden(
            outputs.last_hidden_state,
            speech_starts,
            speech_lengths,
            speech_input_ids.size(1),
        )
        if labels is not None and labels.shape != speech_input_ids.shape:
            raise ValueError(
                "labels and speech_input_ids must have identical shapes."
            )
        logits = None
        loss = None
        fuse = (
            labels is not None
            and self.training
            and self._fused_ce is not None
            and speech_hidden.is_cuda
        )
        if fuse:
            # No logits are produced on purpose -- that is the whole saving.
            loss = self._fused_ce(
                self.speech_head.weight,
                speech_hidden.reshape(-1, speech_hidden.size(-1)),
                labels.reshape(-1),
            )
        else:
            logits = self.speech_head(speech_hidden)
            if labels is not None:
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

        prompt_embeds, attention_mask, position_ids = self._build_generation_prompt(
            text_input_ids,
            text_attention_mask,
            speaker_features,
            speaker_feature_lengths,
            generated,
        )
        output = self.backbone(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        next_logits = self.speech_head(output.last_hidden_state[:, -1]).float()

        for _ in range(max_new_tokens):
            next_logits = next_logits / temperature
            # BOS/PAD are input-only control tokens and must never be emitted.
            next_logits[:, self.config.speech_bos_token_id] = -torch.inf
            next_logits[:, self.config.speech_pad_token_id] = -torch.inf
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
            position_ids = attention_mask.sum(dim=1, keepdim=True) - 1
            output = self.backbone(
                inputs_embeds=self.speech_embedding(next_token),
                attention_mask=attention_mask,
                position_ids=position_ids,
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


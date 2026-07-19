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
import os

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

target_speaker_embedding = None

QWEN3_5_MODEL_ID = "Qwen/Qwen3.5-2B-Base"
QWEN3_TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

QWEN3_5_TTS_SPECIAL_TOKENS = {
    "im_start_token_id": "<|im_start|>",
    "im_end_token_id": "<|im_end|>",
    "tts_pad_token_id": "<tts_pad>",
    "tts_bos_token_id": "<tts_text_bos>",
    "tts_eos_token_id": "<tts_text_eod>",
}


def configure_qwen3_5_text_tokens(config, tokenizer):
    for config_name, token in QWEN3_5_TTS_SPECIAL_TOKENS.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Qwen3.5 tokenizer does not define required token {token!r}.")
        setattr(config, config_name, token_id)


def save_checkpoint_assets(model, processor, output_dir):
    processor.save_pretrained(output_dir)

    speech_tokenizer = model.speech_tokenizer
    speech_tokenizer_dir = os.path.join(output_dir, "speech_tokenizer")
    speech_tokenizer.model.save_pretrained(speech_tokenizer_dir, safe_serialization=True)
    speech_tokenizer.feature_extractor.save_pretrained(speech_tokenizer_dir)

    generation_config_path = os.path.join(output_dir, "generation_config.json")
    with open(generation_config_path, "w", encoding="utf-8") as f:
        json.dump(model.generate_config, f, indent=2, ensure_ascii=False)


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=QWEN3_5_MODEL_ID)
    parser.add_argument("--tts_init_model_path", type=str, default=QWEN3_TTS_MODEL_ID)
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.tts_init_model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    try:
        from transformers import Qwen3_5ForCausalLM
    except ImportError as exc:
        raise ImportError(
            "Qwen3.5 requires transformers>=5.3.0. Run `uv sync` before training."
        ) from exc

    qwen3_5 = Qwen3_5ForCausalLM.from_pretrained(
        args.base_model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    qwen3tts.model.talker.set_qwen3_5_backbone(qwen3_5.model)
    del qwen3_5

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    processor = Qwen3TTSProcessor(tokenizer=tokenizer)
    qwen3tts.processor = processor

    config = qwen3tts.model.config
    configure_qwen3_5_text_tokens(config, tokenizer)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.text_projection(
                    model.talker.get_text_embeddings()(input_text_ids)
                ) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                    use_cache=False,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.config.tts_model_type = "custom_voice"
            unwrapped_model.config.talker_config.spk_id = {
                args.speaker_name: 3000
            }
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            unwrapped_model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=True,
            )
            save_checkpoint_assets(unwrapped_model, processor, output_dir)

if __name__ == "__main__":
    train()

# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os

import torch
from accelerate import Accelerator
from finetuning.dataset import Text2SemanticDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from qwen_tts.core.models import Text2SemanticForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        default="Qwen/Qwen3.5-2B-Base",
    )
    parser.add_argument("--output_model_path", default="output")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_text_tokens", type=int)
    parser.add_argument("--max_semantic_tokens", type=int)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    args = parser.parse_args()
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup_ratio must be in [0, 1).")
    return args


def read_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    if not samples:
        raise ValueError("Training JSONL contains no samples.")
    return samples


def train():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_model_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Text2SemanticForCausalLM.from_qwen_pretrained(
        args.base_model_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.requires_grad_(True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    dataset = Text2SemanticDataset(
        read_jsonl(args.train_jsonl),
        tokenizer,
        speech_bos_token_id=model.config.speech_bos_token_id,
        speech_eos_token_id=model.config.speech_eos_token_id,
        max_text_tokens=args.max_text_tokens,
        max_semantic_tokens=args.max_semantic_tokens,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable != total:
        raise RuntimeError(
            f"Expected full-parameter training, got {trainable:,}/{total:,} trainable."
        )
    accelerator.print(f"Full-parameter training: {trainable:,} parameters")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    total_steps = updates_per_epoch * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    accelerator.init_trackers(
        "text2semantic",
        config={
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    )
    model.train()
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                output = model(**batch, use_cache=False)
                accelerator.backward(output.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss {output.loss.item():.4f}"
                )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            output_dir = os.path.join(
                args.output_model_path, f"checkpoint-epoch-{epoch}"
            )
            unwrapped = accelerator.unwrap_model(model)
            state_dict = accelerator.get_state_dict(model)
            unwrapped.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(output_dir)
            with open(
                os.path.join(output_dir, "generation_config.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "max_new_tokens": 1500,
                        "temperature": 0.8,
                        "top_k": 30,
                        "do_sample": True,
                        "speech_eos_token_id": unwrapped.config.speech_eos_token_id,
                    },
                    handle,
                    indent=2,
                )
    accelerator.end_training()


if __name__ == "__main__":
    train()

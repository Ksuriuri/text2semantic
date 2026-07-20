# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from finetuning.dataset import Text2SemanticDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from qwen_tts.core.models import Text2SemanticForCausalLM


WANDB_PROJECT = "text2semantic"
WANDB_ENTITY = "haoyuanhuang22-jcxy"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        default="Qwen/Qwen3.5-2B-Base",
    )
    parser.add_argument("--output_model_path", default="output")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_text_tokens", type=int)
    parser.add_argument("--max_semantic_tokens", type=int)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--wandb_run_name")
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
    if args.checkpointing_steps <= 0:
        parser.error("--checkpointing_steps must be positive.")
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
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {exc}"
                ) from exc
    if not samples:
        raise ValueError(f"{path} contains no samples.")
    return samples


def build_dataset(path, tokenizer, model_config, args):
    return Text2SemanticDataset(
        read_jsonl(path),
        tokenizer,
        semantic_vocab_size=model_config.semantic_vocab_size,
        speech_bos_token_id=model_config.speech_bos_token_id,
        speech_eos_token_id=model_config.speech_eos_token_id,
        max_text_tokens=args.max_text_tokens,
        max_semantic_tokens=args.max_semantic_tokens,
    )


@torch.inference_mode()
def evaluate(model, dataloader, accelerator):
    model.eval()
    totals = torch.zeros(5, dtype=torch.float64, device=accelerator.device)
    for batch in dataloader:
        output = model(**batch, use_cache=False)
        labels = batch["labels"]
        valid = labels.ne(-100)
        predictions = output.logits.argmax(dim=-1)
        token_loss = F.cross_entropy(
            output.logits.float().reshape(-1, output.logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)
        eos_mask = labels.eq(
            accelerator.unwrap_model(model).config.speech_eos_token_id
        )
        batch_totals = torch.stack(
            (
                token_loss.sum(dim=1),
                (predictions.eq(labels) & valid).sum(dim=1),
                valid.sum(dim=1),
                (predictions.eq(labels) & eos_mask).sum(dim=1),
                eos_mask.sum(dim=1),
            ),
            dim=1,
        )
        totals += accelerator.gather_for_metrics(batch_totals).double().sum(0)
    model.train()
    token_count = totals[2].clamp_min(1)
    return {
        "eval/loss": (totals[0] / token_count).item(),
        "eval/token_accuracy": (totals[1] / token_count).item(),
        "eval/eos_accuracy": (totals[3] / totals[4].clamp_min(1)).item(),
    }


def save_checkpoint(
    accelerator,
    model,
    tokenizer,
    output_dir,
    *,
    epoch,
    step_in_epoch,
    global_step,
):
    accelerator.wait_for_everyone()
    os.makedirs(output_dir, exist_ok=True)
    accelerator.save_state(os.path.join(output_dir, "accelerator_state"))
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
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
        with open(
            os.path.join(output_dir, "trainer_state.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "global_step": global_step,
                },
                handle,
                indent=2,
            )
    accelerator.wait_for_everyone()


def load_resume_state(accelerator, checkpoint):
    state_path = os.path.join(checkpoint, "trainer_state.json")
    accelerator_state = os.path.join(checkpoint, "accelerator_state")
    if not os.path.isfile(state_path) or not os.path.isdir(accelerator_state):
        raise ValueError(
            f"{checkpoint} is not a resumable text2semantic checkpoint."
        )
    with open(state_path, encoding="utf-8") as handle:
        trainer_state = json.load(handle)
    accelerator.load_state(accelerator_state)
    return (
        int(trainer_state["epoch"]),
        int(trainer_state["step_in_epoch"]),
        int(trainer_state["global_step"]),
    )


def train():
    args = parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise EnvironmentError(
            "Set WANDB_API_KEY in the environment before launching training."
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
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

    train_dataset = build_dataset(
        args.train_jsonl, tokenizer, model.config, args
    )
    eval_dataset = build_dataset(
        args.eval_jsonl, tokenizer, model.config, args
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn,
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
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    total_steps = updates_per_epoch * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
    )

    tracker_config = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    accelerator.init_trackers(
        WANDB_PROJECT,
        config=tracker_config,
        init_kwargs={
            "wandb": {
                "entity": WANDB_ENTITY,
                "name": args.wandb_run_name,
            }
        },
    )

    start_epoch = 0
    resume_step = 0
    global_step = 0
    if args.resume_from_checkpoint:
        start_epoch, resume_step, global_step = load_resume_state(
            accelerator, args.resume_from_checkpoint
        )
        accelerator.print(
            f"Resumed from {args.resume_from_checkpoint} at "
            f"epoch={start_epoch}, step={resume_step}, global_step={global_step}"
        )

    model.train()
    for epoch in range(start_epoch, args.num_epochs):
        if epoch == start_epoch and resume_step:
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
            first_step = resume_step
        else:
            active_dataloader = train_dataloader
            first_step = 0

        for step, batch in enumerate(active_dataloader, start=first_step):
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

            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log(
                    {
                        "train/loss": output.loss.detach().float().item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (step + 1) / len(train_dataloader),
                    },
                    step=global_step,
                )
                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        os.path.join(
                            args.output_model_path,
                            f"checkpoint-step-{global_step}",
                        ),
                        epoch=epoch,
                        step_in_epoch=step + 1,
                        global_step=global_step,
                    )

        metrics = evaluate(model, eval_dataloader, accelerator)
        accelerator.log(metrics, step=global_step)
        accelerator.print(
            f"Epoch {epoch} | eval loss {metrics['eval/loss']:.4f} | "
            f"token acc {metrics['eval/token_accuracy']:.4f} | "
            f"EOS acc {metrics['eval/eos_accuracy']:.4f}"
        )
        save_checkpoint(
            accelerator,
            model,
            tokenizer,
            os.path.join(
                args.output_model_path, f"checkpoint-epoch-{epoch}"
            ),
            epoch=epoch + 1,
            step_in_epoch=0,
            global_step=global_step,
        )
        resume_step = 0
    accelerator.end_training()


if __name__ == "__main__":
    train()

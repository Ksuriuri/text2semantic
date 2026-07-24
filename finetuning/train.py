# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from dotenv import load_dotenv
from finetuning.dataset import Text2SemanticDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from qwen_tts.core.models import Text2SemanticForCausalLM
from qwen_tts.semantic_codec import MaskGCTFeatureExtractor


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
    parser.add_argument("--w2v_bert_path", required=True)
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--max_ref_seconds", type=float, default=20.0)
    parser.add_argument("--max_target_seconds", type=float, default=30.0)
    parser.add_argument("--min_speaker_records", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--new_module_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_text_tokens", type=int)
    parser.add_argument("--max_semantic_tokens", type=int)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=10000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_total_limit", type=int, default=2)
    parser.add_argument("--keep_checkpointing_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
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
    if args.max_train_steps <= 0:
        parser.error("--max_train_steps must be positive.")
    if args.num_epochs is not None and args.num_epochs <= 0:
        parser.error("--num_epochs must be positive when set.")
    if args.lr <= 0:
        parser.error("--lr must be positive.")
    if args.new_module_lr <= 0:
        parser.error("--new_module_lr must be positive.")
    if not 0 <= args.adam_beta1 < 1:
        parser.error("--adam_beta1 must be in [0, 1).")
    if not 0 <= args.adam_beta2 < 1:
        parser.error("--adam_beta2 must be in [0, 1).")
    if args.adam_epsilon <= 0:
        parser.error("--adam_epsilon must be positive.")
    if args.checkpointing_steps <= 0:
        parser.error("--checkpointing_steps must be positive.")
    if args.checkpoint_total_limit < 0:
        parser.error("--checkpoint_total_limit must be non-negative.")
    if args.keep_checkpointing_steps <= 0:
        parser.error("--keep_checkpointing_steps must be positive.")
    if args.logging_steps <= 0:
        parser.error("--logging_steps must be positive.")
    if args.eval_steps <= 0:
        parser.error("--eval_steps must be positive.")
    if args.max_ref_seconds <= 0:
        parser.error("--max_ref_seconds must be positive.")
    if args.max_target_seconds <= 0:
        parser.error("--max_target_seconds must be positive.")
    if args.min_speaker_records < 1:
        parser.error("--min_speaker_records must be positive.")
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


def target_audio_path(item):
    return item.get("audio") or item.get("audio_path")


def speaker_key(item):
    speaker_id = item.get("speaker_id")
    if speaker_id is None:
        return None
    language = item.get("language") or item.get("lang")
    return language, speaker_id


def speaker_statistics(*datasets):
    speaker_counts = Counter()
    speaker_audio_paths = defaultdict(list)
    for data in datasets:
        for item in data:
            key = speaker_key(item)
            if key is None:
                continue
            speaker_counts[key] += 1
            audio_path = target_audio_path(item)
            if (
                audio_path is not None
                and audio_path not in speaker_audio_paths[key]
            ):
                speaker_audio_paths[key].append(audio_path)
    return dict(speaker_counts), dict(speaker_audio_paths)


def build_dataset(
    data,
    tokenizer,
    model_config,
    args,
    speaker_counts,
    speaker_audio_paths,
):
    return Text2SemanticDataset(
        data,
        tokenizer,
        semantic_vocab_size=model_config.semantic_vocab_size,
        speech_bos_token_id=model_config.speech_bos_token_id,
        speech_eos_token_id=model_config.speech_eos_token_id,
        speech_pad_token_id=model_config.speech_pad_token_id,
        max_text_tokens=args.max_text_tokens,
        max_semantic_tokens=args.max_semantic_tokens,
        speaker_counts=speaker_counts,
        speaker_audio_paths_by_id=speaker_audio_paths,
        min_speaker_records=args.min_speaker_records,
        max_target_seconds=args.max_target_seconds,
    )


def add_speaker_features(batch, feature_extractor, max_ref_seconds):
    audio_paths = batch.pop("speaker_audio_paths", None)
    if audio_paths is None:
        return batch
    if feature_extractor is None:
        raise ValueError(
            "A speaker feature extractor is required for audio-path batches."
        )
    features, lengths = feature_extractor.encode_files(
        audio_paths,
        max_audio_seconds=max_ref_seconds,
    )
    batch["speaker_features"] = features
    batch["speaker_feature_lengths"] = lengths
    return batch


@torch.inference_mode()
def evaluate(
    model,
    dataloader,
    accelerator,
    feature_extractor=None,
    max_ref_seconds=20.0,
):
    model.eval()
    totals = torch.zeros(5, dtype=torch.float64, device=accelerator.device)
    for batch in dataloader:
        batch = add_speaker_features(
            batch,
            feature_extractor,
            max_ref_seconds,
        )
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


def parameter_group_name(parameter_name):
    if parameter_name.startswith("backbone."):
        return "backbone"
    return "new_modules"


def should_decay_parameter(parameter_name, parameter):
    if parameter.ndim <= 1:
        return False
    lowered = parameter_name.lower()
    no_decay_terms = ("bias", "norm", "embedding", "embeddings")
    return not any(term in lowered for term in no_decay_terms)


def build_optimizer(model, args):
    groups = {
        ("backbone", True): [],
        ("backbone", False): [],
        ("new_modules", True): [],
        ("new_modules", False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_key = (
            parameter_group_name(name),
            should_decay_parameter(name, parameter),
        )
        groups[group_key].append(parameter)

    param_groups = []
    for group_name, lr in (
        ("backbone", args.lr),
        ("new_modules", args.new_module_lr),
    ):
        for use_decay in (True, False):
            parameters = groups[(group_name, use_decay)]
            if not parameters:
                continue
            param_groups.append(
                {
                    "params": parameters,
                    "lr": lr,
                    "weight_decay": args.weight_decay if use_decay else 0.0,
                    "name": f"{group_name}_{'decay' if use_decay else 'no_decay'}",
                    "lr_group": group_name,
                }
            )
    return AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )


def learning_rates_by_group(optimizer):
    lrs = {}
    for group in optimizer.param_groups:
        lr_group = group.get("lr_group")
        if lr_group is None or lr_group in lrs:
            continue
        lrs[lr_group] = group["lr"]
    return lrs


def sorted_checkpoints(output_dir, prefix="checkpoint-step-"):
    if not os.path.isdir(output_dir):
        return []
    checkpoints = []
    for name in os.listdir(output_dir):
        if not name.startswith(prefix):
            continue
        step_text = name[len(prefix) :]
        if not step_text.isdigit():
            continue
        checkpoints.append((int(step_text), os.path.join(output_dir, name)))
    return [path for _, path in sorted(checkpoints)]


def rotate_checkpoints(output_dir, limit):
    if limit == 0:
        checkpoints = sorted_checkpoints(output_dir)
    else:
        checkpoints = sorted_checkpoints(output_dir)[:-limit]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


def run_evaluation(
    model,
    eval_dataloader,
    accelerator,
    speaker_feature_extractor,
    max_ref_seconds,
    global_step,
):
    metrics = evaluate(
        model,
        eval_dataloader,
        accelerator,
        speaker_feature_extractor,
        max_ref_seconds,
    )
    accelerator.log(metrics, step=global_step)
    accelerator.print(
        f"Step {global_step} | eval loss {metrics['eval/loss']:.4f} | "
        f"token acc {metrics['eval/token_accuracy']:.4f} | "
        f"EOS acc {metrics['eval/eos_accuracy']:.4f}"
    )
    return metrics


def train():
    args = parse_args()
    load_dotenv()
    set_seed(args.seed)
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
    speaker_feature_extractor = MaskGCTFeatureExtractor(
        w2v_bert_path=args.w2v_bert_path,
        stats_path=args.stats_path,
        device=accelerator.device,
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

    train_data = read_jsonl(args.train_jsonl)
    eval_data = read_jsonl(args.eval_jsonl)
    train_speaker_counts, train_speaker_audio_paths = speaker_statistics(
        train_data
    )
    eval_speaker_counts, eval_speaker_audio_paths = speaker_statistics(
        eval_data
    )
    train_dataset = build_dataset(
        train_data,
        tokenizer,
        model.config,
        args,
        train_speaker_counts,
        train_speaker_audio_paths,
    )
    eval_dataset = build_dataset(
        eval_data,
        tokenizer,
        model.config,
        args,
        eval_speaker_counts,
        eval_speaker_audio_paths,
    )
    accelerator.print(
        f"Train samples: {len(train_dataset):,}/{train_dataset.raw_size:,} "
        f"after filtering; eval samples: {len(eval_dataset):,}/"
        f"{eval_dataset.raw_size:,} after filtering"
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=train_generator,
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

    optimizer = build_optimizer(model, args)
    total_steps = args.max_train_steps
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
    epoch = start_epoch
    last_eval_step = 0
    while global_step < total_steps and (
        args.num_epochs is None or epoch < args.num_epochs
    ):
        if epoch == start_epoch and resume_step:
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
            first_step = resume_step
        else:
            active_dataloader = train_dataloader
            first_step = 0
        if hasattr(active_dataloader, "set_epoch"):
            active_dataloader.set_epoch(epoch)

        for step, batch in enumerate(active_dataloader, start=first_step):
            if global_step >= total_steps:
                break
            batch = add_speaker_features(
                batch,
                speaker_feature_extractor,
                args.max_ref_seconds,
            )
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
                if global_step % args.logging_steps == 0:
                    group_lrs = learning_rates_by_group(optimizer)
                    log_values = {
                        "train/loss": output.loss.detach().float().item(),
                        "train/epoch": epoch + (step + 1) / len(train_dataloader),
                    }
                    if "backbone" in group_lrs:
                        log_values["train/lr_backbone"] = group_lrs["backbone"]
                    if "new_modules" in group_lrs:
                        log_values["train/lr_new_modules"] = group_lrs[
                            "new_modules"
                        ]
                    accelerator.log(log_values, step=global_step)
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
                    if accelerator.is_main_process:
                        rotate_checkpoints(
                            args.output_model_path, args.checkpoint_total_limit
                        )
                    accelerator.wait_for_everyone()
                if global_step % args.keep_checkpointing_steps == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        os.path.join(
                            args.output_model_path,
                            f"checkpoint-keep-step-{global_step}",
                        ),
                        epoch=epoch,
                        step_in_epoch=step + 1,
                        global_step=global_step,
                    )
                if global_step % args.eval_steps == 0:
                    run_evaluation(
                        model,
                        eval_dataloader,
                        accelerator,
                        speaker_feature_extractor,
                        args.max_ref_seconds,
                        global_step,
                    )
                    last_eval_step = global_step
                if global_step >= total_steps:
                    break

        resume_step = 0
        epoch += 1
    if global_step and global_step != last_eval_step:
        run_evaluation(
            model,
            eval_dataloader,
            accelerator,
            speaker_feature_extractor,
            args.max_ref_seconds,
            global_step,
        )
    accelerator.end_training()


if __name__ == "__main__":
    train()

# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import shutil
import signal
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from dotenv import load_dotenv
from finetuning import manifest_index
from finetuning.checkpoint_policy import (
    ACTION_NONE,
    ACTION_PERSISTENT,
    ACTION_ROLLING,
    PERSISTENT_PREFIX,
    ROLLING_PREFIX,
    CheckpointPolicy,
    checkpoint_dir_name,
)
from finetuning.dataset import Text2SemanticDataset
from finetuning.ref_store import SpeakerRefStore, default_ref_index_path
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
    parser.add_argument("--min_target_seconds", type=float, default=0.5)
    parser.add_argument("--min_speaker_records", type=int, default=2)
    parser.add_argument(
        "--ref_index",
        default=None,
        help=(
            "Path to refs/speaker_index.jsonl (or .parquet). Default: look "
            "next to --train_jsonl for ../refs/speaker_index.jsonl. "
            "Packed speaker refs are the default read path."
        ),
    )
    parser.add_argument(
        "--ref_root",
        default=None,
        help="Trainset refs/ directory. Default: parent of --ref_index.",
    )
    parser.add_argument(
        "--ref_cache",
        default=None,
        help="Local extract cache for packed refs. Default: refs/.extract-cache",
    )
    parser.add_argument(
        "--refs_per_speaker",
        type=int,
        default=0,
        help="Use at most N packed refs per speaker. 0 = all packed refs.",
    )
    parser.add_argument(
        "--loose_refs",
        action="store_true",
        help="Ignore packed refs and use loose audio_path files (legacy).",
    )
    parser.add_argument(
        "--manifest_index_dir",
        default=None,
        help=(
            "Where the shared row index lives. Default: <manifest>.index next "
            "to the manifest itself."
        ),
    )
    parser.add_argument(
        "--rebuild_manifest_index",
        action="store_true",
        help="Rebuild the row index even when the cached one still matches.",
    )
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
    parser.add_argument(
        "--punctuation_dropout_prob",
        type=float,
        default=0.1,
        help=(
            "Probability of removing the written pause marks from a "
            "training sample's text, so the model has to place the pauses "
            "itself.  Line feeds are always kept.  Train split only."
        ),
    )
    parser.add_argument(
        "--punctuation_dropout_keep_word_spaces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep single spaces between words when dropping punctuation, "
            "so space-separated scripts keep their word boundaries.  On by "
            "default; --no-punctuation_dropout_keep_word_spaces drops the "
            "spaces as well."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50,
        help=(
            "Rolling checkpoints land on multiples of this step count, but "
            "only once --checkpointing_min_interval_minutes has also passed. "
            "The step multiple sets the granularity; the interval sets the rate."
        ),
    )
    parser.add_argument(
        "--checkpointing_min_interval_minutes",
        type=float,
        default=30.0,
        help=(
            "Minimum wall-clock minutes between checkpoints of any kind. "
            "0 disables the time gate and saves on every --checkpointing_steps."
        ),
    )
    parser.add_argument("--checkpoint_total_limit", type=int, default=2)
    parser.add_argument("--keep_checkpointing_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume_from_checkpoint",
        help=(
            "Checkpoint directory to resume from, or 'auto' to pick the "
            "newest checkpoint under --output_model_path (nothing to resume "
            "from is not an error, so a preempted job can be relaunched with "
            "the same command line)."
        ),
    )
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
    if args.checkpointing_min_interval_minutes < 0:
        parser.error(
            "--checkpointing_min_interval_minutes must be non-negative."
        )
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
    if args.min_target_seconds < 0:
        parser.error("--min_target_seconds must be non-negative.")
    if args.min_speaker_records < 1:
        parser.error("--min_speaker_records must be positive.")
    if args.refs_per_speaker < 0:
        parser.error("--refs_per_speaker must be >= 0 (0 = all packed refs).")
    if not 0 <= args.punctuation_dropout_prob <= 1:
        parser.error("--punctuation_dropout_prob must be in [0, 1].")
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


def resolve_ref_store(args, *, build_index_if_missing=True):
    if args.loose_refs:
        return None
    index_path = args.ref_index
    if index_path:
        index_path = os.path.expanduser(index_path)
    else:
        discovered = default_ref_index_path(args.train_jsonl)
        index_path = str(discovered) if discovered is not None else None
    if not index_path:
        return None
    return SpeakerRefStore(
        index_path,
        ref_root=args.ref_root,
        cache_dir=args.ref_cache,
        refs_per_speaker=args.refs_per_speaker,
        build_index_if_missing=build_index_if_missing,
    )


def manifest_filter_params(args, ref_store):
    return manifest_index.FilterParams(
        min_target_seconds=args.min_target_seconds,
        max_target_seconds=args.max_target_seconds,
        max_semantic_tokens=args.max_semantic_tokens,
        min_speaker_records=args.min_speaker_records,
        refs_per_speaker=args.refs_per_speaker,
        ref_index=None if ref_store is None else str(ref_store.index_path),
    )


def manifest_index_dir(args, manifest_path):
    if not args.manifest_index_dir:
        return None
    name = os.path.basename(os.path.abspath(manifest_path))
    return os.path.join(args.manifest_index_dir, name + ".index")


def open_manifest(manifest_path, args, ref_store, *, log=None):
    """The manifest as a memmapped, prefiltered row index.

    Every rank opens the same index files and preads the same jsonl, so a node
    holds one copy of the manifest in page cache rather than one Python list per
    rank.
    """
    return manifest_index.load(
        manifest_path,
        params=manifest_filter_params(args, ref_store),
        ref_store=ref_store,
        index_dir=manifest_index_dir(args, manifest_path),
        rebuild=args.rebuild_manifest_index,
        log=log,
    )


def build_dataset(
    data,
    tokenizer,
    model_config,
    args,
    speaker_counts,
    speaker_audio_paths,
    *,
    punctuation_dropout_prob=0.0,
    ref_store=None,
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
        ref_store=ref_store,
        min_speaker_records=args.min_speaker_records,
        max_target_seconds=args.max_target_seconds,
        min_target_seconds=getattr(args, "min_target_seconds", 0.5),
        ref_max_seconds=getattr(args, "max_ref_seconds", 20.0),
        punctuation_dropout_prob=punctuation_dropout_prob,
        punctuation_dropout_keep_word_spaces=(
            args.punctuation_dropout_keep_word_spaces
        ),
        seed=args.seed,
    )


def add_speaker_features(batch, feature_extractor, max_ref_seconds):
    audio_paths = batch.pop("speaker_audio_paths", None)
    waveforms = batch.pop("speaker_waveforms", None)
    if audio_paths is None and waveforms is None:
        return batch
    if feature_extractor is None:
        raise ValueError(
            "A speaker feature extractor is required for audio-path batches."
        )
    if waveforms is not None:
        # Packed refs arrive already decoded from the DataLoader workers.
        features, lengths = feature_extractor.encode_audios(
            waveforms,
            max_audio_seconds=max_ref_seconds,
        )
    else:
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


def sorted_checkpoints(output_dir, prefix=ROLLING_PREFIX):
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


def latest_checkpoint(output_dir):
    """The newest resumable checkpoint, rolling or persistent.

    A persistent save suppresses the rolling save on the same step, so the
    newest state can live under either prefix and both have to be considered.
    Only directories that finished writing (``trainer_state.json`` is written
    last) count, so a checkpoint interrupted mid-write is skipped rather than
    failing the resume.
    """
    candidates = []
    for prefix in (ROLLING_PREFIX, PERSISTENT_PREFIX):
        for path in sorted_checkpoints(output_dir, prefix):
            step = int(os.path.basename(path)[len(prefix) :])
            if os.path.isfile(os.path.join(path, "trainer_state.json")):
                candidates.append((step, path))
    if not candidates:
        return None
    # Ties (same step under both prefixes) resolve to the persistent one, which
    # sorts second by path; either holds identical state.
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


class PreemptionFlag:
    """Set when the node is being taken away.

    A Spot node gets roughly 30 seconds' notice, delivered as SIGTERM to the
    processes on *that* node only -- the other node in a multi-node job hears
    nothing.  So the flag is local, and the training loop has to agree on it
    collectively (see ``decide_checkpoint``) or the ranks that saw the signal
    would enter the save barriers alone and hang.
    """

    def __init__(self):
        self.triggered = False
        self.signal_number = None

    def install(self):
        # SIGUSR1 is the manual equivalent: "checkpoint and stop cleanly".
        for number in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(number, self._handle)
        return self

    def _handle(self, signal_number, _frame):
        self.triggered = True
        self.signal_number = signal_number


def decide_checkpoint(accelerator, policy, global_step, preempting=False):
    """One checkpoint decision, identical on every rank.

    Both inputs are rank-local: wall clock differs between machines, and a
    preemption signal reaches one node only.  Saving is collective, so a
    disagreement here is a hang rather than a wrong file.  One max-reduce
    settles both -- rank 0 is the only rank that proposes an action, and the
    preemption flag is true for everyone if it is true for anyone.
    """
    proposal = policy.action(global_step) if accelerator.is_main_process else 0
    signal_state = torch.tensor(
        [proposal, 1 if preempting else 0],
        dtype=torch.int64,
        device=accelerator.device,
    )
    signal_state = accelerator.reduce(signal_state, reduction="max")
    action, preempting_anywhere = (int(value) for value in signal_state.tolist())
    if preempting_anywhere and action == ACTION_NONE:
        # Best effort: 30 seconds is not enough for 28 GB of optimizer state on
        # every node, so this is a bonus save, not the thing keeping the run
        # safe.  The rolling interval is what bounds the loss.
        action = ACTION_ROLLING
    return action, bool(preempting_anywhere)


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

    # Rank 0 builds the shared indexes (the speaker key table and the manifest
    # row indexes); the other ranks then open the same files. Building on every
    # rank would have eight processes writing the same index at once.
    if accelerator.is_main_process:
        prepared_refs = resolve_ref_store(args, build_index_if_missing=True)
        for manifest_path in (args.train_jsonl, args.eval_jsonl):
            open_manifest(
                manifest_path, args, prepared_refs, log=accelerator.print
            )
        prepared_refs = None
    accelerator.wait_for_everyone()
    ref_store = resolve_ref_store(args, build_index_if_missing=False)
    if ref_store is not None:
        accelerator.print(
            f"Packed refs: index={ref_store.index_path} "
            f"speakers={len(ref_store):,} "
            f"backend={ref_store.index_backend} "
            f"refs_per_speaker={args.refs_per_speaker or 'all packed'}"
        )
    else:
        accelerator.print(
            "Packed refs not found; falling back to loose audio_path files. "
            "Pass --ref_index or place refs/speaker_index.jsonl next to "
            "manifests/ to use the default shard layout."
        )
    train_data = open_manifest(args.train_jsonl, args, ref_store)
    eval_data = open_manifest(args.eval_jsonl, args, ref_store)
    accelerator.print(
        f"Manifest index: train {len(train_data):,} rows "
        f"({train_data.index_dir}), eval {len(eval_data):,} rows"
    )
    train_dataset = build_dataset(
        train_data,
        tokenizer,
        model.config,
        args,
        None,
        None,
        punctuation_dropout_prob=args.punctuation_dropout_prob,
        ref_store=ref_store,
    )
    # The eval split keeps its punctuation: an augmented eval set would
    # make the loss/accuracy curve move for reasons unrelated to training.
    eval_dataset = build_dataset(
        eval_data,
        tokenizer,
        model.config,
        args,
        None,
        None,
        ref_store=ref_store,
    )
    accelerator.print(
        f"Train samples: {len(train_dataset):,}/{train_dataset.raw_size:,} "
        f"after filtering; eval samples: {len(eval_dataset):,}/"
        f"{eval_dataset.raw_size:,} after filtering"
    )
    accelerator.print(
        "Punctuation dropout (train split only): p="
        f"{args.punctuation_dropout_prob}, keep_word_spaces="
        f"{args.punctuation_dropout_keep_word_spaces}"
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
    resume_from = args.resume_from_checkpoint
    if resume_from == "auto":
        # A preempted job is relaunched with the same command line, so "resume
        # from whatever is newest, or start fresh" has to be expressible.
        resume_from = latest_checkpoint(args.output_model_path)
        accelerator.print(
            f"Auto-resume: {resume_from or 'no checkpoint found, starting fresh'}"
        )
    if resume_from:
        start_epoch, resume_step, global_step = load_resume_state(
            accelerator, resume_from
        )
        accelerator.print(
            f"Resumed from {resume_from} at "
            f"epoch={start_epoch}, step={resume_step}, global_step={global_step}"
        )

    preemption = PreemptionFlag().install()
    policy = CheckpointPolicy(
        rolling_steps=args.checkpointing_steps,
        persistent_steps=args.keep_checkpointing_steps,
        min_interval_seconds=args.checkpointing_min_interval_minutes * 60.0,
    )
    accelerator.print(
        f"Checkpoints: rolling on step %{args.checkpointing_steps} and "
        f">={args.checkpointing_min_interval_minutes:g} min since the last "
        f"save, keeping {args.checkpoint_total_limit}; persistent every "
        f"{args.keep_checkpointing_steps} steps; eval every {args.eval_steps} "
        "steps"
    )

    model.train()
    epoch = start_epoch
    last_eval_step = 0
    stopping = False
    while (
        not stopping
        and global_step < total_steps
        and (args.num_epochs is None or epoch < args.num_epochs)
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
                action, preempting = decide_checkpoint(
                    accelerator,
                    policy,
                    global_step,
                    preempting=preemption.triggered,
                )
                if action != ACTION_NONE:
                    save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        os.path.join(
                            args.output_model_path,
                            checkpoint_dir_name(action, global_step),
                        ),
                        epoch=epoch,
                        step_in_epoch=step + 1,
                        global_step=global_step,
                    )
                    policy.record_save()
                    if action == ACTION_ROLLING and accelerator.is_main_process:
                        # Persistent checkpoints use their own prefix and are
                        # never rotated.
                        rotate_checkpoints(
                            args.output_model_path, args.checkpoint_total_limit
                        )
                    accelerator.wait_for_everyone()
                    accelerator.print(
                        f"Step {global_step} | saved "
                        f"{checkpoint_dir_name(action, global_step)}"
                    )
                if preempting:
                    accelerator.print(
                        f"Step {global_step} | preemption signalled, stopping "
                        "after the checkpoint above"
                    )
                    stopping = True
                    break
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
    # A preempted job has seconds left, not the minutes an eval pass takes.
    if not stopping and global_step and global_step != last_eval_step:
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

# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
import shutil
import signal
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from dotenv import load_dotenv
from finetuning import checkpoint_remote, manifest_index
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
from finetuning import source_ref_store, speaker_index
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from qwen_tts.core.models import Text2SemanticForCausalLM
from qwen_tts.semantic_codec import (
    FLOAT_DTYPES,
    MaskGCTFeatureExtractor,
)


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
    parser.add_argument(
        "--speaker_encoder_dtype",
        default="bfloat16",
        choices=sorted(FLOAT_DTYPES),
        help=(
            "Precision of the frozen W2V-BERT that produces speaker "
            "conditioning features. Its fp32 forward was 974 ms of a 3.96 s "
            "step on B200 at batch_size 32 and 20 s refs; bfloat16 is 301 ms "
            "for a 0.031 relative difference on a feature that is mean/std "
            "normalised and fed to a trainable encoder."
        ),
    )
    parser.add_argument(
        "--speaker_mel_in_workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute the ref log-mel in the DataLoader workers instead of "
            "inline in the training step. It is pure CPU work that took 912 ms "
            "of a 3.955 s B200 step with the GPU idle, and the workers are "
            "otherwise idle. Needs --num_workers > 0."
        ),
    )
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
        "--ref_backend",
        choices=("packed", "source_tar"),
        default="packed",
        help=(
            "Where refs come from. 'packed' reads the trainset's ref shards; "
            "'source_tar' reads every clip a speaker has straight out of the "
            "source audio tars by byte range, which needs --ref_source_bucket "
            "(or --ref_source_root for a local mirror)."
        ),
    )
    parser.add_argument(
        "--ref_source_bucket",
        default=None,
        help="GCS bucket holding preprocessed/<dataset>/audio/*.tar.",
    )
    parser.add_argument(
        "--ref_source_project",
        default=None,
        help="GCP project to bill the ref reads to. Default: ambient.",
    )
    parser.add_argument(
        "--ref_source_root",
        default=None,
        help=(
            "Local directory mirroring the bucket layout, used instead of "
            "--ref_source_bucket when the tars are already on this machine."
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
    parser.add_argument(
        "--lr_schedule",
        choices=("cosine", "constant"),
        default="cosine",
        help=(
            "Shape after warmup. 'cosine' decays to zero over the run; "
            "'constant' holds the peak LR, which is what a run whose length "
            "is a budget decision rather than a convergence plan wants -- "
            "stopping or extending it then does not change the LR of the "
            "steps already taken."
        ),
    )
    # Exactly one of these sets the run length, because the cosine schedule is
    # built from it: a step count that outlives the epoch limit decays the LR to
    # zero mid-training, and one that undershoots stops the run with the LR still
    # high. There is deliberately no default -- silently training 100k steps when
    # the caller meant one epoch is the failure this replaces.
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Fixed run length in optimizer steps. Mutually exclusive with "
        "--num_epochs.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Run length in passes over the manifest. The step count, warmup "
        "and cosine decay are derived from the manifest size, so they follow "
        "the dataset instead of needing to be recomputed by hand.",
    )
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
    parser.add_argument(
        "--checkpoint_remote_dir",
        help=(
            "gs:// or s3:// prefix to mirror checkpoints to, whichever cloud "
            "the run is on. A Spot preemption takes "
            "the machine and its disk, so on Spot this is what makes the run "
            "resumable at all; with it set, 'auto' resume reads from here."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume_from_checkpoint",
        help=(
            "Checkpoint directory to resume from, or 'auto' to pick the "
            "newest checkpoint (the newest complete one on "
            "--checkpoint_remote_dir when that is set, otherwise the newest "
            "under --output_model_path). Nothing to resume from is not an "
            "error, so a preempted job can be relaunched with the same "
            "command line."
        ),
    )
    parser.add_argument("--wandb_run_name")
    parser.add_argument(
        "--wandb_run_id",
        default=os.environ.get("WANDB_RUN_ID"),
        help=(
            "Stable W&B run id, so a job relaunched after a Spot preemption "
            "continues the same curve instead of starting a second one. The "
            "run resumes at the global step of the checkpoint it loaded, so "
            "whatever was logged between that checkpoint and the preemption is "
            "sent again; W&B keeps the first value for a step it already has. "
            "Defaults to $WANDB_RUN_ID."
        ),
    )
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
    if (args.max_train_steps is None) == (args.num_epochs is None):
        parser.error(
            "set exactly one of --num_epochs (run length follows the manifest) "
            "or --max_train_steps (fixed step count)."
        )
    if args.max_train_steps is not None and args.max_train_steps <= 0:
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
    if args.checkpoint_remote_dir and not checkpoint_remote.is_remote(
        args.checkpoint_remote_dir
    ):
        parser.error("--checkpoint_remote_dir must be a gs:// or s3:// URI.")
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


def speaker_key(item, key_fields=speaker_index.DEFAULT_KEY_FIELDS):
    return speaker_index.row_key(item, key_fields)


def speaker_statistics(
    *datasets, key_fields=speaker_index.DEFAULT_KEY_FIELDS
):
    speaker_counts = Counter()
    speaker_audio_paths = defaultdict(list)
    for data in datasets:
        for item in data:
            key = speaker_key(item, key_fields)
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
    if args.ref_backend == "source_tar":
        return resolve_source_ref_store(
            args, build_index_if_missing=build_index_if_missing
        )
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


def resolve_source_ref_store(args, *, build_index_if_missing=True):
    """The source-tar ref store: every clip of a speaker, read by byte range."""
    index_path = args.ref_index
    if index_path:
        index_path = os.path.expanduser(index_path)
    else:
        discovered = source_ref_store.default_index_path(args.train_jsonl)
        index_path = str(discovered) if discovered is not None else None
    if not index_path:
        raise SystemExit(
            "--ref_backend source_tar needs --ref_index pointing at "
            f"refs/{source_ref_store.INDEX_NAME}."
        )
    if args.ref_source_root:
        reader = source_ref_store.LocalRangeReader(args.ref_source_root)
    elif args.ref_source_bucket:
        reader = source_ref_store.GcsRangeReader(
            args.ref_source_bucket, project=args.ref_source_project
        )
    else:
        raise SystemExit(
            "--ref_backend source_tar needs --ref_source_bucket or "
            "--ref_source_root."
        )
    return source_ref_store.SourceTarRefStore(
        index_path,
        reader=reader,
        refs_per_speaker=args.refs_per_speaker,
        build_index_if_missing=build_index_if_missing,
    )


def speaker_key_fields(ref_store):
    return tuple(
        getattr(ref_store, "speaker_key_fields", speaker_index.DEFAULT_KEY_FIELDS)
    )


def manifest_filter_params(args, ref_store):
    return manifest_index.FilterParams(
        min_target_seconds=args.min_target_seconds,
        max_target_seconds=args.max_target_seconds,
        max_semantic_tokens=args.max_semantic_tokens,
        min_speaker_records=args.min_speaker_records,
        refs_per_speaker=args.refs_per_speaker,
        ref_index=None if ref_store is None else str(ref_store.index_path),
        speaker_key=",".join(speaker_key_fields(ref_store)),
    )


def manifest_index_dir(args, manifest_path):
    if not args.manifest_index_dir:
        return None
    name = os.path.basename(os.path.abspath(manifest_path))
    return os.path.join(args.manifest_index_dir, name + ".index")


def open_manifest(manifest_path, args, ref_store, *, log=None, rebuild=False):
    """The manifest as a memmapped, prefiltered row index.

    Every rank opens the same index files and preads the same jsonl, so a node
    holds one copy of the manifest in page cache rather than one Python list per
    rank.

    ``rebuild`` belongs to the build phase only: passing it on every rank's open
    would rebuild the index once per rank, one after another, behind the build
    lock.
    """
    return manifest_index.load(
        manifest_path,
        params=manifest_filter_params(args, ref_store),
        ref_store=ref_store,
        index_dir=manifest_index_dir(args, manifest_path),
        rebuild=rebuild,
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
    speaker_mel_extractor=None,
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
        speaker_mel_extractor=speaker_mel_extractor,
        punctuation_dropout_prob=punctuation_dropout_prob,
        punctuation_dropout_keep_word_spaces=(
            args.punctuation_dropout_keep_word_spaces
        ),
        seed=args.seed,
    )


def add_speaker_features(batch, feature_extractor, max_ref_seconds):
    audio_paths = batch.pop("speaker_audio_paths", None)
    waveforms = batch.pop("speaker_waveforms", None)
    mel_features = batch.pop("speaker_input_features", None)
    mel_mask = batch.pop("speaker_feature_attention_mask", None)
    if audio_paths is None and waveforms is None and mel_features is None:
        return batch
    if feature_extractor is None:
        raise ValueError(
            "A speaker feature extractor is required for audio-path batches."
        )
    if mel_features is not None:
        # The log-mel was already computed in a DataLoader worker; only the
        # frozen W2V-BERT forward is left for this process to do.
        features, lengths = feature_extractor.encode_features(
            mel_features,
            mel_mask,
        )
    elif waveforms is not None:
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


def mirror_checkpoint(accelerator, local_dir, remote_prefix, *, log=None):
    """Copy a just-saved checkpoint to object storage.

    Every node uploads its own directory: ``save_state`` writes the model and
    optimizer only from the main process but ``random_states_<rank>.pkl`` from
    every rank, so with node-local disks neither node holds a complete
    checkpoint on its own.  The marker goes on last, from one process, once both
    nodes are done -- it is the only thing that distinguishes a finished upload
    from one a preemption cut in half.
    """
    if not remote_prefix:
        return None
    remote_dir = checkpoint_remote.join(remote_prefix, os.path.basename(local_dir))
    if accelerator.is_local_main_process:
        checkpoint_remote.upload(local_dir, remote_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_remote.mark_complete(remote_dir)
    accelerator.wait_for_everyone()
    if log is not None:
        log(f"Mirrored {os.path.basename(local_dir)} -> {remote_dir}")
    return remote_dir


def fetch_remote_checkpoint(accelerator, remote_prefix, output_dir, *, log=None):
    """Make the newest complete remote checkpoint available locally.

    With a remote prefix configured it is the only source of truth for resume,
    local directories are ignored: a node that survived a preemption may still
    hold a checkpoint whose upload never finished, and letting each node pick
    its own newest local one would have the two nodes resume from different
    weights -- which DDP does not detect, it just trains on wrong gradients.
    """
    remote_dir = checkpoint_remote.latest_complete(remote_prefix)
    if remote_dir is None:
        return None
    local_dir = os.path.join(output_dir, os.path.basename(remote_dir))
    if accelerator.is_local_main_process:
        os.makedirs(local_dir, exist_ok=True)
        if log is not None:
            log(f"Downloading {remote_dir} -> {local_dir}")
        checkpoint_remote.download(remote_dir, local_dir)
    accelerator.wait_for_everyone()
    return local_dir


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


def wandb_init_kwargs(run_name, run_id):
    """`wandb.init` arguments, resuming the same run when an id is given.

    A Spot run is relaunched with the same command line after every preemption.
    W&B mints a fresh run id per process start, so without a stable id each
    relaunch appears as its own curve and the run reads as N short trainings
    instead of one.  `resume="allow"` creates the run the first time and attaches
    to it afterwards, which is what makes the id enough on its own.
    """
    kwargs = {"entity": WANDB_ENTITY, "name": run_name}
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = "allow"
    return kwargs


def steps_for_epochs(
    num_batches, num_processes, gradient_accumulation_steps, num_epochs
):
    """Optimizer steps in `num_epochs` passes over an unsharded dataloader.

    Called before `accelerator.prepare`, so `num_batches` is the whole-manifest
    batch count and the per-rank count has to be derived: accelerate hands every
    rank the same number of batches, rounding up, so a run that would leave one
    rank a batch short still takes the extra step.
    """
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    batches_per_rank = math.ceil(num_batches / num_processes)
    steps_per_epoch = math.ceil(batches_per_rank / gradient_accumulation_steps)
    return steps_per_epoch * num_epochs


def schedule_steps_for_accelerate(optimizer_steps, num_processes, split_batches):
    """Translate optimizer steps into the count `accelerator.prepare` expects.

    A scheduler passed through `accelerate.Accelerator.prepare` is wrapped in
    `AcceleratedScheduler`, which -- unless the dataloader splits batches --
    advances the wrapped scheduler `num_processes` times per optimizer step,
    because it assumes `num_training_steps` was counted on an unsharded
    dataloader.  `steps_for_epochs` (and `--max_train_steps`) count the real
    optimizer steps of the *sharded* run, so handing that number straight to
    `get_cosine_schedule_with_warmup` makes the schedule finish
    `num_processes` times too early: warmup ends almost immediately and the
    cosine sits at zero for the rest of the run.

    Measured, not theorised: the 4x5090 run (30,000 steps, 4 processes) logged
    lr_backbone 4.6976e-6 at step 5,880, which is the cosine value for step
    5,880 x 4 to three significant figures.  At 32 processes the same bug would
    zero the LR about 6,200 steps into a 199,723-step epoch.
    """
    if optimizer_steps < 0:
        # Zero is legal: --warmup_ratio 0 means no warmup at all.
        raise ValueError("optimizer_steps must not be negative")
    if num_processes <= 0:
        raise ValueError("num_processes must be positive")
    return optimizer_steps if split_batches else optimizer_steps * num_processes


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
        dtype=args.speaker_encoder_dtype,
    )
    # Hand the CPU log-mel to the DataLoader workers unless asked not to. With
    # --num_workers 0 there are no workers, so it would run in this process
    # either way and the indirection would only cost a second mel extractor.
    speaker_mel_extractor = (
        speaker_feature_extractor.mel_extractor
        if args.speaker_mel_in_workers and args.num_workers > 0
        else None
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

    # One process per *node* builds the shared indexes (the speaker key table
    # and the manifest row indexes), because the index directories sit on each
    # node's local disk: gating this on the global main process would leave the
    # second node without indexes, and its eight ranks would then all build the
    # same files at once. When the trainset is on a shared filesystem instead,
    # the two node builders serialise on the index build lock and the second one
    # finds the index already there.
    if accelerator.is_local_main_process:
        prepared_refs = resolve_ref_store(args, build_index_if_missing=True)
        for manifest_path in (args.train_jsonl, args.eval_jsonl):
            open_manifest(
                manifest_path,
                args,
                prepared_refs,
                log=accelerator.print,
                rebuild=args.rebuild_manifest_index,
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
        speaker_mel_extractor=speaker_mel_extractor,
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
        speaker_mel_extractor=speaker_mel_extractor,
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
    if args.num_epochs is not None:
        total_steps = steps_for_epochs(
            len(train_dataloader),
            accelerator.num_processes,
            args.gradient_accumulation_steps,
            args.num_epochs,
        )
        accelerator.print(
            f"Run length: {args.num_epochs} epoch(s) over "
            f"{len(train_dataset):,} rows = {total_steps:,} steps at "
            f"{args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
            " rows/step"
        )
    else:
        total_steps = args.max_train_steps
        accelerator.print(f"Run length: {total_steps:,} steps (fixed)")
    warmup_steps = int(total_steps * args.warmup_ratio)
    schedule_total = schedule_steps_for_accelerate(
        total_steps, accelerator.num_processes, accelerator.split_batches
    )
    schedule_warmup = schedule_steps_for_accelerate(
        warmup_steps, accelerator.num_processes, accelerator.split_batches
    )
    accelerator.print(
        f"LR schedule: {args.lr_schedule}, warmup {warmup_steps:,} / total "
        f"{total_steps:,} optimizer steps, built as {schedule_warmup:,} / "
        f"{schedule_total:,} scheduler steps for "
        f"{accelerator.num_processes} process(es)"
    )
    if args.lr_schedule == "constant":
        # Only warmup needs the run length; after it the LR never moves, so a
        # run that is cut short or extended keeps the same LR history.
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=schedule_warmup,
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=schedule_warmup,
            num_training_steps=schedule_total,
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
        init_kwargs={"wandb": wandb_init_kwargs(args.wandb_run_name, args.wandb_run_id)},
    )

    start_epoch = 0
    resume_step = 0
    global_step = 0
    resume_from = args.resume_from_checkpoint
    if resume_from == "auto":
        # A preempted job is relaunched with the same command line, so "resume
        # from whatever is newest, or start fresh" has to be expressible.
        if args.checkpoint_remote_dir:
            resume_from = fetch_remote_checkpoint(
                accelerator,
                args.checkpoint_remote_dir,
                args.output_model_path,
                log=accelerator.print,
            )
        else:
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
        "steps; mirrored to "
        f"{args.checkpoint_remote_dir or 'local disk only'}"
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
                    checkpoint_dir = os.path.join(
                        args.output_model_path,
                        checkpoint_dir_name(action, global_step),
                    )
                    save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        checkpoint_dir,
                        epoch=epoch,
                        step_in_epoch=step + 1,
                        global_step=global_step,
                    )
                    mirror_checkpoint(
                        accelerator,
                        checkpoint_dir,
                        args.checkpoint_remote_dir,
                        log=accelerator.print,
                    )
                    # After the upload, so a save that only made it to local
                    # disk is not counted as one that survives the machine.
                    policy.record_save()
                    if action == ACTION_ROLLING and accelerator.is_main_process:
                        # Persistent checkpoints use their own prefix and are
                        # never rotated.
                        rotate_checkpoints(
                            args.output_model_path, args.checkpoint_total_limit
                        )
                        if args.checkpoint_remote_dir:
                            checkpoint_remote.rotate(
                                args.checkpoint_remote_dir,
                                args.checkpoint_total_limit,
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

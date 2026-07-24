import json
import os

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import Qwen3_5TextConfig

from finetuning.train import (
    add_speaker_features,
    build_optimizer,
    evaluate,
    load_resume_state,
    learning_rates_by_group,
    parse_args,
    rotate_checkpoints,
    save_checkpoint,
    speaker_key,
    speaker_statistics,
    sorted_checkpoints,
)
from qwen_tts.core.models import (
    Text2SemanticConfig,
    Text2SemanticForCausalLM,
)


class SavingTokenizer:
    def save_pretrained(self, path):
        with open(path / "tokenizer_config.json", "w", encoding="utf-8") as handle:
            json.dump({"test": True}, handle)


def test_parse_args_defaults_match_dataset_limits(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_jsonl",
            "train.jsonl",
            "--eval_jsonl",
            "eval.jsonl",
            "--w2v_bert_path",
            "w2v",
            "--stats_path",
            "stats.pt",
        ],
    )
    args = parse_args()
    assert args.lr == 4e-5
    assert args.new_module_lr == 2e-4
    assert args.max_train_steps == 100000
    assert args.num_epochs is None
    assert args.logging_steps == 10
    assert args.eval_steps == 10000
    assert args.checkpointing_steps == 1000
    assert args.checkpoint_total_limit == 2
    assert args.keep_checkpointing_steps == 10000
    assert args.seed == 42
    assert args.max_ref_seconds == 20.0
    assert args.max_target_seconds == 30.0
    assert args.min_speaker_records == 2


def test_speaker_statistics_are_split_local():
    train_counts, train_paths = speaker_statistics(
        [
            {
                "speaker_id": "speaker-a",
                "language": "en",
                "audio_path": "train-en.wav",
            },
            {
                "speaker_id": "speaker-a",
                "language": "zh",
                "audio_path": "train-zh.wav",
            },
        ]
    )
    eval_counts, eval_paths = speaker_statistics(
        [{"speaker_id": "speaker-a", "language": "en", "audio_path": "eval.wav"}]
    )

    assert speaker_key({"speaker_id": "speaker-a", "language": "en"}) == (
        "en",
        "speaker-a",
    )
    assert train_counts == {("en", "speaker-a"): 1, ("zh", "speaker-a"): 1}
    assert train_paths == {
        ("en", "speaker-a"): ["train-en.wav"],
        ("zh", "speaker-a"): ["train-zh.wav"],
    }
    assert eval_counts == {("en", "speaker-a"): 1}
    assert eval_paths == {("en", "speaker-a"): ["eval.wav"]}


def tiny_model():
    qwen_config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        max_position_embeddings=128,
    )
    return Text2SemanticForCausalLM(
        Text2SemanticConfig(
            qwen_config=qwen_config.to_dict(),
            semantic_vocab_size=16,
            speech_bos_token_id=16,
            speech_eos_token_id=17,
            speech_pad_token_id=18,
            speaker_input_dim=8,
            speaker_conformer_output_size=8,
            speaker_conformer_linear_units=16,
            speaker_conformer_attention_heads=2,
            speaker_conformer_num_blocks=1,
            speaker_conformer_input_layer="linear",
            speaker_num_latents=2,
            speaker_latent_dim=32,
            speaker_perceiver_depth=1,
            speaker_perceiver_ff_mult=2,
        )
    )


def test_accelerator_checkpoint_restores_full_training_state(tmp_path):
    accelerator = Accelerator(cpu=True)
    model = tiny_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )
    original = next(model.parameters()).detach().clone()

    save_checkpoint(
        accelerator,
        model,
        SavingTokenizer(),
        tmp_path,
        epoch=2,
        step_in_epoch=7,
        global_step=19,
    )
    with torch.no_grad():
        next(model.parameters()).add_(1)

    state = load_resume_state(accelerator, tmp_path)
    assert state == (2, 7, 19)
    assert torch.equal(next(model.parameters()).detach(), original)
    assert (tmp_path / "accelerator_state").is_dir()
    assert (tmp_path / "trainer_state.json").is_file()


def test_optimizer_uses_lr_groups_and_no_decay_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_jsonl",
            "train.jsonl",
            "--eval_jsonl",
            "eval.jsonl",
            "--w2v_bert_path",
            "w2v",
            "--stats_path",
            "stats.pt",
        ],
    )
    args = parse_args()
    optimizer = build_optimizer(tiny_model(), args)
    group_names = {group["name"] for group in optimizer.param_groups}

    assert group_names == {
        "backbone_decay",
        "backbone_no_decay",
        "new_modules_decay",
        "new_modules_no_decay",
    }
    assert learning_rates_by_group(optimizer) == {
        "backbone": 4e-5,
        "new_modules": 2e-4,
    }
    assert any(group["weight_decay"] == 0 for group in optimizer.param_groups)
    assert any(group["weight_decay"] == 0.01 for group in optimizer.param_groups)


def test_rotate_checkpoints_keeps_latest_regular_steps(tmp_path):
    for step in (1000, 2000, 3000, 4000):
        (tmp_path / f"checkpoint-step-{step}").mkdir()
    (tmp_path / "checkpoint-keep-step-4000").mkdir()

    rotate_checkpoints(tmp_path, limit=2)

    assert [os.path.basename(path) for path in sorted_checkpoints(tmp_path)] == [
        "checkpoint-step-3000",
        "checkpoint-step-4000",
    ]
    assert (tmp_path / "checkpoint-keep-step-4000").is_dir()


def test_evaluation_reports_semantic_and_eos_metrics():
    accelerator = Accelerator(cpu=True)
    model = tiny_model()
    samples = [
        {
            "text_input_ids": torch.tensor([2, 3]),
            "text_attention_mask": torch.tensor([1, 1]),
            "speech_input_ids": torch.tensor([16, 4, 5]),
            "speech_attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([4, 5, 17]),
            "speaker_features": torch.randn(5, 8),
            "speaker_feature_lengths": torch.tensor(5),
        }
    ]
    dataloader = DataLoader(samples, batch_size=1)
    model, dataloader = accelerator.prepare(model, dataloader)
    metrics = evaluate(model, dataloader, accelerator)
    assert metrics["eval/loss"] > 0
    assert 0 <= metrics["eval/token_accuracy"] <= 1
    assert 0 <= metrics["eval/eos_accuracy"] <= 1


def test_add_speaker_features_replaces_audio_paths():
    class Extractor:
        def encode_files(self, paths, max_audio_seconds):
            assert paths == ["ref.wav"]
            assert max_audio_seconds == 12.0
            return torch.ones(1, 4, 8), torch.tensor([4])

    batch = {"speaker_audio_paths": ["ref.wav"]}
    result = add_speaker_features(batch, Extractor(), 12.0)
    assert "speaker_audio_paths" not in result
    assert result["speaker_features"].shape == (1, 4, 8)
    assert result["speaker_feature_lengths"].tolist() == [4]

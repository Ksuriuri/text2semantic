import json

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import Qwen3_5TextConfig

from finetuning.train import evaluate, load_resume_state, save_checkpoint
from qwen_tts.core.models import (
    Text2SemanticConfig,
    Text2SemanticForCausalLM,
)


class SavingTokenizer:
    def save_pretrained(self, path):
        with open(path / "tokenizer_config.json", "w", encoding="utf-8") as handle:
            json.dump({"test": True}, handle)


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
            speech_pad_token_id=17,
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
        }
    ]
    dataloader = DataLoader(samples, batch_size=1)
    model, dataloader = accelerator.prepare(model, dataloader)
    metrics = evaluate(model, dataloader, accelerator)
    assert metrics["eval/loss"] > 0
    assert 0 <= metrics["eval/token_accuracy"] <= 1
    assert 0 <= metrics["eval/eos_accuracy"] <= 1

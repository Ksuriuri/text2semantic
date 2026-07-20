import torch
from torch import nn
from transformers import Qwen3_5TextConfig

from finetuning.dataset import Text2SemanticDataset
from qwen_tts.core.models import (
    Text2SemanticConfig,
    Text2SemanticForCausalLM,
)
from qwen_tts.semantic_codec import RepCodec


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [2, len(messages[0]["content"]) + 2]


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
    config = Text2SemanticConfig(
        qwen_config=qwen_config.to_dict(),
        semantic_vocab_size=16,
        speech_bos_token_id=16,
        speech_eos_token_id=17,
        speech_pad_token_id=17,
    )
    return Text2SemanticForCausalLM(config)


def test_dataset_alignment_and_mask():
    dataset = Text2SemanticDataset(
        [
            {"text": "hello", "semantic_codes": [3, 4]},
            {"text": "x", "semantic_codes": [5]},
        ],
        DummyTokenizer(),
        speech_bos_token_id=8192,
        speech_eos_token_id=8193,
    )
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["speech_input_ids"].tolist() == [
        [8192, 3, 4],
        [8192, 5, 8193],
    ]
    assert batch["labels"].tolist() == [
        [3, 4, 8193],
        [5, 8193, -100],
    ]
    assert batch["speech_attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_forward_backward_and_independent_speech_parameters():
    model = tiny_model()
    output = model(
        text_input_ids=torch.tensor([[2, 3]]),
        speech_input_ids=torch.tensor([[16, 4, 5]]),
        labels=torch.tensor([[4, 5, 17]]),
    )
    assert output.logits.shape == (1, 3, 18)
    output.loss.backward()
    assert model.speech_embedding.weight.grad is not None
    assert model.speech_head.weight.grad is not None
    assert model.get_input_embeddings().weight.grad is not None
    assert not any(
        forbidden in key
        for key in model.state_dict()
        for forbidden in ("code_predictor", "speaker_encoder", "quantizer")
    )


def test_generation_stops_at_eos():
    model = tiny_model()

    class EosHead(nn.Module):
        def forward(self, hidden):
            logits = torch.full((*hidden.shape[:-1], 18), -100.0)
            logits[..., 17] = 100.0
            return logits

    model.speech_head = EosHead()
    generated = model.generate_semantic(
        torch.tensor([[2, 3]]),
        max_new_tokens=5,
        do_sample=False,
    )
    assert len(generated) == 1
    assert generated[0].numel() == 0


def test_checkpoint_round_trip(tmp_path):
    model = tiny_model()
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = Text2SemanticForCausalLM.from_pretrained(tmp_path)
    assert torch.equal(
        model.speech_embedding.weight,
        restored.speech_embedding.weight,
    )
    assert restored.config.semantic_vocab_size == 16


def test_repcodec_indices_are_in_range():
    codec = RepCodec(
        codebook_size=32,
        hidden_size=16,
        codebook_dim=4,
        vocos_dim=8,
        vocos_intermediate_dim=16,
        vocos_num_layers=1,
    ).eval()
    codes, _ = codec.quantize(torch.randn(1, 5, 16))
    assert codes.shape == (1, 5)
    assert int(codes.min()) >= 0
    assert int(codes.max()) < 32


from array import array

import pytest
import torch
from torch import nn
from transformers import BatchFeature, Qwen3_5TextConfig

from finetuning.dataset import Text2SemanticDataset
from qwen_tts.core.models import (
    Text2SemanticConfig,
    Text2SemanticForCausalLM,
)
from qwen_tts.inference.text2semantic_model import Text2SemanticModel
from qwen_tts.semantic_codec import (
    MaskGCTFeatureExtractor,
    MaskGCTSemanticTokenizer,
    RepCodec,
)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = "dummy"

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("The Qwen chat template must not be used for TTS.")

    def __call__(self, prompt, add_special_tokens):
        assert add_special_tokens is False
        assert prompt.startswith(
            "<|im_start|>system\nSpeak out the provided text.<|im_end|>\n"
        )
        assert prompt.endswith("<|im_start|>assistant\n")
        assert "<think>" not in prompt and "</think>" not in prompt
        return {"input_ids": [2] if "\nx<|im_end|>" in prompt else [2, 3]}


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
    return Text2SemanticForCausalLM(config)


def speaker_inputs(batch_size=1):
    return {
        "speaker_features": torch.randn(batch_size, 5, 8),
        "speaker_feature_lengths": torch.full((batch_size,), 5, dtype=torch.long),
    }


def test_dataset_alignment_and_mask():
    dataset = Text2SemanticDataset(
        [
            {
                "audio": "target-1.wav",
                "ref_audio": "ref-1.wav",
                "text": "hello",
                "speaker_id": "speaker-a",
                "semantic_codes": [3, 4],
            },
            {
                "audio": "target-2.wav",
                "text": "x",
                "speaker_id": "speaker-a",
                "semantic_codes": [5],
            },
        ],
        DummyTokenizer(),
        speech_bos_token_id=8192,
        speech_eos_token_id=8193,
        speech_pad_token_id=8194,
    )
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["speech_input_ids"].tolist() == [
        [8192, 3, 4],
        [8192, 5, 8194],
    ]
    assert batch["labels"].tolist() == [
        [3, 4, 8193],
        [5, 8193, -100],
    ]
    assert batch["speech_attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["text_input_ids"].tolist() == [[2, 3], [2, 0]]
    assert batch["text_attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["speaker_audio_paths"] == ["ref-1.wav", "target-1.wav"]


def test_dataset_reads_compact_codes_and_filters_speakers_and_duration(tmp_path):
    code_path = tmp_path / "codes.bin"
    with open(code_path, "wb") as handle:
        array("H", [3, 4, 5, 6, 7, 8]).tofile(handle)

    dataset = Text2SemanticDataset(
        [
            {
                "audio_path": "target-1.wav",
                "text": "hello",
                "speaker_id": "speaker-a",
                "duration": 10.0,
                "semantic_code_path": str(code_path),
                "semantic_code_offset": 1,
                "semantic_code_length": 2,
            },
            {
                "audio_path": "prompt-only.wav",
                "text": "too long",
                "speaker_id": "speaker-a",
                "duration": 31.0,
                "semantic_code_path": str(code_path),
                "semantic_code_offset": 3,
                "semantic_code_length": 1,
            },
            {
                "audio_path": "single-speaker.wav",
                "text": "skip",
                "speaker_id": "speaker-b",
                "duration": 5.0,
                "semantic_code_path": str(code_path),
                "semantic_code_offset": 4,
                "semantic_code_length": 1,
            },
        ],
        DummyTokenizer(),
        semantic_vocab_size=16,
        speech_bos_token_id=16,
        speech_eos_token_id=17,
        speech_pad_token_id=18,
    )

    assert len(dataset) == 1
    batch = dataset.collate_fn([dataset[0]])
    assert batch["speech_input_ids"].tolist() == [[16, 4, 5]]
    assert batch["labels"].tolist() == [[4, 5, 17]]
    assert batch["speaker_audio_paths"] == ["prompt-only.wav"]


def test_dataset_filters_overlong_semantic_targets_instead_of_truncating():
    with pytest.raises(ValueError, match="No usable samples"):
        Text2SemanticDataset(
            [
                {
                    "audio": "target.wav",
                    "ref_audio": "ref.wav",
                    "text": "hello",
                    "semantic_codes": [3, 4, 5],
                }
            ],
            DummyTokenizer(),
            max_semantic_tokens=2,
        )


def test_dataset_rejects_out_of_bounds_compact_code_ranges(tmp_path):
    code_path = tmp_path / "codes.bin"
    with open(code_path, "wb") as handle:
        array("H", [3, 4]).tofile(handle)

    dataset = Text2SemanticDataset(
        [
            {
                "audio": "target.wav",
                "ref_audio": "ref.wav",
                "text": "hello",
                "semantic_code_path": str(code_path),
                "semantic_code_offset": 1,
                "semantic_code_length": 3,
            }
        ],
        DummyTokenizer(),
    )

    with pytest.raises(ValueError, match="out of bounds"):
        dataset[0]


def test_dataset_filters_samples_without_independent_reference():
    with pytest.raises(ValueError, match="No usable samples"):
        Text2SemanticDataset(
            [
                {
                    "audio": "target.wav",
                    "text": "no speaker reference",
                    "semantic_codes": [3, 4],
                },
                {
                    "audio": "same.wav",
                    "ref_audio": "same.wav",
                    "text": "explicit target leak",
                    "semantic_codes": [5, 6],
                },
                {
                    "audio": "only.wav",
                    "text": "single speaker",
                    "speaker_id": "speaker-a",
                    "semantic_codes": [7, 8],
                },
            ],
            DummyTokenizer(),
        )


def test_forward_backward_and_independent_speech_parameters():
    model = tiny_model()
    output = model(
        text_input_ids=torch.tensor([[2, 3]]),
        speech_input_ids=torch.tensor([[16, 4, 5]]),
        labels=torch.tensor([[4, 5, 17]]),
        **speaker_inputs(),
    )
    assert output.logits.shape == (1, 3, 19)
    output.loss.backward()
    assert model.speech_embedding.weight.grad is not None
    assert model.speech_head.weight.grad is not None
    assert model.get_input_embeddings().weight.grad is not None
    assert next(model.speaker_encoder.parameters()).grad is not None
    assert not any(
        forbidden in key
        for key in model.state_dict()
        for forbidden in ("code_predictor", "quantizer")
    )
    assert any(key.startswith("speaker_encoder.") for key in model.state_dict())


def test_generation_stops_at_eos():
    model = tiny_model()

    class EosHead(nn.Module):
        def forward(self, hidden):
            logits = torch.full((*hidden.shape[:-1], 19), -100.0)
            logits[..., 17] = 100.0
            return logits

    model.speech_head = EosHead()
    generated = model.generate_semantic(
        torch.tensor([[2, 3]]),
        max_new_tokens=5,
        do_sample=False,
        **speaker_inputs(),
    )
    assert len(generated) == 1
    assert generated[0].numel() == 0


def test_generation_never_emits_pad_token():
    model = tiny_model()

    class PadBiasedHead(nn.Module):
        def forward(self, hidden):
            logits = torch.full((*hidden.shape[:-1], 19), -100.0)
            logits[..., 18] = 100.0
            logits[..., 17] = 90.0
            return logits

    model.speech_head = PadBiasedHead()
    generated = model.generate_semantic(
        torch.tensor([[2, 3]]),
        max_new_tokens=5,
        do_sample=False,
        **speaker_inputs(),
    )
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
    assert torch.equal(
        next(model.speaker_encoder.parameters()),
        next(restored.speaker_encoder.parameters()),
    )


def test_training_right_padding_and_generation_left_padding():
    model = tiny_model()
    features = torch.randn(2, 7, 8)
    lengths = torch.tensor([7, 4])
    latents = model.speaker_encoder(features, lengths)
    assert latents.shape == (2, 2, 32)

    _, training_mask, _, _ = model._build_training_inputs(
        torch.tensor([[2, 3, 0], [4, 5, 6]]),
        torch.tensor([[1, 1, 0], [1, 1, 1]]),
        torch.tensor([[16, 4, 18], [16, 5, 6]]),
        torch.tensor([[1, 1, 0], [1, 1, 1]]),
        features,
        lengths,
    )
    assert training_mask.tolist() == [
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    output = model(
        text_input_ids=torch.tensor([[2, 3, 0], [4, 5, 6]]),
        text_attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        speech_input_ids=torch.tensor([[16, 4, 18], [16, 5, 6]]),
        speech_attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        labels=torch.tensor([[4, 17, -100], [5, 6, 17]]),
        speaker_features=features,
        speaker_feature_lengths=lengths,
    )
    assert output.logits.shape == (2, 3, 19)
    assert torch.count_nonzero(output.logits[0, 2]) == 0

    _, generation_mask, generation_position_ids = model._build_generation_prompt(
        torch.tensor([[2, 3, 0], [4, 5, 6]]),
        torch.tensor([[1, 1, 0], [1, 1, 1]]),
        features,
        lengths,
        torch.tensor([[16], [16]]),
    )
    assert generation_mask.tolist() == [
        [0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]
    assert generation_position_ids.tolist() == [
        [0, 0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]


def test_inference_wrapper_broadcasts_reference_audio():
    class Model:
        device = torch.device("cpu")

        def generate_semantic(self, input_ids, **kwargs):
            assert input_ids.shape[0] == 2
            assert kwargs["speaker_features"].shape == (2, 4, 8)
            assert kwargs["speaker_feature_lengths"].tolist() == [4, 4]
            return [torch.tensor([1]), torch.tensor([2])]

    class Extractor:
        def encode_files(self, paths, max_audio_seconds):
            assert paths == ["ref.wav", "ref.wav"]
            assert max_audio_seconds == 15.0
            return torch.ones(2, 4, 8), torch.tensor([4, 4])

    wrapper = Text2SemanticModel(Model(), DummyTokenizer(), Extractor())
    result = wrapper.generate(["first", "second"], ref_audio="ref.wav")
    assert [tokens.tolist() for tokens in result] == [[1], [2]]


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


def test_semantic_tokenizer_forces_fp32_and_trims_padding(monkeypatch):
    class FeatureExtractor:
        def __call__(self, audio, sampling_rate, return_tensors):
            return BatchFeature(
                {
                    "input_features": torch.ones(1, 5, 3),
                    "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
                }
            )

    class SemanticModel:
        def __call__(self, input_features, attention_mask, output_hidden_states):
            assert input_features.dtype == torch.float32
            hidden_states = [None] * 18
            hidden_states[17] = torch.ones(1, 5, 4, dtype=torch.float32)
            return type("Output", (), {"hidden_states": hidden_states})()

    class Codec:
        def quantize(self, features):
            assert features.dtype == torch.float32
            return torch.tensor([[1, 2, 3, 4, 5]]), features

    tokenizer = MaskGCTSemanticTokenizer.__new__(MaskGCTSemanticTokenizer)
    tokenizer.device = torch.device("cpu")
    tokenizer.feature_extractor = FeatureExtractor()
    tokenizer.semantic_model = SemanticModel()
    tokenizer.mean = torch.zeros(4)
    tokenizer.std = torch.ones(4)
    tokenizer.codec = Codec()
    tokenizer.codebook_size = 8192
    monkeypatch.setattr(
        "qwen_tts.semantic_codec.librosa.load",
        lambda *args, **kwargs: (torch.zeros(160).numpy(), 16000),
    )

    codes = tokenizer.encode_file("dummy.wav")
    assert codes.tolist() == [1, 2, 3]


def test_maskgct_feature_extractor_batches_and_masks_padding(monkeypatch):
    class FeatureExtractor:
        def __call__(
            self,
            audios,
            sampling_rate,
            padding,
            return_attention_mask,
            return_tensors,
        ):
            assert len(audios) == 2
            assert sampling_rate == 16000
            assert padding and return_attention_mask and return_tensors == "pt"
            return BatchFeature(
                {
                    "input_features": torch.ones(2, 5, 3),
                    "attention_mask": torch.tensor(
                        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
                    ),
                }
            )

    class SemanticModel:
        def __call__(self, input_features, attention_mask, output_hidden_states):
            assert input_features.dtype == torch.float32
            hidden_states = [None] * 18
            hidden_states[17] = torch.ones(2, 5, 4)
            return type("Output", (), {"hidden_states": hidden_states})()

    extractor = MaskGCTFeatureExtractor.__new__(MaskGCTFeatureExtractor)
    extractor.device = torch.device("cpu")
    extractor.feature_extractor = FeatureExtractor()
    extractor.semantic_model = SemanticModel()
    extractor.mean = torch.zeros(4)
    extractor.std = torch.ones(4)
    monkeypatch.setattr(
        "qwen_tts.semantic_codec.librosa.load",
        lambda *args, **kwargs: (torch.zeros(160).numpy(), 16000),
    )

    features, lengths = extractor.encode_files(["a.wav", "b.wav"])
    assert features.shape == (2, 5, 4)
    assert lengths.tolist() == [3, 5]
    assert torch.count_nonzero(features[0, 3:]) == 0


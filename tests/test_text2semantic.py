import random
from array import array

import numpy as np
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
    SpeakerMelExtractor,
)
from qwen_tts.text_augment import strip_pause_marks


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


def test_dataset_filters_samples_without_usable_text():
    dataset = Text2SemanticDataset(
        [
            {
                "audio": "null-text.wav",
                "text": None,
                "speaker_id": "speaker-a",
                "semantic_codes": [3],
            },
            {
                "audio": "empty-text.wav",
                "text": "",
                "speaker_id": "speaker-a",
                "semantic_codes": [4],
            },
            {
                "audio": "blank-text.wav",
                "text": "   ",
                "speaker_id": "speaker-a",
                "semantic_codes": [5],
            },
            {
                "audio": "keep-1.wav",
                "text": "hello",
                "speaker_id": "speaker-a",
                "semantic_codes": [6],
            },
            {
                "audio": "keep-2.wav",
                "text": "hello",
                "speaker_id": "speaker-a",
                "semantic_codes": [7],
            },
        ],
        DummyTokenizer(),
    )

    assert [item["audio"] for item in dataset.data] == ["keep-1.wav", "keep-2.wav"]
    assert dataset.filtered_size == 3
    # Every surviving row must be materializable: the null-text rows used to pass
    # filtering and blow up inside a DataLoader worker instead.
    for index in range(len(dataset)):
        dataset[index]


def test_dataset_spreads_reference_clips_over_the_speaker():
    data = [
        {
            "audio": f"clip-{index}.wav",
            "text": "hello",
            "speaker_id": "speaker-a",
            "semantic_codes": [3],
        }
        for index in range(12)
    ]

    dataset = Text2SemanticDataset(data, DummyTokenizer())
    references = [dataset[index]["speaker_audio_path"] for index in range(len(dataset))]

    assert len(set(references)) > 1, "reference clips must not collapse onto one path"
    for index, reference in enumerate(references):
        assert reference != data[index]["audio"]
    # Seeded, so the choice is reproducible across ranks and resumes.
    again = Text2SemanticDataset(data, DummyTokenizer(), seed=dataset.seed)
    assert [again[i]["speaker_audio_path"] for i in range(len(again))] == references
    shifted = Text2SemanticDataset(data, DummyTokenizer(), seed=dataset.seed + 1)
    assert [shifted[i]["speaker_audio_path"] for i in range(len(shifted))] != references


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


def test_gradient_checkpointing_covers_the_speaker_encoder():
    # transformers' own gradient_checkpointing_enable only reaches the backbone,
    # and the speaker encoder is where 20 s references cost 13.5 GiB. Same
    # gradients either way -- the encoder's dropout is 0, so the recomputed
    # forward is identical rather than merely equivalent.
    torch.manual_seed(0)
    reference = tiny_model()
    reference.train()
    batch = dict(
        text_input_ids=torch.tensor([[2, 3]]),
        speech_input_ids=torch.tensor([[16, 4, 5]]),
        labels=torch.tensor([[4, 5, 17]]),
        **speaker_inputs(),
    )
    assert reference.speaker_gradient_checkpointing is False
    reference(**batch).loss.backward()
    plain = {
        name: parameter.grad.clone()
        for name, parameter in reference.speaker_encoder.named_parameters()
        if parameter.grad is not None
    }
    assert plain

    calls = []
    checkpointed = tiny_model()
    checkpointed.load_state_dict(reference.state_dict())
    checkpointed.train()
    checkpointed.gradient_checkpointing_enable()
    assert checkpointed.speaker_gradient_checkpointing is True
    original = torch.utils.checkpoint.checkpoint

    def counting_checkpoint(function, *args, **kwargs):
        calls.append(function)
        return original(function, *args, **kwargs)

    torch.utils.checkpoint.checkpoint = counting_checkpoint
    try:
        checkpointed(**batch).loss.backward()
    finally:
        torch.utils.checkpoint.checkpoint = original
    assert checkpointed.speaker_encoder in calls

    for name, parameter in checkpointed.speaker_encoder.named_parameters():
        assert torch.equal(parameter.grad, plain[name]), name

    checkpointed.gradient_checkpointing_disable()
    assert checkpointed.speaker_gradient_checkpointing is False


def test_checkpointing_accepts_features_from_the_frozen_extractor():
    # MaskGCTFeatureExtractor.encode_* are @torch.inference_mode(), so every
    # speaker feature tensor a training step sees is an inference tensor, and
    # checkpoint() refuses to save one for backward. Caught on 8 GPUs at batch 48,
    # which is a slow way to find it.
    model = tiny_model()
    model.gradient_checkpointing_enable()
    model.train()
    with torch.inference_mode():
        features = torch.randn(1, 5, 8)
        # An inference tensor too, and the one that actually broke the run: the
        # conversions in _encode_speaker_prefix only clear the flag when they copy,
        # and .to(dtype=torch.long) on a long tensor returns it unchanged.
        lengths = torch.full((1,), 5, dtype=torch.long)
    assert torch.is_inference(features) and torch.is_inference(lengths)
    output = model(
        text_input_ids=torch.tensor([[2, 3]]),
        speech_input_ids=torch.tensor([[16, 4, 5]]),
        labels=torch.tensor([[4, 5, 17]]),
        speaker_features=features,
        speaker_feature_lengths=lengths,
    )
    output.loss.backward()
    assert next(model.speaker_encoder.parameters()).grad is not None


def test_the_speaker_encoder_is_not_checkpointed_under_no_grad():
    # checkpoint() with grad disabled is pure recompute for nothing, and
    # generation calls this path on every sampled token.
    model = tiny_model()
    model.gradient_checkpointing_enable()
    model.train()
    calls = []
    original = torch.utils.checkpoint.checkpoint
    torch.utils.checkpoint.checkpoint = lambda function, *a, **k: calls.append(
        function
    ) or original(function, *a, **k)
    try:
        with torch.no_grad():
            model(
                text_input_ids=torch.tensor([[2, 3]]),
                speech_input_ids=torch.tensor([[16, 4, 5]]),
                **speaker_inputs(),
            )
    finally:
        torch.utils.checkpoint.checkpoint = original
    assert calls == []


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

    mel_extractor = SpeakerMelExtractor.__new__(SpeakerMelExtractor)
    mel_extractor.feature_extractor = FeatureExtractor()

    extractor = MaskGCTFeatureExtractor.__new__(MaskGCTFeatureExtractor)
    extractor.device = torch.device("cpu")
    extractor.dtype = torch.float32
    extractor.mel_extractor = mel_extractor
    extractor.feature_extractor = mel_extractor.feature_extractor
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


class RecordingTokenizer:
    """Tokenizer that keeps every prompt it was handed."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, add_special_tokens):
        assert add_special_tokens is False
        self.prompts.append(prompt)
        return {"input_ids": [2, 3]}


def _dropout_dataset(tokenizer, text="你好，世界！", **kwargs):
    return Text2SemanticDataset(
        [
            {
                "audio": "target-1.wav",
                "ref_audio": "ref-1.wav",
                "text": text,
                "speaker_id": "speaker-a",
                "semantic_codes": [3, 4],
            }
        ],
        tokenizer,
        min_speaker_records=1,
        **kwargs,
    )


def test_strip_pause_marks_removes_punctuation_and_spaces():
    assert strip_pause_marks("你好，世界！\n真的吗？") == "你好世界\n真的吗"
    assert strip_pause_marks("Hello, world!\tHow are you?") == "HelloworldHowareyou"
    # Ideographic space and NBSP are separators too.
    assert strip_pause_marks("a\u3000b\u00a0c") == "abc"
    # Wave dashes act as prosody marks even though Unicode calls them symbols.
    assert strip_pause_marks("好啊~好啊～") == "好啊好啊"
    # Digits and content-bearing symbols stay: dropping them would change what
    # is spoken, not how it is paced.
    assert strip_pause_marks("50% off, $3 + $4 = $7!") == "50%off$3+$4=$7"
    assert strip_pause_marks("R&B, 9/11, user_name@host") == "R&B9/11user_name@host"


def test_strip_pause_marks_keeps_marks_glued_inside_a_word():
    # Orthography, not prosody: dropping these changes what is read out.
    assert strip_pause_marks("It is 3.14, not 12:30.") == "Itis3.14not12:30"
    assert strip_pause_marks("don't stop state-of-the-art!") == "don'tstopstate-of-the-art"
    # The exemption needs all three characters to be ASCII, so a CJK comma
    # goes whether its neighbours are hanzi or Latin letters (mixed text).
    assert strip_pause_marks("好，好") == "好好"
    assert strip_pause_marks("hello，world") == "helloworld"
    assert (
        strip_pause_marks("中英 mixed，text", keep_word_spaces=True)
        == "中英 mixedtext"
    )


def test_strip_pause_marks_can_keep_word_boundaries():
    assert (
        strip_pause_marks("Hello, world!\n How  are you?", keep_word_spaces=True)
        == "Hello world\nHow are you"
    )
    assert strip_pause_marks("你好，世界。", keep_word_spaces=True) == "你好世界"


def test_strip_pause_marks_keeps_the_line_layout():
    # Line feeds are pauses the writer already committed to, so they stay,
    # and blank lines stay blank lines.
    assert strip_pause_marks("第一行。\n\n第二行！") == "第一行\n\n第二行"
    # A CRLF transcript keeps the LF and loses the CR.
    assert strip_pause_marks("one, two\r\nthree.") == "onetwo\nthree"
    assert (
        strip_pause_marks("one, two\r\nthree.", keep_word_spaces=True)
        == "one two\nthree"
    )
    # Only the punctuation goes on an all-punctuation line.
    assert strip_pause_marks("a\n……\nb") == "a\n\nb"


def test_strip_pause_marks_may_strip_everything():
    assert strip_pause_marks("……！！") == ""


def test_punctuation_dropout_is_off_unless_requested():
    tokenizer = RecordingTokenizer()
    dataset = _dropout_dataset(tokenizer)
    assert dataset.punctuation_dropout_prob == 0.0
    for _ in range(50):
        dataset[0]
    assert tokenizer.prompts
    assert all("\n你好，世界！<|im_end|>" in prompt for prompt in tokenizer.prompts)


def test_punctuation_dropout_strips_every_read_at_probability_one():
    tokenizer = RecordingTokenizer()
    dataset = _dropout_dataset(tokenizer, punctuation_dropout_prob=1.0)
    for _ in range(10):
        dataset[0]
    assert all("\n你好世界<|im_end|>" in prompt for prompt in tokenizer.prompts)


def test_punctuation_dropout_hits_roughly_the_configured_rate():
    tokenizer = RecordingTokenizer()
    dataset = _dropout_dataset(tokenizer, punctuation_dropout_prob=0.1)
    random.seed(12345)
    for _ in range(400):
        dataset[0]
    stripped = sum("\n你好世界<|im_end|>" in p for p in tokenizer.prompts)
    # Binomial(400, 0.1): mean 40, sd 6.  Seeded, so this is deterministic.
    assert 20 <= stripped <= 60


def test_punctuation_dropout_keeps_word_spaces_by_default():
    tokenizer = RecordingTokenizer()
    dataset = _dropout_dataset(
        tokenizer,
        text="Hello, world!",
        punctuation_dropout_prob=1.0,
    )
    assert dataset.punctuation_dropout_keep_word_spaces is True
    dataset[0]
    assert "\nHello world<|im_end|>" in tokenizer.prompts[-1]


def test_punctuation_dropout_can_drop_word_spaces_too():
    tokenizer = RecordingTokenizer()
    dataset = _dropout_dataset(
        tokenizer,
        text="Hello, world!",
        punctuation_dropout_prob=1.0,
        punctuation_dropout_keep_word_spaces=False,
    )
    dataset[0]
    assert "\nHelloworld<|im_end|>" in tokenizer.prompts[-1]


def test_punctuation_dropout_keeps_text_that_would_strip_to_nothing():
    tokenizer = RecordingTokenizer()
    dataset = _dropout_dataset(
        tokenizer, text="……！！", punctuation_dropout_prob=1.0
    )
    # An all-punctuation transcript must not become an empty prompt.
    dataset[0]
    assert "\n……！！<|im_end|>" in tokenizer.prompts[-1]


def test_punctuation_dropout_keeps_the_line_feeds_of_a_sample():
    tokenizer = RecordingTokenizer()
    dataset = _dropout_dataset(
        tokenizer, text="你好，世界！\n真的吗？", punctuation_dropout_prob=1.0
    )
    dataset[0]
    assert "\n你好世界\n真的吗<|im_end|>" in tokenizer.prompts[-1]


def test_punctuation_dropout_rejects_probabilities_outside_the_unit_range():
    with pytest.raises(ValueError):
        _dropout_dataset(RecordingTokenizer(), punctuation_dropout_prob=1.5)
    with pytest.raises(ValueError):
        _dropout_dataset(RecordingTokenizer(), punctuation_dropout_prob=-0.1)


def _row_loop_training_inputs(
    speaker_embeds,
    text_embeds,
    speech_embeds,
    text_attention_mask,
    speech_attention_mask,
):
    """The row loop that _build_training_inputs used to be, kept as the oracle.

    The vectorised version has to agree with this bit for bit, for every shape
    of padding, or the speedup is worthless.
    """
    text_lengths = text_attention_mask.sum(dim=1).long()
    speech_lengths = speech_attention_mask.sum(dim=1).long()
    total_lengths = text_lengths + speech_lengths + speaker_embeds.size(1)
    max_total_length = int(total_lengths.max().item())
    inputs_embeds = text_embeds.new_zeros(
        text_embeds.size(0), max_total_length, text_embeds.size(-1)
    )
    attention_mask = text_attention_mask.new_zeros(
        text_embeds.size(0), max_total_length
    )
    speech_starts = []
    for row in range(text_embeds.size(0)):
        text_length = int(text_lengths[row])
        speech_length = int(speech_lengths[row])
        sequence = torch.cat(
            (
                speaker_embeds[row],
                text_embeds[row, :text_length],
                speech_embeds[row, :speech_length],
            ),
            dim=0,
        )
        inputs_embeds[row, : sequence.size(0)] = sequence
        attention_mask[row, : sequence.size(0)] = 1
        speech_starts.append(speaker_embeds.size(1) + text_length)
    return inputs_embeds, attention_mask, speech_starts, speech_lengths


@pytest.mark.parametrize(
    "text_lengths,speech_lengths",
    [
        ([2, 2], [3, 3]),          # no padding at all
        ([3, 1], [4, 2]),          # both sides ragged
        ([1, 3], [4, 1]),          # the longest text is not the longest speech
        ([4, 4, 1, 2], [1, 5, 5, 3]),
        ([2], [3]),                # batch of one
    ],
)
def test_vectorised_sequence_assembly_matches_the_row_loop(
    text_lengths, speech_lengths
):
    torch.manual_seed(0)
    model = tiny_model()
    batch_size = len(text_lengths)
    text_width = max(text_lengths)
    speech_width = max(speech_lengths)

    text_input_ids = torch.randint(0, 32, (batch_size, text_width))
    speech_input_ids = torch.randint(0, 16, (batch_size, speech_width))
    text_attention_mask = torch.zeros(batch_size, text_width, dtype=torch.long)
    speech_attention_mask = torch.zeros(batch_size, speech_width, dtype=torch.long)
    for row, length in enumerate(text_lengths):
        text_attention_mask[row, :length] = 1
    for row, length in enumerate(speech_lengths):
        speech_attention_mask[row, :length] = 1

    speaker = speaker_inputs(batch_size)
    with torch.no_grad():
        got = model._build_training_inputs(
            text_input_ids,
            text_attention_mask,
            speech_input_ids,
            speech_attention_mask,
            speaker["speaker_features"],
            speaker["speaker_feature_lengths"],
        )
        speaker_embeds = model._encode_speaker_prefix(
            speaker["speaker_features"],
            speaker["speaker_feature_lengths"],
        )
        text_embeds = model.get_input_embeddings()(text_input_ids)
        speech_embeds = model.speech_embedding(speech_input_ids)
        want = _row_loop_training_inputs(
            speaker_embeds.to(dtype=text_embeds.dtype),
            text_embeds,
            speech_embeds.to(dtype=text_embeds.dtype),
            text_attention_mask,
            speech_attention_mask,
        )

    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])
    assert got[2].tolist() == want[2]
    assert torch.equal(got[3], want[3])


@pytest.mark.parametrize(
    "text_lengths,speech_lengths",
    [([2, 2], [3, 3]), ([3, 1], [4, 2]), ([1, 4, 2], [5, 1, 3])],
)
def test_gathered_speech_hidden_matches_the_row_loop(text_lengths, speech_lengths):
    torch.manual_seed(1)
    batch_size = len(text_lengths)
    speech_width = max(speech_lengths)
    prefix_length = 4
    hidden_size = 6

    speech_lengths_t = torch.tensor(speech_lengths, dtype=torch.long)
    speech_starts = prefix_length + torch.tensor(text_lengths, dtype=torch.long)
    total = int((speech_starts + speech_lengths_t).max())
    hidden_states = torch.randn(batch_size, total, hidden_size)

    want = hidden_states.new_zeros(batch_size, speech_width, hidden_size)
    for row in range(batch_size):
        start = int(speech_starts[row])
        length = int(speech_lengths_t[row])
        want[row, :length] = hidden_states[row, start : start + length]

    got = Text2SemanticForCausalLM._gather_speech_hidden(
        hidden_states, speech_starts, speech_lengths_t, speech_width
    )
    assert torch.equal(got, want)


def test_sequence_assembly_keeps_gradients_flowing_to_every_input():
    """The scatters must not silently detach an input the loop used to reach."""
    torch.manual_seed(2)
    model = tiny_model()
    text_input_ids = torch.randint(0, 32, (2, 3))
    speech_input_ids = torch.randint(0, 16, (2, 4))
    text_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    speech_attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    speaker = speaker_inputs(2)
    speaker["speaker_features"].requires_grad_(True)

    inputs_embeds, _, _, _ = model._build_training_inputs(
        text_input_ids,
        text_attention_mask,
        speech_input_ids,
        speech_attention_mask,
        speaker["speaker_features"],
        speaker["speaker_feature_lengths"],
    )
    inputs_embeds.sum().backward()

    assert speaker["speaker_features"].grad is not None
    assert torch.isfinite(speaker["speaker_features"].grad).all()
    assert model.speech_embedding.weight.grad is not None
    assert model.speech_embedding.weight.grad.abs().sum() > 0
    assert model.get_input_embeddings().weight.grad.abs().sum() > 0


def test_worker_computed_mel_reaches_the_same_features(monkeypatch):
    """The split must be a move, not a change: same numbers, different process.

    encode_audios() is now mel_extractor() followed by encode_features(), and
    the DataLoader worker calls the first half. If the two halves disagreed, the
    speaker conditioning would silently change the moment --num_workers > 0.
    """

    class FeatureExtractor:
        def __call__(
            self,
            audios,
            sampling_rate,
            padding,
            return_attention_mask,
            return_tensors,
        ):
            width = max(len(audio) for audio in audios)
            features = torch.zeros(len(audios), width, 3)
            mask = torch.zeros(len(audios), width, dtype=torch.long)
            for row, audio in enumerate(audios):
                features[row, : len(audio)] = torch.arange(1, len(audio) + 1)[
                    :, None
                ].float()
                mask[row, : len(audio)] = 1
            return BatchFeature(
                {"input_features": features, "attention_mask": mask}
            )

    class SemanticModel:
        def __call__(self, input_features, attention_mask, output_hidden_states):
            hidden_states = [None] * 18
            hidden_states[17] = input_features.float().sum(dim=-1, keepdim=True)
            hidden_states[17] = hidden_states[17].expand(-1, -1, 4).contiguous()
            return type("Output", (), {"hidden_states": hidden_states})()

    mel_extractor = SpeakerMelExtractor.__new__(SpeakerMelExtractor)
    mel_extractor.feature_extractor = FeatureExtractor()
    extractor = MaskGCTFeatureExtractor.__new__(MaskGCTFeatureExtractor)
    extractor.device = torch.device("cpu")
    extractor.dtype = torch.float32
    extractor.mel_extractor = mel_extractor
    extractor.feature_extractor = mel_extractor.feature_extractor
    extractor.semantic_model = SemanticModel()
    extractor.mean = torch.zeros(4)
    extractor.std = torch.ones(4)

    audios = [np.arange(4, dtype=np.float32), np.arange(7, dtype=np.float32)]
    inline_features, inline_lengths = extractor.encode_audios(
        audios, max_audio_seconds=None
    )
    # What a worker would put in the batch, then what the step does with it.
    worker_mel, worker_mask = mel_extractor(audios, max_audio_seconds=None)
    split_features, split_lengths = extractor.encode_features(
        worker_mel, worker_mask
    )

    assert torch.equal(inline_features, split_features)
    assert torch.equal(inline_lengths, split_lengths)
    assert inline_lengths.tolist() == [4, 7]


def test_chunking_the_frozen_forward_changes_nothing_but_the_batch_it_sees():
    """W2V-BERT rows are independent, so chunking must be a pure memory trade.

    The point of the chunk is the relative_key attention buffer, which is sized
    by the batch handed to the frozen model: 8.4 GB at batch 32 and 20 s refs,
    and the peak that stops the training batch from growing. If a chunk ever
    changed a feature, speaker conditioning would depend on --batch_size.
    """
    batches = []

    class SemanticModel:
        def __call__(self, input_features, attention_mask, output_hidden_states):
            batches.append(input_features.size(0))
            hidden_states = [None] * 18
            hidden_states[17] = input_features.float().sum(dim=-1, keepdim=True)
            hidden_states[17] = hidden_states[17].expand(-1, -1, 4).contiguous()
            return type("Output", (), {"hidden_states": hidden_states})()

    def extractor(chunk_size):
        made = MaskGCTFeatureExtractor.__new__(MaskGCTFeatureExtractor)
        made.device = torch.device("cpu")
        made.dtype = torch.float32
        made.semantic_model = SemanticModel()
        made.mean = torch.zeros(4)
        made.std = torch.ones(4)
        made.chunk_size = chunk_size
        return made

    torch.manual_seed(0)
    mel = torch.randn(7, 5, 3)
    mask = torch.ones(7, 5, dtype=torch.long)
    mask[3, 2:] = 0

    whole_features, whole_lengths = extractor(0).encode_features(mel, mask)
    assert batches == [7]
    batches.clear()
    chunked_features, chunked_lengths = extractor(3).encode_features(mel, mask)

    assert batches == [3, 3, 1]
    assert torch.equal(whole_features, chunked_features)
    assert torch.equal(whole_lengths, chunked_lengths)


def test_the_mel_extractor_truncates_to_max_ref_seconds():
    # The truncation used to live in encode_audios; if it had not moved with the
    # mel, a 20 s cap would silently become "whatever the shard holds".
    class FeatureExtractor:
        def __call__(self, audios, **kwargs):
            width = max(len(audio) for audio in audios)
            return BatchFeature(
                {
                    "input_features": torch.zeros(len(audios), width, 3),
                    "attention_mask": torch.ones(
                        len(audios), width, dtype=torch.long
                    ),
                }
            )

    mel_extractor = SpeakerMelExtractor.__new__(SpeakerMelExtractor)
    mel_extractor.feature_extractor = FeatureExtractor()
    audios = [np.zeros(16000 * 3, dtype=np.float32)]
    features, _ = mel_extractor(audios, max_audio_seconds=1.0)
    assert features.size(1) == 16000
    with pytest.raises(ValueError):
        mel_extractor(audios, max_audio_seconds=0)
    with pytest.raises(ValueError):
        mel_extractor([], max_audio_seconds=1.0)


def test_speech_id_range_is_validated_once_not_per_forward():
    # The range check calls int() on the id tensor, which on CUDA blocks until
    # the queue drains. It must therefore run on the first forward and then stop.
    model = tiny_model()
    assert model._speech_ids_validated is False
    calls = []
    original = torch.Tensor.min

    def counting_min(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    torch.Tensor.min = counting_min
    try:
        for _ in range(3):
            model(
                text_input_ids=torch.tensor([[2, 3]]),
                speech_input_ids=torch.tensor([[16, 4, 5]]),
                labels=torch.tensor([[4, 5, 17]]),
                **speaker_inputs(),
            )
    finally:
        torch.Tensor.min = original
    assert model._speech_ids_validated is True
    assert len(calls) == 1, f"validated {len(calls)} times, expected once"

    # A bad id in the very first batch is still fatal.
    fresh = tiny_model()
    with pytest.raises(ValueError):
        fresh(
            text_input_ids=torch.tensor([[2, 3]]),
            speech_input_ids=torch.tensor([[16, 4, 999]]),
            labels=torch.tensor([[4, 5, 17]]),
            **speaker_inputs(),
        )


def test_eval_mode_still_returns_logits_for_the_accuracy_metrics():
    # evaluate() reads output.logits, so the fused loss must never be taken in
    # eval mode. On CPU the fused path is off anyway, which is what keeps these
    # tests meaningful without a GPU.
    model = tiny_model()
    model.eval()
    output = model(
        text_input_ids=torch.tensor([[2, 3]]),
        speech_input_ids=torch.tensor([[16, 4, 5]]),
        labels=torch.tensor([[4, 5, 17]]),
        **speaker_inputs(),
    )
    assert output.logits is not None
    assert output.logits.shape == (1, 3, 19)
    assert output.loss is not None


def test_mismatched_label_shape_is_rejected_on_both_loss_paths():
    model = tiny_model()
    for train_mode in (True, False):
        model.train(train_mode)
        with pytest.raises(ValueError):
            model(
                text_input_ids=torch.tensor([[2, 3]]),
                speech_input_ids=torch.tensor([[16, 4, 5]]),
                labels=torch.tensor([[4, 5]]),
                **speaker_inputs(),
            )

import random

import pytest
import torch
from transformers import Qwen3_5TextConfig

from qwen_tts.core.models import Text2SemanticConfig, Text2SemanticForCausalLM
from qwen_tts.text_conditioning import (
    CONDITIONING_SPECIAL_TOKENS,
    EMOTION_END_TOKEN,
    EMOTION_START_TOKEN,
    LANGUAGE_TOKENS,
    TextConditioner,
    add_conditioning_tokens,
    augment_description,
    condition_inference_text,
    drop_pause_markers,
    render_closed_markers,
    resize_text_embeddings,
    validate_conditioning_tokens,
)


TABLE = {
    "version": 3,
    "label_7": {},
    "events": {
        "laughter": {
            "forms": {
                "en": ["laughter", "a light laugh"],
                "ja": ["笑い声", "軽く笑う"],
            }
        }
    },
    "spans": {
        "平静": {
            "match": {"zh": ["平静"], "en": ["calm"]},
            "forms": {"zh": ["淡然"], "ja": ["穏やか"], "en": ["composed"]},
        },
        "好奇": {
            "match": {"zh": ["好奇"]},
            "forms": {"zh": ["想弄明白"], "ja": ["興味深い"]},
        },
    },
    "templates": {},
    "full_description_overrides": {
        "en": {"calm": ["unruffled"]},
    },
}


def test_conditioner_adds_atomic_language_and_fish_event_without_speaking_bracket():
    item = {
        "id": "fish-1",
        "text": "[laughing] Hello  [laughing] world.",
        "language": "en",
        "emotion": {"tags": ["laughing"], "events": []},
    }
    conditioner = TextConditioner(
        language_tag_prob=1.0,
        emotion_conditioning=True,
        emotion_synonym_prob=0.0,
        synonym_table=TABLE,
        deterministic=True,
    )
    assert conditioner(item) == (
        "<|lang_en|><|emo_start|>laughter<|emo_end|>Hello"
        "<|emo_start|>laughter<|emo_end|>world."
    )


def test_fish_surface_tag_uses_canonical_event_synonyms():
    item = {
        "id": "fish-2",
        "text": "I tried. [stifled laugh] It did not work.",
        "language": "en",
        "emotion": {"tags": ["stifled laugh"]},
    }
    conditioner = TextConditioner(
        emotion_conditioning=True,
        emotion_synonym_prob=1.0,
        synonym_table=TABLE,
        deterministic=True,
    )
    value = conditioner(item)
    assert value == (
        "I tried.<|emo_start|>a light laugh<|emo_end|>It did not work."
    )
    assert "[stifled laugh]" not in value


def test_multiple_fish_tags_keep_their_original_order_and_positions():
    item = {
        "id": "fish-3",
        "text": "A. [giggle] B. [sighing] C.",
        "language": "en",
        "emotion": {"tags": ["giggle", "sighing"], "events": []},
    }
    conditioner = TextConditioner(
        emotion_conditioning=True,
        emotion_synonym_prob=0.0,
        synonym_table=TABLE,
    )
    assert conditioner(item) == (
        "A.<|emo_start|>laughter<|emo_end|>B."
        "<|emo_start|>sighing<|emo_end|>C."
    )


def test_description_longest_match_respects_replacement_cap_and_target_language():
    value = augment_description(
        "平静而好奇又平静",
        "ja",
        TABLE,
        random.Random(0),
        replace_prob=1.0,
        max_replacements=2,
    )
    assert value == "穏やか而興味深い又平静"


def test_deterministic_eval_conditioning_is_stable_by_row_id():
    conditioner = TextConditioner(
        language_tag_prob=0.6,
        emotion_conditioning=True,
        emotion_synonym_prob=0.0,
        synonym_table=TABLE,
        deterministic=True,
        seed=7,
    )
    item = {
        "id": "game-1",
        "text": "台词",
        "language": "ja",
        "emotion": {"description_zh": "平静而好奇", "events": []},
    }
    assert conditioner(item) == conditioner(item)


def test_inference_controls_are_explicit_and_validated():
    assert condition_inference_text("hello", language="en", emotion="calm") == (
        "<|lang_en|><|emo_start|>calm<|emo_end|>hello"
    )
    with pytest.raises(ValueError, match="language must be one of"):
        condition_inference_text("hello", language="xx")
    with pytest.raises(ValueError, match="emotion must be non-empty"):
        condition_inference_text("hello", emotion="  ")


def test_inference_converts_inline_brackets_and_auto_language_is_noop():
    assert condition_inference_text(
        "你好，[轻轻叹气]再试一次。[breathing]",
        language="auto",
    ) == (
        "你好，<|emo_start|>轻轻叹气<|emo_end|>再试一次。"
        "<|emo_start|>breathing<|emo_end|>"
    )
    assert condition_inference_text("[ 平静 ]hello", language="zh") == (
        "<|lang_zh|><|emo_start|>平静<|emo_end|>hello"
    )
    assert condition_inference_text("literal [] and [ ] stay") == (
        "literal [] and [ ] stay"
    )


def test_inline_emotion_stays_at_its_original_position():
    assert condition_inference_text(
        "The weather is lovely today. [sighing]Shall we take a walk?"
    ) == (
        "The weather is lovely today."
        "<|emo_start|>sighing<|emo_end|>Shall we take a walk?"
    )


def test_inference_strips_tag_spaces_and_merges_adjacent_brackets():
    assert condition_inference_text("[ 平静 ][ sigh ]hello") == (
        "<|emo_start|>平静, sigh<|emo_end|>hello"
    )
    assert condition_inference_text("A. [calm] [pause] B.") == (
        "A.<|emo_start|>calm, pause<|emo_end|>B."
    )


def test_closed_markers_merge_whitespace_neighbors_and_keep_alts():
    text = "I came to discuss it but it is not the right time yet."
    annotations = [
        {
            "type": "emotion",
            "label": "serious",
            "insert_char_index": 0,
            "confidence": 0.52,
            "alternatives": [
                {"label": "intense", "confidence": 0.28},
                {"label": "dramatic", "confidence": 0.16},
            ],
        },
        {"type": "event", "label": "pause", "insert_char_index": 20, "confidence": 0.9, "alternatives": []},
        {
            "type": "emotion",
            "label": "disappointed",
            "insert_char_index": 21,
            "confidence": 0.51,
            "alternatives": [
                {"label": "serious", "confidence": 0.32},
                {"label": "calm", "confidence": 0.12},
            ],
        },
    ]
    kept = render_closed_markers(text, annotations, drop_leading=False)
    assert kept.startswith("<|emo_start|>serious<|emo_end|>")
    assert "<|emo_start|>pause, disappointed, serious<|emo_end|>" in kept
    dropped = render_closed_markers(text, annotations, drop_leading=True)
    assert not dropped.startswith("<|emo_start|>serious<|emo_end|>")
    assert "<|emo_start|>pause, disappointed, serious<|emo_end|>" in dropped


def test_conditioner_uses_annotations_and_can_drop_leading():
    item = {
        "id": "sr-1",
        "text": "Hello there.",
        "language": "en",
        "annotations": [
            {"type": "emotion", "label": "calm", "insert_char_index": 0, "alternatives": []},
            {
                "type": "event",
                "label": "sigh",
                "insert_char_index": 0,
                "alternatives": [{"label": "chuckles", "confidence": 0.35}],
            },
            {"type": "emotion", "label": "gentle", "insert_char_index": 6, "alternatives": []},
        ],
    }
    keep = TextConditioner(
        emotion_conditioning=True,
        drop_leading_tag_prob=0.0,
        deterministic=True,
    )
    assert keep(item) == (
        "<|emo_start|>calm, sigh, chuckles<|emo_end|>Hello"
        "<|emo_start|>gentle<|emo_end|>there."
    )
    drop = TextConditioner(
        emotion_conditioning=True,
        drop_leading_tag_prob=1.0,
        deterministic=True,
    )
    assert drop(item) == "Hello<|emo_start|>gentle<|emo_end|>there."


def test_leading_tag_stays_when_only_later_tags_are_pause():
    text = "Hello there."
    only_pause = [
        {"type": "emotion", "label": "calm", "insert_char_index": 0, "alternatives": []},
        {"type": "event", "label": "pause", "insert_char_index": 5, "alternatives": []},
    ]
    assert render_closed_markers(text, only_pause, drop_leading=True).startswith(
        "<|emo_start|>calm<|emo_end|>"
    )
    only_lead = [
        {"type": "emotion", "label": "calm", "insert_char_index": 0, "alternatives": []},
        {"type": "event", "label": "sigh", "insert_char_index": 0, "alternatives": []},
    ]
    assert render_closed_markers(text, only_lead, drop_leading=True).startswith(
        "<|emo_start|>calm, sigh<|emo_end|>"
    )


def test_pause_dropout_has_three_distinct_modes():
    annotations = [
        {"type": "emotion", "label": "calm", "insert_char_index": 0},
        {"type": "event", "label": "pause", "insert_char_index": 4},
        {"type": "event", "label": "pause", "insert_char_index": 10},
        {"type": "emotion", "label": "sad", "insert_char_index": 11},
    ]
    none = drop_pause_markers(
        annotations, random.Random(0), drop_all_prob=1.0, drop_partial_prob=0.0
    )
    assert [item["label"] for item in none] == ["calm", "sad"]
    kept = drop_pause_markers(
        annotations, random.Random(0), drop_all_prob=0.0, drop_partial_prob=0.0
    )
    assert sum(item["label"] == "pause" for item in kept) == 2
    partial = drop_pause_markers(
        annotations, random.Random(1), drop_all_prob=0.0, drop_partial_prob=1.0
    )
    n_pause = sum(item["label"] == "pause" for item in partial)
    assert 1 <= n_pause <= 2
    single = [
        {"type": "emotion", "label": "calm", "insert_char_index": 0},
        {"type": "event", "label": "pause", "insert_char_index": 3},
    ]
    assert drop_pause_markers(
        single, random.Random(0), drop_all_prob=0.0, drop_partial_prob=1.0
    ) == single


class TinyTokenizer:
    def __init__(self, size=32):
        self.tokens = {f"tok-{i}": i for i in range(size)}
        self.inverse = {value: key for key, value in self.tokens.items()}

    def __len__(self):
        return len(self.tokens)

    def add_special_tokens(self, payload):
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.tokens:
                index = len(self.tokens)
                self.tokens[token] = index
                self.inverse[index] = token
                added += 1
        return added

    def __call__(self, text, add_special_tokens=False):
        if text in self.tokens:
            return {"input_ids": [self.tokens[text]]}
        return {"input_ids": [0, 1]}

    def convert_ids_to_tokens(self, token_id):
        return self.inverse[token_id]


def tiny_model(vocab_size=32):
    qwen = Qwen3_5TextConfig(
        vocab_size=vocab_size,
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
        qwen_config=qwen.to_dict(),
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


def test_special_tokens_resize_text_embedding_and_nested_saved_config():
    tokenizer = TinyTokenizer()
    model = tiny_model()
    assert add_conditioning_tokens(tokenizer) == len(CONDITIONING_SPECIAL_TOKENS)
    validate_conditioning_tokens(tokenizer)
    delta = resize_text_embeddings(model, tokenizer)
    assert delta == len(CONDITIONING_SPECIAL_TOKENS)
    assert model.get_input_embeddings().num_embeddings == len(tokenizer)
    assert model.config.qwen_config["vocab_size"] == len(tokenizer)
    for token in (*LANGUAGE_TOKENS.values(), EMOTION_START_TOKEN, EMOTION_END_TOKEN):
        assert len(tokenizer(token, add_special_tokens=False)["input_ids"]) == 1

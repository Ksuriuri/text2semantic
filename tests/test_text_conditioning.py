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
        "<|lang_en|><|emo_start|>laughter<|emo_end|> Hello  "
        "<|emo_start|>laughter<|emo_end|> world."
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
        "I tried. <|emo_start|>a light laugh<|emo_end|> It did not work."
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
        "A. <|emo_start|>laughter<|emo_end|> B. "
        "<|emo_start|>sighing<|emo_end|> C."
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
        "The weather is lovely today. "
        "<|emo_start|>sighing<|emo_end|>Shall we take a walk?"
    )


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

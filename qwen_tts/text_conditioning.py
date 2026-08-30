# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Language and affect controls for text-to-semantic prompts.

The manifest remains the source of truth: this module only changes the text
that reaches the model. Training can therefore redraw the optional language
tag and affect synonyms every epoch without rewriting annotations, while eval
and inference stay deterministic.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


LANGUAGES = ("ar", "de", "en", "es", "fr", "ja", "ko", "pt", "ru", "zh")
LANGUAGE_TOKENS = {language: f"<|lang_{language}|>" for language in LANGUAGES}
EMOTION_START_TOKEN = "<|emo_start|>"
EMOTION_END_TOKEN = "<|emo_end|>"
INLINE_EMOTION_RE = re.compile(r"\[([^\[\]\r\n]+)\]")
CONDITIONING_SPECIAL_TOKENS = (
    *LANGUAGE_TOKENS.values(),
    EMOTION_START_TOKEN,
    EMOTION_END_TOKEN,
)

# Fish S2.1 writes surface cues rather than the canonical event names used by
# the game annotations and v3 synonym table. Keep this mapping semantic and
# conservative: it only collapses obvious inflections or near-identical vocal
# events, while unknown cues remain available verbatim.
EVENT_ALIASES = {
    "clears_throat": "throat_clearing",
    "cough": "coughing",
    "crying_loudly": "crying",
    "exhale": "breathing",
    "giggle": "laughter",
    "groan": "groaning",
    "huff": "breathing",
    "inhale": "breathing",
    "laugh": "laughter",
    "laughing": "laughter",
    "long_laugh": "laughter",
    "moan": "groaning",
    "moaning": "groaning",
    "pant": "panting",
    "sigh": "sighing",
    "sniff": "sniffing",
    "sniffle": "sniffing",
    "snort": "snorting",
    "sobbing": "crying",
    "stifled_laugh": "laughter",
    "chuckling": "laughter",
}


def add_conditioning_tokens(tokenizer):
    """Register every control marker as one indivisible tokenizer token."""
    return tokenizer.add_special_tokens(
        {"additional_special_tokens": list(CONDITIONING_SPECIAL_TOKENS)}
    )


def validate_conditioning_tokens(tokenizer):
    """Refuse inference when a checkpoint tokenizer lacks the control tokens."""
    broken = []
    for token in CONDITIONING_SPECIAL_TOKENS:
        ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != token:
            broken.append(token)
    if broken:
        raise ValueError(
            "Checkpoint tokenizer does not contain the conditioning special "
            f"tokens: {', '.join(broken)}"
        )


def resize_text_embeddings(model, tokenizer):
    """Resize the Qwen text embedding and persist its new nested config size."""
    wanted = len(tokenizer)
    current = model.get_input_embeddings().num_embeddings
    if current == wanted:
        return 0
    model.backbone.resize_token_embeddings(wanted, mean_resizing=False)
    # Text2SemanticConfig embeds Qwen's config rather than exposing its vocab
    # size at the top level. Updating only the live backbone would save weights
    # that a later from_pretrained() reconstructs at the old, smaller size.
    model.config.qwen_config = model.backbone.config.to_dict()
    return wanted - current


def load_synonym_table(path):
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 3:
        raise ValueError("emotion synonym table must be version 3")
    for field in ("events", "spans", "templates", "full_description_overrides"):
        if not isinstance(payload.get(field), dict):
            raise ValueError(f"emotion synonym table is missing {field}")
    return payload


def _forms(record, language, *, fallback="zh"):
    forms = (record or {}).get("forms") or {}
    values = forms.get(language) or forms.get(fallback) or forms.get("en") or []
    return [str(value).strip() for value in values if str(value).strip()]


def _event_surface(event, language, table, rng, replace_prob):
    key = str(event).strip().lower().replace(" ", "_")
    key = EVENT_ALIASES.get(key, key)
    record = (table or {}).get("events", {}).get(key)
    forms = _forms(record, language, fallback="en")
    if not forms:
        return key.replace("_", " ")
    if len(forms) > 1 and rng.random() < replace_prob:
        return rng.choice(forms[1:])
    return forms[0]


def augment_description(
    text,
    language,
    table,
    rng,
    *,
    replace_prob=0.7,
    max_replacements=2,
):
    """Apply a whole-string override or longest-match span replacements."""
    if not text or not table or replace_prob <= 0.0 or max_replacements <= 0:
        return text

    overrides = table.get("full_description_overrides", {}).get(language, {})
    choices = overrides.get(text) or []
    if choices and rng.random() < replace_prob:
        return rng.choice(choices)

    candidates = []
    for group in ("spans", "templates"):
        for record in table.get(group, {}).values():
            replacements = _forms(record, language)
            if not replacements:
                continue
            for variants in (record.get("match") or {}).values():
                for match in variants:
                    match = str(match)
                    if match:
                        candidates.append((match, replacements))
    candidates.sort(key=lambda value: len(value[0]), reverse=True)

    out = []
    position = 0
    replaced = 0
    while position < len(text):
        found = None
        for match, replacements in candidates:
            if text.startswith(match, position):
                found = (match, replacements)
                break
        if found is None:
            out.append(text[position])
            position += 1
            continue
        match, replacements = found
        if replaced < max_replacements and rng.random() < replace_prob:
            out.append(rng.choice(replacements))
            replaced += 1
        else:
            out.append(match)
        position += len(match)
    return "".join(out)


def _strip_fish_tags(text, tags):
    cleaned = text
    for tag in tags:
        pattern = re.compile(r"\[\s*" + re.escape(str(tag)) + r"\s*\]", re.I)
        cleaned = pattern.sub(" ", cleaned)
    # Removing a tag in the middle can leave doubled spaces. Do not otherwise
    # normalise punctuation or line layout; those are the transcript itself.
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _condition_fish_tags(text, tags, language, table, rng, replace_prob):
    """Replace Fish surface cues with control spans at their true positions."""
    conditioned = text
    for tag in tags:
        pattern = re.compile(r"\[\s*" + re.escape(str(tag)) + r"\s*\]", re.I)

        def replace(_match, value=tag):
            surface = _event_surface(
                value,
                language,
                table,
                rng,
                replace_prob,
            )
            return f"{EMOTION_START_TOKEN}{surface}{EMOTION_END_TOKEN}"

        conditioned = pattern.sub(replace, conditioned)
    return conditioned


def emotion_text(item, table, rng, *, synonym_prob=0.7, max_replacements=2):
    emotion = item.get("emotion") or {}
    if not isinstance(emotion, dict):
        return ""
    language = str(item.get("language") or "zh").lower()
    tags = emotion.get("tags") or []
    events = list(emotion.get("events") or [])
    if tags:
        events.extend(tag for tag in tags if tag not in events)

    description = emotion.get(f"description_{language}")
    if not description:
        description = emotion.get("description_zh") or emotion.get("description_en")
    description = str(description or "").strip()
    if description:
        description = augment_description(
            description,
            language,
            table,
            rng,
            replace_prob=synonym_prob,
            max_replacements=max_replacements,
        )
    elif emotion.get("label_7"):
        label = str(emotion["label_7"]).strip().lower()
        forms = _forms((table or {}).get("label_7", {}).get(label), language)
        description = forms[0] if forms else label

    event_texts = [
        _event_surface(event, language, table, rng, synonym_prob)
        for event in events
        if str(event).strip()
    ]
    parts = [part for part in (description, "; ".join(event_texts)) if part]
    return "; ".join(parts)


class TextConditioner:
    def __init__(
        self,
        *,
        language_tag_prob=0.0,
        emotion_conditioning=False,
        emotion_synonym_prob=0.0,
        emotion_max_replacements=2,
        synonym_table=None,
        deterministic=False,
        seed=42,
    ):
        for name, value in (
            ("language_tag_prob", language_tag_prob),
            ("emotion_synonym_prob", emotion_synonym_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.language_tag_prob = language_tag_prob
        self.emotion_conditioning = emotion_conditioning
        self.emotion_synonym_prob = emotion_synonym_prob
        self.emotion_max_replacements = emotion_max_replacements
        self.synonym_table = synonym_table
        self.deterministic = deterministic
        self.seed = seed

    def _rng(self, item, index):
        if not self.deterministic:
            return random
        uid = item.get("id", index)
        return random.Random(f"text-conditioning:{self.seed}:{uid}")

    def __call__(self, item, index=None):
        rng = self._rng(item, index)
        language = str(item.get("language") or "").lower()
        text = item["text"]
        emotion = item.get("emotion") or {}
        tags = emotion.get("tags") if isinstance(emotion, dict) else None
        if tags:
            if self.emotion_conditioning:
                text = _condition_fish_tags(
                    text,
                    tags,
                    language,
                    self.synonym_table,
                    rng,
                    self.emotion_synonym_prob,
                )
            else:
                text = _strip_fish_tags(text, tags)

        prefix = ""
        if self.language_tag_prob > 0.0 and rng.random() < self.language_tag_prob:
            token = LANGUAGE_TOKENS.get(language)
            if token is None:
                raise ValueError(f"unsupported language for conditioning: {language!r}")
            prefix += token
        if self.emotion_conditioning:
            prefix_item = item
            if tags:
                prefix_emotion = dict(emotion)
                prefix_emotion["tags"] = []
                prefix_item = dict(item)
                prefix_item["emotion"] = prefix_emotion
            value = emotion_text(
                prefix_item,
                self.synonym_table,
                rng,
                synonym_prob=self.emotion_synonym_prob,
                max_replacements=self.emotion_max_replacements,
            )
            if value:
                prefix += f"{EMOTION_START_TOKEN}{value}{EMOTION_END_TOKEN}"
        return prefix + text


def _replace_inline_emotions(text):
    """Turn ``[affect]`` cues into model control spans in place."""

    def replace(match):
        value = match.group(1).strip()
        if not value:
            return match.group(0)
        return f"{EMOTION_START_TOKEN}{value}{EMOTION_END_TOKEN}"

    return INLINE_EMOTION_RE.sub(replace, text)


def condition_inference_text(text, *, language=None, emotion=None):
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    text = _replace_inline_emotions(text)
    prefix = ""
    if language is not None:
        language = str(language).strip().lower()
        if language and language != "auto" and language not in LANGUAGE_TOKENS:
            raise ValueError(
                f"language must be one of {', '.join(LANGUAGES)}, got {language!r}"
            )
        if language and language != "auto":
            prefix += LANGUAGE_TOKENS[language]
    if emotion is not None:
        emotion = str(emotion).strip()
        if not emotion:
            raise ValueError("emotion must be non-empty when provided")
        prefix += f"{EMOTION_START_TOKEN}{emotion}{EMOTION_END_TOKEN}"
    return prefix + text

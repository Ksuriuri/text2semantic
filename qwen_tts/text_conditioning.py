# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Language and affect controls for text-to-semantic prompts.

The manifest remains the source of truth: this module only changes the text
that reaches the model. Training can therefore redraw the optional language
tag and marker order every epoch without rewriting annotations, while eval
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
INLINE_EMOTION_RE = re.compile(r"\[([^\[\]]*)\]", re.DOTALL)
ALT_MIN_CONFIDENCE = 0.3
DROP_LEADING_TAG_PROB = 0.0
PAUSE_DROP_ALL_PROB = 0.25
PAUSE_DROP_PARTIAL_PROB = 0.35
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


def _dedupe_labels(labels):
    seen = set()
    out = []
    for label in labels:
        value = str(label or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def marker_labels(marker, alt_min_confidence=ALT_MIN_CONFIDENCE):
    """Primary label plus alternatives at or above the confidence floor."""
    labels = [str(marker.get("label") or "").strip()]
    for alt in marker.get("alternatives") or []:
        if not isinstance(alt, dict):
            continue
        if float(alt.get("confidence") or 0.0) < alt_min_confidence:
            continue
        labels.append(str(alt.get("label") or "").strip())
    return _dedupe_labels(labels)


def _order_same_index_markers(markers, rng):
    """Order markers that share one character index.

    Training passes an rng so emotion vs event is not fixed. Callers without
    an rng (tests, offline render) keep emotion before event.
    """
    items = list(markers)
    if len(items) <= 1:
        return items
    if rng is not None:
        rng.shuffle(items)
        return items
    type_rank = {"emotion": 0, "event": 1}
    items.sort(key=lambda marker: type_rank.get(str(marker.get("type") or ""), 2))
    return items


def group_adjacent_markers(annotations, transcript, rng=None):
    """Cluster markers that share an index or have only whitespace between them."""
    buckets = {}
    for marker in annotations or []:
        if not str(marker.get("label") or "").strip():
            continue
        index = int(marker.get("insert_char_index") or 0)
        buckets.setdefault(index, []).append(marker)
    items = []
    for index in sorted(buckets):
        items.extend(_order_same_index_markers(buckets[index], rng))
    groups = []
    for marker in items:
        index = int(marker.get("insert_char_index") or 0)
        if not groups:
            groups.append({"index": index, "markers": [marker]})
            continue
        previous = groups[-1]["index"]
        between = transcript[previous:index] if 0 <= previous <= index <= len(transcript) else "x"
        if between == "" or between.isspace():
            groups[-1]["markers"].append(marker)
        else:
            groups.append({"index": index, "markers": [marker]})
    return groups


def format_emotion_span(labels):
    return f"{EMOTION_START_TOKEN}{', '.join(labels)}{EMOTION_END_TOKEN}"


def merge_adjacent_emotion_spans(text):
    """Join control spans that have no non-space text between them."""
    text = re.sub(
        rf"({re.escape(EMOTION_END_TOKEN)})[ \t]*({re.escape(EMOTION_START_TOKEN)})",
        r"\1\2",
        text,
    )
    pattern = re.compile(
        rf"(?:{re.escape(EMOTION_START_TOKEN)}.*?{re.escape(EMOTION_END_TOKEN)}){{2,}}"
    )

    def replace(match):
        inners = re.findall(
            re.escape(EMOTION_START_TOKEN) + r"(.*?)" + re.escape(EMOTION_END_TOKEN),
            match.group(0),
        )
        labels = []
        for inner in inners:
            labels.extend(part.strip() for part in inner.split(","))
        return format_emotion_span(_dedupe_labels(labels))

    return pattern.sub(replace, text)


def strip_spaces_around_emotion_spans(text):
    return re.sub(
        rf"[ \t]*({re.escape(EMOTION_START_TOKEN)}.*?{re.escape(EMOTION_END_TOKEN)})[ \t]*",
        r"\1",
        text,
    )


def normalize_emotion_spans(text):
    return strip_spaces_around_emotion_spans(merge_adjacent_emotion_spans(text))


def normalize_bracket_tags(text):
    """Strip spaces inside/around real ``[tag]`` cues and merge neighbors.

    Empty ``[]`` / ``[ ]`` stay literal so they are not treated as labels.
    """

    def normalize_inner(match):
        value = match.group(1).strip()
        return f"[{value}]" if value else match.group(0)

    text = re.sub(r"\[([^\[\]\r\n]*)\]", normalize_inner, text)
    nonempty = r"\[[^\[\]\s][^\[\]]*\]"
    text = re.sub(rf"[ \t]+({nonempty})", r"\1", text)
    text = re.sub(rf"({nonempty})[ \t]+", r"\1", text)

    def merge(match):
        labels = []
        for inner in re.findall(r"\[([^\[\]]+)\]", match.group(0)):
            labels.extend(part.strip() for part in inner.split(","))
        return "[" + ", ".join(_dedupe_labels(labels)) + "]"

    return re.sub(rf"(?:{nonempty}){{2,}}", merge, text)


def drop_pause_markers(
    annotations,
    rng,
    *,
    drop_all_prob=PAUSE_DROP_ALL_PROB,
    drop_partial_prob=PAUSE_DROP_PARTIAL_PROB,
    partial_rate=0.5,
):
    """Three-way pause dropout: drop all, drop a subset, or keep all.

    Partial mode keeps at least one pause when the sentence has any, so it
    stays distinct from the all-drop mode on single-pause rows.
    """
    if not 0.0 <= drop_all_prob + drop_partial_prob <= 1.0:
        raise ValueError("pause drop probabilities must sum to at most 1")
    items = list(annotations or [])
    draw = rng.random()
    if draw < drop_all_prob:
        return [item for item in items if str(item.get("label") or "") != "pause"]
    if draw < drop_all_prob + drop_partial_prob:
        pauses = [item for item in items if str(item.get("label") or "") == "pause"]
        others = [item for item in items if str(item.get("label") or "") != "pause"]
        if len(pauses) <= 1:
            return items
        kept = [item for item in pauses if rng.random() >= partial_rate]
        if not kept:
            kept = [rng.choice(pauses)]
        kept_ids = {id(item) for item in kept}
        return others + [item for item in pauses if id(item) in kept_ids]
    return items


def has_inline_non_pause(groups, alt_min_confidence=ALT_MIN_CONFIDENCE):
    """True when a later group still has a real emotion/event, not only pause."""
    for group in groups[1:]:
        for marker in group["markers"]:
            for label in marker_labels(marker, alt_min_confidence):
                if label != "pause":
                    return True
    return False


def render_closed_markers(
    transcript,
    annotations,
    *,
    alt_min_confidence=ALT_MIN_CONFIDENCE,
    drop_leading=False,
    rng=None,
):
    """Insert closed-vocab spans at their character indices."""
    text = transcript if isinstance(transcript, str) else ""
    groups = group_adjacent_markers(annotations, text, rng=rng)
    if (
        drop_leading
        and groups
        and groups[0]["index"] == 0
        and has_inline_non_pause(groups, alt_min_confidence)
    ):
        groups = groups[1:]
    for group in reversed(groups):
        labels = []
        for marker in group["markers"]:
            labels.extend(marker_labels(marker, alt_min_confidence))
        labels = _dedupe_labels(labels)
        if not labels:
            continue
        index = max(0, min(group["index"], len(text)))
        text = text[:index] + format_emotion_span(labels) + text[index:]
    return normalize_emotion_spans(text)


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
        drop_leading_tag_prob=0.0,
        alt_min_confidence=ALT_MIN_CONFIDENCE,
        pause_drop_all_prob=0.0,
        pause_drop_partial_prob=0.0,
    ):
        for name, value in (
            ("language_tag_prob", language_tag_prob),
            ("emotion_synonym_prob", emotion_synonym_prob),
            ("drop_leading_tag_prob", drop_leading_tag_prob),
            ("alt_min_confidence", alt_min_confidence),
            ("pause_drop_all_prob", pause_drop_all_prob),
            ("pause_drop_partial_prob", pause_drop_partial_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if pause_drop_all_prob + pause_drop_partial_prob > 1.0:
            raise ValueError("pause drop probabilities must sum to at most 1")
        self.language_tag_prob = language_tag_prob
        self.emotion_conditioning = emotion_conditioning
        self.emotion_synonym_prob = 0.0  # Legacy argument retained; augmentation disabled.
        self.emotion_max_replacements = emotion_max_replacements
        self.synonym_table = synonym_table
        self.deterministic = deterministic
        self.seed = seed
        self.drop_leading_tag_prob = drop_leading_tag_prob
        self.alt_min_confidence = alt_min_confidence
        self.pause_drop_all_prob = pause_drop_all_prob
        self.pause_drop_partial_prob = pause_drop_partial_prob

    def _rng(self, item, index):
        if not self.deterministic:
            return random
        uid = item.get("id", index)
        return random.Random(f"text-conditioning:{self.seed}:{uid}")

    def __call__(self, item, index=None):
        rng = self._rng(item, index)
        language = str(item.get("language") or "").lower()
        text = item["text"]
        annotations = item.get("annotations")
        emotion = item.get("emotion") or {}
        tags = emotion.get("tags") if isinstance(emotion, dict) else None
        used_closed_markers = False
        if self.emotion_conditioning and isinstance(annotations, list) and annotations:
            transcript = item.get("official_transcript")
            if not isinstance(transcript, str) or not transcript:
                transcript = text
            annotations = drop_pause_markers(
                annotations,
                rng,
                drop_all_prob=self.pause_drop_all_prob,
                drop_partial_prob=self.pause_drop_partial_prob,
            )
            drop_leading = rng.random() < self.drop_leading_tag_prob
            text = render_closed_markers(
                transcript,
                annotations,
                alt_min_confidence=self.alt_min_confidence,
                drop_leading=drop_leading,
                rng=rng,
            )
            used_closed_markers = True
        elif tags:
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
        if self.emotion_conditioning and not used_closed_markers:
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
        text = condition_inline_spans(
            text,
            rng=None if self.deterministic else rng,
            pause_drop_all_prob=(
                self.pause_drop_all_prob
                if self.emotion_conditioning and not used_closed_markers else 0.0
            ),
            pause_drop_partial_prob=(
                self.pause_drop_partial_prob
                if self.emotion_conditioning and not used_closed_markers else 0.0
            ),
        )
        return prefix + text


def _replace_inline_emotions(text):
    """Turn ``[affect]`` cues into model control spans in place."""

    def replace(match):
        value = match.group(1).strip()
        return f"{EMOTION_START_TOKEN}{value}{EMOTION_END_TOKEN}"

    return INLINE_EMOTION_RE.sub(replace, text)


def condition_inline_spans(
    text, *, rng=None, pause_drop_all_prob=0.0, pause_drop_partial_prob=0.0
):
    """Convert every bracket pair; treat arbitrary contents as opaque labels."""
    text = _replace_inline_emotions(text)
    span = re.escape(EMOTION_START_TOKEN) + r"(.*?)" + re.escape(EMOTION_END_TOKEN)
    matches = list(re.finditer(span, text, re.DOTALL))
    drop_positions = set()
    if rng is not None and (pause_drop_all_prob or pause_drop_partial_prob):
        pauses = [
            {"label": m.group(1).strip(), "position": i}
            for i, m in enumerate(matches)
        ]
        kept = drop_pause_markers(
            pauses, rng, drop_all_prob=pause_drop_all_prob,
            drop_partial_prob=pause_drop_partial_prob,
        )
        drop_positions = set(range(len(matches))) - {m["position"] for m in kept}
    for i in sorted(drop_positions, reverse=True):
        m = matches[i]
        text = text[:m.start()] + text[m.end():]
    cluster = re.compile(r"(?:" + re.escape(EMOTION_START_TOKEN) + r".*?" + re.escape(EMOTION_END_TOKEN) + r"[ \t]*)+", re.DOTALL)
    def merge(match):
        labels = re.findall(span, match.group(0), re.DOTALL)
        labels = list(dict.fromkeys(label.strip() for label in labels))
        if rng is not None and len(labels) > 1:
            rng.shuffle(labels)
        return format_emotion_span(labels)
    text = cluster.sub(merge, text)
    return re.sub(r"[ \t]*(" + re.escape(EMOTION_START_TOKEN) + r".*?" + re.escape(EMOTION_END_TOKEN) + r")[ \t]*", r"\1", text, flags=re.DOTALL)


def condition_inference_text(text, *, language=None, emotion=None):
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    text = condition_inline_spans(text)
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

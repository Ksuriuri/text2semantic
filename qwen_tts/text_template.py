# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Fish Speech-style ChatML prompts for text-to-semantic generation."""

FISH_SPEECH_SYSTEM_PROMPT = "Speak out the provided text."


def build_tts_messages(text):
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string.")
    return [
        {"role": "system", "content": FISH_SPEECH_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def build_tts_prompt(text):
    build_tts_messages(text)
    return (
        f"<|im_start|>system\n{FISH_SPEECH_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def tokenize_tts_prompt(tokenizer, text):
    """Tokenize explicit ChatML without Qwen's automatic thinking block."""
    prompt = build_tts_prompt(text)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("Tokenized TTS prompt must not be empty.")
    return ids

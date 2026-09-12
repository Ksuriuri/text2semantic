# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Training-time text augmentation for text-to-semantic.

Users type prompts with no punctuation at all, and the model then has to pick
the pauses itself.  Training only on fully punctuated transcripts makes that a
distribution shift, so a fraction of the training samples has every written
pause cue removed.

"Pause cue" is narrower than "punctuation": a mark that is read out as a word,
or that lives inside a word, is part of *what* is spoken rather than of how it
is paced, and dropping it would teach the model to mispronounce content.
"""

import re

# Only explicit pause punctuation is eligible. Hyphens, brackets, quotes,
# colons, slashes, symbols and whitespace are not pause marks.
_PAUSE_MARKS = frozenset("，、。！？,.!?—…")
_CONTROL_SPAN_RE = re.compile(r"\[[^\[\]]*\]|<\|emo_start\|>.*?<\|emo_end\|>", re.DOTALL)


def _is_pause_mark(text, index, char):
    # Decimal separators are spoken content, not a prosodic pause.
    decimal = (
        char == "." and 0 < index < len(text) - 1
        and text[index - 1].isdigit() and text[index + 1].isdigit()
    )
    return char in _PAUSE_MARKS and not decimal


def strip_pause_marks(text, *, keep_word_spaces=False):
    """Remove only explicit pause marks outside complete control spans.

    Keep whitespace, hyphens and other punctuation unchanged. The legacy
    keep_word_spaces argument is accepted for checkpoint/CLI compatibility.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    def strip(part):
        return "".join(c for i, c in enumerate(part) if not _is_pause_mark(part, i, c))
    parts = []
    cursor = 0
    for match in _CONTROL_SPAN_RE.finditer(text):
        parts.extend((strip(text[cursor:match.start()]), match.group(0)))
        cursor = match.end()
    parts.append(strip(text[cursor:]))
    return "".join(parts)

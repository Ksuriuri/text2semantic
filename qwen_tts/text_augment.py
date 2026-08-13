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

import unicodedata

# Always removed.  The wave dashes act as prosody marks in CJK text even though
# Unicode files them as math symbols.
_EXTRA_MARKS = frozenset("~～")

# Never removed, although Unicode calls them punctuation (Po/Pc): each of these
# is spoken as a word or belongs to an identifier.  "50% off" must not become
# "50 off", and "R&B" must not become "RB".
_KEPT_MARKS = frozenset("%@&#/_")

# Candidate categories: P* (every punctuation class), Z* (every separator,
# including NBSP and the ideographic space) and Cc (\n, \r, \t, \v, \f).
# Letters, digits and symbols ($ + = < > °) are never touched.
_REMOVED_CATEGORY_STARTS = ("P", "Z")


def _is_ascii_word_char(char):
    return char.isascii() and char.isalnum()


def _inside_ascii_word(text, index, char):
    """True for an ASCII mark glued between two ASCII alphanumerics.

    That is the orthographic case, not the prosodic one: 3.14, 12:30, don't,
    state-of-the-art, AC/DC.  All three characters have to be ASCII: a CJK comma
    between two hanzi is a pause cue, and so is a fullwidth comma between two
    Latin letters in mixed text ("hello，world").
    """
    if not char.isascii():
        return False
    if index == 0 or index + 1 >= len(text):
        return False
    return _is_ascii_word_char(text[index - 1]) and _is_ascii_word_char(
        text[index + 1]
    )


def _is_pause_mark(text, index, char):
    if char in _EXTRA_MARKS:
        return True
    if char in _KEPT_MARKS:
        return False
    category = unicodedata.category(char)
    if category[0] not in _REMOVED_CATEGORY_STARTS and category != "Cc":
        return False
    return not _inside_ascii_word(text, index, char)


def strip_pause_marks(text, *, keep_word_spaces=False):
    """Return `text` with pause punctuation, spaces and line breaks removed.

    `keep_word_spaces=True` collapses runs of whitespace into a single space
    instead of deleting them, which preserves word boundaries in
    space-separated scripts while still removing all pause punctuation.

    The result can legitimately be empty (a transcript that is nothing but
    punctuation); callers decide what to do with that.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    kept = []
    for index, char in enumerate(text):
        if char.isspace():
            if keep_word_spaces:
                kept.append(" ")
            continue
        if _is_pause_mark(text, index, char):
            continue
        kept.append(char)
    stripped = "".join(kept)
    return " ".join(stripped.split()) if keep_word_spaces else stripped

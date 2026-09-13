#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Inference-time text normalizer for t2s.

Rules accumulate from the #TTS研究:7ef45303 thread. Keep each rule small and
documented; do not silently rewrite eval dumps unless the caller asks.
"""
from __future__ import annotations

import re
import unicodedata

# Tab and other indent whitespace (not ordinary space, not newline).
_TAB_LIKE = {
    ord("\t"),
    ord("\v"),
    ord("\f"),
    ord("\u00a0"),
    ord("\u1680"),
    ord("\u2000"),
    ord("\u2001"),
    ord("\u2002"),
    ord("\u2003"),
    ord("\u2004"),
    ord("\u2005"),
    ord("\u2006"),
    ord("\u2007"),
    ord("\u2008"),
    ord("\u2009"),
    ord("\u200a"),
    ord("\u202f"),
    ord("\u205f"),
    ord("\u3000"),
}

_PAUSE_PUNCT = set("，。！？；、：,.!?;:…—～~·")
_CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_SPACE_AFTER_PUNCT = re.compile(r"(?<=[，。！？；、：…])[ \t]+")
_MULTI_SPACE_BETWEEN_CJK = re.compile(
    r"(?<=[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]) {2,}"
    r"(?=[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af])"
)
_MULTI_SPACE = re.compile(r" {2,}")

_SMALL = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
_SCALES = [
    (10**12, "trillion"),
    (10**9, "billion"),
    (10**6, "million"),
    (10**3, "thousand"),
]

_PROTECTED_PATTERNS = (
    # Protect structured syntax before interpreting '-' or '%'.
    re.compile(r"https?://[^\s]+", re.IGNORECASE),
    re.compile(r"www\.[^\s]+", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"(?<!\w)/(?:[A-Za-z0-9._~-]+/)*[A-Za-z0-9._~-]+/?"),
    re.compile(r"\b[A-Za-z]:\\(?:[^\s\\]+\\)*[^\s\\]*"),
    re.compile(r"(?<!\w)--?[A-Za-z][A-Za-z0-9-]*\b"),
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)+(?:[-+][A-Za-z0-9.-]+)?\b", re.IGNORECASE),
    re.compile(r"%[0-9A-Fa-f]{2}"),
    re.compile(r"%(?:[-+#0 ]*\d*(?:\.\d+)?[sdifoxXeEgGc])"),
    re.compile(r"\b\d+\s+%\s+\d+\b"),
    re.compile(r"\b\d+-\d+/\d+\b"),
)

_ZH_DIGITS = "零一二三四五六七八九"


def _is_cjk(ch: str) -> bool:
    return bool(ch) and _CJK_RE.match(ch) is not None


def _is_pause_punct(ch: str) -> bool:
    if not ch:
        return False
    if ch in _PAUSE_PUNCT:
        return True
    return unicodedata.category(ch).startswith("P")


def _normalize_tabs(text: str) -> str:
    """Drop indent tabs; only insert a Chinese comma when a pause is missing.

    After existing punctuation, a tab or expanded indent is layout, not a
    missing mark — replacing it would yield '，，'. Between two CJK clauses
    with no punct, '，' is the right pause. Between Latin words, use a space.
    """
    if not text:
        return text
    text = text.translate({cp: ord("\t") for cp in _TAB_LIKE})
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "\t":
            out.append(ch)
            i += 1
            continue
        j = i + 1
        while j < n and text[j] == "\t":
            j += 1
        left = out[-1] if out else ""
        right = text[j] if j < n else ""
        if _is_pause_punct(left) or _is_pause_punct(right):
            pass
        elif _is_cjk(left) and _is_cjk(right):
            out.append("，")
        elif left and left not in " \n" and right and right not in " \n":
            out.append(" ")
        i = j
    text = "".join(out)
    text = _SPACE_AFTER_PUNCT.sub("", text)
    text = _MULTI_SPACE_BETWEEN_CJK.sub("，", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text


def _cardinal(n: int) -> str:
    if n < 0:
        return "minus " + _cardinal(-n)
    if n < 20:
        return _SMALL[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_SMALL[ones]}" if ones else "")
    if n < 1000:
        h, rest = divmod(n, 100)
        if rest == 0:
            return f"{_SMALL[h]} hundred"
        return f"{_SMALL[h]} hundred {_cardinal(rest)}"
    for value, name in _SCALES:
        if n >= value:
            q, r = divmod(n, value)
            head = f"{_cardinal(q)} {name}"
            return head if r == 0 else f"{head} {_cardinal(r)}"
    return str(n)


def _digits_to_en(frac: str) -> str:
    return " ".join(_SMALL[int(d)] for d in frac)


def _integer_to_zh(n: int) -> str:
    """Return a compact spoken Chinese cardinal for non-negative integers."""
    if n < 10:
        return _ZH_DIGITS[n]
    if n < 100:
        q, r = divmod(n, 10)
        head = "十" if q == 1 else _ZH_DIGITS[q] + "十"
        return head if r == 0 else head + _ZH_DIGITS[r]
    if n < 1000:
        q, r = divmod(n, 100)
        head = _ZH_DIGITS[q] + "百"
        if r == 0:
            return head
        return head + ("零" if r < 10 else "") + _integer_to_zh(r)
    if n < 10000:
        q, r = divmod(n, 1000)
        head = _ZH_DIGITS[q] + "千"
        if r == 0:
            return head
        return head + ("零" if r < 100 else "") + _integer_to_zh(r)
    if n < 10**8:
        q, r = divmod(n, 10000)
        head = _integer_to_zh(q) + "万"
        if r == 0:
            return head
        return head + ("零" if r < 1000 else "") + _integer_to_zh(r)
    q, r = divmod(n, 10**8)
    head = _integer_to_zh(q) + "亿"
    if r == 0:
        return head
    return head + ("零" if r < 10**7 else "") + _integer_to_zh(r)


def _protect_syntax(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Hide URLs/code-like spans behind private-use characters.

    A single private-use character is deliberately used for each span: numeric
    placeholder ids would themselves be consumed by range/number rules.
    """
    protected: list[tuple[str, str]] = []

    def hide(match: re.Match) -> str:
        token = chr(0xE000 + len(protected))
        protected.append((token, match.group(0)))
        return token

    for pattern in _PROTECTED_PATTERNS:
        text = pattern.sub(hide, text)
    return text, protected


def _restore_syntax(text: str, protected: list[tuple[str, str]]) -> str:
    for token, value in reversed(protected):
        text = text.replace(token, value)
    return text



# September 13: numeric spelling belongs to the model. Only explicit binary
# subtraction and standalone percentages remain, alongside whitespace cleanup.
_NUMBER = r"\d+(?:\.\d+)?[%％]?"
_MATH_EXPR_RE = re.compile(
    rf"(?<![A-Za-z0-9.])[-+]?{_NUMBER}(?:\s*[+\-×÷*/=]\s*[-+]?{_NUMBER})+(?![A-Za-z0-9%％]|\.\d)"
)
_PERCENT_RE = re.compile(r"(?<![A-Za-z0-9.%％])(-?\d+(?:\.\d+)?)\s*[%％]")
_NUMERIC_CHAIN_RE = re.compile(
    rf"(?<![A-Za-z0-9.])[-+]?{_NUMBER}(?:\s*[-–—~～至到]\s*[-+]?{_NUMBER})+(?![A-Za-z0-9%％]|\.\d)"
)
_ISO_DATE_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}[-–—](?:0?[1-9]|1[0-2])[-–—](?:0?[1-9]|[12]\d|3[01])(?!\d)")


def _percent_phrase(match: re.Match, chinese: bool) -> str:
    value = match.group(1)
    sign = "-" if value.startswith("-") else ""
    unsigned = value.lstrip("-")
    # Preserve ALL decimal digits and trailing zeros. Never round or truncate.
    if "." in unsigned:
        spoken = unsigned
    elif chinese:
        spoken = "百" if unsigned == "100" else _integer_to_zh(int(unsigned))
    else:
        spoken = _cardinal(int(unsigned))
    return sign + ("百分之" + spoken if chinese else spoken + " percent")


def _normalize_subtraction_percent(text: str) -> str:
    if not text:
        return text
    chinese = bool(_CJK_RE.search(text)) or not bool(re.search(r"[A-Za-z]", text))
    text, protected = _protect_syntax(text)

    def hide(match: re.Match) -> str:
        token = chr(0xE000 + len(protected))
        protected.append((token, match.group(0)))
        return token

    # Dates and ambiguous numeric chains remain byte-for-byte intact. Do not
    # let their minus signs or individual percentage endpoints be rewritten.
    text = _ISO_DATE_RE.sub(hide, text)

    def math_sub(match: re.Match) -> str:
        expr = match.group(0)
        explicit = bool(re.search(r"[=+×÷*/]", expr)) or bool(
            re.search(r"\d[%％]?\s+-\s+\d", expr)
        )
        if not explicit:
            return expr
        # A minus following an operand is binary subtraction; leading minus
        # or minus after another operator is unary and must remain unchanged.
        expr = re.sub(r"(?<=[0-9%％])\s*-\s*(?=[+-]?\d)",
                      "减" if chinese else " minus ", expr)
        expr = _PERCENT_RE.sub(lambda m: _percent_phrase(m, chinese), expr)
        token = chr(0xE000 + len(protected))
        protected.append((token, expr))
        return token

    text = _MATH_EXPR_RE.sub(math_sub, text)
    text = _NUMERIC_CHAIN_RE.sub(hide, text)
    text = _PERCENT_RE.sub(lambda m: _percent_phrase(m, chinese), text)
    return _restore_syntax(text, protected)


def normalize(text: str) -> str:
    if text is None:
        return ""
    return _normalize_subtraction_percent(_normalize_tabs(str(text)))


def describe(before: str, after: str) -> str:
    if before == after:
        return ""
    notes = []
    tabbed = _normalize_tabs(str(before))
    if tabbed != str(before):
        notes.append("tab/indent stripped")
    if _normalize_subtraction_percent(tabbed) != tabbed:
        notes.append("subtraction/percent expanded")
    return "; ".join(notes) or "normalized"

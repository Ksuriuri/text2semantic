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

_BYTE_UNITS = {
    "KB": ("kilobyte", "kilobytes"),
    "MB": ("megabyte", "megabytes"),
    "GB": ("gigabyte", "gigabytes"),
    "TB": ("terabyte", "terabytes"),
    "PB": ("petabyte", "petabytes"),
    "EB": ("exabyte", "exabytes"),
    "KIB": ("kibibyte", "kibibytes"),
    "MIB": ("mebibyte", "mebibytes"),
    "GIB": ("gibibyte", "gibibytes"),
    "TIB": ("tebibyte", "tebibytes"),
    "PIB": ("pebibyte", "pebibytes"),
}
_BIT_RATE = {
    "KBPS": ("kilobit per second", "kilobits per second"),
    "MBPS": ("megabit per second", "megabits per second"),
    "GBPS": ("gigabit per second", "gigabits per second"),
    "TBPS": ("terabit per second", "terabits per second"),
}
_BYTE_RATE = {
    "KB/S": ("kilobyte per second", "kilobytes per second"),
    "MB/S": ("megabyte per second", "megabytes per second"),
    "GB/S": ("gigabyte per second", "gigabytes per second"),
    "TB/S": ("terabyte per second", "terabytes per second"),
    "KIB/S": ("kibibyte per second", "kibibytes per second"),
    "MIB/S": ("mebibyte per second", "mebibytes per second"),
    "GIB/S": ("gibibyte per second", "gibibytes per second"),
}
_HZ = {
    "KHZ": "kilohertz",
    "MHZ": "megahertz",
    "GHZ": "gigahertz",
    "THZ": "terahertz",
}
_MONEY_SCALE = {
    "K": "thousand",
    "M": "million",
    "B": "billion",
    "T": "trillion",
}

# $89M / US$ 1.2B / €50K. Do this before bare units so T is trillion not TB.
_MONEY_SCALE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:US\s*)?([$€£])\s*(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*([KMBT])\b"
)
_MONEY_PLAIN_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:US\s*)?([$€£])\s*(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?(?!\s*[KMBT]|[A-Za-z])"
)
# 16GB, 1.5 TiB. Not 5G / 4K.
_NUM_BYTE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*"
    r"(KiB|MiB|GiB|TiB|PiB|KB|MB|GB|TB|PB|EB)\b"
)
_BYTE_RATE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*"
    r"(KiB|MiB|GiB|KB|MB|GB|TB)/s\b"
)
_BIT_RATE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*"
    r"(kbps|Mbps|Gbps|Tbps)\b"
)
_HZ_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*"
    r"(kHz|MHz|GHz|THz)\b"
)
# 89M → million, not MB / Mbps / MHz.
_NUM_MILLION_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*M\b(?![BbHhp])"
)
_BARE_BYTE_RE = re.compile(r"(?<![A-Za-z0-9])(KiB|MiB|GiB|TiB|PiB|KB|MB|GB|TB|PB|EB)\b")


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


def _number_to_en(whole: str, frac: str | None) -> str:
    n = int(whole.replace(",", "") or "0")
    spoken = _cardinal(n)
    if frac:
        frac = frac.rstrip("0")
        if frac:
            spoken = f"{spoken} point {_digits_to_en(frac)}"
    return spoken


def _plural_unit(count_whole: str, count_frac: str | None, singular: str, plural: str) -> str:
    n = int(count_whole.replace(",", "") or "0")
    if n == 1 and not (count_frac and count_frac.rstrip("0")):
        return singular
    return plural


_MINOR = {
    "$": ("cent", "cents"),
    "€": ("cent", "cents"),
    "£": ("penny", "pence"),
}


def _currency_word(sym: str) -> str:
    return {"$": "dollars", "€": "euros", "£": "pounds"}.get(sym, "dollars")


def _currency_word_one(sym: str) -> str:
    return {"$": "dollar", "€": "euro", "£": "pound"}.get(sym, "dollar")


def _cents_from_frac(frac: str | None) -> int:
    if not frac:
        return 0
    return int((frac + "00")[:2])


def _money_phrase(sym: str, whole: str, frac: str | None, scale: str | None) -> str:
    if scale:
        num = _number_to_en(whole, frac)
        return f"{num} {_MONEY_SCALE[scale.upper()]} {_currency_word(sym)}"
    n = int(whole.replace(",", "") or "0")
    cents = _cents_from_frac(frac)
    major_one = _currency_word_one(sym)
    major = _currency_word(sym)
    minor_one, minor = _MINOR.get(sym, ("cent", "cents"))
    if n == 0 and cents == 0:
        return f"zero {major}"
    if n == 0:
        return f"{_cardinal(cents)} {minor_one if cents == 1 else minor}"
    head = f"{_cardinal(n)} {major_one if n == 1 else major}"
    if cents == 0:
        return head
    return f"{head} and {_cardinal(cents)} {minor_one if cents == 1 else minor}"


def _sub_money_scale(m: re.Match) -> str:
    return _money_phrase(m.group(1), m.group(2), m.group(3), m.group(4))


def _sub_money_plain(m: re.Match) -> str:
    return _money_phrase(m.group(1), m.group(2), m.group(3), None)


def _sub_num_byte(m: re.Match) -> str:
    unit = m.group(3).upper()
    singular, plural = _BYTE_UNITS[unit]
    spoken = _number_to_en(m.group(1), m.group(2))
    name = _plural_unit(m.group(1), m.group(2), singular, plural)
    return f"{spoken} {name}"


def _sub_byte_rate(m: re.Match) -> str:
    key = (m.group(3) + "/S").upper()
    spoken = _number_to_en(m.group(1), m.group(2))
    singular, plural = _BYTE_RATE[key]
    name = _plural_unit(m.group(1), m.group(2), singular, plural)
    return f"{spoken} {name}"


def _sub_bit_rate(m: re.Match) -> str:
    spoken = _number_to_en(m.group(1), m.group(2))
    singular, plural = _BIT_RATE[m.group(3).upper()]
    name = _plural_unit(m.group(1), m.group(2), singular, plural)
    return f"{spoken} {name}"


def _sub_hz(m: re.Match) -> str:
    spoken = _number_to_en(m.group(1), m.group(2))
    return f"{spoken} {_HZ[m.group(3).upper()]}"


def _sub_million(m: re.Match) -> str:
    return f"{_number_to_en(m.group(1), m.group(2))} million"


def _sub_bare_byte(m: re.Match) -> str:
    return _BYTE_UNITS[m.group(1).upper()][0]


def _normalize_units(text: str) -> str:
    """Expand storage / money / rate abbreviations into spoken English.

    Thread 7ef45303: KB/MB/GB/TB → kilobyte…; $89M → eighty-nine million dollars.
    Also: $K/B/T, KiB family, MB/s, Mbps, MHz, and 89M as million (not MB).
    Leaves 5G / 4K alone (no trailing B).
    """
    if not text:
        return text
    text = _MONEY_SCALE_RE.sub(_sub_money_scale, text)
    text = _MONEY_PLAIN_RE.sub(_sub_money_plain, text)
    text = _BYTE_RATE_RE.sub(_sub_byte_rate, text)
    text = _BIT_RATE_RE.sub(_sub_bit_rate, text)
    text = _HZ_RE.sub(_sub_hz, text)
    text = _NUM_BYTE_RE.sub(_sub_num_byte, text)
    text = _NUM_MILLION_RE.sub(_sub_million, text)
    text = _BARE_BYTE_RE.sub(_sub_bare_byte, text)
    return text


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = _normalize_tabs(str(text))
    return _normalize_units(text)


def describe(before: str, after: str) -> str:
    if before == after:
        return ""
    notes = []
    tabbed = _normalize_tabs(str(before))
    if tabbed != str(before):
        notes.append("tab/indent stripped")
    if _normalize_units(tabbed) != tabbed:
        notes.append("units/money expanded")
    return "; ".join(notes) or "normalized"

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

_ISO_DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-–—](0?[1-9]|1[0-2])[-–—](0?[1-9]|[12]\d|3[01])(?!\d)")
_MATH_EXPR_RE = re.compile(
    r"(?<![A-Za-z0-9.])[-+]?\d+(?:\.\d+)?[%％]?"
    r"(?:\s*[+\-×÷*/=]\s*[-+]?\d+(?:\.\d+)?[%％]?)+(?![\w%％])"
)
_PERCENT_RANGE_RE = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*[%％]?\s*(?:-|–|—|~|～|至|到)\s*"
    r"(\d+(?:\.\d+)?)\s*[%％]"
)
_PERCENT_RE = re.compile(r"(?<![A-Za-z0-9.%％])(-?\d+(?:\.\d+)?)\s*[%％]")
_TEMPERATURE_RE = re.compile(
    r"(?<![A-Za-z0-9])-(\d+(?:\.\d+)?)\s*(°\s*C|℃|°\s*F|℉)", re.IGNORECASE
)
_NEGATIVE_RE = re.compile(r"(^|[=(>,，。！？；;:+×÷*/])\s*-(\d+(?:\.\d+)?)(?![\d%％])")
_YEAR_RANGE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[-–—~～至到]\s*((?:19|20)\d{2})\s*年")
_CHAPTER_RANGE_RE = re.compile(r"第\s*(\d+(?:\.\d+)?)\s*[-–—~～至到]\s*(\d+(?:\.\d+)?)\s*([章节页集期])")
_TIME_RANGE_RE = re.compile(
    r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)\s*[-–—~～至到]\s*"
    r"([01]?\d|2[0-3]):([0-5]\d)(?!\d)"
)
_RANGE_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*[-–—~～至到]\s*(\d+(?:\.\d+)?)\s*"
    r"(只|天|小时|分钟|秒|岁|人|个|件|条|台|公里|千米|厘米|毫米|米|千克|公斤|克|"
    r"kg|km|cm|mm|ms|GB|MB|KB|TB|g|m|页|章|集|期|次|倍|种|项|款|美元|元|"
    r"days?|hours?|minutes?|seconds?|years?|people|items?|times?|chapters?|pages?)(?![A-Za-z])",
    re.IGNORECASE,
)
_ALPHA_RANGE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z])\s*[-–—~～至到]\s*([A-Za-z])(?![A-Za-z0-9])"
)
_PHONE_RE = re.compile(r"(?<!\d)(\d{2,4}(?:-\d{3,4}){2,})(?!\d)")
_HYPHENATED_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+(?![A-Za-z0-9])"
)
_WORD_HYPHEN_RE = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])")

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


def _number_to_zh(value: str) -> str:
    whole, dot, frac = value.partition(".")
    spoken = _integer_to_zh(int(whole or "0"))
    if dot and frac:
        spoken += "点" + "".join(_ZH_DIGITS[int(d)] for d in frac)
    return spoken


def _spoken_number(value: str, chinese: bool) -> str:
    if chinese:
        return _number_to_zh(value)
    whole, dot, frac = value.partition(".")
    return _number_to_en(whole, frac if dot else None)


def _spoken_percent(value: str, chinese: bool) -> str:
    negative = value.startswith("-")
    value = value.lstrip("-")
    spoken = _spoken_number(value, chinese)
    if chinese:
        if value == "100":
            spoken = "百"
        return ("负" if negative else "") + "百分之" + spoken
    return ("negative " if negative else "") + spoken + " percent"


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


def _math_phrase(expr: str, chinese: bool) -> str:
    tokens = re.findall(r"\d+(?:\.\d+)?[%％]?|[+\-×÷*/=]", expr)
    words: list[str] = []
    previous_was_operator = True
    operators_zh = {"+": "加", "-": "减", "×": "乘", "*": "乘", "÷": "除以", "/": "除以", "=": "等于"}
    operators_en = {
        "+": "plus",
        "-": "minus",
        "×": "times",
        "*": "times",
        "÷": "divided by",
        "/": "divided by",
        "=": "equals",
    }
    for token in tokens:
        if token == "-" and previous_was_operator:
            words.append("负" if chinese else "negative")
            previous_was_operator = True
        elif token in operators_zh:
            words.append((operators_zh if chinese else operators_en)[token])
            previous_was_operator = True
        else:
            if token.endswith(("%", "％")):
                words.append(_spoken_percent(token[:-1], chinese))
            else:
                words.append(_spoken_number(token, chinese))
            previous_was_operator = False
    return "".join(words) if chinese else " ".join(words)


def _normalize_hyphen_percent(text: str) -> str:
    """Classify hyphens and percent signs using the approved rule priority.

    High-confidence contexts are expanded. Ambiguous score/mixed-fraction/bare
    numeric forms are intentionally left alone until a domain-specific example
    gives enough context to distinguish them.
    """
    if not text:
        return text
    # With no lexical language signal, this WebUI defaults numeric-only input
    # to Chinese. A surrounding Latin word switches the phrase to English.
    chinese = bool(_CJK_RE.search(text)) or not bool(re.search(r"[A-Za-z]", text))
    text, protected = _protect_syntax(text)

    # ISO dates are dates, never two numeric ranges. English text is preserved
    # because changing it to Chinese date markers would switch languages.
    if chinese:
        text = _ISO_DATE_RE.sub(
            lambda m: f"{m.group(1)}年{int(m.group(2))}月{int(m.group(3))}日", text
        )
    else:
        text = _ISO_DATE_RE.sub(lambda m: m.group(0).replace("–", "-").replace("—", "-"), text)

    # Expressions with '=' / another explicit operator, or a spaced minus, are
    # arithmetic. A compact lone 3-1 stays ambiguous and is not guessed.
    def math_sub(match: re.Match) -> str:
        expr = match.group(0)
        explicit = bool(re.search(r"[=+×÷*/]", expr)) or bool(re.search(r"\d[%％]?\s+-\s+\d", expr))
        return _math_phrase(expr, chinese) if explicit else expr

    text = _MATH_EXPR_RE.sub(math_sub, text)

    def percent_range_sub(match: re.Match) -> str:
        left = _spoken_number(match.group(1), chinese)
        right = _spoken_number(match.group(2), chinese)
        if chinese:
            return f"百分之{left}到百分之{right}"
        return f"{left} to {right} percent"

    text = _PERCENT_RANGE_RE.sub(percent_range_sub, text)

    def temperature_sub(match: re.Match) -> str:
        value = _spoken_number(match.group(1), chinese)
        celsius = "C" in match.group(2).upper() or "℃" in match.group(2)
        if chinese:
            return f"零下{value}{'摄氏度' if celsius else '华氏度'}"
        return f"minus {value} degrees {'Celsius' if celsius else 'Fahrenheit'}"

    text = _TEMPERATURE_RE.sub(temperature_sub, text)

    def negative_sub(match: re.Match) -> str:
        prefix = match.group(1)
        value = _spoken_number(match.group(2), chinese)
        return prefix + (("负" + value) if chinese else ("negative " + value))

    text = _NEGATIVE_RE.sub(negative_sub, text)

    def year_range_sub(match: re.Match) -> str:
        if chinese:
            return f"{match.group(1)}年到{match.group(2)}年"
        return f"{_cardinal(int(match.group(1)))} to {_cardinal(int(match.group(2)))}"

    text = _YEAR_RANGE_RE.sub(year_range_sub, text)

    def chapter_range_sub(match: re.Match) -> str:
        left = _spoken_number(match.group(1), chinese)
        right = _spoken_number(match.group(2), chinese)
        if chinese:
            return f"第{left}到{right}{match.group(3)}"
        return f"{left} to {right} {match.group(3)}"

    text = _CHAPTER_RANGE_RE.sub(chapter_range_sub, text)

    def time_range_sub(match: re.Match) -> str:
        h1, m1, h2, m2 = (int(match.group(i)) for i in range(1, 5))
        if chinese:
            left = _integer_to_zh(h1) + "点" + ("" if m1 == 0 else _integer_to_zh(m1) + "分")
            right = _integer_to_zh(h2) + "点" + ("" if m2 == 0 else _integer_to_zh(m2) + "分")
            return left + "到" + right
        return f"{_cardinal(h1)} {_cardinal(m1):s} to {_cardinal(h2)} {_cardinal(m2):s}"

    text = _TIME_RANGE_RE.sub(time_range_sub, text)

    def unit_range_sub(match: re.Match) -> str:
        # Known score contexts are not quantity ranges, even if followed by CJK.
        before = text[max(0, match.start() - 5) : match.start()]
        if re.search(r"(?:比分|战胜|负于|击败)\s*$", before):
            return match.group(0)
        left = _spoken_number(match.group(1), chinese)
        right = _spoken_number(match.group(2), chinese)
        unit = match.group(3)
        if unit.upper() in _BYTE_UNITS:
            unit = _BYTE_UNITS[unit.upper()][1]
        unit_space = " " if unit[0].isascii() and unit[0].isalpha() else ""
        return f"{left}{'到' if chinese else ' to '}{right}{unit_space}{unit}"

    text = _RANGE_UNIT_RE.sub(unit_range_sub, text)
    text = _ALPHA_RANGE_RE.sub(
        lambda m: f"{m.group(1)}{'到' if chinese else ' to '}{m.group(2)}", text
    )

    def phone_sub(match: re.Match) -> str:
        raw = match.group(1)
        if chinese:
            return "".join(_ZH_DIGITS[int(ch)] if ch.isdigit() else " " for ch in raw)
        return " dash ".join(_digits_to_en(part) for part in raw.split("-"))

    text = _PHONE_RE.sub(phone_sub, text)

    def token_sub(match: re.Match) -> str:
        raw = match.group(0)
        if not any(ch.isdigit() for ch in raw):
            return raw.replace("-", " ")
        parts = raw.split("-")
        if chinese:
            return " ".join(
                re.sub(
                    r"\d+",
                    lambda n: " " + "".join(_ZH_DIGITS[int(d)] for d in n.group(0)),
                    part,
                ).strip()
                for part in parts
            )
        return " dash ".join(
            re.sub(r"\d+", lambda n: " " + _digits_to_en(n.group(0)), part).strip()
            for part in parts
        )

    text = _HYPHENATED_TOKEN_RE.sub(token_sub, text)
    text = _WORD_HYPHEN_RE.sub(" ", text)
    text = _PERCENT_RE.sub(lambda m: _spoken_percent(m.group(1), chinese), text)

    # Unicode dashes not consumed by an unambiguous rule are prosodic pauses.
    text = re.sub(r"\s*[–—]\s*", "，" if chinese else ", ", text)
    return _restore_syntax(text, protected)


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = _normalize_tabs(str(text))
    text = _normalize_hyphen_percent(text)
    return _normalize_units(text)


def describe(before: str, after: str) -> str:
    if before == after:
        return ""
    notes = []
    tabbed = _normalize_tabs(str(before))
    if tabbed != str(before):
        notes.append("tab/indent stripped")
    hyphen_normalized = _normalize_hyphen_percent(tabbed)
    if hyphen_normalized != tabbed:
        notes.append("hyphen/percent expanded")
    if _normalize_units(hyphen_normalized) != hyphen_normalized:
        notes.append("units/money expanded")
    return "; ".join(notes) or "normalized"

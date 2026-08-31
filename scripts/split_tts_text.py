#!/usr/bin/env python3
"""Split long TTS target text before generation.

Chinese / CJK: punct-split if >50 non-space chars; force-split if a piece >100.
English-like: punct-split if >30 words; force-split if a piece >50 words.
"""
from __future__ import annotations

import re

CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]")
PUNCT_SPLIT = re.compile(r"(?<=[。！？；;.!?…])\s*")
SOFT_CJK = 50
HARD_CJK = 100
SOFT_WORDS = 30
HARD_WORDS = 50


def is_cjk(text: str) -> bool:
    ns = [ch for ch in text if not ch.isspace()]
    if not ns:
        return False
    return sum(1 for ch in ns if CJK_RE.match(ch)) / len(ns) >= 0.3


def nchars_cjk(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def nwords(text: str) -> int:
    return len(text.split())


def needs_split(text: str, kind: str | None = None) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    cjk = is_cjk(text) if kind is None else kind == "cjk"
    if cjk:
        return nchars_cjk(text) > SOFT_CJK
    return nwords(text) > SOFT_WORDS


def split_punct(text: str) -> list[str]:
    parts = [p.strip() for p in PUNCT_SPLIT.split(text) if p.strip()]
    return parts or [text.strip()]


def _measure(text: str, cjk: bool) -> int:
    return nchars_cjk(text) if cjk else nwords(text)


def force_chunks(text: str, limit: int, cjk: bool) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if _measure(text, cjk) <= limit:
        return [text]
    out: list[str] = []
    buf = ""
    for ch in text:
        trial = buf + ch
        if _measure(trial, cjk) <= limit:
            buf = trial
            continue
        cut = _back_break(buf) if buf else ""
        if cut:
            out.append(cut.strip())
            buf = (buf[len(cut) :] + ch).lstrip()
        else:
            if buf.strip():
                out.append(buf.strip())
            buf = ch
    if buf.strip():
        out.append(buf.strip())
    return out or [text]


def _back_break(buf: str) -> str:
    for i in range(len(buf) - 1, 0, -1):
        if buf[i] in "。！？；;.!?…，,、 ":
            return buf[: i + 1]
    return ""


def plan_segments(text: str, kind: str | None = None) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    cjk = is_cjk(text) if kind is None else kind == "cjk"
    if cjk:
        if nchars_cjk(text) <= SOFT_CJK:
            return [text]
        pieces = split_punct(text)
        out: list[str] = []
        for piece in pieces:
            if nchars_cjk(piece) > HARD_CJK:
                out.extend(force_chunks(piece, HARD_CJK, cjk=True))
            elif piece:
                out.append(piece)
        return out or [text]
    if nwords(text) <= SOFT_WORDS:
        return [text]
    pieces = split_punct(text)
    out = []
    for piece in pieces:
        if nwords(piece) > HARD_WORDS:
            out.extend(force_chunks(piece, HARD_WORDS, cjk=False))
        elif piece:
            out.append(piece)
    return out or [text]


def classify(text: str) -> str:
    return "cjk" if is_cjk(text) else "latin"


def join_prefix(prefix: str, target: str) -> str:
    prefix = (prefix or "").strip()
    target = (target or "").strip()
    if not prefix:
        return target
    if not target:
        return prefix
    if prefix[-1].isspace() or target[0].isspace():
        return f"{prefix}{target}"
    if is_cjk(prefix) or is_cjk(target):
        return f"{prefix}{target}"
    if prefix[-1] in "。！？；;.!?…，,、":
        return f"{prefix} {target}" if target[0].isascii() else f"{prefix}{target}"
    return f"{prefix} {target}"


def plan_segments_with_prefix(prefix: str, target: str) -> list[str]:
    """Split the target by the usual length rule; always prepend prefix text."""
    prefix = (prefix or "").strip()
    target = (target or "").strip()
    if not prefix:
        return plan_segments(target)
    combined = join_prefix(prefix, target)
    if not needs_split(combined):
        return [combined]
    return [join_prefix(prefix, piece) for piece in plan_segments(target)]

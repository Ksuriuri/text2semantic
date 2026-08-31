#!/usr/bin/env python3
"""Cut a reference clip + Whisper text for speaker-similarity continuation.

<=10s: whole wav is the prefix.
>10s: longest prefix inside the first 10s ending at the last pause, plus
~120ms of trailing audio (silence if present).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

WINDOW_S = 10.0
TRAIL_S = 0.12
GAP_S = 0.25
PUNCT = "。！？；;.!?…，,、"
_DEFAULT_CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
CACHE_DIR = Path(os.environ.get("SIMBOOST_CACHE", _DEFAULT_CACHE_ROOT / "noiz-tts" / "simboost"))

_lock = threading.Lock()
_pipe = None


def _truthy(value) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _cache_key(path: str) -> str:
    st = os.stat(path)
    h = hashlib.sha1()
    h.update(os.path.abspath(path).encode())
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    with _lock:
        if _pipe is not None:
            return _pipe
        import torch
        from transformers import pipeline

        model_id = os.environ.get("SIMBOOST_WHISPER", "openai/whisper-large-v3")
        device = os.environ.get("SIMBOOST_WHISPER_DEVICE", "cuda:0")
        _pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
        )
        return _pipe


def _load_wav(path: str) -> tuple[np.ndarray, int]:
    import librosa

    audio, sr = librosa.load(path, sr=None, mono=True)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _words_from_result(result: dict) -> list[dict]:
    words = []
    for ch in result.get("chunks") or []:
        ts = ch.get("timestamp")
        if not isinstance(ts, (list, tuple)) or len(ts) < 2:
            continue
        start, end = ts[0], ts[1]
        text = (ch.get("text") or "").strip()
        if start is None or end is None or not text:
            continue
        words.append({"text": text, "start": float(start), "end": float(end)})
    return words


def _transcribe(path: str, language: str | None = None) -> dict:
    pipe = _get_pipe()
    gen = {"task": "transcribe"}
    if language:
        gen["language"] = language
    # dest has no ffmpeg; feed PCM so transformers does not shell out.
    audio, sr = _load_wav(path)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    payload = {"array": audio, "sampling_rate": sr}
    with _lock:
        try:
            result = pipe(payload, return_timestamps="word", generate_kwargs=gen)
        except Exception:
            result = pipe(
                {"raw": audio, "sampling_rate": sr},
                return_timestamps="word",
                generate_kwargs=gen,
            )
    text = (result.get("text") or "").strip()
    words = _words_from_result(result)
    if not words and text:
        audio, sr = _load_wav(path)
        words = [{"text": text, "start": 0.0, "end": float(len(audio) / sr)}]
    return {"text": text, "words": words}


def _cut_end(duration: float, words: list[dict]) -> tuple[float, str, str]:
    """Return (end_sec, clip_text, rule) for a wav longer than WINDOW_S."""
    in_win = [w for w in words if w["end"] <= WINDOW_S + 1e-3]
    if not in_win:
        return min(WINDOW_S, duration), "", "first_10s_no_words"

    punct = [w for w in in_win if w["text"][-1] in PUNCT]
    if punct:
        last = punct[-1]
        return last["end"], _join_words(in_win[: in_win.index(last) + 1]), "last_punct"

    gap_i = None
    for i in range(1, len(in_win)):
        if in_win[i]["start"] - in_win[i - 1]["end"] >= GAP_S:
            gap_i = i - 1
    if gap_i is not None:
        last = in_win[gap_i]
        return last["end"], _join_words(in_win[: gap_i + 1]), "last_gap"

    last = in_win[-1]
    return last["end"], _join_words(in_win), "last_word"


def _join_words(words: list[dict]) -> str:
    if not words:
        return ""
    # Whisper chunks already include leading spaces for English.
    return "".join(w["text"] if w["text"].startswith(" ") else (" " + w["text"]) for w in words).strip()


def _write_clip(audio: np.ndarray, sr: int, end_s: float, dest: Path) -> None:
    end = min(len(audio), int(round((end_s + TRAIL_S) * sr)))
    clip = audio[:end]
    if end_s + TRAIL_S > len(audio) / sr:
        pad = int(round((end_s + TRAIL_S) * sr)) - len(audio)
        if pad > 0:
            clip = np.concatenate([clip, np.zeros(pad, dtype=np.float32)])
    dest.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(str(dest), clip, sr)


def prepare_ref(path: str, *, language: str | None = None, cache_dir: Path | None = None) -> dict:
    """Return clip wav path + prefix text. Cached per source mtime/size."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    key = _cache_key(path)
    meta_path = cache_dir / f"{key}.json"
    clip_path = cache_dir / f"{key}.wav"
    if meta_path.exists() and clip_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if os.path.isfile(meta.get("clip_path", str(clip_path))):
            return meta

    audio, sr = _load_wav(path)
    duration = float(len(audio) / sr) if sr else 0.0
    asr = _transcribe(path, language=language)
    if duration <= WINDOW_S + 1e-3:
        end_s = duration
        clip_text = asr["text"]
        rule = "whole_le_10s"
        import soundfile as sf

        cache_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(clip_path), audio, sr)
    else:
        end_s, clip_text, rule = _cut_end(duration, asr["words"])
        if not clip_text:
            clip_text = asr["text"]
        _write_clip(audio, sr, end_s, clip_path)

    meta = {
        "src": os.path.abspath(path),
        "clip_path": str(clip_path),
        "clip_text": clip_text,
        "full_text": asr["text"],
        "duration": duration,
        "end_s": end_s,
        "rule": rule,
        "n_words": len(asr["words"]),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

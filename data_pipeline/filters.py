"""Row-level and speaker-level quality gates.

Three filters are applied, in the order requested for this dataset build:

1. **Sample rate** - keep utterances recorded at >= 22.05 kHz.
2. **Duration and speaker support** - keep utterances with
   ``3s < duration < 30s`` that belong to a speaker who has at least one *other*
   utterance, where at least one utterance of that speaker is longer than 6s.
3. **ASR agreement** - keep utterances whose transcription error rate is below
   0.5.  When the metadata carries an original transcript, the Cohere ASR result
   is scored against it; otherwise the Granite result is scored against the
   Cohere result.  Rows with neither reference are dropped, because their
   quality cannot be established.

Filter 1 and the duration half of filter 2 are per-row and evaluated during the
streaming scan.  The speaker-support half of filter 2 needs the whole corpus, so
it runs as a second pass over the compact index produced by the scan.

The word/character error rates are computed with a plain Levenshtein DP so the
pipeline has no dependency beyond the project's existing requirements.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

#: Sentinels that the upstream exporter writes for missing values.
_NULLISH = {"", "none", "null", "nan", "n/a"}

_PUNCT_RE = re.compile(
    r"[!-/:-@\[-`{-~ -⁯　-〿＀-￯]+"
)
_SPACE_RE = re.compile(r"\s+")

#: Scripts that are not word-delimited; WER is meaningless for them so the
#: character error rate is used as the primary metric instead.
_CJK_LANGS = {"zh", "ja", "ko", "yue", "zh-cn", "zh-tw", "cmn", "jp", "kr"}


def is_nullish(value):
    """True when a metadata value stands for "absent"."""
    if value is None:
        return True
    return str(value).strip().lower() in _NULLISH


def as_float(value):
    """Parse a metadata number that may arrive as a string, else ``None``."""
    if is_nullish(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def as_text(value):
    """Return a stripped string, or ``""`` when the value is a null sentinel."""
    if is_nullish(value):
        return ""
    return str(value).strip()


def normalise_text(text):
    """Lowercase, strip punctuation and collapse whitespace.

    Both ASR engines differ in casing and punctuation policy (Granite emits
    lowercase and no terminal punctuation, Cohere emits cased text with
    punctuation), so scoring raw strings would report a large error rate for
    transcripts that are in fact identical.
    """
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _levenshtein(reference, hypothesis):
    """Edit distance between two sequences, O(min(len)) memory."""
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,       # deletion
                    current[j - 1] + 1,    # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def _rate(reference_tokens, hypothesis_tokens):
    if not reference_tokens:
        # An empty reference cannot be scored; report 1.0 so the row is dropped
        # rather than silently accepted.
        return 1.0
    return _levenshtein(reference_tokens, hypothesis_tokens) / len(reference_tokens)


def wer(reference, hypothesis):
    """Word error rate over normalised text."""
    return _rate(normalise_text(reference).split(), normalise_text(hypothesis).split())


def cer(reference, hypothesis):
    """Character error rate over normalised text with spaces removed."""
    reference = normalise_text(reference).replace(" ", "")
    hypothesis = normalise_text(hypothesis).replace(" ", "")
    return _rate(list(reference), list(hypothesis))


def error_rate(reference, hypothesis, language=None):
    """Return ``(metric_name, value)`` using the metric suited to ``language``.

    For space-delimited languages the word error rate decides; for CJK the
    character error rate does.  ``min(WER, CER)`` is deliberately *not* used:
    taking the friendlier of two metrics would let badly aligned pairs through.
    """
    lang = (language or "").strip().lower()
    if lang in _CJK_LANGS:
        return "cer", cer(reference, hypothesis)
    return "wer", wer(reference, hypothesis)


@dataclass
class FilterConfig:
    """Thresholds for the three gates. Defaults are the requested values."""

    min_sample_rate: int = 22050
    min_duration: float = 3.0
    max_duration: float = 30.0
    #: At least one utterance of the speaker must exceed this length.
    speaker_long_utterance_seconds: float = 6.0
    #: A usable speaker needs the target plus one reference utterance.
    min_speaker_records: int = 2
    max_error_rate: float = 0.5
    #: Drop rows that have no scoreable transcript pair. When False such rows
    #: are kept and marked ``asr_scored=False``.
    require_asr_score: bool = True


@dataclass
class FilterCounters:
    """Per-dataset tally of why rows were dropped."""

    total: int = 0
    kept: int = 0
    no_audio_path: int = 0
    bad_sample_rate: int = 0
    bad_duration: int = 0
    no_speaker: int = 0
    no_asr_hypothesis: int = 0
    no_asr_reference: int = 0
    asr_above_threshold: int = 0
    dropped_speaker_singleton: int = 0
    dropped_speaker_no_long: int = 0
    seconds_total: float = 0.0
    seconds_kept: float = 0.0
    extra: dict = field(default_factory=dict)

    def add(self, other):
        for key, value in vars(other).items():
            if key == "extra":
                for k, v in value.items():
                    self.extra[k] = self.extra.get(k, 0) + v
            else:
                setattr(self, key, getattr(self, key) + value)
        return self

    def as_dict(self):
        payload = {k: v for k, v in vars(self).items() if k != "extra"}
        payload.update(self.extra)
        return payload


@dataclass
class RowRecord:
    """The compact per-utterance record kept in memory during the scan.

    Only what the later stages need: identity, the speaker key, duration and
    the measured error rate.  The transcript itself is re-read from GCS during
    the encode stage so that a full-corpus scan never has to hold 75 GB of text.
    """

    __slots__ = (
        "dataset",
        "shard",
        "utt_id",
        "audio_path",
        "language",
        "speaker_id",
        "duration",
        "sample_rate",
        "metric",
        "error_rate",
        "asr_scored",
    )

    def __init__(
        self,
        dataset,
        shard,
        utt_id,
        audio_path,
        language,
        speaker_id,
        duration,
        sample_rate,
        metric,
        error_rate_,
        asr_scored,
    ):
        self.dataset = dataset
        self.shard = shard
        self.utt_id = utt_id
        self.audio_path = audio_path
        self.language = language
        self.speaker_id = speaker_id
        self.duration = duration
        self.sample_rate = sample_rate
        self.metric = metric
        self.error_rate = error_rate_
        self.asr_scored = asr_scored

    @property
    def speaker_key(self):
        """``(language, speaker_id)`` - the key the training dataset groups on."""
        return (self.language, self.speaker_id)

    def as_dict(self):
        return {
            "dataset": self.dataset,
            "shard": self.shard,
            "id": self.utt_id,
            "audio_path": self.audio_path,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "duration": round(self.duration, 4),
            "sample_rate": self.sample_rate,
            "metric": self.metric,
            "error_rate": None if self.error_rate is None else round(self.error_rate, 4),
            "asr_scored": self.asr_scored,
        }


def screen_row(row, cohere_text, granite_text, config, counters, dataset, shard):
    """Apply the per-row gates; return a :class:`RowRecord` or ``None``.

    ``counters`` is updated in place with the reason for every rejection.
    """
    counters.total += 1

    audio_path = as_text(row.get("audio_path"))
    if not audio_path:
        counters.no_audio_path += 1
        return None

    duration = as_float(row.get("duration"))
    if duration is not None:
        counters.seconds_total += duration

    sample_rate = as_float(row.get("sample_rate"))
    if sample_rate is None or sample_rate < config.min_sample_rate:
        counters.bad_sample_rate += 1
        return None

    if duration is None or not (config.min_duration < duration < config.max_duration):
        counters.bad_duration += 1
        return None

    speaker_id = as_text(row.get("speaker_id"))
    if not speaker_id:
        # Without a speaker id the "another utterance of the same speaker"
        # requirement can never be satisfied, and the training dataset would
        # drop the row anyway for lack of a reference clip.
        counters.no_speaker += 1
        return None

    language = as_text(row.get("language")) or None
    original_text = as_text(row.get("text"))
    cohere_text = as_text(cohere_text)
    granite_text = as_text(granite_text)

    if not cohere_text:
        counters.no_asr_hypothesis += 1
        return None

    if original_text:
        reference, hypothesis = original_text, cohere_text
    elif granite_text:
        # No ground-truth transcript: cross-check the two engines instead.
        reference, hypothesis = granite_text, cohere_text
    else:
        counters.no_asr_reference += 1
        if config.require_asr_score:
            return None
        record = RowRecord(
            dataset, shard, as_text(row.get("id")), audio_path, language,
            speaker_id, duration, int(sample_rate), None, None, False,
        )
        counters.kept += 1
        counters.seconds_kept += duration
        return record

    metric, value = error_rate(reference, hypothesis, language)
    if value >= config.max_error_rate:
        counters.asr_above_threshold += 1
        return None

    counters.kept += 1
    counters.seconds_kept += duration
    return RowRecord(
        dataset, shard, as_text(row.get("id")), audio_path, language,
        speaker_id, duration, int(sample_rate), metric, value, True,
    )


def apply_speaker_gate(records, config, counters=None):
    """Second pass of filter 2: speaker support.

    A record survives when its speaker has at least ``min_speaker_records``
    surviving utterances and at least one of them is longer than
    ``speaker_long_utterance_seconds``.
    """
    counts = {}
    has_long = set()
    for record in records:
        key = record.speaker_key
        counts[key] = counts.get(key, 0) + 1
        if record.duration > config.speaker_long_utterance_seconds:
            has_long.add(key)

    kept = []
    dropped_singleton = 0
    dropped_no_long = 0
    for record in records:
        key = record.speaker_key
        if counts[key] < config.min_speaker_records:
            dropped_singleton += 1
            continue
        if key not in has_long:
            dropped_no_long += 1
            continue
        kept.append(record)

    if counters is not None:
        counters.dropped_speaker_singleton += dropped_singleton
        counters.dropped_speaker_no_long += dropped_no_long
    return kept

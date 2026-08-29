# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

import io
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from finetuning import speaker_index
from qwen_tts.text_augment import strip_pause_marks
from qwen_tts.text_template import tokenize_tts_prompt


class Text2SemanticDataset(Dataset):
    """Text prefix and MaskGCT semantic-code teacher-forcing dataset."""

    def __init__(
        self,
        data,
        tokenizer,
        *,
        semantic_vocab_size=8192,
        speech_bos_token_id=8192,
        speech_eos_token_id=8193,
        speech_pad_token_id=8194,
        max_text_tokens=None,
        max_semantic_tokens=None,
        speaker_counts=None,
        speaker_audio_paths_by_id=None,
        ref_store=None,
        min_speaker_records=2,
        max_target_seconds=30.0,
        min_target_seconds=0.5,
        ref_max_seconds=20.0,
        speaker_mel_extractor=None,
        punctuation_dropout_prob=0.0,
        punctuation_dropout_keep_word_spaces=True,
        text_conditioner=None,
        seed=42,
    ):
        # A prefiltered source (finetuning.manifest_index.ManifestIndex) has
        # already applied every filter below and materializes one row at a time.
        # Walking it here would parse all 94M rows of the t2s-v1 manifest three
        # times over and then keep the survivors in a list, which is exactly the
        # per-process copy the index exists to avoid.
        self.prefiltered = bool(getattr(data, "prefiltered", False))
        self.raw_size = int(getattr(data, "raw_size", len(data)))
        self.tokenizer = tokenizer
        self.semantic_vocab_size = semantic_vocab_size
        self.speech_bos_token_id = speech_bos_token_id
        self.speech_eos_token_id = speech_eos_token_id
        self.speech_pad_token_id = speech_pad_token_id
        self.max_text_tokens = max_text_tokens
        self.max_semantic_tokens = max_semantic_tokens
        self.ref_store = ref_store
        # When set, collate_fn computes the ref log-mel here in the worker
        # instead of leaving it inline in the training step.
        self.speaker_mel_extractor = speaker_mel_extractor
        # The ref backend owns the key: packed refs are keyed by speaker within
        # a language, source-tar refs by speaker within a dataset as well.
        self.speaker_key_fields = tuple(
            getattr(
                ref_store, "speaker_key_fields", speaker_index.DEFAULT_KEY_FIELDS
            )
        )
        if self.prefiltered:
            self.speaker_counts = speaker_counts or {}
            self.speaker_audio_paths_by_id = speaker_audio_paths_by_id or {}
        else:
            self.speaker_counts = speaker_counts or self._count_speakers(data)
            self.speaker_audio_paths_by_id = (
                speaker_audio_paths_by_id
                if speaker_audio_paths_by_id is not None
                else (
                    {}
                    if ref_store is not None
                    else self._collect_speaker_audio_paths(data)
                )
            )
        self.min_speaker_records = min_speaker_records
        self.max_target_seconds = max_target_seconds
        self.min_target_seconds = min_target_seconds
        self.ref_max_seconds = ref_max_seconds
        # Packed refs are read out of the shard and decoded here, in the
        # DataLoader worker, so no ref ever lands on disk.
        self.ref_audio_in_memory = ref_store is not None and hasattr(
            ref_store, "read_ref"
        )
        if not 0.0 <= punctuation_dropout_prob <= 1.0:
            raise ValueError("punctuation_dropout_prob must be in [0, 1].")
        self.punctuation_dropout_prob = punctuation_dropout_prob
        self.punctuation_dropout_keep_word_spaces = (
            punctuation_dropout_keep_word_spaces
        )
        self.text_conditioner = text_conditioner
        self.seed = seed
        # A row whose ref cannot be read is replaced by another row, this many
        # tries before giving up. Eight is generous for a lost shard and still
        # small enough that a manifest with no readable refs at all fails fast
        # instead of walking the whole dataset per sample.
        self.ref_substitution_attempts = 8
        self.substituted_rows = 0
        self._semantic_code_cache = {}
        if self.prefiltered:
            self.data = data
        else:
            self.data = [item for item in data if self._is_usable(item)]
        if not len(self.data):
            raise ValueError("No usable samples remain after dataset filtering.")
        self.filtered_size = self.raw_size - len(self.data)

    @staticmethod
    def _target_audio_path(item):
        return item.get("audio") or item.get("audio_path")

    def _speaker_key(self, item):
        return speaker_index.row_key(item, self.speaker_key_fields)

    def _count_speakers(self, data):
        counts = {}
        for item in data:
            speaker_key = self._speaker_key(item)
            if speaker_key is not None:
                counts[speaker_key] = counts.get(speaker_key, 0) + 1
        return counts

    def _collect_speaker_audio_paths(self, data):
        paths_by_id = {}
        for item in data:
            speaker_key = self._speaker_key(item)
            audio_path = self._target_audio_path(item)
            if speaker_key is None or audio_path is None:
                continue
            paths_by_id.setdefault(speaker_key, [])
            if audio_path not in paths_by_id[speaker_key]:
                paths_by_id[speaker_key].append(audio_path)
        return paths_by_id

    @staticmethod
    def _has_usable_text(item):
        text = item.get("text")
        return isinstance(text, str) and bool(text.strip())

    def _is_usable(self, item):
        if not self._has_usable_text(item):
            return False
        if not self._has_speaker_ref(item):
            return False
        speaker_key = self._speaker_key(item)
        if (
            speaker_key is not None
            and self.speaker_counts.get(speaker_key, 0) < self.min_speaker_records
        ):
            return False
        duration = item.get("duration")
        if (
            duration is not None
            and self.max_target_seconds is not None
            and float(duration) > self.max_target_seconds
        ):
            return False
        if (
            duration is not None
            and self.min_target_seconds is not None
            and float(duration) < self.min_target_seconds
        ):
            return False
        semantic_length = self._semantic_length(item)
        if (
            semantic_length is not None
            and self.max_semantic_tokens is not None
            and semantic_length > self.max_semantic_tokens
        ):
            return False
        return True

    @staticmethod
    def _semantic_length(item):
        if "semantic_codes" in item:
            return len(item["semantic_codes"])
        if "semantic_code_length" in item:
            return int(item["semantic_code_length"])
        return None

    def __len__(self):
        return len(self.data)

    def _augmented_text(self, item, index=None):
        """The sample's text, sometimes with every written pause cue removed.

        The draw happens per read rather than per row, so over epochs the model
        sees the same sentence both punctuated and bare and has to learn the
        pacing itself.  DataLoader workers get their own `random` seed derived
        from the base seed, so a run stays reproducible.
        """
        text = item["text"]
        if (
            self.punctuation_dropout_prob > 0.0
            and random.random() < self.punctuation_dropout_prob
        ):
            stripped = strip_pause_marks(
                text,
                keep_word_spaces=self.punctuation_dropout_keep_word_spaces,
            )
            # A transcript that is nothing but punctuation strips down to
            # nothing and would raise inside _tokenize_text; keep the original.
            if stripped.strip():
                text = stripped
        if self.text_conditioner is None:
            return text
        conditioned = dict(item)
        conditioned["text"] = text
        return self.text_conditioner(conditioned, index=index)

    def _tokenize_text(self, text):
        ids = tokenize_tts_prompt(self.tokenizer, text)
        if self.max_text_tokens is not None:
            ids = ids[: self.max_text_tokens]
        if not ids:
            raise ValueError("Tokenized text must not be empty.")
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, index):
        sample = self._sample(index)
        if sample is not None:
            return sample
        # The row passed the manifest's ref filter -- which can only consult the
        # speaker index -- and then turned out to have no ref that is actually
        # readable. On t2s-v1 this is one shard the packer rewrote after a restart:
        # 24 speakers lost every member the index still names for them, and one such
        # row took a 64-rank job down in a DataLoader worker.
        #
        # A ref from a different speaker is not an option: it would teach the model
        # the wrong voice for the text. So the row itself is replaced, by a
        # deterministic draw from elsewhere in the manifest rather than by index + 1,
        # because rows of one speaker sit together and the neighbour is likely to be
        # broken in the same way. Seeded by string, which random.Random hashes
        # stably, so every rank and every relaunch substitutes identically.
        rng = random.Random(f"ref-substitute:{self.seed}:{index}")
        for _ in range(self.ref_substitution_attempts):
            other = rng.randrange(len(self.data))
            sample = self._sample(other)
            if sample is not None:
                self._count_substitution(index, other)
                return sample
        raise ValueError(
            "Each sample needs a packed ref (speaker_index), "
            "'ref_audio', or another clip of the same speaker; "
            f"row {index} and {self.ref_substitution_attempts} substitutes for it "
            "all lack one, which is a broken manifest rather than a lost shard."
        )

    def _count_substitution(self, index, other):
        self.substituted_rows += 1
        # Loud once, then at every power of ten: a handful of rows is a known bad
        # shard, while a count that keeps climbing is a different problem and must
        # not be invisible just because the run stayed up.
        count = self.substituted_rows
        if 10 ** (len(str(count)) - 1) == count:
            print(
                f"[dataset] row {index} has no readable ref; trained on row "
                f"{other} instead ({count} substitutions so far)",
                flush=True,
            )

    def _sample(self, index):
        """The training sample for ``index``, or None when its ref is unreadable."""
        item = self.data[index]
        if not self._has_usable_text(item):
            raise ValueError("Each sample needs a non-empty string 'text'.")
        speaker_audio = None
        speaker_audio_path = None
        if self.ref_audio_in_memory:
            speaker_audio = self._speaker_audio(item, index)
            missing_ref = speaker_audio is None
        else:
            speaker_audio_path = self._speaker_audio_path(item, index)
            missing_ref = speaker_audio_path is None
        if missing_ref:
            return None
        codes = self._semantic_codes(item)
        if codes.numel() == 0:
            raise ValueError("semantic_codes must not be empty.")
        if int(codes.min()) < 0 or int(codes.max()) >= self.semantic_vocab_size:
            raise ValueError(
                f"semantic_codes must be in [0, {self.semantic_vocab_size - 1}]."
            )
        return {
            "text_input_ids": self._tokenize_text(
                self._augmented_text(item, index=index)
            ),
            "speech_input_ids": torch.cat(
                (torch.tensor([self.speech_bos_token_id]), codes)
            ),
            "labels": torch.cat(
                (codes, torch.tensor([self.speech_eos_token_id]))
            ),
            "speaker_audio_path": speaker_audio_path,
            "speaker_audio": speaker_audio,
        }

    def _speaker_audio(self, item, index=None):
        """The ref clip as a 16 kHz mono waveform, read from the shard in place.

        An explicit ``ref_audio`` path still wins, so a manifest that names its
        own ref file keeps working.
        """
        explicit = item.get("ref_audio") or item.get("ref_audio_path")
        if explicit is not None:
            if explicit == self._target_audio_path(item):
                return None
            return self._decode_audio(explicit, explicit)
        speaker_key = self._speaker_key(item)
        if speaker_key is None:
            return None
        rng = None if index is None else random.Random(self.seed + index)
        exclude = item.get("id")
        picked = self.ref_store.read_ref(
            speaker_key,
            exclude=None if exclude is None else str(exclude),
            rng=rng,
        )
        if picked is None:
            return None
        name, payload = picked
        return self._decode_audio(io.BytesIO(payload), name)

    def _decode_audio(self, source, name):
        audio, _ = librosa.load(
            source,
            sr=16000,
            mono=True,
            duration=self.ref_max_seconds,
        )
        if audio.size == 0:
            raise ValueError(f"ref audio decoded to nothing: {name}")
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _has_speaker_ref(self, item):
        explicit = item.get("ref_audio") or item.get("ref_audio_path")
        target_audio_path = self._target_audio_path(item)
        if explicit is not None:
            return explicit != target_audio_path
        speaker_key = self._speaker_key(item)
        if speaker_key is None:
            return False
        if self.ref_store is not None:
            exclude = item.get("id")
            return self.ref_store.has_usable_ref(
                speaker_key,
                exclude=None if exclude is None else str(exclude),
            )
        return self._speaker_audio_path(item) is not None

    def _speaker_audio_path(self, item, index=None):
        """Reference clip for `item`, or None when the speaker has no other clip.

        With `index` given, the clip is drawn from a per-index seeded rotation so
        a speaker's rows spread over all of its clips instead of piling onto the
        first one.  Filtering calls this without an index: only the existence of
        a distinct clip matters there, not which one.
        """
        explicit = item.get("ref_audio") or item.get("ref_audio_path")
        target_audio_path = self._target_audio_path(item)
        if explicit is not None:
            return explicit if explicit != target_audio_path else None
        speaker_key = self._speaker_key(item)
        if speaker_key is None:
            return None
        if self.ref_store is not None:
            rng = None
            if index is not None:
                rng = random.Random(self.seed + index)
            exclude = None
            target_id = item.get("id")
            if target_id is not None:
                exclude = str(target_id)
            return self.ref_store.pick_path(
                speaker_key, exclude=exclude, rng=rng
            )
        candidates = self.speaker_audio_paths_by_id.get(speaker_key, ())
        if not candidates:
            return None
        start = 0
        if index is not None:
            start = random.Random(self.seed + index).randrange(len(candidates))
        for offset in range(len(candidates)):
            audio_path = candidates[(start + offset) % len(candidates)]
            if audio_path != target_audio_path:
                return audio_path
        return None

    def _semantic_codes(self, item):
        if "semantic_codes" in item:
            return torch.tensor(item["semantic_codes"], dtype=torch.long)
        required = (
            "semantic_code_path",
            "semantic_code_offset",
            "semantic_code_length",
        )
        if not all(key in item for key in required):
            raise ValueError(
                "Each sample needs 'semantic_codes' or compact semantic code fields."
            )
        path = item["semantic_code_path"]
        codes = self._semantic_code_cache.get(path)
        if codes is None:
            codes = np.memmap(path, dtype="<u2", mode="r")
            self._semantic_code_cache[path] = codes
        offset = int(item["semantic_code_offset"])
        length = int(item["semantic_code_length"])
        if offset < 0 or length <= 0 or offset + length > len(codes):
            raise ValueError(
                "Compact semantic code range is out of bounds: "
                f"path={path}, offset={offset}, length={length}, "
                f"available={len(codes)}."
            )
        return torch.tensor(codes[offset : offset + length], dtype=torch.long)

    def collate_fn(self, samples):
        batch_size = len(samples)
        max_text = max(x["text_input_ids"].numel() for x in samples)
        max_speech = max(x["speech_input_ids"].numel() for x in samples)
        text_pad = self.tokenizer.pad_token_id
        if text_pad is None:
            text_pad = self.tokenizer.eos_token_id
        if text_pad is None:
            raise ValueError(
                "The Qwen tokenizer must define pad_token_id or eos_token_id."
            )

        text_ids = torch.full((batch_size, max_text), text_pad, dtype=torch.long)
        text_mask = torch.zeros((batch_size, max_text), dtype=torch.long)
        speech_ids = torch.full(
            (batch_size, max_speech),
            self.speech_pad_token_id,
            dtype=torch.long,
        )
        speech_mask = torch.zeros((batch_size, max_speech), dtype=torch.long)
        labels = torch.full((batch_size, max_speech), -100, dtype=torch.long)

        for row, sample in enumerate(samples):
            text_length = sample["text_input_ids"].numel()
            speech_length = sample["speech_input_ids"].numel()
            text_ids[row, :text_length] = sample["text_input_ids"]
            text_mask[row, :text_length] = 1
            speech_ids[row, :speech_length] = sample["speech_input_ids"]
            speech_mask[row, :speech_length] = 1
            labels[row, :speech_length] = sample["labels"]

        batch = {
            "text_input_ids": text_ids,
            "text_attention_mask": text_mask,
            "speech_input_ids": speech_ids,
            "speech_attention_mask": speech_mask,
            "labels": labels,
        }
        # In-memory refs travel as waveforms; the loose-file path still sends
        # paths for the encoder to open itself.
        if samples[0].get("speaker_audio") is not None:
            waveforms = [sample["speaker_audio"] for sample in samples]
            if self.speaker_mel_extractor is None:
                batch["speaker_waveforms"] = waveforms
            else:
                features, mask = self.speaker_mel_extractor(
                    waveforms,
                    max_audio_seconds=self.ref_max_seconds,
                )
                batch["speaker_input_features"] = features
                batch["speaker_feature_attention_mask"] = mask
        else:
            batch["speaker_audio_paths"] = [
                sample["speaker_audio_path"] for sample in samples
            ]
        return batch

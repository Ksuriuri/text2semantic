# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Speaker-contiguous ref shards for text2semantic training.

Layout (relative to the trainset root)::

    refs/speaker_index.jsonl
    refs/shards/refs-000123.tar

Index row::

    {"language": "zh", "speaker_id": "spk_1",
     "shard": "refs-000123.tar",
     "members": ["spk_1/000.flac", "spk_1/001.flac"]}

A speaker's refs always live in one shard. Training picks among ``members``
(``refs_per_speaker``; 0 means all packed refs). Members are extracted into a
local cache so the existing W2V-BERT file encoder keeps working.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path


def speaker_key_from_row(row):
    speaker_id = row.get("speaker_id")
    if speaker_id is None:
        return None
    language = row.get("language") or row.get("lang")
    return language, str(speaker_id)


def default_ref_index_path(train_jsonl):
    """Discover ``refs/speaker_index.jsonl`` next to a manifests/ jsonl."""
    start = Path(train_jsonl).expanduser().resolve().parent
    names = ("speaker_index.jsonl", "speaker_index.parquet")
    for base in (start.parent / "refs", start / "refs"):
        for name in names:
            candidate = base / name
            if candidate.is_file():
                return candidate
    return None


def default_ref_root(index_path):
    index_path = Path(index_path).expanduser().resolve()
    if index_path.parent.name == "refs":
        return index_path.parent
    return index_path.parent


class SpeakerRefStore:
    def __init__(
        self,
        index_path,
        *,
        ref_root=None,
        cache_dir=None,
        refs_per_speaker=0,
    ):
        self.index_path = Path(index_path).expanduser().resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(f"ref index not found: {self.index_path}")
        self.ref_root = (
            Path(ref_root).expanduser().resolve()
            if ref_root
            else default_ref_root(self.index_path)
        )
        self.shard_dir = (
            self.ref_root / "shards"
            if (self.ref_root / "shards").is_dir()
            else self.ref_root
        )
        if refs_per_speaker < 0:
            raise ValueError("refs_per_speaker must be >= 0 (0 = all packed).")
        self.refs_per_speaker = int(refs_per_speaker)
        if cache_dir is None:
            cache_dir = os.environ.get(
                "T2S_REF_CACHE", str(self.ref_root / ".extract-cache")
            )
        self.cache_dir = Path(cache_dir)
        self._index = self._load_index(self.index_path)

    def __len__(self):
        return len(self._index)

    def __contains__(self, key):
        return self._lookup(key) is not None

    def members(self, key):
        entry = self._lookup(key)
        if entry is None:
            return ()
        names = list(entry["members"])
        if self.refs_per_speaker > 0:
            names = names[: self.refs_per_speaker]
        return tuple(names)

    def has_usable_ref(self, key, *, exclude=None):
        return self.pick_member(key, exclude=exclude) is not None

    def pick_member(self, key, *, exclude=None, rng=None):
        names = list(self.members(key))
        if exclude:
            names = [name for name in names if name != exclude]
        if not names:
            return None
        if rng is None:
            return names[0]
        return names[rng.randrange(len(names))]

    def pick_path(self, key, *, exclude=None, rng=None):
        entry = self._lookup(key)
        member = self.pick_member(key, exclude=exclude, rng=rng)
        if entry is None or member is None:
            return None
        return self.extract_member(entry["shard"], member)

    def extract_member(self, shard_name, member):
        member = str(member).replace("\\", "/").lstrip("/")
        if not member or member.startswith("../") or "/../" in member:
            raise ValueError(f"unsafe ref member {member!r} in {shard_name}")
        dest = self.cache_dir / Path(shard_name).stem / member
        if dest.is_file():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shard_path = self._shard_path(shard_name)
        with tarfile.open(shard_path, "r:*") as archive:
            try:
                info = archive.getmember(member)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"{member} not in {shard_path}"
                ) from exc
            if not info.isfile():
                raise ValueError(f"ref member is not a file: {member}")
            extracted = archive.extractfile(info)
            if extracted is None:
                raise FileNotFoundError(f"cannot read {member} from {shard_path}")
            payload = extracted.read()
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(payload)
        tmp.replace(dest)
        return dest

    def _lookup(self, key):
        if key is None:
            return None
        language, speaker_id = key
        speaker_id = str(speaker_id)
        direct = self._index.get((language, speaker_id))
        if direct is not None:
            return direct
        if language is not None:
            return self._index.get((None, speaker_id))
        return None

    def _shard_path(self, shard_name):
        name = Path(shard_name)
        if name.is_absolute() and name.is_file():
            return name
        for candidate in (
            self.shard_dir / name.name,
            self.ref_root / name.name,
            self.ref_root / shard_name,
        ):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"ref shard {shard_name} not found under {self.shard_dir}"
        )

    @staticmethod
    def _load_index(path):
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return SpeakerRefStore._load_parquet(path)
        if suffix in {".jsonl", ".json"}:
            return SpeakerRefStore._load_jsonl(path)
        raise ValueError(f"unsupported ref index type: {path}")

    @staticmethod
    def _load_jsonl(path):
        index = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path} on line {line_number}: {exc}"
                    ) from exc
                SpeakerRefStore._add_row(index, row, source=f"{path}:{line_number}")
        if not index:
            raise ValueError(f"{path} contains no speaker rows.")
        return index

    @staticmethod
    def _load_parquet(path):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Reading speaker_index.parquet requires pyarrow."
            ) from exc
        table = pq.read_table(path)
        index = {}
        for row in table.to_pylist():
            SpeakerRefStore._add_row(index, row, source=str(path))
        if not index:
            raise ValueError(f"{path} contains no speaker rows.")
        return index

    @staticmethod
    def _add_row(index, row, *, source):
        speaker_id = row.get("speaker_id")
        if speaker_id is None or speaker_id == "":
            raise ValueError(f"missing speaker_id in {source}")
        shard = row.get("shard") or row.get("shard_name")
        members = row.get("members") or row.get("member") or []
        if isinstance(members, str):
            members = [members]
        members = [str(item) for item in members if item]
        if not shard or not members:
            raise ValueError(f"missing shard/members in {source}")
        language = row.get("language")
        if language is None:
            language = row.get("lang")
        key = (language, str(speaker_id))
        index[key] = {"shard": str(shard), "members": members}

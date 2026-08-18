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
(``refs_per_speaker``; 0 means all packed refs).

``read_member`` returns a member's bytes straight out of the shard, using the
offsets from :mod:`finetuning.ref_member_index`, so nothing is written to disk:
a full extraction would be a second 8.5 TB copy of the refs, and at one training
epoch every extracted file would be read about once, so the copy would never pay
for itself. ``extract_member``/``pick_path`` keep the on-disk behaviour for the
loose-file encoder path and for tests.
"""

from __future__ import annotations

import json
import os
import tarfile
from collections import OrderedDict
from pathlib import Path

from finetuning import ref_member_index, speaker_index as speaker_index_table


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
        member_index_dir=None,
        max_open_shards=8,
        max_member_indexes=64,
        index_backend="auto",
        build_index_if_missing=True,
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
        self.member_index_dir = Path(
            member_index_dir
            if member_index_dir is not None
            else ref_member_index.index_dir_for(self.ref_root)
        )
        self.max_open_shards = max(1, int(max_open_shards))
        self.max_member_indexes = max(1, int(max_member_indexes))
        # Both caches are per process: DataLoader workers are forked, and an
        # inherited file handle or parsed index would be shared state.
        self._member_indexes = OrderedDict()
        self._shard_handles = OrderedDict()
        self._owner_pid = os.getpid()
        # A big speaker index stays on disk: t2s-v1's is 5.2 GB of json, which
        # becomes ~12 GB of dict per process, once per rank and per worker.
        if index_backend == "auto":
            index_backend = (
                "memmap"
                if self.index_path.suffix.lower() in {".jsonl", ".json"}
                and self.index_path.stat().st_size
                > speaker_index_table.MEMMAP_THRESHOLD_BYTES
                else "ram"
            )
        if index_backend not in {"memmap", "ram"}:
            raise ValueError(f"unknown index_backend {index_backend!r}")
        self.index_backend = index_backend
        if index_backend == "memmap":
            self._index = None
            self._table = speaker_index_table.load(
                self.index_path,
                build_if_missing=build_index_if_missing,
                log=None,
            )
        else:
            self._table = None
            self._index = self._load_index(self.index_path)

    def __len__(self):
        return len(self._table if self._index is None else self._index)

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

    def ref_ids(self, key):
        """The row ids these members are, for comparing against ``exclude``."""
        return tuple(ref_member_index.member_row_id(name) for name in self.members(key))

    def pick_member(self, key, *, exclude=None, rng=None):
        names = list(self.members(key))
        if exclude:
            # A shard member is named after the row it came from
            # (<dataset>/<language>/<speaker>/<id>.flac), so the row id has to be
            # compared against that name's id and not against the whole path --
            # otherwise nothing ever matches and a clip can end up as its own
            # ref, which teaches the model to copy.
            exclude = str(exclude)
            names = [
                name
                for name in names
                if name != exclude
                and ref_member_index.member_row_id(name) != exclude
            ]
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

    def pick_ref(self, key, *, exclude=None, rng=None):
        """``(shard name, member name)`` for a usable ref, or None."""
        entry = self._lookup(key)
        member = self.pick_member(key, exclude=exclude, rng=rng)
        if entry is None or member is None:
            return None
        return entry["shard"], member

    def read_ref(self, key, *, exclude=None, rng=None):
        """``(member name, bytes)`` for a usable ref, or None. Nothing is written."""
        picked = self.pick_ref(key, exclude=exclude, rng=rng)
        if picked is None:
            return None
        shard_name, member = picked
        return member, self.read_member(shard_name, member)

    def read_member(self, shard_name, member):
        """A member's bytes, read in place from the shard."""
        member = self._safe_member(member, shard_name)
        shard_path = self._shard_path(shard_name)
        members = self._member_index(shard_path)
        location = members.get(member)
        if location is None:
            raise FileNotFoundError(f"{member} not in {shard_path}")
        offset, size = location
        payload = os.pread(self._shard_handle(shard_path), size, offset)
        if len(payload) != size:
            raise OSError(
                f"short read for {member} in {shard_path}: "
                f"{len(payload)} of {size} bytes"
            )
        return payload

    def _member_index(self, shard_path):
        self._reset_after_fork()
        key = str(shard_path)
        cached = self._member_indexes.get(key)
        if cached is not None:
            self._member_indexes.move_to_end(key)
            return cached
        members = ref_member_index.read_shard_index(
            shard_path, self.member_index_dir
        )
        self._member_indexes[key] = members
        while len(self._member_indexes) > self.max_member_indexes:
            self._member_indexes.popitem(last=False)
        return members

    def _shard_handle(self, shard_path):
        self._reset_after_fork()
        key = str(shard_path)
        handle = self._shard_handles.get(key)
        if handle is not None:
            self._shard_handles.move_to_end(key)
            return handle
        handle = os.open(shard_path, os.O_RDONLY)
        self._shard_handles[key] = handle
        while len(self._shard_handles) > self.max_open_shards:
            _, evicted = self._shard_handles.popitem(last=False)
            os.close(evicted)
        return handle

    def _reset_after_fork(self):
        pid = os.getpid()
        if pid == self._owner_pid:
            return
        # The parent's descriptors belong to the parent; drop them without
        # closing, since the parent is still using them.
        self._shard_handles = OrderedDict()
        self._member_indexes = OrderedDict()
        self._owner_pid = pid

    @staticmethod
    def _safe_member(member, shard_name):
        member = str(member).replace("\\", "/").lstrip("/")
        if not member or member.startswith("../") or "/../" in member:
            raise ValueError(f"unsafe ref member {member!r} in {shard_name}")
        return member

    def extract_member(self, shard_name, member):
        member = self._safe_member(member, shard_name)
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
        if self._index is None:
            return self._table.get((language, speaker_id))
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

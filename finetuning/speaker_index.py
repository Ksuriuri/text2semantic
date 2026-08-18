# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Memmapped lookup over a large refs/speaker_index.jsonl.

t2s-v1 ships a 5.2 GB speaker index (~11.6M speakers, each with a shard name and
a handful of member paths). Loaded into a dict that is roughly 12 GB of Python
objects per process, and eight ranks with their DataLoader workers would each
end up with their own copy as refcounting dirties the shared pages.

So the rows stay on disk and only a sorted key table is built beside them::

    <index>.index/meta.json
    <index>.index/keys.blob         NUL-joined key fields, sorted
    <index>.index/key_offsets.u64   where each key starts in keys.blob
    <index>.index/row_offsets.u64   where the speaker's json row starts
    <index>.index/row_lengths.u32   how long that row is

A lookup binary-searches the memmapped key table and parses exactly one row, so
every rank shares one page cache copy and holds nothing per process but a small
LRU of recently used speakers.

Which fields make up the key and what a row means are the caller's business:
the packed refs key on ``(language, speaker_id)`` and hold a shard plus member
names, while the source-tar refs key on ``(dataset, language, speaker_id)`` and
hold ranged locations (see finetuning/source_ref_store.py). The key fields are
recorded in meta.json, because asking for different ones has to rebuild the
table even though the source file has not changed at all.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from array import array
from collections import OrderedDict
from pathlib import Path

import numpy as np

from finetuning import index_build

try:  # matches finetuning.manifest_index; the stdlib fallback is fine
    import orjson

    def _loads(payload):
        return orjson.loads(payload)

except ImportError:  # pragma: no cover - depends on the environment

    def _loads(payload):
        return json.loads(payload)


FORMAT_VERSION = 2
META_NAME = "meta.json"
KEYS_NAME = "keys.blob"
KEY_OFFSETS_NAME = "key_offsets.u64"
ROW_OFFSETS_NAME = "row_offsets.u64"
ROW_LENGTHS_NAME = "row_lengths.u32"

# Below this a plain dict costs little and keeps the code path short.
MEMMAP_THRESHOLD_BYTES = 64 << 20


# The packed refs identify a speaker by language; the source-tar refs need the
# dataset too, because a speaker_id like "0001" exists in several datasets and
# they are not the same person.
DEFAULT_KEY_FIELDS = ("language", "speaker_id")


def index_dir_for(jsonl_path):
    return Path(str(Path(jsonl_path).expanduser().resolve()) + ".index")


def encode_key(*parts):
    """Sortable key bytes; a missing part is the empty string, never absent."""
    return "\x00".join("" if part is None else str(part) for part in parts).encode(
        "utf-8"
    )


def row_key_parts(row, key_fields):
    parts = []
    for field in key_fields:
        value = row.get(field)
        if not value and field == "language":
            value = row.get("lang")
        # An empty field is the same as an absent one, which is how the key has
        # always treated a missing language; encode_key writes both as "".
        parts.append(value if value or field == "speaker_id" else None)
    return parts


def row_key(row, key_fields=DEFAULT_KEY_FIELDS):
    """The lookup key for a manifest row, or None when it names no speaker.

    Every part is a string or None, so a manifest with integer speaker ids keys
    the same way the index does.
    """
    speaker_id = row.get("speaker_id")
    if speaker_id is None or speaker_id == "":
        return None
    return tuple(
        None if part is None else str(part) for part in row_key_parts(row, key_fields)
    )


def shard_members_row(row):
    """The packed-refs row shape: one shard, a few member names."""
    members = row.get("members") or row.get("member") or []
    if isinstance(members, str):
        members = [members]
    return {
        "shard": str(row.get("shard") or row.get("shard_name")),
        "members": [str(item) for item in members if item],
    }


def build(jsonl_path, index_dir=None, *, key_fields=DEFAULT_KEY_FIELDS, log=print):
    """Write the sorted key table for `jsonl_path` and return its directory."""
    key_fields = tuple(key_fields)
    jsonl_path = Path(jsonl_path).expanduser().resolve()
    index_dir = (
        Path(index_dir).expanduser().resolve()
        if index_dir is not None
        else index_dir_for(jsonl_path)
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / META_NAME
    if meta_path.exists():
        meta_path.unlink()

    entries = []
    started = time.monotonic()
    with open(jsonl_path, "rb") as handle:
        offset = 0
        for line in handle:
            length = len(line)
            start = offset
            offset += length
            stripped = line.strip()
            if not stripped:
                continue
            row = _loads(stripped)
            speaker_id = row.get("speaker_id")
            if speaker_id is None or speaker_id == "":
                raise ValueError(
                    f"missing speaker_id in {jsonl_path} at byte {start}"
                )
            entries.append(
                (
                    encode_key(*row_key_parts(row, key_fields)),
                    start,
                    len(line.rstrip(b"\r\n")),
                )
            )
            if log is not None and len(entries) % 2_000_000 == 0:
                log(
                    f"  read {len(entries):,} speakers, "
                    f"{time.monotonic() - started:.0f}s"
                )
    if not entries:
        raise ValueError(f"{jsonl_path} contains no speaker rows.")
    entries.sort(key=lambda entry: entry[0])

    key_offsets = array("Q")
    row_offsets = array("Q")
    row_lengths = array("I")
    publisher = index_build.IndexPublisher(index_dir)
    with publisher, open(publisher.path(KEYS_NAME), "wb") as keys_out:
        for key, row_offset, row_length in entries:
            key_offsets.append(keys_out.tell())
            keys_out.write(key)
            row_offsets.append(row_offset)
            row_lengths.append(row_length)
        key_offsets.append(keys_out.tell())
        publisher.path(KEY_OFFSETS_NAME).write_bytes(key_offsets.tobytes())
        publisher.path(ROW_OFFSETS_NAME).write_bytes(row_offsets.tobytes())
        publisher.path(ROW_LENGTHS_NAME).write_bytes(row_lengths.tobytes())
    publisher.publish()

    source = jsonl_path.stat()
    meta_path.write_text(
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "key_fields": list(key_fields),
                "source": str(jsonl_path),
                "source_size": source.st_size,
                "source_mtime_ns": source.st_mtime_ns,
                "speakers": len(entries),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if log is not None:
        log(
            f"Indexed {jsonl_path.name}: {len(entries):,} speakers in "
            f"{time.monotonic() - started:.0f}s -> {index_dir}"
        )
    return index_dir


class MemmappedSpeakerIndex:
    """Binary-searched, on-disk replacement for the speaker index dict."""

    def __init__(
        self,
        jsonl_path,
        index_dir=None,
        *,
        cache_rows=4096,
        row_adapter=shard_members_row,
    ):
        self.row_adapter = row_adapter
        self.jsonl_path = Path(jsonl_path).expanduser().resolve()
        self.index_dir = (
            Path(index_dir).expanduser().resolve()
            if index_dir is not None
            else index_dir_for(self.jsonl_path)
        )
        self.meta = json.loads((self.index_dir / META_NAME).read_text())
        self.key_fields = tuple(self.meta.get("key_fields") or DEFAULT_KEY_FIELDS)
        self.keys = np.memmap(self.index_dir / KEYS_NAME, dtype=np.uint8, mode="r")
        self.key_offsets = np.memmap(
            self.index_dir / KEY_OFFSETS_NAME, dtype="<u8", mode="r"
        )
        self.row_offsets = np.memmap(
            self.index_dir / ROW_OFFSETS_NAME, dtype="<u8", mode="r"
        )
        self.row_lengths = np.memmap(
            self.index_dir / ROW_LENGTHS_NAME, dtype="<u4", mode="r"
        )
        if len(self.key_offsets) != len(self.row_offsets) + 1:
            raise ValueError(f"corrupt speaker index: {self.index_dir}")
        self.cache_rows = max(1, int(cache_rows))
        self._rows = OrderedDict()
        self._fd = None
        self._fd_pid = None

    def __len__(self):
        return len(self.row_offsets)

    def __contains__(self, key):
        return self.get(key) is not None

    def get(self, key):
        """The adapted row for a key tuple matching ``key_fields``, or None."""
        if key is None:
            return None
        parts = list(key)
        if len(parts) != len(self.key_fields):
            raise ValueError(
                f"key {key!r} does not match key fields {self.key_fields}"
            )
        row = self._get_encoded(encode_key(*parts))
        if row is not None:
            return row
        # Same fallback as the in-memory index: a row written without a language
        # still answers a lookup that has one.
        try:
            language_at = self.key_fields.index("language")
        except ValueError:
            return None
        if parts[language_at] is None:
            return None
        parts[language_at] = None
        return self._get_encoded(encode_key(*parts))

    def _get_encoded(self, key):
        cached = self._rows.get(key)
        if cached is not None:
            self._rows.move_to_end(key)
            return cached
        position = self._search(key)
        if position is None:
            return None
        entry = self.row_adapter(_loads(self._read_row(position)))
        self._rows[key] = entry
        while len(self._rows) > self.cache_rows:
            self._rows.popitem(last=False)
        return entry

    def _key_at(self, position):
        start = int(self.key_offsets[position])
        end = int(self.key_offsets[position + 1])
        return self.keys[start:end].tobytes()

    def _search(self, key):
        low = 0
        high = len(self) - 1
        while low <= high:
            middle = (low + high) // 2
            candidate = self._key_at(middle)
            if candidate == key:
                return middle
            if candidate < key:
                low = middle + 1
            else:
                high = middle - 1
        return None

    def _read_row(self, position):
        pid = os.getpid()
        if self._fd is None or self._fd_pid != pid:
            self._fd = os.open(self.jsonl_path, os.O_RDONLY)
            self._fd_pid = pid
        return os.pread(
            self._fd,
            int(self.row_lengths[position]),
            int(self.row_offsets[position]),
        )


def _stale(meta, jsonl_path, key_fields=DEFAULT_KEY_FIELDS):
    if meta.get("format_version") != FORMAT_VERSION:
        return "format version changed"
    # Keys are what the binary search compares, and a table keyed on
    # (language, speaker_id) answers a (dataset, language, speaker_id) lookup
    # with someone else's refs rather than a miss. Nothing about the source file
    # changes when the caller asks for different fields, so check them here.
    if tuple(meta.get("key_fields") or DEFAULT_KEY_FIELDS) != tuple(key_fields):
        return "key fields changed"
    source = Path(jsonl_path).stat()
    if meta.get("source_size") != source.st_size:
        return "speaker index size changed"
    if meta.get("source_mtime_ns") != source.st_mtime_ns:
        return "speaker index mtime changed"
    return None


def load(
    jsonl_path,
    index_dir=None,
    *,
    key_fields=DEFAULT_KEY_FIELDS,
    row_adapter=shard_members_row,
    build_if_missing=True,
    rebuild=False,
    log=print,
):
    """Open the key table, building it when missing or stale.

    Safe from every rank on every node; the build is serialised by an exclusive
    lock on the index directory (see finetuning/index_build.py).
    """
    jsonl_path = Path(jsonl_path).expanduser().resolve()
    index_dir = (
        Path(index_dir).expanduser().resolve()
        if index_dir is not None
        else index_dir_for(jsonl_path)
    )
    key_fields = tuple(key_fields)
    reason = _build_reason(index_dir, jsonl_path, rebuild, key_fields)
    if reason is not None:
        if not build_if_missing:
            raise FileNotFoundError(
                f"speaker index table for {jsonl_path} is not usable "
                f"({reason}); build it on the main process first."
            )
        with index_build.build_lock(index_dir, log=log) as lock:
            if lock.waited:
                reason = _build_reason(
                    index_dir, jsonl_path, rebuild and not lock.waited, key_fields
                )
            if reason is not None:
                if log is not None:
                    log(
                        f"Building speaker index table for {jsonl_path.name}: "
                        f"{reason}"
                    )
                build(jsonl_path, index_dir, key_fields=key_fields, log=log)
    return MemmappedSpeakerIndex(jsonl_path, index_dir, row_adapter=row_adapter)


def _build_reason(index_dir, jsonl_path, rebuild, key_fields=DEFAULT_KEY_FIELDS):
    """Why this key table needs building, or None when it is usable."""
    if rebuild:
        return "rebuild requested"
    meta_path = Path(index_dir) / META_NAME
    if not meta_path.is_file():
        return "index missing"
    try:
        return _stale(json.loads(meta_path.read_text()), jsonl_path, key_fields)
    except (ValueError, OSError):
        return "index unreadable"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the memmapped key table for refs/speaker_index.jsonl."
    )
    parser.add_argument("speaker_index")
    parser.add_argument("--index_dir", default=None)
    parser.add_argument(
        "--key_field",
        action="append",
        dest="key_fields",
        default=None,
        help=(
            "row field to key on; repeat in order. Default: "
            f"{' '.join(DEFAULT_KEY_FIELDS)}"
        ),
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    load(
        args.speaker_index,
        args.index_dir,
        key_fields=tuple(args.key_fields or DEFAULT_KEY_FIELDS),
        rebuild=args.rebuild,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

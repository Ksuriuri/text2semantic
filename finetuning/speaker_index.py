# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Memmapped lookup over a large refs/speaker_index.jsonl.

t2s-v1 ships a 5.2 GB speaker index (~11.6M speakers, each with a shard name and
a handful of member paths). Loaded into a dict that is roughly 12 GB of Python
objects per process, and eight ranks with their DataLoader workers would each
end up with their own copy as refcounting dirties the shared pages.

So the rows stay on disk and only a sorted key table is built beside them::

    <index>.index/meta.json
    <index>.index/keys.blob         "<language>\\0<speaker_id>" keys, sorted
    <index>.index/key_offsets.u64   where each key starts in keys.blob
    <index>.index/row_offsets.u64   where the speaker's json row starts
    <index>.index/row_lengths.u32   how long that row is

A lookup binary-searches the memmapped key table and parses exactly one row, so
every rank shares one page cache copy and holds nothing per process but a small
LRU of recently used speakers.
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


FORMAT_VERSION = 1
META_NAME = "meta.json"
KEYS_NAME = "keys.blob"
KEY_OFFSETS_NAME = "key_offsets.u64"
ROW_OFFSETS_NAME = "row_offsets.u64"
ROW_LENGTHS_NAME = "row_lengths.u32"

# Below this a plain dict costs little and keeps the code path short.
MEMMAP_THRESHOLD_BYTES = 64 << 20


def index_dir_for(jsonl_path):
    return Path(str(Path(jsonl_path).expanduser().resolve()) + ".index")


def encode_key(language, speaker_id):
    language = "" if language is None else str(language)
    return f"{language}\x00{speaker_id}".encode("utf-8")


def build(jsonl_path, index_dir=None, *, log=print):
    """Write the sorted key table for `jsonl_path` and return its directory."""
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
            language = row.get("language")
            if language is None:
                language = row.get("lang")
            entries.append(
                (
                    encode_key(language, speaker_id),
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

    def __init__(self, jsonl_path, index_dir=None, *, cache_rows=4096):
        self.jsonl_path = Path(jsonl_path).expanduser().resolve()
        self.index_dir = (
            Path(index_dir).expanduser().resolve()
            if index_dir is not None
            else index_dir_for(self.jsonl_path)
        )
        self.meta = json.loads((self.index_dir / META_NAME).read_text())
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
        """``{"shard": ..., "members": [...]}`` for a (language, speaker_id) key."""
        if key is None:
            return None
        language, speaker_id = key
        row = self._get_encoded(encode_key(language, speaker_id))
        if row is None and language is not None:
            # Same fallback as the in-memory index: a row packed without a
            # language still answers a lookup that has one.
            row = self._get_encoded(encode_key(None, speaker_id))
        return row

    def _get_encoded(self, key):
        cached = self._rows.get(key)
        if cached is not None:
            self._rows.move_to_end(key)
            return cached
        position = self._search(key)
        if position is None:
            return None
        payload = self._read_row(position)
        row = _loads(payload)
        members = row.get("members") or row.get("member") or []
        if isinstance(members, str):
            members = [members]
        entry = {
            "shard": str(row.get("shard") or row.get("shard_name")),
            "members": [str(item) for item in members if item],
        }
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


def _stale(meta, jsonl_path):
    if meta.get("format_version") != FORMAT_VERSION:
        return "format version changed"
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
    reason = _build_reason(index_dir, jsonl_path, rebuild)
    if reason is not None:
        if not build_if_missing:
            raise FileNotFoundError(
                f"speaker index table for {jsonl_path} is not usable "
                f"({reason}); build it on the main process first."
            )
        with index_build.build_lock(index_dir, log=log) as lock:
            if lock.waited:
                reason = _build_reason(
                    index_dir, jsonl_path, rebuild and not lock.waited
                )
            if reason is not None:
                if log is not None:
                    log(
                        f"Building speaker index table for {jsonl_path.name}: "
                        f"{reason}"
                    )
                build(jsonl_path, index_dir, log=log)
    return MemmappedSpeakerIndex(jsonl_path, index_dir)


def _build_reason(index_dir, jsonl_path, rebuild):
    """Why this key table needs building, or None when it is usable."""
    if rebuild:
        return "rebuild requested"
    meta_path = Path(index_dir) / META_NAME
    if not meta_path.is_file():
        return "index missing"
    try:
        return _stale(json.loads(meta_path.read_text()), jsonl_path)
    except (ValueError, OSError):
        return "index unreadable"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the memmapped key table for refs/speaker_index.jsonl."
    )
    parser.add_argument("speaker_index")
    parser.add_argument("--index_dir", default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    load(args.speaker_index, args.index_dir, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

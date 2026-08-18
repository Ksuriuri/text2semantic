# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""One shared, memmapped row index per manifest jsonl.

Reading a manifest with ``read_jsonl`` keeps every row in a Python list, which
costs roughly 1.9 KB per 573-byte row: the 54 GB t2s-v1 train manifest needs
~174 GiB that way, so eight ranks on one node cannot hold even one copy between
them.  This module builds a small on-disk index instead::

    <manifest>.index/meta.json      build parameters and row counts
    <manifest>.index/offsets.u64    byte offset of each kept row
    <manifest>.index/lengths.u32    byte length of each kept row

Every rank memmaps the same two arrays and ``pread``s rows out of the same
jsonl, so the node holds a single copy in page cache no matter how many ranks
and DataLoader workers read it.  1.1 GB of index for 94M rows, and a row is
parsed only when it is actually sampled.

Filtering runs once, at build time, and only surviving rows get an entry, so
``__getitem__`` does no filtering work and the training loop never sees a row it
cannot use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from finetuning import index_build, speaker_index

try:  # orjson parses these rows ~4x faster; the stdlib is a fine fallback.
    import orjson

    def _loads(payload):
        return orjson.loads(payload)

except ImportError:  # pragma: no cover - depends on the environment

    def _loads(payload):
        return json.loads(payload)


# 2: the ref probe compares row ids instead of member names, so an index built
# by version 1 keeps rows whose only ref is the row itself.
FORMAT_VERSION = 2
META_NAME = "meta.json"
OFFSETS_NAME = "offsets.u64"
LENGTHS_NAME = "lengths.u32"
_FLUSH_ROWS = 1 << 20


def index_dir_for(jsonl_path):
    return Path(str(Path(jsonl_path).expanduser().resolve()) + ".index")


@dataclass(frozen=True)
class FilterParams:
    """Everything that decides whether a manifest row is kept.

    Recorded in ``meta.json`` so a run that changes, say, ``min_target_seconds``
    rebuilds the index instead of silently training on the previous filter.
    """

    min_target_seconds: float | None = 0.5
    max_target_seconds: float | None = 30.0
    max_semantic_tokens: int | None = None
    min_speaker_records: int = 2
    refs_per_speaker: int = 0
    ref_index: str | None = None
    # Comma-joined rather than a tuple: as_meta() goes through json, and a tuple
    # would come back as a list and compare unequal to itself on every load.
    speaker_key: str = ",".join(speaker_index.DEFAULT_KEY_FIELDS)

    def as_meta(self):
        return asdict(self)

    @property
    def speaker_key_fields(self):
        return tuple(field for field in self.speaker_key.split(",") if field)


class ManifestIndex:
    """Random access to the kept rows of a manifest jsonl.

    ``prefiltered`` tells :class:`Text2SemanticDataset` that the filtering has
    already happened, so it must not walk the rows again.
    """

    prefiltered = True

    def __init__(self, jsonl_path, index_dir=None, *, code_root=None):
        self.jsonl_path = Path(jsonl_path).expanduser().resolve()
        self.index_dir = (
            Path(index_dir).expanduser().resolve()
            if index_dir is not None
            else index_dir_for(self.jsonl_path)
        )
        self.meta = json.loads((self.index_dir / META_NAME).read_text())
        self.offsets = np.memmap(
            self.index_dir / OFFSETS_NAME, dtype="<u8", mode="r"
        )
        self.lengths = np.memmap(
            self.index_dir / LENGTHS_NAME, dtype="<u4", mode="r"
        )
        if len(self.offsets) != len(self.lengths):
            raise ValueError(f"corrupt manifest index: {self.index_dir}")
        # Compact code paths are stored relative to the manifests/ directory
        # ("../codes/ears/ears-000003.u2.bin"), so they only resolve against the
        # manifest, never against whatever directory the job was launched from.
        self.code_root = (
            Path(code_root).expanduser().resolve()
            if code_root is not None
            else self.jsonl_path.parent
        )
        self.raw_size = int(self.meta.get("raw_rows", len(self.offsets)))
        self.filtered_size = self.raw_size - len(self.offsets)
        self._path_cache = {}
        self._fd = None
        self._fd_pid = None

    def __len__(self):
        return len(self.offsets)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        count = len(self)
        index = int(index)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError(f"row {index} out of range ({count} rows)")
        payload = os.pread(
            self._handle(),
            int(self.lengths[index]),
            int(self.offsets[index]),
        )
        row = _loads(payload)
        path = row.get("semantic_code_path")
        if path is not None:
            row["semantic_code_path"] = self._resolve(path)
        return row

    def _handle(self):
        # DataLoader workers are forked, and a file descriptor shared across
        # processes shares its offset too; pread does not use it, but a stale fd
        # from before the fork is still worth avoiding.
        pid = os.getpid()
        if self._fd is None or self._fd_pid != pid:
            self._fd = os.open(self.jsonl_path, os.O_RDONLY)
            self._fd_pid = pid
        return self._fd

    def _resolve(self, path):
        resolved = self._path_cache.get(path)
        if resolved is None:
            if os.path.isabs(path):
                resolved = path
            else:
                resolved = os.path.normpath(
                    os.path.join(self.code_root, path)
                )
            self._path_cache[path] = resolved
        return resolved


def _speaker_key(row, key_fields=speaker_index.DEFAULT_KEY_FIELDS):
    return speaker_index.row_key(row, key_fields)


def _fast_string_field(line, field):
    """A flat ``"field": "value"`` out of the raw line, or None if not found.

    Returns ``False`` for a value this shortcut must not decode (an escape), so
    the caller can tell "absent" from "give up and parse properly".
    """
    marker = f'"{field}": "'
    start = line.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = line.find('"', start)
    if end < 0:
        return False
    value = line[start:end]
    return False if "\\" in value else value


def _fast_speaker_key(line, key_fields=speaker_index.DEFAULT_KEY_FIELDS):
    """The key tuple straight out of the raw line, or None.

    The packer writes flat rows with string values, so pulling the fields out
    textually skips a full parse of every row in the counting pass.  A row that
    does not match this shape falls back to the real parser.
    """
    parts = []
    for field in key_fields:
        value = _fast_string_field(line, field)
        if not value and field == "language":
            # Same fallback as the parsed path; without it a manifest that
            # writes "lang" would count under one key and filter under another,
            # and every row would look like a single-record speaker.
            value = _fast_string_field(line, "lang")
        if value is False:
            return None
        if field == "speaker_id" and not value:
            # Required, and an empty one keys as nothing at all.
            return None
        parts.append(value or None)
    return tuple(parts)


def _count_speakers(jsonl_path, log, key_fields=speaker_index.DEFAULT_KEY_FIELDS):
    counts = {}
    started = time.monotonic()
    with open(jsonl_path, encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            key = _fast_speaker_key(line, key_fields)
            if key is None:
                key = _speaker_key(_loads(line), key_fields)
            if key is not None:
                counts[key] = counts.get(key, 0) + 1
            if log is not None and row_number % (20 * _FLUSH_ROWS) == 0:
                log(
                    f"  counting speakers: {row_number:,} rows, "
                    f"{len(counts):,} speakers, "
                    f"{time.monotonic() - started:.0f}s"
                )
    return counts


class _RefProbe:
    """Cached 'does this speaker have a ref other than `exclude`' lookup.

    ``SpeakerRefStore.has_usable_ref`` rebuilds the member list on every call,
    which is far too slow across 94M rows; a speaker's ref count answers the
    question outright unless it has exactly one ref.

    The store is asked for row ids rather than member names, because ``exclude``
    is a row id: a speaker whose only ref is the row being filtered has no
    usable ref, and keeping that row would fail at read time instead.
    """

    def __init__(self, ref_store):
        self.ref_store = ref_store
        self._ids = getattr(ref_store, "ref_ids", None) or ref_store.members
        self._cache = {}

    def usable(self, key, exclude):
        if key is None:
            return False
        entry = self._cache.get(key)
        if entry is None:
            ids = self._ids(key)
            entry = (len(ids), ids[0] if len(ids) == 1 else None)
            self._cache[key] = entry
        count, only = entry
        if count == 0:
            return False
        if count > 1:
            return True
        return only != exclude


def _keep(row, params, speaker_counts, ref_probe):
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return False
    key = _speaker_key(row, params.speaker_key_fields)
    if ref_probe is not None:
        target_id = row.get("id")
        if not ref_probe.usable(
            key, None if target_id is None else str(target_id)
        ):
            return False
    elif key is None:
        return False
    if (
        key is not None
        and params.min_speaker_records > 1
        and speaker_counts.get(key, 0) < params.min_speaker_records
    ):
        return False
    duration = row.get("duration")
    if duration is not None:
        duration = float(duration)
        if (
            params.max_target_seconds is not None
            and duration > params.max_target_seconds
        ):
            return False
        if (
            params.min_target_seconds is not None
            and duration < params.min_target_seconds
        ):
            return False
    if params.max_semantic_tokens is not None:
        length = row.get("semantic_code_length")
        if length is not None and int(length) > params.max_semantic_tokens:
            return False
    return True


def build(
    jsonl_path,
    *,
    params,
    ref_store=None,
    index_dir=None,
    log=print,
):
    """Write the row index for `jsonl_path` and return its directory.

    ``meta.json`` is written last, so a build that dies half way leaves an
    index that :func:`load` rejects rather than one it half trusts.
    """
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
    publisher = index_build.IndexPublisher(index_dir)

    speaker_counts = {}
    if params.min_speaker_records > 1:
        if log is not None:
            log(f"Indexing {jsonl_path}: counting speaker records")
        speaker_counts = _count_speakers(
            jsonl_path, log, params.speaker_key_fields
        )
        if log is not None:
            log(f"  {len(speaker_counts):,} speakers")

    ref_probe = None if ref_store is None else _RefProbe(ref_store)
    offsets = array("Q")
    lengths = array("I")
    # The arrays are written in native layout and read back as "<u8"/"<u4".
    if sys.byteorder != "little" or offsets.itemsize != 8 or lengths.itemsize != 4:
        raise RuntimeError("manifest_index needs little-endian 8/4-byte arrays.")
    raw_rows = 0
    started = time.monotonic()
    with publisher, open(publisher.path(OFFSETS_NAME), "wb") as offsets_out, open(
        publisher.path(LENGTHS_NAME), "wb"
    ) as lengths_out:

        def flush():
            offsets_out.write(offsets.tobytes())
            lengths_out.write(lengths.tobytes())
            del offsets[:]
            del lengths[:]

        with open(jsonl_path, "rb") as handle:
            offset = 0
            for line in handle:
                length = len(line)
                start = offset
                offset += length
                stripped = line.strip()
                if not stripped:
                    continue
                raw_rows += 1
                row = _loads(stripped)
                if not _keep(row, params, speaker_counts, ref_probe):
                    continue
                offsets.append(start)
                lengths.append(len(line.rstrip(b"\r\n")))
                if len(offsets) >= _FLUSH_ROWS:
                    flush()
                    if log is not None:
                        log(
                            f"  indexed {raw_rows:,} rows, "
                            f"{offsets_out.tell() // 8:,} kept, "
                            f"{time.monotonic() - started:.0f}s"
                        )
            flush()
        kept = offsets_out.tell() // 8

    if kept == 0:
        publisher.discard()
        raise ValueError(
            f"No usable rows in {jsonl_path} under {params.as_meta()}."
        )
    # Only now do the row arrays become visible; meta.json below is what makes
    # them usable.
    publisher.publish()
    source = jsonl_path.stat()
    meta = {
        "format_version": FORMAT_VERSION,
        "source": str(jsonl_path),
        "source_size": source.st_size,
        "source_mtime_ns": source.st_mtime_ns,
        "raw_rows": raw_rows,
        "kept_rows": kept,
        "speakers": len(speaker_counts),
        "filters": params.as_meta(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    if log is not None:
        log(
            f"Indexed {jsonl_path}: kept {kept:,}/{raw_rows:,} rows in "
            f"{time.monotonic() - started:.0f}s -> {index_dir}"
        )
    return index_dir


def _stale(meta, jsonl_path, params):
    if meta.get("format_version") != FORMAT_VERSION:
        return "format version changed"
    source = Path(jsonl_path).stat()
    if meta.get("source_size") != source.st_size:
        return "manifest size changed"
    if meta.get("source_mtime_ns") != source.st_mtime_ns:
        return "manifest mtime changed"
    if meta.get("filters") != params.as_meta():
        return "filter parameters changed"
    return None


def load(
    jsonl_path,
    *,
    params,
    ref_store=None,
    index_dir=None,
    code_root=None,
    rebuild=False,
    log=print,
):
    """Open the index for `jsonl_path`, building it when it is missing or stale.

    Safe to call from every rank on every node: the build takes an exclusive
    lock on the index directory, and a rank that waited for that lock re-checks
    staleness rather than rebuilding what it just waited for.  Calling it on one
    process first is still cheaper -- the other ranks then find the index
    already there -- but it is no longer a correctness requirement.
    """
    jsonl_path = Path(jsonl_path).expanduser().resolve()
    index_dir = (
        Path(index_dir).expanduser().resolve()
        if index_dir is not None
        else index_dir_for(jsonl_path)
    )
    reason = _build_reason(index_dir, jsonl_path, params, rebuild)
    if reason is not None:
        with index_build.build_lock(index_dir, log=log) as lock:
            if lock.waited:
                # Whoever held the lock was almost certainly building this same
                # index; a second build would only overwrite it with itself.
                reason = _build_reason(
                    index_dir, jsonl_path, params, rebuild and not lock.waited
                )
            if reason is not None:
                if log is not None:
                    log(f"Building manifest index for {jsonl_path.name}: {reason}")
                build(
                    jsonl_path,
                    params=params,
                    ref_store=ref_store,
                    index_dir=index_dir,
                    log=log,
                )
    return ManifestIndex(jsonl_path, index_dir, code_root=code_root)


def _build_reason(index_dir, jsonl_path, params, rebuild):
    """Why this index needs building, or None when it is usable as it stands."""
    if rebuild:
        return "rebuild requested"
    meta_path = Path(index_dir) / META_NAME
    if not meta_path.is_file():
        return "index missing"
    try:
        return _stale(json.loads(meta_path.read_text()), jsonl_path, params)
    except (ValueError, OSError):
        return "index unreadable"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the shared row index for a manifest jsonl."
    )
    parser.add_argument("manifest")
    parser.add_argument("--index_dir", default=None)
    parser.add_argument("--min_target_seconds", type=float, default=0.5)
    parser.add_argument("--max_target_seconds", type=float, default=30.0)
    parser.add_argument("--max_semantic_tokens", type=int, default=None)
    parser.add_argument("--min_speaker_records", type=int, default=2)
    parser.add_argument("--refs_per_speaker", type=int, default=0)
    parser.add_argument("--ref_index", default=None)
    parser.add_argument("--loose_refs", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    ref_store = None
    ref_index = args.ref_index
    if not args.loose_refs:
        from finetuning.ref_store import (
            SpeakerRefStore,
            default_ref_index_path,
        )

        if ref_index is None:
            discovered = default_ref_index_path(args.manifest)
            ref_index = None if discovered is None else str(discovered)
        if ref_index is not None:
            ref_store = SpeakerRefStore(
                ref_index, refs_per_speaker=args.refs_per_speaker
            )
    params = FilterParams(
        min_target_seconds=args.min_target_seconds,
        max_target_seconds=args.max_target_seconds,
        max_semantic_tokens=args.max_semantic_tokens,
        min_speaker_records=args.min_speaker_records,
        refs_per_speaker=args.refs_per_speaker,
        ref_index=ref_index,
    )
    load(
        args.manifest,
        params=params,
        ref_store=ref_store,
        index_dir=args.index_dir,
        rebuild=args.rebuild,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

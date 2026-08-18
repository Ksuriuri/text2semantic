# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Refs read straight out of the source audio tars, by byte range.

The packed refs (finetuning/ref_store.py) hold a fixed handful of clips per
speaker, chosen when the trainset was built. This store instead points at the
original ``preprocessed/<dataset>/audio/*.tar`` archives, so every clip a
speaker has is a possible ref -- for a speaker with 300 clips that is 300
candidates instead of 8, and no ref data is copied anywhere.

Index row (``refs/source_speaker_index.jsonl``)::

    {"dataset": "ears", "language": "en", "speaker_id": "p001",
     "refs": [["preprocessed/ears/audio/ears-000000.tar",
               "ears__p001_emo_adoration_freeform", 512, 382599], ...]}

A ref is ``(tar, row id, offset, size)``. The offsets come from the per-tar
sidecars written by the header scan; keeping them in the index means a ref costs
exactly one ranged GET at training time, with no sidecar to fetch and no tar to
open.

Two things this store must get right:

* The key is ``(dataset, language, speaker_id)``. Speaker ids are only unique
  within a dataset -- ``p001`` exists in several of them and is not the same
  person -- so a two-part key would quietly hand out another speaker's voice.
* A ref must not be the clip being predicted. That is why the second field is
  the row id and not the tar member name: the packer sanitized ids into
  filenames (``Genshin__en/#Unknown/vo_x`` became
  ``Genshin__en_Unknown_vo_x.flac``, and a podcast member drops the episode hash
  its id repeats), so a member name cannot be turned back into the id it came
  from, and comparing ``exclude`` against one excluded nothing on three of the
  four largest datasets. The index carries the id, so the comparison is exact.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

from finetuning import speaker_index as speaker_index_table

KEY_FIELDS = ("dataset", "language", "speaker_id")
INDEX_NAME = "source_speaker_index.jsonl"


def source_refs_row(row):
    """The source-tar row shape: a list of ``(tar, row id, offset, size)``."""
    refs = []
    for entry in row.get("refs") or ():
        tar, row_id, offset, size = entry
        refs.append((str(tar), str(row_id), int(offset), int(size)))
    return {"refs": tuple(refs)}


def default_index_path(train_jsonl):
    """Discover ``refs/source_speaker_index.jsonl`` next to a manifests/ jsonl."""
    start = Path(train_jsonl).expanduser().resolve().parent
    for base in (start.parent / "refs", start / "refs"):
        candidate = base / INDEX_NAME
        if candidate.is_file():
            return candidate
    return None


class GcsRangeReader:
    """One ranged GET per ref, with a client per process.

    DataLoader workers are forked, and a google-cloud-storage client carries an
    HTTP session that must not be shared across a fork, so the client is rebuilt
    when the pid changes.
    """

    def __init__(self, bucket, *, project=None):
        self.bucket_name = bucket
        self.project = project
        self._bucket = None
        self._pid = None

    def _bucket_for_this_process(self):
        pid = os.getpid()
        if self._bucket is None or self._pid != pid:
            from google.cloud import storage

            self._bucket = storage.Client(project=self.project).bucket(
                self.bucket_name
            )
            self._pid = pid
        return self._bucket

    def read(self, blob_name, offset, size):
        if size <= 0:
            raise ValueError(f"refusing a {size}-byte read of {blob_name}")
        blob = self._bucket_for_this_process().blob(blob_name)
        # end is inclusive in the GCS API.
        payload = blob.download_as_bytes(start=offset, end=offset + size - 1)
        if len(payload) != size:
            raise OSError(
                f"short read for {blob_name}@{offset}: "
                f"{len(payload)} of {size} bytes"
            )
        return payload


class LocalRangeReader:
    """The same interface over a local directory of tars, for a mirrored copy."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self._handles = OrderedDict()
        self._pid = os.getpid()
        self.max_open = 8

    def read(self, blob_name, offset, size):
        pid = os.getpid()
        if pid != self._pid:
            # The parent still owns those descriptors; drop, do not close.
            self._handles = OrderedDict()
            self._pid = pid
        handle = self._handles.get(blob_name)
        if handle is None:
            handle = os.open(self.root / blob_name, os.O_RDONLY)
            self._handles[blob_name] = handle
            while len(self._handles) > self.max_open:
                _, evicted = self._handles.popitem(last=False)
                os.close(evicted)
        else:
            self._handles.move_to_end(blob_name)
        payload = os.pread(handle, size, offset)
        if len(payload) != size:
            raise OSError(
                f"short read for {blob_name}@{offset}: "
                f"{len(payload)} of {size} bytes"
            )
        return payload


class SourceTarRefStore:
    """``read_ref``-compatible ref store over the source tars."""

    speaker_key_fields = KEY_FIELDS

    def __init__(
        self,
        index_path,
        *,
        reader,
        refs_per_speaker=0,
        index_backend="auto",
        build_index_if_missing=True,
        log=None,
    ):
        self.index_path = Path(index_path).expanduser().resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(f"source ref index not found: {self.index_path}")
        if refs_per_speaker < 0:
            raise ValueError("refs_per_speaker must be >= 0 (0 = every clip).")
        self.refs_per_speaker = int(refs_per_speaker)
        self.reader = reader
        if index_backend == "auto":
            index_backend = (
                "memmap"
                if self.index_path.stat().st_size
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
                key_fields=KEY_FIELDS,
                row_adapter=source_refs_row,
                build_if_missing=build_index_if_missing,
                log=log,
            )
        else:
            self._table = None
            self._index = self._load_index(self.index_path)

    def __len__(self):
        return len(self._table if self._index is None else self._index)

    def __contains__(self, key):
        return self._lookup(key) is not None

    def refs(self, key):
        entry = self._lookup(key)
        if entry is None:
            return ()
        refs = entry["refs"]
        if self.refs_per_speaker > 0:
            refs = refs[: self.refs_per_speaker]
        return tuple(refs)

    def ref_ids(self, key):
        """The row ids these refs are, so a ref probe can count or compare them."""
        return tuple(ref[1] for ref in self.refs(key))

    def has_usable_ref(self, key, *, exclude=None):
        return self.pick_ref(key, exclude=exclude) is not None

    def pick_ref(self, key, *, exclude=None, rng=None):
        """``(tar, row id, offset, size)`` for a usable ref, or None."""
        candidates = [
            ref for ref in self.refs(key) if not self._excluded(ref[1], exclude)
        ]
        if not candidates:
            return None
        if rng is None:
            return candidates[0]
        return candidates[rng.randrange(len(candidates))]

    def read_ref(self, key, *, exclude=None, rng=None):
        """``(row id, bytes)`` for a usable ref, or None."""
        picked = self.pick_ref(key, exclude=exclude, rng=rng)
        if picked is None:
            return None
        tar, row_id, offset, size = picked
        return row_id, self.reader.read(tar, offset, size)

    @staticmethod
    def _excluded(row_id, exclude):
        return exclude is not None and row_id == str(exclude)

    def _lookup(self, key):
        if key is None:
            return None
        if len(key) != len(KEY_FIELDS):
            raise ValueError(f"key {key!r} does not match key fields {KEY_FIELDS}")
        if self._index is None:
            return self._table.get(key)
        return self._index.get(tuple(key))

    @staticmethod
    def _load_index(path):
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
                key = speaker_index_table.row_key(row, KEY_FIELDS)
                if key is None:
                    raise ValueError(f"missing speaker_id in {path}:{line_number}")
                entry = source_refs_row(row)
                if not entry["refs"]:
                    raise ValueError(f"no refs in {path}:{line_number}")
                index[key] = entry
        if not index:
            raise ValueError(f"{path} contains no speaker rows.")
        return index

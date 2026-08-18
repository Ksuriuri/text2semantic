# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Byte offsets of the members inside each ref shard.

Reading one ref out of a 544 MB shard used to mean ``tarfile.open`` plus
``getmember``, and ``getmember`` walks every header in the archive: one seek per
member, several hundred of them, for a single 200 KB clip.  Under a globally
shuffled sampler that happens once per sample.

A shard's headers never change after packing, so they are scanned once into a
sidecar::

    refs/.member-index/refs-000123.json
    {"shard_size": 544210944, "members": {"spk_1/000.flac": [1536, 214528], ...}}

After that a ref read is one ``pread`` at a known offset, and the sidecars cost
about 40 KB per shard (~650 MB for the full t2s-v1 refs set) instead of the
8.5 TB a full extraction would put on disk.

A sidecar records the shard size it was built from and is rebuilt when that no
longer matches, and a truncated shard is rejected outright: scanning a tar that
is still being downloaded yields a short member list without raising, and a
sidecar built from one would silently hide every ref past the cut.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

INDEX_DIR_NAME = ".member-index"
FORMAT_VERSION = 1


AUDIO_SUFFIXES = frozenset(
    {"flac", "wav", "mp3", "ogg", "opus", "m4a", "aac", "webm"}
)


def member_row_id(member):
    """The manifest row id a tar member holds.

    Both ref layouts name a member after the row it came from, so this is what
    lets a ref be excluded from being its own target: packed shards use
    ``<dataset>/<language>/<speaker>/<id>.flac`` and the source tars use
    ``<id>.flac``.  The id already carries its own dataset prefix
    (``ears__p045_emo_adoration_freeform``, and note
    ``laion_emolia__ZH_...`` under dataset ``laion_emolia_zh``), so nothing may
    be stripped off the front: doing so produced an id that matched no row, and
    a ref could then be handed out as its own target.

    Only a known audio extension comes off the end, because a row id may
    perfectly well contain a dot.
    """
    name = str(member).replace("\\", "/").rsplit("/", 1)[-1]
    stem, dot, suffix = name.rpartition(".")
    if dot and suffix.lower() in AUDIO_SUFFIXES:
        return stem
    return name


def index_dir_for(ref_root):
    override = os.environ.get("T2S_REF_MEMBER_INDEX")
    if override:
        return Path(override).expanduser()
    return Path(ref_root).expanduser() / INDEX_DIR_NAME


def index_path_for(shard_path, index_dir):
    return Path(index_dir) / (Path(shard_path).stem + ".json")


def scan_shard(shard_path):
    """{member name: (data offset, size)} for every regular file in the tar."""
    shard_path = Path(shard_path)
    shard_size = shard_path.stat().st_size
    members = {}
    # "r:" refuses a compressed archive on purpose: a member's byte offset is
    # only meaningful in an uncompressed tar, which is what the packer writes.
    with tarfile.open(shard_path, "r:") as archive:
        for info in archive:
            if info.isfile():
                members[info.name] = (info.offset_data, info.size)
    if not members:
        raise ValueError(f"no files in ref shard {shard_path}")
    # A cut that lands on a block boundary makes tarfile stop early and report no
    # error at all, so completeness has to be checked against the shard itself:
    # a finished tar is whole-blocks long, ends with two zero blocks, and holds
    # every member's data inside its own length.
    if shard_size % tarfile.BLOCKSIZE:
        raise ValueError(
            f"ref shard {shard_path} is {shard_size} bytes, not a whole number "
            f"of {tarfile.BLOCKSIZE}-byte blocks (still being written?)"
        )
    trailer = 2 * tarfile.BLOCKSIZE
    with open(shard_path, "rb") as handle:
        handle.seek(-min(trailer, shard_size), os.SEEK_END)
        if handle.read() != b"\0" * trailer:
            raise ValueError(
                f"ref shard {shard_path} does not end in the two zero blocks "
                f"that close a tar (truncated or still being written?)"
            )
    for name, (offset, size) in members.items():
        if offset + size > shard_size:
            raise ValueError(
                f"ref shard {shard_path} is truncated: {name} needs "
                f"{offset + size} bytes but the shard is {shard_size}"
            )
    return members


def build_shard_index(shard_path, index_dir, *, overwrite=False):
    """Write the sidecar for one shard; returns its path."""
    out_path = index_path_for(shard_path, index_dir)
    if out_path.is_file() and not overwrite:
        return out_path
    members = scan_shard(shard_path)
    payload = {
        "format_version": FORMAT_VERSION,
        "shard": Path(shard_path).name,
        "shard_size": Path(shard_path).stat().st_size,
        "members": {name: list(value) for name, value in members.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The pid matters: sidecars are built lazily by whichever rank first samples
    # a speaker in that shard, so with 16 ranks and their DataLoader workers two
    # processes can scan the same shard at once.  A shared ".part" name would
    # have both of them truncating and writing one file, and the rename would
    # then publish a mixture of the two.
    tmp_path = out_path.with_suffix(f".json.part-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, separators=(",", ":")))
    tmp_path.replace(out_path)
    return out_path


def read_shard_index(shard_path, index_dir, *, build_if_missing=True):
    """{member: (offset, size)} for one shard, building the sidecar on demand."""
    out_path = index_path_for(shard_path, index_dir)
    if not out_path.is_file():
        if not build_if_missing:
            raise FileNotFoundError(f"missing ref member index: {out_path}")
        build_shard_index(shard_path, index_dir)
    payload = json.loads(out_path.read_text())
    if payload.get("format_version") != FORMAT_VERSION or payload.get(
        "shard_size"
    ) != Path(shard_path).stat().st_size:
        build_shard_index(shard_path, index_dir, overwrite=True)
        payload = json.loads(out_path.read_text())
    return {
        name: (int(value[0]), int(value[1]))
        for name, value in payload["members"].items()
    }


def _build_one(job):
    shard_path, index_dir, overwrite = job
    try:
        build_shard_index(shard_path, index_dir, overwrite=overwrite)
    except Exception as exc:  # keep going; report the shard that failed
        return shard_path, str(exc)
    return shard_path, None


def build_all(shard_dir, index_dir=None, *, workers=16, overwrite=False, log=print):
    """Pre-build every sidecar under `shard_dir`.

    Worth running once after the trainset lands: the first training epoch would
    otherwise pay one header scan per shard, spread over the ranks.
    """
    shard_dir = Path(shard_dir).expanduser().resolve()
    if index_dir is None:
        index_dir = index_dir_for(shard_dir.parent)
    index_dir = Path(index_dir).expanduser().resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(shard_dir.glob("*.tar"))
    if not shards:
        raise ValueError(f"no *.tar shards under {shard_dir}")
    todo = [
        (shard, index_dir, overwrite)
        for shard in shards
        if overwrite or not index_path_for(shard, index_dir).is_file()
    ]
    if log is not None:
        log(
            f"{len(shards):,} shards, {len(todo):,} to scan, "
            f"{workers} workers -> {index_dir}"
        )
    failures = []
    started = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for shard_path, error in pool.map(_build_one, todo, chunksize=4):
            done += 1
            if error is not None:
                failures.append((shard_path, error))
                if log is not None:
                    log(f"  FAILED {shard_path}: {error}")
            elif log is not None and done % 250 == 0:
                elapsed = time.monotonic() - started
                rate = done / elapsed if elapsed else 0.0
                log(
                    f"  {done:,}/{len(todo):,} scanned, {elapsed:.0f}s, "
                    f"{rate:.1f} shards/s"
                )
    if log is not None:
        log(
            f"member index done: {done - len(failures):,} ok, "
            f"{len(failures):,} failed, {time.monotonic() - started:.0f}s"
        )
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Scan ref shards once so training can pread members."
    )
    parser.add_argument("shard_dir", help="trainset refs/shards directory")
    parser.add_argument("--index_dir", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    failures = build_all(
        args.shard_dir,
        index_dir=args.index_dir,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

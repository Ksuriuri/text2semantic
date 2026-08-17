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
    {"members": {"spk_1/000.flac": [1536, 214528], ...}}

After that a ref read is one ``pread`` at a known offset, and the sidecars cost
about 40 KB per shard (~650 MB for the full t2s-v1 refs set) instead of the
8.5 TB a full extraction would put on disk.
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
    members = {}
    # "r:" refuses a compressed archive on purpose: a member's byte offset is
    # only meaningful in an uncompressed tar, which is what the packer writes.
    with tarfile.open(shard_path, "r:") as archive:
        for info in archive:
            if info.isfile():
                members[info.name] = (info.offset_data, info.size)
    if not members:
        raise ValueError(f"no files in ref shard {shard_path}")
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
        "members": {name: list(value) for name, value in members.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.part")
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
    if payload.get("format_version") != FORMAT_VERSION:
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

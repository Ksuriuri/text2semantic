#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Build ``refs/source_speaker_index.jsonl`` for --ref_backend source_tar.

Joins two things that already exist, so no audio is read and nothing is packed:

* the per-tar member sidecars from the header scan,
  ``index/source-tar-members/<dataset>/<tar>.json`` -> ``{member: [offset, size]}``
* the trainset manifests, ``trainsets/t2s-v1/manifests/{train,eval}.jsonl``,
  which say which rows survived filtering and what speaker each one belongs to

A source tar member is named ``<row id>.flac``, and the manifest tells us the
row's ``(dataset, language, speaker_id)`` and duration, so one pass over each
side is enough::

    {"dataset": "ears", "language": "en", "speaker_id": "ears__p001",
     "refs": [["preprocessed/ears/audio/ears-000000.tar",
               "ears__p001_emo_adoration_freeform.flac", 512, 382599], ...]}

Only manifest rows become candidate refs. Clips the trainset filter dropped are
often perfectly good audio -- a bad transcript does not matter for a ref -- but
their duration and sample rate were never checked, so they stay out.

The duration window is the pack-time ref window ([3, 30] s), not the target
window: a 0.5 s clip is a fine thing to predict and a useless thing to condition
on.

Runs per dataset and skips a dataset whose part file is already written, so a
preempted Spot VM resumes instead of starting over.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetuning.ref_member_index import member_row_id  # noqa: E402

try:
    import orjson

    def _loads(payload):
        return orjson.loads(payload)

except ImportError:  # pragma: no cover - depends on the environment

    def _loads(payload):
        return json.loads(payload)


BUCKET = "noiz-taiwan-audio-data"
PROJECT = "noiz-430406"
SIDECAR_PREFIX = "index/source-tar-members"
AUDIO_TEMPLATE = "preprocessed/{dataset}/audio/{shard}"
# The window pack_trainset used to choose refs, not the target window.
REF_MIN_SECONDS = 3.0
REF_MAX_SECONDS = 30.0


def _quiet(*_args):
    """log=None means silence, as everywhere else in this codebase."""


class GcsBlobs:
    """The blob operations this script needs, over a GCS bucket."""

    def __init__(self, bucket=BUCKET, project=PROJECT):
        from google.cloud import storage

        self.client = storage.Client(project=project)
        self.bucket = self.client.bucket(bucket)
        self.name = f"gs://{bucket}"

    def list(self, prefix):
        return [
            blob.name
            for blob in self.client.list_blobs(self.bucket, prefix=prefix)
            if blob.size
        ]

    def read(self, name):
        return self.bucket.blob(name).download_as_bytes()

    def lines(self, name):
        with self.bucket.blob(name).open("rb") as handle:
            for line in handle:
                yield line


class LocalBlobs:
    """The same operations over a directory tree, for tests and a mirror."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.name = str(self.root)

    def list(self, prefix):
        base = self.root / prefix
        if not base.is_dir():
            return []
        return sorted(
            str(path.relative_to(self.root))
            for path in base.rglob("*")
            if path.is_file() and path.stat().st_size
        )

    def read(self, name):
        return (self.root / name).read_bytes()

    def lines(self, name):
        with (self.root / name).open("rb") as handle:
            for line in handle:
                yield line


def split_manifests(blobs, manifests, work_dir, *, log=print):
    """Write ``<work>/rows/<dataset>.tsv`` of the rows that may become refs.

    One pass over the manifests, so the 50 GiB train.jsonl is read once no
    matter how many datasets come out of it.

    Returns ``(rows_dir, rebuilt)``; ``rebuilt`` is False when a previous run
    had already split these same manifests.
    """
    log = log or _quiet
    rows_dir = Path(work_dir) / "rows"
    done_marker = rows_dir / "_COMPLETE"
    manifests = list(manifests)
    if done_marker.is_file():
        previous = json.loads(done_marker.read_text()).get("manifests")
        if previous == manifests:
            log(f"rows already split: {rows_dir}")
            return rows_dir, False
        # Resuming is only safe when it resumes the same job.
        log(f"re-splitting: manifests changed from {previous} to {manifests}")
        done_marker.unlink()
    if rows_dir.is_dir():
        for stale in rows_dir.glob("*.tsv"):
            stale.unlink()
    rows_dir.mkdir(parents=True, exist_ok=True)

    handles = {}
    counts = {
        "manifests": manifests,
        "rows": 0,
        "kept": 0,
        "no_speaker": 0,
        "bad_duration": 0,
    }
    started = time.monotonic()
    try:
        for manifest in manifests:
            log(f"splitting {manifest}")
            for line in blobs.lines(manifest):
                if not line.strip():
                    continue
                counts["rows"] += 1
                row = _loads(line)
                duration = row.get("duration")
                if (
                    duration is None
                    or float(duration) < REF_MIN_SECONDS
                    or float(duration) > REF_MAX_SECONDS
                ):
                    counts["bad_duration"] += 1
                    continue
                dataset = row.get("dataset")
                speaker = row.get("speaker_id")
                row_id = row.get("id")
                if not dataset or not speaker or not row_id:
                    counts["no_speaker"] += 1
                    continue
                language = row.get("language") or row.get("lang") or ""
                handle = handles.get(dataset)
                if handle is None:
                    handle = (rows_dir / f"{dataset}.tsv").open(
                        "w", encoding="utf-8"
                    )
                    handles[dataset] = handle
                # No field here can hold a tab: ids, languages and speaker ids
                # are all path/label shaped.
                handle.write(f"{row_id}\t{language}\t{speaker}\n")
                counts["kept"] += 1
                if counts["rows"] % 20_000_000 == 0:
                    log(
                        f"  {counts['rows']:,} rows, {counts['kept']:,} ref "
                        f"candidates, {time.monotonic() - started:.0f}s"
                    )
    finally:
        for handle in handles.values():
            handle.close()
    done_marker.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    log(
        f"split {counts['rows']:,} rows -> {counts['kept']:,} ref candidates "
        f"in {time.monotonic() - started:.0f}s"
    )
    return rows_dir, True


def read_rows(path):
    """``{row id: (language, speaker_id)}`` for one dataset."""
    wanted = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row_id, language, speaker = line.rstrip("\n").split("\t")
            wanted[row_id] = (language or None, speaker)
    return wanted


def collect_dataset(
    blobs,
    dataset,
    wanted,
    *,
    prefix=SIDECAR_PREFIX,
    workers=16,
    log=print,
):
    """``{(dataset, language, speaker): [(tar, member, offset, size), ...]}``."""
    sidecars = [
        name
        for name in blobs.list(f"{prefix.rstrip('/')}/{dataset}/")
        if name.endswith(".json")
    ]
    log = log or _quiet
    refs = {}
    tars = {}
    seen_members = 0
    matched = 0
    started = time.monotonic()

    def fetch(name):
        return name, blobs.read(name)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (name, payload) in enumerate(
            pool.map(fetch, sidecars), start=1
        ):
            sidecar = _loads(payload)
            shard = sidecar.get("shard") or Path(name).stem + ".tar"
            # The sidecar names the tar, not its path, and the path is not
            # derivable from the tar name: laion_emolia-000869.tar lives under
            # preprocessed/laion_emolia_zh/audio/.
            tar = tars.setdefault(
                shard, AUDIO_TEMPLATE.format(dataset=dataset, shard=shard)
            )
            for member, location in (sidecar.get("members") or {}).items():
                seen_members += 1
                found = wanted.get(member_row_id(member))
                if found is None:
                    continue
                language, speaker = found
                offset, size = location
                refs.setdefault((dataset, language, speaker), []).append(
                    (tar, member, int(offset), int(size))
                )
                matched += 1
            if done % 2000 == 0 or done == len(sidecars):
                log(
                    f"  [{dataset}] {done}/{len(sidecars)} tars, "
                    f"{matched:,}/{seen_members:,} members matched, "
                    f"{len(refs):,} speakers, "
                    f"{time.monotonic() - started:.0f}s"
                )
    return refs, {
        "tars": len(sidecars),
        "members": seen_members,
        "matched": matched,
        "wanted": len(wanted),
        "speakers": len(refs),
    }


def write_part(refs, path, *, max_refs_per_speaker=0, seed=42):
    """One dataset's index rows, sorted so a rebuild is byte-identical."""
    rng = random.Random(seed)
    written = 0
    with open(path, "w", encoding="utf-8") as handle:
        for key in sorted(refs):
            dataset, language, speaker = key
            members = sorted(refs[key], key=lambda ref: ref[1])
            if 0 < max_refs_per_speaker < len(members):
                # A sample rather than the first N: the first N of a sorted list
                # is one recording session for most speakers.
                members = sorted(
                    rng.sample(members, max_refs_per_speaker),
                    key=lambda ref: ref[1],
                )
            handle.write(
                json.dumps(
                    {
                        "dataset": dataset,
                        "language": language,
                        "speaker_id": speaker,
                        "refs": [list(ref) for ref in members],
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
            written += 1
    return written


def build(
    blobs,
    work_dir,
    *,
    manifests,
    datasets=None,
    prefix=SIDECAR_PREFIX,
    max_refs_per_speaker=0,
    workers=16,
    seed=42,
    log=print,
):
    """Split the manifests, join each dataset, and concatenate the parts."""
    log = log or _quiet
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    rows_dir, rebuilt = split_manifests(blobs, manifests, work_dir, log=log)
    parts_dir = work_dir / "parts"
    if rebuilt:
        # Those parts were joined against other rows, and an mtime comparison is
        # too coarse to tell: drop them outright.
        shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(exist_ok=True)

    available = sorted(path.stem for path in rows_dir.glob("*.tsv"))
    if datasets:
        missing = sorted(set(datasets) - set(available))
        if missing:
            raise SystemExit(
                f"no manifest rows for {missing}; the manifests hold {available}"
            )
        available = [name for name in available if name in set(datasets)]
    if not available:
        raise SystemExit(f"no ref candidates in {manifests}")

    stats = {}
    for dataset in available:
        part = parts_dir / f"{dataset}.jsonl"
        stat_path = parts_dir / f"{dataset}.json"
        if part.is_file() and stat_path.is_file():
            stats[dataset] = json.loads(stat_path.read_text())
            log(f"[{dataset}] already built: {stats[dataset]}")
            continue
        wanted = read_rows(rows_dir / f"{dataset}.tsv")
        refs, counts = collect_dataset(
            blobs, dataset, wanted, prefix=prefix, workers=workers, log=log
        )
        if counts["matched"] == 0:
            raise SystemExit(
                f"[{dataset}] {counts['wanted']:,} manifest rows matched none of "
                f"{counts['members']:,} tar members -- check the sidecar prefix "
                f"and the member naming before trusting an empty index"
            )
        tmp = part.with_suffix(".jsonl.part")
        counts["rows"] = write_part(
            refs,
            tmp,
            max_refs_per_speaker=max_refs_per_speaker,
            seed=seed,
        )
        tmp.replace(part)
        stat_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
        stats[dataset] = counts
        log(f"[{dataset}] {counts}")

    index_path = work_dir / "source_speaker_index.jsonl"
    tmp = index_path.with_suffix(".jsonl.part")
    with tmp.open("wb") as sink:
        for dataset in available:
            with (parts_dir / f"{dataset}.jsonl").open("rb") as part:
                while True:
                    chunk = part.read(1 << 20)
                    if not chunk:
                        break
                    sink.write(chunk)
    tmp.replace(index_path)
    summary = {
        "index": str(index_path),
        "bytes": index_path.stat().st_size,
        "speakers": sum(value["rows"] for value in stats.values()),
        "refs": sum(value["matched"] for value in stats.values()),
        "per_dataset": stats,
    }
    (work_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(
        f"{summary['speakers']:,} speakers, {summary['refs']:,} refs, "
        f"{summary['bytes'] / 1024 ** 3:.2f} GiB -> {index_path}"
    )
    return index_path, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work_dir", default=os.path.expanduser("~/t2s-source-refs")
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="repeatable; defaults to the t2s-v1 train and eval manifests",
    )
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument(
        "--local_root",
        default=None,
        help="read from this directory instead of GCS",
    )
    parser.add_argument("--sidecar_prefix", default=SIDECAR_PREFIX)
    parser.add_argument("--max_refs_per_speaker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    blobs = (
        LocalBlobs(args.local_root)
        if args.local_root
        else GcsBlobs(args.bucket, args.project)
    )
    manifests = args.manifest or [
        "trainsets/t2s-v1/manifests/train.jsonl",
        "trainsets/t2s-v1/manifests/eval.jsonl",
    ]
    build(
        blobs,
        args.work_dir,
        manifests=manifests,
        datasets=args.dataset,
        prefix=args.sidecar_prefix,
        max_refs_per_speaker=args.max_refs_per_speaker,
        workers=args.workers,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

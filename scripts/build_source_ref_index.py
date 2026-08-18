#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Build ``refs/source_speaker_index.jsonl`` for --ref_backend source_tar.

Joins three things that already exist, so no audio is read and nothing is packed:

* the per-tar member sidecars from the header scan,
  ``index/source-tar-members/<dataset>/<tar>.json`` -> ``{member: [offset, size]}``
* the per-tar metadata written when the tar was packed,
  ``preprocessed/<dataset>/metadata/<tar>.jsonl``, whose rows carry both ``id``
  and ``audio_path`` (``audio/<tar>.tar/<member>``)
* the trainset manifests, ``trainsets/t2s-v1/manifests/{train,eval}.jsonl``,
  which say which rows survived filtering and what speaker each one belongs to

The metadata is what makes the join exact. A member is *not* named after its row
id: the packer sanitizes the id into a filename, so
``Genshin__en/#Unknown/vo_NTAQ008_15_olorun_01`` is stored as
``Genshin__en_Unknown_vo_NTAQ008_15_olorun_01.flac``, and a podcast id repeats
its episode hash where the member does not. Deriving one from the other matched
0 of Genshin's 644,716 members; the metadata row states both, so nothing is
guessed.

An index row holds ``(tar, row id, offset, size)`` per ref -- the row id rather
than the member name, because the id is what a caller has when it needs to keep
a clip from being its own reference, and the bytes are already fully addressed
by the tar and the range::

    {"dataset": "ears", "language": "en", "speaker_id": "ears__p001",
     "refs": [["preprocessed/ears/audio/ears-000000.tar",
               "ears__p001_emo_adoration_freeform", 512, 382599], ...]}

Only manifest rows become candidate refs. Clips the trainset filter dropped are
often perfectly good audio -- a bad transcript does not matter for a ref -- but
their duration and sample rate were never checked, so they stay out.

The duration window is the pack-time ref window ([3, 30] s), not the target
window: a 0.5 s clip is a fine thing to predict and a useless thing to condition
on.

Runs per dataset and skips a dataset whose part file is already written, so a
preempted Spot VM resumes instead of starting over.

The manifest pass is split into byte ranges across processes (--split_workers),
because parsing 99M JSON rows is what costs the time here, not the download.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
METADATA_PREFIX = "preprocessed/{dataset}/metadata/"
METADATA_TEMPLATE = "preprocessed/{dataset}/metadata/{stem}.jsonl"
# The window pack_trainset used to choose refs, not the target window.
REF_MIN_SECONDS = 3.0
REF_MAX_SECONDS = 30.0
# One split range, sized so a 54 GB manifest becomes ~135 of them: small enough
# to fill every core, large enough that the per-range GET is not the cost.
PART_BYTES = 384 * 1024 * 1024
READ_CHUNK = 32 * 1024 * 1024
WRITE_BUFFER = 4 * 1024 * 1024


def _quiet(*_args):
    """log=None means silence, as everywhere else in this codebase."""


class GcsBlobs:
    """The blob operations this script needs, over a GCS bucket."""

    def __init__(self, bucket=BUCKET, project=PROJECT):
        from google.cloud import storage

        self.bucket_name = bucket
        self.project = project
        self.client = storage.Client(project=project)
        self.bucket = self.client.bucket(bucket)
        self.name = f"gs://{bucket}"

    def spec(self):
        """How to rebuild this backend in a worker process."""
        return ("gcs", self.bucket_name, self.project)

    def list(self, prefix):
        return [
            blob.name
            for blob in self.client.list_blobs(self.bucket, prefix=prefix)
            if blob.size
        ]

    def read(self, name):
        return self.bucket.blob(name).download_as_bytes()

    def size(self, name):
        blob = self.bucket.blob(name)
        blob.reload()
        return blob.size

    def reader(self, name):
        return self.bucket.blob(name).open("rb", chunk_size=READ_CHUNK)


class LocalBlobs:
    """The same operations over a directory tree, for tests and a mirror."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.name = str(self.root)

    def spec(self):
        return ("local", str(self.root))

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

    def size(self, name):
        return (self.root / name).stat().st_size

    def reader(self, name):
        return (self.root / name).open("rb")


def blobs_from_spec(spec):
    """The backend a worker process rebuilds from ``blobs.spec()``."""
    if spec[0] == "gcs":
        return GcsBlobs(spec[1], spec[2])
    return LocalBlobs(spec[1])


def _lines_in_range(handle, lo, hi):
    """The lines that *start* inside ``[lo, hi)``.

    A worker whose range starts at ``lo`` seeks to ``lo - 1`` and discards
    through the next newline, because that line started in the previous range.
    It then reads whole lines while the position at the start of the line is
    below ``hi``, so the line straddling ``hi`` is parsed exactly once -- here,
    and not by the worker that owns the next range.
    """
    if lo:
        handle.seek(lo - 1)
        handle.readline()
    position = handle.tell()
    while position < hi:
        line = handle.readline()
        if not line:
            break
        position += len(line)
        yield line


def _split_lines(lines, writer, counts):
    """Write the ref candidates among ``lines``, counting what was dropped."""
    for line in lines:
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
        # No field here can hold a tab: ids, languages and speaker ids are all
        # path/label shaped.
        writer(dataset).write(f"{row_id}\t{language}\t{speaker}\n")
        counts["kept"] += 1


def split_range(job):
    """Parse one byte range into ``<part_dir>/<dataset>.tsv``."""
    spec, manifest, lo, hi, part_dir = job
    part_dir = Path(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    counts = {"rows": 0, "kept": 0, "no_speaker": 0, "bad_duration": 0}
    handles = {}

    def writer(dataset):
        handle = handles.get(dataset)
        if handle is None:
            handle = (part_dir / f"{dataset}.tsv").open(
                "w", encoding="utf-8", buffering=WRITE_BUFFER
            )
            handles[dataset] = handle
        return handle

    try:
        with blobs_from_spec(spec).reader(manifest) as handle:
            _split_lines(_lines_in_range(handle, lo, hi), writer, counts)
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def plan_ranges(blobs, manifests, parts_dir, *, part_bytes=PART_BYTES):
    jobs = []
    for manifest in manifests:
        size = blobs.size(manifest)
        lo = 0
        while lo < size:
            hi = min(lo + part_bytes, size)
            jobs.append(
                (
                    blobs.spec(),
                    manifest,
                    lo,
                    hi,
                    # Zero-padded so the merge concatenates in manifest order,
                    # which is what makes a rebuild byte-identical.
                    str(Path(parts_dir) / f"{len(jobs):06d}"),
                )
            )
            lo = hi
    return jobs


def merge_ranges(parts_dir, rows_dir):
    """Concatenate the per-range files into one file per dataset."""
    by_dataset = {}
    for part in sorted(Path(parts_dir).iterdir()):
        for path in sorted(part.glob("*.tsv")):
            by_dataset.setdefault(path.stem, []).append(path)
    for dataset, paths in sorted(by_dataset.items()):
        with (Path(rows_dir) / f"{dataset}.tsv").open("wb") as sink:
            for path in paths:
                with path.open("rb") as handle:
                    shutil.copyfileobj(handle, sink, 8 * 1024 * 1024)
    return sorted(by_dataset)


def split_manifests(
    blobs,
    manifests,
    work_dir,
    *,
    workers=1,
    part_bytes=PART_BYTES,
    log=print,
):
    """Write ``<work>/rows/<dataset>.tsv`` of the rows that may become refs.

    The manifests are cut into byte ranges and each range is parsed in its own
    process, because this pass is CPU-bound in the JSON parser rather than on
    the network: one stream through the 54 GB train.jsonl held 97% of a single
    core at 3 MB/s (~4.8 h), while 135 ranges over 48 processes did the same
    99.4M rows in 397 s.

    Returns ``(rows_dir, rebuilt)``; ``rebuilt`` is False when a previous run
    had already split these same manifests.
    """
    log = log or _quiet
    rows_dir = Path(work_dir) / "rows"
    done_marker = rows_dir / "_COMPLETE"
    manifests = list(manifests)
    # A dataset gets added by appending rows to train.jsonl, so the manifest
    # keeps its name and only grows: matching on the name alone would report
    # "already split" and quietly build the index from the old rows.
    sizes = {manifest: blobs.size(manifest) for manifest in manifests}
    if done_marker.is_file():
        marker = json.loads(done_marker.read_text())
        previous = marker.get("manifests")
        previous_sizes = marker.get("manifest_sizes")
        if previous == manifests and previous_sizes == sizes:
            log(f"rows already split: {rows_dir}")
            return rows_dir, False
        # Resuming is only safe when it resumes the same job.
        log(
            f"re-splitting: manifests changed from {previous} ({previous_sizes}) "
            f"to {manifests} ({sizes})"
        )
        done_marker.unlink()
    if rows_dir.is_dir():
        for stale in rows_dir.glob("*.tsv"):
            stale.unlink()
    rows_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = Path(work_dir) / "rows_parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)

    counts = {
        "manifests": manifests,
        "manifest_sizes": sizes,
        "rows": 0,
        "kept": 0,
        "no_speaker": 0,
        "bad_duration": 0,
    }
    jobs = plan_ranges(blobs, manifests, parts_dir, part_bytes=part_bytes)
    log(f"splitting {len(manifests)} manifests in {len(jobs)} ranges")
    started = time.monotonic()
    done = 0
    for finished in _each_range(jobs, workers):
        done += 1
        for key, value in finished.items():
            counts[key] += value
        if done % 20 == 0 or done == len(jobs):
            log(
                f"  {done}/{len(jobs)} ranges, {counts['rows']:,} rows, "
                f"{counts['kept']:,} ref candidates, "
                f"{time.monotonic() - started:.0f}s"
            )
    merge_ranges(parts_dir, rows_dir)
    shutil.rmtree(parts_dir, ignore_errors=True)
    done_marker.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    log(
        f"split {counts['rows']:,} rows -> {counts['kept']:,} ref candidates "
        f"in {time.monotonic() - started:.0f}s"
    )
    return rows_dir, True


def _each_range(jobs, workers):
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(split_range, jobs)
        return
    for job in jobs:
        yield split_range(job)


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
    """``{(dataset, language, speaker): [(tar, row id, offset, size), ...]}``.

    One tar at a time, joining its metadata (row id -> member) against its
    sidecar (member -> byte range). Both are keyed by the tar's stem, so a tar
    with a sidecar but no metadata contributes nothing and is counted rather
    than guessed at.
    """
    log = log or _quiet
    sidecar_dir = f"{prefix.rstrip('/')}/{dataset}"
    stems = sorted(
        Path(name).stem
        for name in blobs.list(f"{sidecar_dir}/")
        if name.endswith(".json")
    )
    described = {
        Path(name).name[: -len(".jsonl")]
        for name in blobs.list(METADATA_PREFIX.format(dataset=dataset))
        if name.endswith(".jsonl")
    }
    paired = [stem for stem in stems if stem in described]
    refs = {}
    seen_members = 0
    matched = 0
    unlocated = 0
    started = time.monotonic()

    def fetch(stem):
        return (
            stem,
            blobs.read(f"{sidecar_dir}/{stem}.json"),
            blobs.read(METADATA_TEMPLATE.format(dataset=dataset, stem=stem)),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (stem, sidecar_payload, metadata_payload) in enumerate(
            pool.map(fetch, paired), start=1
        ):
            sidecar = _loads(sidecar_payload)
            shard = sidecar.get("shard") or f"{stem}.tar"
            # The sidecar names the tar, not its path, and the path is not
            # derivable from the tar name: laion_emolia-000869.tar lives under
            # preprocessed/laion_emolia_zh/audio/.
            tar = AUDIO_TEMPLATE.format(dataset=dataset, shard=shard)
            locations = sidecar.get("members") or {}
            seen_members += len(locations)
            for line in metadata_payload.splitlines():
                if not line.strip():
                    continue
                row = _loads(line)
                row_id = row.get("id")
                found = wanted.get(row_id)
                if found is None:
                    continue
                # audio_path is "audio/<tar>.tar/<member>"; only the member is
                # taken from it, because the tar's path and its offsets come
                # from the sidecar that was scanned.
                member = str(row.get("audio_path") or "").rsplit("/", 1)[-1]
                location = locations.get(member)
                if location is None:
                    unlocated += 1
                    continue
                language, speaker = found
                offset, size = location
                refs.setdefault((dataset, language, speaker), []).append(
                    (tar, row_id, int(offset), int(size))
                )
                matched += 1
            if done % 2000 == 0 or done == len(paired):
                log(
                    f"  [{dataset}] {done}/{len(paired)} tars, "
                    f"{matched:,}/{seen_members:,} members matched, "
                    f"{len(refs):,} speakers, "
                    f"{time.monotonic() - started:.0f}s"
                )
    if len(paired) < len(stems):
        # Not fatal on its own -- a tar packed after the last metadata dump has
        # no rows in the manifests either -- but it must never pass unnoticed.
        log(
            f"  [{dataset}] {len(stems) - len(paired)} of {len(stems)} tars "
            "have no metadata jsonl and were skipped"
        )
    if unlocated:
        log(
            f"  [{dataset}] {unlocated:,} manifest rows name a member that is "
            "not in the scanned tar"
        )
    return refs, {
        "tars": len(paired),
        "tars_without_metadata": len(stems) - len(paired),
        "members": seen_members,
        "matched": matched,
        "unlocated": unlocated,
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
    split_workers=None,
    seed=42,
    log=print,
):
    """Split the manifests, join each dataset, and concatenate the parts."""
    log = log or _quiet
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    rows_dir, rebuilt = split_manifests(
        blobs,
        manifests,
        work_dir,
        # The split is CPU-bound and the join is not, so the split gets
        # processes and as many as the box has cores.
        workers=workers if split_workers is None else split_workers,
        log=log,
    )
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
                f"{counts['members']:,} tar members across {counts['tars']:,} "
                "tars -- check the sidecar prefix and whether the metadata rows "
                "carry the same ids as the manifests, before trusting an empty "
                "index"
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
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="threads for the per-tar fetch, and the default for --split_workers",
    )
    parser.add_argument(
        "--split_workers",
        type=int,
        default=None,
        help="processes for the manifest split; one per core is right",
    )
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
        split_workers=args.split_workers,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

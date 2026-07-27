"""Stage 1: stream every shard, apply the per-row gates, emit a compact index.

Scale drives the design.  The corpus holds roughly 84 M utterances across 42 k
shards, and the metadata plus ASR jsonl that must be read totals about 75 GB
(``laion_emolia`` alone is ~69 GB of it).  So the scan:

* never downloads the audio - only ``metadata/`` and ``asr/`` jsonl;
* processes one shard at a time in a worker process and writes a gzipped index
  file per shard, so peak memory stays at one shard (~2000 rows);
* keeps only the fields later stages need, which shrinks 75 GB of source text to
  a few GB of index;
* applies the speaker-support gate afterwards in :func:`speaker_gate`, which
  needs corpus-wide counts and therefore streams the index twice instead of
  holding it in memory.

Run ``python -m data_pipeline.scan --help`` for the CLI.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from data_pipeline import gcs
from data_pipeline.filters import (
    FilterConfig,
    FilterCounters,
    as_float,
    as_text,
    screen_row,
)

_WORKER = {}


def _worker_init(gcs_key, project, bucket, root):
    client = gcs.make_client(gcs_key, project)
    _WORKER["corpus"] = gcs.Corpus(client, bucket=bucket, root=root)


def _asr_index(corpus, blob_name):
    """Map utterance id to transcript for one ASR shard."""
    if blob_name is None:
        return {}
    table = {}
    for row in corpus.read_jsonl(blob_name):
        utt_id = as_text(row.get("id"))
        if not utt_id:
            continue
        if as_text(row.get("status")) not in ("", "transcribed"):
            continue
        table[utt_id] = row.get("text")
    return table


def scan_shard(shard, config, out_dir):
    """Screen a single shard and write its surviving rows to a gzipped index."""
    corpus = _WORKER["corpus"]
    counters = FilterCounters()
    sample_rates = Counter()
    languages = Counter()

    cohere = _asr_index(corpus, shard.cohere_blob)
    granite = _asr_index(corpus, shard.granite_blob)

    out_path = Path(out_dir) / shard.dataset / f"{shard.dataset}-{shard.index:06d}.jsonl.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")

    kept = 0
    with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=1) as sink:
        for row in corpus.read_jsonl(shard.metadata_blob):
            utt_id = as_text(row.get("id"))
            record = screen_row(
                row,
                cohere.get(utt_id),
                granite.get(utt_id),
                config,
                counters,
                shard.dataset,
                shard.index,
            )
            rate = as_float(row.get("sample_rate"))
            if rate is not None:
                sample_rates[int(rate)] += 1
            languages[as_text(row.get("language")) or "?"] += 1
            if record is None:
                continue
            sink.write(json.dumps(record.as_dict(), ensure_ascii=False))
            sink.write("\n")
            kept += 1
    os.replace(tmp_path, out_path)

    counters.extra["cohere_available"] = 1 if shard.cohere_blob else 0
    return {
        "dataset": shard.dataset,
        "shard": shard.index,
        "kept": kept,
        "counters": counters.as_dict(),
        "sample_rates": dict(sample_rates),
        "languages": dict(languages),
        "has_granite": bool(shard.granite_blob),
        "metadata_bytes": shard.metadata_bytes,
        "index_path": str(out_path),
    }


def _scan_shard_entry(args):
    shard, config, out_dir = args
    try:
        return scan_shard(shard, config, out_dir)
    except Exception as error:  # keep one bad shard from killing the run
        return {
            "dataset": shard.dataset,
            "shard": shard.index,
            "error": f"{type(error).__name__}: {error}",
        }


def speaker_gate(index_dir, config, out_path, datasets=None):
    """Apply the corpus-wide speaker-support gate over the scan index.

    Pass 1 accumulates per-speaker counts and whether the speaker has a long
    utterance.  Pass 2 rewrites the surviving rows.  Speaker keys are
    ``(language, speaker_id)``, matching ``Text2SemanticDataset._speaker_key``.
    """
    index_dir = Path(index_dir)
    files = sorted(index_dir.rglob("*.jsonl.gz"))
    if datasets:
        wanted = set(datasets)
        files = [f for f in files if f.parent.name in wanted]

    counts = Counter()
    has_long = set()
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row.get("language"), row["speaker_id"])
                counts[key] += 1
                if row["duration"] > config.speaker_long_utterance_seconds:
                    has_long.add(key)

    stats = {
        "speakers_seen": len(counts),
        "speakers_kept": 0,
        "rows_in": sum(counts.values()),
        "rows_kept": 0,
        "dropped_singleton": 0,
        "dropped_no_long": 0,
        "seconds_kept": 0.0,
        "per_dataset": defaultdict(lambda: {"rows": 0, "seconds": 0.0, "speakers": set()}),
    }
    keep_key = {
        key for key, n in counts.items()
        if n >= config.min_speaker_records and key in has_long
    }
    stats["speakers_kept"] = len(keep_key)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(out_path).endswith(".gz") else open
    with opener(out_path, "wt", encoding="utf-8") as sink:
        for path in files:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = (row.get("language"), row["speaker_id"])
                    if key not in keep_key:
                        if counts[key] < config.min_speaker_records:
                            stats["dropped_singleton"] += 1
                        else:
                            stats["dropped_no_long"] += 1
                        continue
                    sink.write(line if line.endswith("\n") else line + "\n")
                    stats["rows_kept"] += 1
                    stats["seconds_kept"] += row["duration"]
                    bucket = stats["per_dataset"][row["dataset"]]
                    bucket["rows"] += 1
                    bucket["seconds"] += row["duration"]
                    bucket["speakers"].add(key)

    stats["per_dataset"] = {
        name: {
            "rows": value["rows"],
            "hours": round(value["seconds"] / 3600.0, 4),
            "speakers": len(value["speakers"]),
        }
        for name, value in sorted(stats["per_dataset"].items())
    }
    stats["hours_kept"] = round(stats["seconds_kept"] / 3600.0, 4)
    return stats


def build_parser():
    parser = argparse.ArgumentParser(
        description="Scan the GCS corpus and emit a filtered utterance index.",
    )
    parser.add_argument("--out-dir", required=True, help="Where to write the index tree.")
    parser.add_argument("--gcs-key", default=None, help="Service-account json path.")
    parser.add_argument("--project", default=gcs.PROJECT)
    parser.add_argument("--bucket", default=gcs.BUCKET)
    parser.add_argument("--root", default=gcs.ROOT)
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset of datasets to scan (default: all).",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--max-shards-per-dataset", type=int, default=None,
        help="Debug aid: scan only the first N shards of each dataset.",
    )
    parser.add_argument("--min-sample-rate", type=int, default=22050)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--speaker-long-seconds", type=float, default=6.0)
    parser.add_argument("--min-speaker-records", type=int, default=2)
    parser.add_argument("--max-error-rate", type=float, default=0.5)
    parser.add_argument(
        "--keep-unscored", action="store_true",
        help="Keep rows that have neither an original transcript nor Granite ASR.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Do not re-scan shards whose index file already exists.",
    )
    parser.add_argument(
        "--gate-only", action="store_true",
        help="Skip the scan and only run the speaker gate over an existing index.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = FilterConfig(
        min_sample_rate=args.min_sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        speaker_long_utterance_seconds=args.speaker_long_seconds,
        min_speaker_records=args.min_speaker_records,
        max_error_rate=args.max_error_rate,
        require_asr_score=not args.keep_unscored,
    )
    out_dir = Path(args.out_dir)
    index_dir = out_dir / "index"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.gate_only:
        client = gcs.make_client(args.gcs_key, args.project)
        corpus = gcs.Corpus(client, bucket=args.bucket, root=args.root)
        datasets = args.datasets or list(gcs.DATASETS)

        jobs = []
        engines = {}
        for dataset in datasets:
            engines[dataset] = corpus.asr_engines(dataset)
            shards = corpus.shards(dataset)
            if args.max_shards_per_dataset:
                shards = shards[: args.max_shards_per_dataset]
            for shard in shards:
                target = index_dir / dataset / f"{dataset}-{shard.index:06d}.jsonl.gz"
                if args.skip_existing and target.exists():
                    continue
                jobs.append((shard, config, str(index_dir)))
            print(
                f"[plan] {dataset:16s} shards={len(shards):6d} "
                f"asr_engines={engines[dataset]}",
                flush=True,
            )
        print(f"[plan] shards to scan: {len(jobs)}", flush=True)

        totals = defaultdict(FilterCounters)
        sample_rates = defaultdict(Counter)
        errors = []
        started = time.time()
        done = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(args.gcs_key, args.project, args.bucket, args.root),
        ) as pool:
            futures = [pool.submit(_scan_shard_entry, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if "error" in result:
                    errors.append(result)
                    print(f"[error] {result['dataset']}-{result['shard']}: "
                          f"{result['error']}", flush=True)
                    continue
                bucket = totals[result["dataset"]]
                for key, value in result["counters"].items():
                    if hasattr(bucket, key):
                        setattr(bucket, key, getattr(bucket, key) + value)
                    else:
                        bucket.extra[key] = bucket.extra.get(key, 0) + value
                for rate, n in result["sample_rates"].items():
                    sample_rates[result["dataset"]][int(rate)] += n
                if done % 200 == 0 or done == len(jobs):
                    rate = done / max(time.time() - started, 1e-6)
                    print(
                        f"[scan] {done}/{len(jobs)} shards "
                        f"({rate:.1f}/s, eta {(len(jobs)-done)/max(rate,1e-6)/60:.1f} min)",
                        flush=True,
                    )

        report = {
            "config": vars(config),
            "asr_engines": engines,
            "errors": errors,
            "per_dataset": {
                name: {
                    "counters": counters.as_dict(),
                    "hours_total": round(counters.seconds_total / 3600.0, 2),
                    "hours_kept_prespeaker": round(counters.seconds_kept / 3600.0, 2),
                    "sample_rates": dict(sorted(sample_rates[name].items())),
                }
                for name, counters in sorted(totals.items())
            },
        }
        (out_dir / "scan_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[scan] wrote {out_dir / 'scan_report.json'}", flush=True)

    print("[gate] applying speaker-support gate ...", flush=True)
    stats = speaker_gate(
        index_dir, config, out_dir / "filtered_index.jsonl.gz", args.datasets
    )
    (out_dir / "gate_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

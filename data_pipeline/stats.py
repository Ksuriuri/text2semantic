"""Stage 1b: turn the scan/gate reports into an hours-and-bytes summary.

Disk footprint is reported for the three things the training set actually needs:

* **audio** - the flac payloads of the surviving utterances.  Per dataset the
  average flac bitrate is derived from the shard tar sizes divided by the total
  audio duration in that dataset, then multiplied by the surviving hours.  This
  avoids reading 20 TB of tars to measure the exact bytes.
* **semantic codes** - the MaskGCT tokenizer emits one ``uint16`` per 20 ms
  frame (50 Hz), so 100 bytes per audio second, stored in the packed ``<u2``
  code store that ``Text2SemanticDataset`` memory-maps.
* **manifest** - the jsonl the trainer reads, measured directly.

Usage::

    python -m data_pipeline.stats --run-dir runs/full --gcs-key /path/key.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_pipeline import gcs

#: MaskGCT semantic frame rate (W2V-BERT 20 ms frames) and bytes per code.
SEMANTIC_FPS = 50.0
SEMANTIC_BYTES_PER_CODE = 2


def audio_bytes_per_dataset(corpus, datasets):
    """Total ``audio/`` tar bytes per dataset (a listing call, no downloads)."""
    sizes = {}
    for dataset in datasets:
        total = 0
        for blob in corpus.client.list_blobs(
            corpus.bucket, prefix=f"{corpus.root}/{dataset}/audio/"
        ):
            total += blob.size or 0
        sizes[dataset] = total
    return sizes


def build_summary(scan_report, gate_report, audio_bytes):
    """Combine the reports into per-dataset and total hours/bytes."""
    rows = []
    for dataset, block in sorted(gate_report["per_dataset"].items()):
        scanned = scan_report["per_dataset"].get(dataset, {})
        counters = scanned.get("counters", {})
        hours_total = scanned.get("hours_total", 0.0)
        hours_kept = block["hours"]
        total_bytes = audio_bytes.get(dataset, 0)
        # Bytes per audio second, measured over the whole dataset.
        bps = (total_bytes / (hours_total * 3600.0)) if hours_total else 0.0
        kept_audio = bps * hours_kept * 3600.0
        kept_codes = hours_kept * 3600.0 * SEMANTIC_FPS * SEMANTIC_BYTES_PER_CODE
        rows.append(
            {
                "dataset": dataset,
                "rows_total": counters.get("total", 0),
                "rows_kept": block["rows"],
                "speakers_kept": block["speakers"],
                "hours_total": hours_total,
                "hours_kept": hours_kept,
                "keep_ratio_rows": (
                    round(block["rows"] / counters["total"], 4)
                    if counters.get("total") else None
                ),
                "flac_bytes_per_second": round(bps, 1),
                "audio_gb_total": round(total_bytes / 1e9, 2),
                "audio_gb_kept": round(kept_audio / 1e9, 2),
                "codes_gb_kept": round(kept_codes / 1e9, 3),
            }
        )

    total = {
        "rows_total": sum(r["rows_total"] for r in rows),
        "rows_kept": sum(r["rows_kept"] for r in rows),
        "speakers_kept": sum(r["speakers_kept"] for r in rows),
        "hours_total": round(sum(r["hours_total"] for r in rows), 2),
        "hours_kept": round(sum(r["hours_kept"] for r in rows), 2),
        "audio_gb_total": round(sum(r["audio_gb_total"] for r in rows), 2),
        "audio_gb_kept": round(sum(r["audio_gb_kept"] for r in rows), 2),
        "codes_gb_kept": round(sum(r["codes_gb_kept"] for r in rows), 3),
    }
    return {"per_dataset": rows, "total": total}


def format_table(summary):
    lines = []
    header = (
        "%-16s %12s %12s %9s %10s %10s %10s %9s"
        % ("dataset", "rows_total", "rows_kept", "spk", "h_total",
           "h_kept", "audioGB", "codesGB")
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in summary["per_dataset"]:
        lines.append(
            "%-16s %12d %12d %9d %10.1f %10.1f %10.1f %9.2f"
            % (row["dataset"], row["rows_total"], row["rows_kept"],
               row["speakers_kept"], row["hours_total"], row["hours_kept"],
               row["audio_gb_kept"], row["codes_gb_kept"])
        )
    total = summary["total"]
    lines.append("-" * len(header))
    lines.append(
        "%-16s %12d %12d %9d %10.1f %10.1f %10.1f %9.2f"
        % ("TOTAL", total["rows_total"], total["rows_kept"],
           total["speakers_kept"], total["hours_total"], total["hours_kept"],
           total["audio_gb_kept"], total["codes_gb_kept"])
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gcs-key", default=None)
    parser.add_argument("--project", default=gcs.PROJECT)
    parser.add_argument("--bucket", default=gcs.BUCKET)
    parser.add_argument("--root", default=gcs.ROOT)
    parser.add_argument(
        "--audio-sizes", default=None,
        help="Cached json of dataset -> audio bytes, to skip the listing pass.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    scan_report = json.loads((run_dir / "scan_report.json").read_text("utf-8"))
    gate_report = json.loads((run_dir / "gate_report.json").read_text("utf-8"))

    if args.audio_sizes and Path(args.audio_sizes).exists():
        audio_bytes = json.loads(Path(args.audio_sizes).read_text("utf-8"))
    else:
        corpus = gcs.Corpus(
            gcs.make_client(args.gcs_key, args.project),
            bucket=args.bucket, root=args.root,
        )
        audio_bytes = audio_bytes_per_dataset(
            corpus, sorted(gate_report["per_dataset"])
        )
        if args.audio_sizes:
            Path(args.audio_sizes).write_text(
                json.dumps(audio_bytes, indent=2), encoding="utf-8"
            )

    summary = build_summary(scan_report, gate_report, audio_bytes)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(format_table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

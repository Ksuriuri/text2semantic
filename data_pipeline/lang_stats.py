"""Stage 1c: per-language hours and disk footprint for a subset of datasets.

``stats.py`` answers "how much data survives, per dataset".  This module answers
"which languages does a chosen subset cover, and what does that subset cost on
disk" - the question you ask when deciding which datasets to encode.

It reads only the scan index (``<run-dir>/index/<dataset>/*.jsonl.gz``); no audio
and no GPU.  Two things make it more than a group-by over ``gate_report.json``:

* **The speaker gate is re-applied to the subset.**  The gate keys on
  ``(language, speaker_id)`` and needs >= ``--min-speaker-records`` clips plus one
  clip over ``--long-seconds``.  A corpus-wide run can satisfy that by pooling
  datasets the subset excludes, so reusing the corpus-wide gate would overcount.
  (For the seven character-labelled datasets it happens to make no difference -
  11,484 speakers either way - but that is a measured fact, not a given.)
* **Footprint is split into what you must keep versus what merely streams
  through.**  Training memory-maps the semantic codes and reads one reference
  clip per speaker to extract speaker features online, so the resident cost is
  ``codes + reference``.  The full audio column is what the encode stage
  downloads and discards, i.e. a one-off network cost, not storage.

Byte accounting matches ``stats.py``: 100 bytes per audio second for codes
(50 Hz x uint16), and a per-dataset flac bytes-per-second derived from the
``audio/`` tar sizes divided by that dataset's total duration.

Usage::

    python -m data_pipeline.lang_stats --run-dir runs/full \\
        --audio-sizes runs/audio_sizes.json --datasets vctk ears
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data_pipeline.stats import SEMANTIC_BYTES_PER_CODE, SEMANTIC_FPS

#: The datasets whose ``speaker_id`` is a genuine character/person label.
TRUSTED_DATASETS = (
    "Genshin", "StarRail", "WutheringWaves",
    "ears", "expresso", "hi_fi_tts", "vctk",
)

BYTES_PER_SECOND_OF_CODES = SEMANTIC_FPS * SEMANTIC_BYTES_PER_CODE


def _audio_bytes(sizes, dataset):
    """Accept both ``{ds: int}`` and ``{ds: {"bytes": int, ...}}`` caches."""
    raw = sizes[dataset]
    return raw if isinstance(raw, int) else raw["bytes"]


def index_shards(run_dir, datasets):
    shards = []
    for dataset in datasets:
        shards += sorted((Path(run_dir) / "index" / dataset).glob("*.jsonl.gz"))
    return shards


def _rows(paths):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def subset_speaker_gate(paths, min_records=2, long_seconds=6.0):
    """Speakers that pass the support gate *within this subset of datasets*."""
    counts = Counter()
    has_long = set()
    for row in _rows(paths):
        key = (row.get("language"), row["speaker_id"])
        counts[key] += 1
        if row["duration"] > long_seconds:
            has_long.add(key)
    keep = {k for k, n in counts.items() if n >= min_records and k in has_long}
    return keep, len(counts)


def collect(paths, keep, bytes_per_second, reference_clips_per_speaker=2):
    """Aggregate kept rows into ``(dataset, language)`` cells with byte costs.

    Reference bytes are estimated per cell as ``clips x mean clip duration x
    bytes-per-second``, because the reference clips are drawn from that cell.
    """
    cells = defaultdict(lambda: {"rows": 0, "seconds": 0.0, "speakers": set()})
    for row in _rows(paths):
        key = (row.get("language"), row["speaker_id"])
        if key not in keep:
            continue
        cell = cells[(row["dataset"], row.get("language"))]
        cell["rows"] += 1
        cell["seconds"] += row["duration"]
        cell["speakers"].add(key)

    out = {}
    for (dataset, language), cell in cells.items():
        bps = bytes_per_second[dataset]
        mean_duration = cell["seconds"] / cell["rows"]
        clips = min(reference_clips_per_speaker * len(cell["speakers"]), cell["rows"])
        out[(dataset, language)] = {
            "rows": cell["rows"],
            "seconds": cell["seconds"],
            "speakers": cell["speakers"],
            "reference_clips": clips,
            "audio_bytes": cell["seconds"] * bps,
            "reference_bytes": clips * mean_duration * bps,
            "codes_bytes": cell["seconds"] * BYTES_PER_SECOND_OF_CODES,
        }
    return out


def roll_up(cells, axis):
    """Sum cells along ``axis`` (0 = dataset, 1 = language), keeping the other."""
    other = 1 - axis
    groups = defaultdict(lambda: {
        "rows": 0, "seconds": 0.0, "speakers": set(), "reference_clips": 0,
        "audio_bytes": 0.0, "reference_bytes": 0.0, "codes_bytes": 0.0,
        "members": set(),
    })
    for key, cell in cells.items():
        group = groups[key[axis]]
        group["members"].add(key[other])
        group["speakers"] |= cell["speakers"]
        for field in ("rows", "reference_clips", "seconds", "audio_bytes",
                      "reference_bytes", "codes_bytes"):
            group[field] += cell[field]
    return groups


def totals(groups):
    total = {
        "rows": 0, "seconds": 0.0, "speakers": set(), "reference_clips": 0,
        "audio_bytes": 0.0, "reference_bytes": 0.0, "codes_bytes": 0.0,
    }
    for group in groups.values():
        total["speakers"] |= group["speakers"]
        for field in ("rows", "reference_clips", "seconds", "audio_bytes",
                      "reference_bytes", "codes_bytes"):
            total[field] += group[field]
    return total


def format_table(groups, label, members_label):
    header = "%-16s %10s %10s %9s %9s %9s %9s   %s" % (
        label, "rows", "hours", "spk", "codesGB", "refGB", "audioGB",
        members_label)
    lines = [header, "-" * len(header)]
    ordered = sorted(groups.items(), key=lambda kv: -kv[1]["seconds"])
    for key, group in ordered:
        lines.append("%-16s %10d %10.1f %9d %9.3f %9.2f %9.1f   %s" % (
            key, group["rows"], group["seconds"] / 3600.0, len(group["speakers"]),
            group["codes_bytes"] / 1e9, group["reference_bytes"] / 1e9,
            group["audio_bytes"] / 1e9, ",".join(sorted(group["members"]))))
    total = totals(groups)
    lines.append("-" * len(header))
    lines.append("%-16s %10d %10.1f %9d %9.3f %9.2f %9.1f" % (
        "TOTAL", total["rows"], total["seconds"] / 3600.0,
        len(total["speakers"]), total["codes_bytes"] / 1e9,
        total["reference_bytes"] / 1e9, total["audio_bytes"] / 1e9))
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="Scan run directory.")
    parser.add_argument(
        "--audio-sizes", required=True,
        help="Json cache of dataset -> audio bytes written by data_pipeline.stats.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=list(TRUSTED_DATASETS),
        help="Datasets to include (default: the 7 character-labelled ones).",
    )
    parser.add_argument("--reference-clips-per-speaker", type=int, default=2)
    parser.add_argument("--min-speaker-records", type=int, default=2)
    parser.add_argument("--long-seconds", type=float, default=6.0)
    parser.add_argument("--out", default=None, help="Optional json output path.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    scan_report = json.loads((run_dir / "scan_report.json").read_text("utf-8"))
    sizes = json.loads(Path(args.audio_sizes).read_text("utf-8"))

    paths = index_shards(run_dir, args.datasets)
    print(f"[plan] {len(args.datasets)} datasets, {len(paths)} index shards",
          flush=True)

    keep, seen = subset_speaker_gate(
        paths, args.min_speaker_records, args.long_seconds
    )
    print(f"[gate] speakers seen {seen} kept {len(keep)}", flush=True)

    bytes_per_second = {}
    for dataset in args.datasets:
        hours_total = scan_report["per_dataset"][dataset]["hours_total"]
        bytes_per_second[dataset] = (
            _audio_bytes(sizes, dataset) / (hours_total * 3600.0)
            if hours_total else 0.0
        )

    cells = collect(paths, keep, bytes_per_second, args.reference_clips_per_speaker)
    by_language = roll_up(cells, axis=1)
    by_dataset = roll_up(cells, axis=0)
    total = totals(by_language)

    print()
    print(format_table(by_language, "language", "datasets"))
    print()
    print(format_table(by_dataset, "dataset", "languages"))
    print()
    resident = total["codes_bytes"] + total["reference_bytes"]
    print("reference clips per speaker : %d" % args.reference_clips_per_speaker)
    print("kept hours                  : %.1f h" % (total["seconds"] / 3600.0))
    print("codes (packed uint16)       : %.2f GB" % (total["codes_bytes"] / 1e9))
    print("reference audio to keep     : %.2f GB" % (total["reference_bytes"] / 1e9))
    print("min on-disk for training    : %.2f GB" % (resident / 1e9))
    print("ALL kept audio (streamed)   : %.2f GB" % (total["audio_bytes"] / 1e9))

    if args.out:
        payload = {
            "datasets": args.datasets,
            "speakers_kept": len(keep),
            "by_language": {
                lang: {
                    "rows": g["rows"],
                    "hours": round(g["seconds"] / 3600.0, 4),
                    "speakers": len(g["speakers"]),
                    "datasets": sorted(g["members"]),
                    "codes_gb": round(g["codes_bytes"] / 1e9, 4),
                    "reference_gb": round(g["reference_bytes"] / 1e9, 4),
                    "audio_gb": round(g["audio_bytes"] / 1e9, 2),
                }
                for lang, g in sorted(by_language.items())
            },
            "by_dataset": {
                ds: {
                    "rows": g["rows"],
                    "hours": round(g["seconds"] / 3600.0, 4),
                    "speakers": len(g["speakers"]),
                    "languages": sorted(g["members"]),
                    "codes_gb": round(g["codes_bytes"] / 1e9, 4),
                    "reference_gb": round(g["reference_bytes"] / 1e9, 4),
                    "audio_gb": round(g["audio_bytes"] / 1e9, 2),
                }
                for ds, g in sorted(by_dataset.items())
            },
            "total": {
                "rows": total["rows"],
                "hours": round(total["seconds"] / 3600.0, 4),
                "speakers": len(total["speakers"]),
                "codes_gb": round(total["codes_bytes"] / 1e9, 4),
                "reference_gb": round(total["reference_bytes"] / 1e9, 4),
                "resident_gb": round(resident / 1e9, 4),
                "audio_gb": round(total["audio_bytes"] / 1e9, 2),
            },
        }
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[out] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

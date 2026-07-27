"""Stage 2: materialise audio, tokenise it and write the training manifest.

Input is the filtered index from :mod:`data_pipeline.scan`.  For each shard the
worker:

1. streams the shard's ``audio/<ds>-NNNNNN.tar`` from GCS once and extracts only
   the surviving members;
2. runs :class:`qwen_tts.semantic_codec.MaskGCTSemanticTokenizer` over them;
3. appends the codes to a packed ``<u2`` code store and records
   ``semantic_code_path`` / ``semantic_code_offset`` / ``semantic_code_length``;
4. keeps the audio only when it is needed as a speaker reference clip.

Why the code store instead of inline ``semantic_codes``: at ~50 codes per second
the whole corpus of codes is a few GB, while inlining them into jsonl inflates
the manifest by an order of magnitude and makes it slow to load.
``Text2SemanticDataset`` already supports the memory-mapped form.

Why some audio is retained: the training dataset extracts speaker features from
a *reference* audio file online, and that file must be a different utterance of
the same speaker.  The features cannot be frozen offline because the speaker
encoder is trainable.  ``--reference-clips-per-speaker`` therefore keeps a small
number of clips per speaker instead of the full corpus.

Usage::

    python -m data_pipeline.encode \
        --index runs/full/filtered_index.jsonl.gz \
        --out-dir /mnt/data_3t_1/t2s_train \
        --model-dir checkpoints/maskgct \
        --gcs-key /path/gcs-key.json --device cuda:0
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from data_pipeline import gcs, pairing

CODE_DTYPE = "<u2"


def load_index(path):
    """Group index rows by ``(dataset, shard)`` so each tar is streamed once."""
    opener = gzip.open if str(path).endswith(".gz") else open
    grouped = defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped[(row["dataset"], row["shard"])].append(row)
    return grouped


def choose_reference_clips(rows_by_shard, per_speaker, min_w_distance=0,
                           drop_consecutive_groups=False):
    """Pick which utterances must keep their audio, as speaker references.

    The longest clips of a speaker make the best references, and the gate has
    already guaranteed at least one clip longer than 6 s per speaker.

    With ``min_w_distance > 0`` the picks are additionally spread along the
    source recording (see ``pairing.spread_reference_clips``).  Longest-first
    alone tends to return adjacent cuts of one continuous utterance -- 85.5% of
    laion's ``B_S`` groups are exactly that -- which makes the reference nearly
    identical to the target and teaches copying instead of timbre transfer.
    With ``drop_consecutive_groups`` a group that is one unbroken ``_W`` run does
    NOT get backfilled: it loses the speaker instead of taking a near-repeat
    reference.  Off by default because it discards data; it is the right trade
    only where the backfilled pair is low value, which the adjacency structure
    establishes for laion's slice half (see ``pairing.spread_reference_clips``).

    Returns the id set plus a report of what the constraint cost.
    """
    by_speaker = defaultdict(list)
    for rows in rows_by_shard.values():
        for row in rows:
            by_speaker[(row.get("language"), row["speaker_id"])].append(row)
    keep = set()
    dropped_speakers = 0
    for rows in by_speaker.values():
        if min_w_distance > 0:
            backfill = not (drop_consecutive_groups
                            and pairing.is_consecutive_run(rows))
            picked = pairing.spread_reference_clips(rows, per_speaker,
                                                    min_w_distance,
                                                    backfill=backfill)
            if not backfill and len(picked) < per_speaker:
                dropped_speakers += 1
        else:
            picked = sorted(rows, key=lambda r: -r["duration"])[:per_speaker]
        for row in picked:
            keep.add(row["id"])
    report = None
    if min_w_distance > 0:
        report = pairing.summarize(by_speaker, min_w_distance)
        if drop_consecutive_groups:
            report["speakers_dropped_consecutive"] = dropped_speakers
    return keep, report


def build_tokenizer(model_dir, device):
    from qwen_tts.semantic_codec import MaskGCTSemanticTokenizer

    model_dir = Path(model_dir)
    return MaskGCTSemanticTokenizer(
        w2v_bert_path=model_dir / "w2v-bert-2.0",
        stats_path=model_dir / "wav2vec2bert_stats.pt",
        repcodec_config_path=model_dir / "semantic_codec" / "config.yaml",
        repcodec_checkpoint_path=model_dir / "semantic_codec" / "model.safetensors",
        device=device,
    )


def shard_texts(corpus, dataset, shard, wanted_ids):
    """Recover the training transcript for the surviving ids of one shard.

    The scan index intentionally stores no text - carrying 75 GB of transcripts
    through the index would defeat its purpose - so the transcript is re-read
    here.  The original metadata transcript wins when present, otherwise the
    Cohere ASR result is used (the scan has already verified that one of the two
    exists and that the pair agrees).
    """
    from data_pipeline.filters import as_text

    root, name = corpus.root, dataset
    meta_blob = f"{root}/{name}/metadata/{name}-{shard:06d}.jsonl"
    texts = {}
    missing = set(wanted_ids)
    for row in corpus.read_jsonl(meta_blob):
        utt_id = as_text(row.get("id"))
        if utt_id not in missing:
            continue
        text = as_text(row.get("text"))
        if text:
            texts[utt_id] = text
            missing.discard(utt_id)
    if missing:
        cohere_blob = (
            f"{root}/{name}/asr/{gcs.COHERE_ENGINE}/{name}-{shard:06d}.jsonl"
        )
        for row in corpus.read_jsonl(cohere_blob):
            utt_id = as_text(row.get("id"))
            if utt_id in missing:
                text = as_text(row.get("text"))
                if text:
                    texts[utt_id] = text
    return texts


def encode_shard(corpus, tokenizer, dataset, shard, rows, out_dir,
                 reference_ids, keep_audio_dir, batch_size=8):
    """Encode one shard; returns the manifest rows and the code array."""
    texts = shard_texts(corpus, dataset, shard, {r["id"] for r in rows})
    wanted = {}
    for row in rows:
        member = gcs.tar_member_name(row["audio_path"])
        if member is None:
            continue
        wanted[member] = Path(member).name
        row["_member"] = member

    with tempfile.TemporaryDirectory(prefix=f"{dataset}-{shard:06d}-") as scratch:
        written = corpus.extract_audio(dataset, shard, wanted, scratch)
        local = {}
        for member, path in written.items():
            local[member] = path

        manifest = []
        codes_chunks = []
        offset = 0
        for row in rows:
            member = row.get("_member")
            audio_path = local.get(member)
            if audio_path is None:
                continue
            text = texts.get(row["id"])
            if not text:
                # Nothing to condition on; the trainer requires a transcript.
                continue
            try:
                codes = tokenizer.encode_file(audio_path).numpy().astype(CODE_DTYPE)
            except Exception as error:
                print(f"[warn] encode failed {row['id']}: "
                      f"{type(error).__name__}: {error}", flush=True)
                continue

            kept_audio = None
            if row["id"] in reference_ids and keep_audio_dir is not None:
                target = Path(keep_audio_dir) / dataset / Path(member).name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(audio_path, target)
                kept_audio = str(target)

            manifest.append(
                {
                    "id": row["id"],
                    "text": text,
                    "dataset": dataset,
                    "language": row.get("language"),
                    "speaker_id": row["speaker_id"],
                    "duration": row["duration"],
                    "sample_rate": row["sample_rate"],
                    "audio": kept_audio,
                    "semantic_code_offset": offset,
                    "semantic_code_length": int(codes.size),
                }
            )
            codes_chunks.append(codes)
            offset += int(codes.size)

    if not codes_chunks:
        return [], None
    return manifest, np.concatenate(codes_chunks)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", required=True, help="filtered_index.jsonl.gz")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-dir", required=True,
                        help="Directory produced by scripts/download_models.py")
    parser.add_argument("--gcs-key", default=None)
    parser.add_argument("--project", default=gcs.PROJECT)
    parser.add_argument("--bucket", default=gcs.BUCKET)
    parser.add_argument("--root", default=gcs.ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-clips-per-speaker", type=int, default=2)
    parser.add_argument("--min-w-distance", type=int,
                        default=pairing.DEFAULT_MIN_W_DISTANCE,
                        help="Minimum source-window (_W) distance between a "
                             "speaker's reference clips. 0 disables the "
                             "constraint and restores plain longest-first, "
                             "which on laion picks adjacent cuts of one "
                             "utterance.")
    parser.add_argument("--drop-consecutive-groups", action="store_true",
                        help="Drop a speaker whose clips form one unbroken _W "
                             "run instead of backfilling a near-repeat "
                             "reference. Discards data, so off by default; it "
                             "is the better trade only for laion's slice half, "
                             "where the backfilled pair is low value -- 85.5% of "
                             "those groups are one continuous run, so the "
                             "backfill is a near-repeat of the target.")
    parser.add_argument("--no-keep-audio", action="store_true",
                        help="Do not retain any audio (manifest will lack refs).")
    parser.add_argument("--shard-limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    (out_dir / "codes").mkdir(parents=True, exist_ok=True)
    manifest_dir = out_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    keep_audio_dir = None if args.no_keep_audio else out_dir / "reference_audio"

    grouped = load_index(args.index)
    if args.no_keep_audio:
        reference_ids, pair_report = set(), None
    else:
        reference_ids, pair_report = choose_reference_clips(
            grouped, args.reference_clips_per_speaker, args.min_w_distance,
            args.drop_consecutive_groups)
    keys = sorted(grouped)
    if args.shard_limit:
        keys = keys[: args.shard_limit]
    print(f"[encode] shards={len(keys)} rows={sum(len(grouped[k]) for k in keys)} "
          f"reference_clips={len(reference_ids)}", flush=True)
    if pair_report:
        # Printed because the constraint silently costs recall otherwise: a
        # speaker with 0 usable pairs contributes only its reference.
        print(f"[encode] pairing constraint: {pair_report}", flush=True)

    corpus = gcs.Corpus(
        gcs.make_client(args.gcs_key, args.project),
        bucket=args.bucket, root=args.root,
    )
    tokenizer = build_tokenizer(args.model_dir, args.device)

    for dataset, shard in keys:
        code_path = out_dir / "codes" / dataset / f"{dataset}-{shard:06d}.u2.bin"
        manifest_path = manifest_dir / dataset / f"{dataset}-{shard:06d}.jsonl"
        if args.skip_existing and code_path.exists() and manifest_path.exists():
            continue
        code_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest, codes = encode_shard(
            corpus, tokenizer, dataset, shard, grouped[(dataset, shard)],
            out_dir, reference_ids, keep_audio_dir,
        )
        if codes is None:
            print(f"[encode] {dataset}-{shard:06d}: nothing encoded", flush=True)
            continue
        tmp = code_path.with_suffix(".tmp")
        codes.tofile(tmp)
        os.replace(tmp, code_path)
        with open(manifest_path, "w", encoding="utf-8") as sink:
            for row in manifest:
                row["semantic_code_path"] = str(code_path)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[encode] {dataset}-{shard:06d}: rows={len(manifest)} "
              f"codes={codes.size}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

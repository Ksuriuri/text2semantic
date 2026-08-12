#!/usr/bin/env python3
"""Turn the bucket's MaskGCT code shards into a trainer manifest.

The bucket layout is a *shared data base*: one packed `.u2.bin` plus one index
`.jsonl` per audio tar, every sample encoded, nothing filtered. A training run
needs something narrower and slightly differently spelled, and both consumers
in `noiz-tts` already read the packed layout correctly but expect their own
field names and path conventions:

  text2semantic/finetuning/dataset.py    `_semantic_codes()` memmaps `<u2` and
      slices `[offset : offset+length]`. It passes `semantic_code_path` to
      `np.memmap` **verbatim** -- it does not resolve relative paths -- so the
      manifest must carry a path that resolves from the training process's cwd.
      Filters on `duration` (`max_target_seconds`), `semantic_code_length`
      (`max_semantic_tokens`), and needs >= `min_speaker_records` rows per
      speaker plus a second audio file for the speaker reference.

  semantic2any/semantic2any/data/s2mel_dataset.py    `_load_semantic_codes()`
      memmaps `<u2` the same way, and `_resolve_path()` does resolve relative
      paths, but against the *manifest's* directory. It reads `semantic_fps`
      (not our `semantic_frame_rate`), and `_has_semantic_codes()` requires
      `semantic_lookup_path` + `semantic_lookup_sha256`; a batch mixing two
      lookup paths raises "A batch must use one semantic lookup table and
      checksum".

So three concrete gaps, all on the writer's side, none needing a change to
either trainer:

  1. `semantic_frame_rate` -> also emit `semantic_fps` (same value: 25.0 for
     indextts25, 50.0 for maskgct).
  2. `semantic_code_path` is a bare filename in the bucket index (deliberately,
     so a shard stays relocatable) -> rewrite to a real path.
  3. no decode-side artifact in the bucket -> `--emit-lookup` materializes one
     from the codec checkpoint and stamps its sha256 on every row. It is a
     function of the checkpoint, not of the data, which is why it does not
     belong in the per-tar shards.

What that artifact is depends on the codec, and both are byte-compatible with
what `semantic2any/scripts/precompute_maskgct_codes.py` writes:

  * maskgct     `maskgct_lookup.pt`, the frozen 8192x1024 post-projection
                embedding `quantizer.vq2emb(arange(8192))`.
  * indextts25  `indextts25_decoder.pt`, the decode-side *weights*
                (quantizer + decoder + up). This codec has no per-code table:
                decode runs a ConvNeXt stack over the whole sequence, so a code
                has no context-free embedding to tabulate.

Usage::

    # local shards -> manifest for semantic2any (paths relative to the manifest)
    python features/build_manifest.py --codes-dir /data/maskgct/vctk \\
        --out /data/manifests/vctk.jsonl --emit-lookup

    # for text2semantic, which memmaps the string as-is
    python features/build_manifest.py --codes-dir /data/maskgct/vctk \\
        --out /data/manifests/vctk.jsonl --path-style absolute \\
        --max-target-seconds 30 --min-speaker-records 2

    # straight from the bucket (downloads each shard once into --cache-dir)
    python features/build_manifest.py --dataset vctk --out vctk.jsonl \\
        --cache-dir /data/maskgct-cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from read_codes import CODE_DTYPE, DEFAULT_ROOT, FEATURE_DIR, fetch_shard

#: (artifact name, code fps) per codec.  indextts25 is IndexTTS-2.5's
#: EnhancedCodec and the default, matching the tokenizer default in
#: qwen_tts.semantic_codec.
SEMANTIC_CODECS = {
    "indextts25": ("indextts25_decoder.pt", 25.0),
    "maskgct": ("maskgct_lookup.pt", 50.0),
}
DEFAULT_SEMANTIC_CODEC = "indextts25"

#: Spellings of the same codec that may appear in bucket rows.
SEMANTIC_CODEC_ALIASES = {
    "indextts2.5": "indextts25",
    "indextts-2.5": "indextts25",
    "indextts_2.5": "indextts25",
    "enhancedcodec": "indextts25",
}


def canonical_semantic_codec(name):
    return SEMANTIC_CODEC_ALIASES.get(str(name).strip().lower(),
                                      str(name).strip().lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_codec(checkpoint_dir: Path, codec_type: str, device: str = "cpu"):
    """Build the codec from a scripts/download_models.py checkpoint dir.

    Needs this project's ``qwen_tts.semantic_codec``, which is the only copy that
    carries the decode side. The SpeechData copy of this script (see README) can
    still build a manifest and reuse an existing artifact via ``--lookup-path``;
    it just cannot materialize one until the same change lands upstream.
    """
    try:
        from qwen_tts.semantic_codec import (
            RepCodec,
            load_codec_checkpoint,
            load_codec_config,
            resolve_codec_assets,
        )
    except ImportError as error:  # running from the SpeechData copy
        raise RuntimeError(
            "--emit-lookup needs qwen_tts.semantic_codec (the decode side lives "
            "there). Run this from text2semantic, or pass --lookup-path to reuse "
            f"an artifact built there: {error}"
        ) from error

    config_path, checkpoint_path = resolve_codec_assets(checkpoint_dir, codec_type)
    codec = RepCodec(**load_codec_config(config_path, codec_type)).to(device)
    load_codec_checkpoint(codec, checkpoint_path)
    codec.eval()
    return codec, Path(checkpoint_path)


def build_lookup(out_path: Path, *, checkpoint_dir: Path,
                 codec_type: str = DEFAULT_SEMANTIC_CODEC,
                 device: str = "cpu") -> Path:
    """Materialize the decode-side artifact from the codec checkpoint.

    Written once per manifest tree, not per shard: it depends only on the
    RepCodec weights. Skipped if a matching file already exists, and a
    *mismatching* one is an error rather than an overwrite -- a manifest whose
    rows claim a checksum the file no longer has would fail deep inside
    training instead of here.

    For maskgct this is the [8192, 1024] lookup table. For indextts25 there is no
    such table (its decode is context dependent), so the artifact carries the
    decode-side weights in the shape semantic2any's IndexTTS25CodeDecoder loads.
    """
    # Imported lazily and *after* the reuse check: building the artifact needs
    # the RepCodec weights and semantic_codec.py, but reusing an existing one
    # needs neither, so a manifest built next to an existing artifact runs
    # anywhere.
    if out_path.exists():
        return out_path

    import torch

    codec_type = canonical_semantic_codec(codec_type)
    codec, checkpoint_path = _load_codec(checkpoint_dir, codec_type, device)

    if codec_type == "indextts25":
        state = {
            key: value.detach().cpu().clone()
            for key, value in codec.state_dict().items()
            if key.startswith(("quantizer.", "decoder.", "up."))
        }
        if not state:
            raise ValueError("Refusing to write an empty decoder bundle")
        payload = {
            "codec_type": "indextts25",
            "frames_per_code": int(codec.downsample_scale),
            "semantic_dim": int(codec.hidden_size),
            "arch": {
                "codebook_size": int(codec.codebook_size),
                "hidden_size": int(codec.hidden_size),
                "codebook_dim": int(codec.codebook_dim),
                "vocos_dim": int(codec.vocos_dim),
                "vocos_intermediate_dim": int(codec.vocos_intermediate_dim),
                "vocos_num_layers": int(codec.vocos_num_layers),
                "num_quantizers": int(codec.num_quantizers),
                "downsample_scale": int(codec.downsample_scale),
            },
            "state_dict": state,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
        }
    else:
        with torch.no_grad():
            codes = torch.arange(int(codec.quantizer.codebook_size),
                                 device=device, dtype=torch.long)
            # vq2emb takes [n_quantizers, B, T]; one codebook, one "batch" of 8192.
            lookup = codec.quantizer.vq2emb(codes.view(1, 1, -1))
            lookup = lookup.transpose(1, 2).reshape(codes.numel(), -1)
            payload = {"lookup": lookup.detach().cpu().float().contiguous()}

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    return out_path


def iter_shards(codes_dir: Path):
    for index_path in sorted(codes_dir.glob("*.jsonl")):
        yield index_path


def manifest_rows(index_path: Path, *, codes_dir: Path, manifest_dir: Path,
                  path_style: str, codec_type: str = DEFAULT_SEMANTIC_CODEC,
                  semantic_fps: float = 25.0):
    """Yield trainer-shaped rows for one shard's index."""
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "encoded" or row.get("semantic_code_offset") is None:
            continue  # skipped_short_audio carries no codes

        blob = (codes_dir / row["semantic_code_path"]).resolve()
        if path_style == "absolute":
            code_path = str(blob)
        else:
            code_path = os.path.relpath(blob, manifest_dir)
        row = dict(row)
        row["semantic_code_path"] = code_path
        # s2mel_dataset reads `semantic_fps`; keep `semantic_frame_rate` too so
        # the row stays readable by anything written against the bucket schema.
        row["semantic_fps"] = row.get("semantic_frame_rate", semantic_fps)
        row["semantic_codebooks"] = 1
        # A bucket row that already names its codec wins: it was written by the
        # job that produced these ids, and disagreeing with it would mislabel
        # them. Normalized, because that job spells it "indextts2.5".
        row["semantic_codec"] = canonical_semantic_codec(
            row.get("semantic_codec") or codec_type
        )
        if row["semantic_codec"] != codec_type:
            raise ValueError(
                f"{index_path} was encoded with {row['semantic_codec']!r} but "
                f"--semantic-codec is {codec_type!r}; the decode-side artifact "
                "would not match the codes"
            )
        yield row


def keep(row: dict, args: argparse.Namespace) -> str | None:
    """Return the name of the rule that drops this row, or None to keep it.

    The rules mirror what the two trainers apply themselves, so that filtering
    here only ever removes rows they would have dropped anyway -- the manifest
    is a training-run decision, the bucket stays complete.
    """
    duration = row.get("duration")
    duration = float(duration) if duration not in (None, "", "None") else None
    if duration is not None:
        if args.min_target_seconds is not None and duration < args.min_target_seconds:
            return "min_target_seconds"
        if args.max_target_seconds is not None and duration > args.max_target_seconds:
            return "max_target_seconds"
    length = int(row["semantic_code_length"])
    if args.max_semantic_tokens is not None and length > args.max_semantic_tokens:
        return "max_semantic_tokens"
    if args.min_sample_rate is not None:
        # Source-recording quality, not a tokenizer requirement -- the codes were
        # produced from audio loaded at 16 kHz regardless, so upsampled low-rate
        # material is indistinguishable to the tokenizer but still sounds worse.
        # A row with no sample_rate at all is dropped rather than assumed good.
        sample_rate = row.get("sample_rate")
        try:
            sample_rate = float(sample_rate) if sample_rate not in (None, "", "None") else None
        except (TypeError, ValueError):
            sample_rate = None
        if sample_rate is None or sample_rate < args.min_sample_rate:
            return "min_sample_rate"
    if args.require_speaker_id:
        speaker = row.get("speaker_id")
        # Upstream exports write missing values as the *string* "None".
        if speaker in (None, "", "None"):
            return "no_speaker_id"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("source")
    source.add_argument("--codes-dir", type=Path,
                        help="local directory of <shard>.jsonl + <shard>.u2.bin")
    source.add_argument("--dataset", help="fetch the dataset's shards from GCS")
    source.add_argument("--shard", action="append", default=None,
                        help="limit to these shards (repeatable)")
    source.add_argument("--cache-dir", type=Path, default=Path("/tmp/maskgct-codes-cache"))
    source.add_argument("--root", default=DEFAULT_ROOT)
    source.add_argument("--token", default="gcs-key.json",
                        help="service-account json; ignored when the file is "
                             "absent, so a GCE VM uses its attached account")

    parser.add_argument("--out", type=Path, required=True, help="manifest jsonl to write")
    parser.add_argument("--path-style", choices=("relative", "absolute"),
                        default="relative",
                        help="relative: resolved against the manifest dir "
                             "(semantic2any). absolute: required by text2semantic, "
                             "which memmaps the string verbatim.")
    parser.add_argument("--semantic-codec", default=DEFAULT_SEMANTIC_CODEC,
                        choices=sorted(SEMANTIC_CODECS),
                        help="codec that produced these shards. indextts25 "
                             "(default) has 25 Hz codes and ships decoder "
                             "weights; maskgct has 50 Hz codes and a lookup "
                             "table. A shard that names its own codec must "
                             "agree with this.")
    parser.add_argument("--emit-lookup", action="store_true",
                        help="write the decode-side artifact next to the "
                             "manifest and stamp semantic_lookup_path/_sha256 "
                             "on every row (semantic2any requires both)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/maskgct"))
    parser.add_argument("--lookup-path", type=Path, default=None,
                        help="reuse an existing lookup table instead of building one")

    flt = parser.add_argument_group("filters (default: keep everything)")
    flt.add_argument("--min-target-seconds", type=float, default=None)
    flt.add_argument("--max-target-seconds", type=float, default=None,
                     help="text2semantic's own default is 30.0")
    flt.add_argument("--max-semantic-tokens", type=int, default=None)
    flt.add_argument("--min-sample-rate", type=int, default=None,
                     help="drop rows recorded below N Hz; text2semantic's "
                          "data_pipeline uses 22050. Rows with no sample_rate "
                          "are dropped too, since quality can't be established")
    flt.add_argument("--require-speaker-id", action="store_true",
                     help="drop rows whose speaker_id is missing or the string "
                          "'None'; text2semantic cannot use them (it needs a "
                          "speaker reference clip)")
    flt.add_argument("--min-speaker-records", type=int, default=None,
                     help="drop speakers with fewer than N kept rows; "
                          "text2semantic needs >=2 to pair a reference clip")
    args = parser.parse_args()

    if not args.codes_dir and not args.dataset:
        parser.error("pass --codes-dir or --dataset")

    manifest_dir = args.out.parent.resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    codec_type = canonical_semantic_codec(args.semantic_codec)
    lookup_name, semantic_fps = SEMANTIC_CODECS[codec_type]

    lookup_path = args.lookup_path
    lookup_sha = None
    if args.emit_lookup:
        lookup_path = lookup_path or (manifest_dir / lookup_name)
        lookup_path = build_lookup(Path(lookup_path),
                                   checkpoint_dir=args.checkpoint_dir,
                                   codec_type=codec_type)
        lookup_sha = sha256_file(Path(lookup_path))

    if args.codes_dir:
        codes_dir = args.codes_dir.resolve()
        indexes = list(iter_shards(codes_dir))
        if args.shard:
            wanted = {s if s.endswith(".jsonl") else f"{s}.jsonl" for s in args.shard}
            indexes = [p for p in indexes if p.name in wanted]
    else:
        import gcsfs

        # A key file is the fallback, not the default: on a GCE VM the attached
        # service account works through the metadata server, and passing a
        # nonexistent path makes gcsfs raise "token is either not valid, or
        # expired" instead of falling back. Same precedence as features/run.sh.
        token = args.token
        if token and token != "anon" and not Path(token).is_file():
            token = None
        fs = gcsfs.GCSFileSystem(token=token)
        remote = f"{args.root}/{args.dataset}/{FEATURE_DIR}"
        shards = args.shard or sorted(
            Path(name).stem for name in fs.ls(remote)
            if name.endswith(".jsonl") and "/errors/" not in name
        )
        codes_dir = (args.cache_dir / args.dataset).resolve()
        indexes = [fetch_shard(args.dataset, shard, codes_dir,
                               root=args.root, token=args.token)
                   for shard in shards]

    dropped = Counter()
    rows = []
    for index_path in indexes:
        for row in manifest_rows(index_path, codes_dir=codes_dir,
                                 manifest_dir=manifest_dir,
                                 path_style=args.path_style,
                                 codec_type=codec_type,
                                 semantic_fps=semantic_fps):
            reason = keep(row, args)
            if reason:
                dropped[reason] += 1
                continue
            if lookup_sha:
                lp = Path(lookup_path).resolve()
                row["semantic_lookup_path"] = (
                    str(lp) if args.path_style == "absolute"
                    else os.path.relpath(lp, manifest_dir))
                row["semantic_lookup_sha256"] = lookup_sha
            rows.append(row)

    if args.min_speaker_records:
        counts = Counter(row.get("speaker_id") for row in rows)
        before = len(rows)
        rows = [row for row in rows
                if counts[row.get("speaker_id")] >= args.min_speaker_records]
        dropped["min_speaker_records"] += before - len(rows)

    tmp_out = args.out.with_suffix(args.out.suffix + ".part")
    with tmp_out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_out, args.out)

    codes = sum(int(row["semantic_code_length"]) for row in rows)
    print(json.dumps({
        "manifest": str(args.out),
        "shards": len(indexes),
        "rows": len(rows),
        "semantic_codec": codec_type,
        "semantic_fps": semantic_fps,
        "audio_hours": round(codes / semantic_fps / 3600, 2),
        "dropped": dict(dropped),
        "path_style": args.path_style,
        "lookup": str(lookup_path) if lookup_sha else None,
        "code_dtype": CODE_DTYPE,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

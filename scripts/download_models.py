#!/usr/bin/env python3
"""Download every model weight the data pipeline needs into one directory.

The semantic tokenizer is assembled from upstream repositories:

* ``facebook/w2v-bert-2.0``  - the frozen speech encoder; layer 17 hidden
  states are the tokenizer input.
* ``amphion/MaskGCT``        - the single-codebook RepCodec quantizer
  (``--semantic-codec maskgct``, 50 Hz codes).
* ``IndexTeam/IndexTTS-2.5`` - the EnhancedCodec quantizer ``codec.pth``
  (the default, 25 Hz codes).  Same architecture with ``downsample_scale: 2``.
* ``IndexTeam/IndexTTS-2``   - the W2V-BERT feature mean/std statistics
  (``wav2vec2bert_stats.pt``).  Note this file is *not* published in the
  amphion/MaskGCT repo; IndexTTS-2 is where it is distributed.

All three are public on the Hugging Face Hub.  Layout produced here matches what
``data_pipeline.encode --model-dir`` and ``finetuning/prepare_data.py`` expect::

    <model-dir>/
        w2v-bert-2.0/{config.json,preprocessor_config.json,model.safetensors}
        wav2vec2bert_stats.pt
        enhanced_codec/{config.yaml,codec.pth}        # indextts25 (default)
        semantic_codec/{config.yaml,model.safetensors}  # maskgct

Neither ``config.yaml`` is published upstream - both projects fix the
architecture in code - so they are written from ``configs/*_semantic.yaml`` in
this project.

Usage::

    python scripts/download_models.py --out-dir checkpoints/maskgct
    # only the codec you intend to use:
    python scripts/download_models.py --out-dir ... --semantic-codec indextts25
    # behind the GFW:
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py --out-dir ...
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

W2V_BERT_REPO = "facebook/w2v-bert-2.0"
W2V_BERT_FILES = (
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
)

#: (repo id, repo path, local destination) for assets both codecs need.
TOKENIZER_FILES = (
    ("IndexTeam/IndexTTS-2", "wav2vec2bert_stats.pt", "wav2vec2bert_stats.pt"),
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: (repo id, repo path, local destination, architecture config) per codec.
CODEC_ASSETS = {
    "indextts25": (
        "IndexTeam/IndexTTS-2.5", "codec.pth", "enhanced_codec/codec.pth",
        REPO_ROOT / "configs" / "enhanced_codec_semantic.yaml",
    ),
    "maskgct": (
        "amphion/MaskGCT", "semantic_codec/model.safetensors",
        "semantic_codec/model.safetensors",
        REPO_ROOT / "configs" / "repcodec_semantic.yaml",
    ),
}


def fetch(repo_id, filename, out_path, revision=None):
    from huggingface_hub import hf_hub_download

    out_path = Path(out_path)
    if out_path.exists():
        print(f"[skip] {out_path} already present")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {repo_id}:{filename}")
    cached = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    shutil.copyfile(cached, out_path)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--stats-from", default=None,
        help="Copy wav2vec2bert_stats.pt from a local path instead of the Hub.",
    )
    parser.add_argument(
        "--semantic-codec", default="both",
        choices=("both", *CODEC_ASSETS),
        help="Which quantizer(s) to fetch. Default both, so an existing "
             "maskgct model-dir keeps working while indextts25 is the "
             "tokenizer default.",
    )
    parser.add_argument(
        "--codec-from", default=None,
        help="Copy the quantizer checkpoint from a local path instead of the "
             "Hub (e.g. an existing IndexTTS-2.5 checkpoints/codec.pth).",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in W2V_BERT_FILES:
        fetch(W2V_BERT_REPO, filename, out_dir / "w2v-bert-2.0" / filename,
              args.revision)

    for repo_id, repo_path, local in TOKENIZER_FILES:
        if local == "wav2vec2bert_stats.pt" and args.stats_from:
            target = out_dir / local
            if not target.exists():
                shutil.copyfile(args.stats_from, target)
                print(f"[copy] {args.stats_from} -> {target}")
            continue
        fetch(repo_id, repo_path, out_dir / local, args.revision)

    wanted = (
        tuple(CODEC_ASSETS) if args.semantic_codec == "both"
        else (args.semantic_codec,)
    )
    if args.codec_from and len(wanted) != 1:
        parser.error("--codec-from needs a single --semantic-codec")
    for codec in wanted:
        repo_id, repo_path, local, config_source = CODEC_ASSETS[codec]
        target = out_dir / local
        if args.codec_from:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(args.codec_from, target)
                print(f"[copy] {args.codec_from} -> {target}")
        else:
            fetch(repo_id, repo_path, target, args.revision)
        config_target = target.parent / "config.yaml"
        shutil.copyfile(config_source, config_target)
        print(f"[copy] {config_source} -> {config_target}")

    print("\nmodel-dir ready:")
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out_dir)}  {path.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

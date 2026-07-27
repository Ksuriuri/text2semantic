#!/usr/bin/env python3
"""Download every model weight the data pipeline needs into one directory.

The MaskGCT semantic tokenizer is assembled from two upstream repositories:

* ``facebook/w2v-bert-2.0``  - the frozen speech encoder; layer 17 hidden
  states are the tokenizer input.
* ``amphion/MaskGCT``        - the single-codebook RepCodec quantizer.
* ``IndexTeam/IndexTTS-2``   - the W2V-BERT feature mean/std statistics
  (``wav2vec2bert_stats.pt``).  Note this file is *not* published in the
  amphion/MaskGCT repo; IndexTTS-2 is where it is distributed.

All three are public on the Hugging Face Hub.  Layout produced here matches what
``data_pipeline.encode --model-dir`` and ``finetuning/prepare_data.py`` expect::

    <model-dir>/
        w2v-bert-2.0/{config.json,preprocessor_config.json,model.safetensors}
        wav2vec2bert_stats.pt
        semantic_codec/{config.yaml,model.safetensors}

``semantic_codec/config.yaml`` is not published by the upstream repo - the
architecture is fixed in code there - so it is written from
``configs/repcodec_semantic.yaml`` in this project.

Usage::

    python scripts/download_models.py --out-dir checkpoints/maskgct
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

#: (repo id, repo path, local destination) for the tokenizer assets.
TOKENIZER_FILES = (
    ("amphion/MaskGCT", "semantic_codec/model.safetensors",
     "semantic_codec/model.safetensors"),
    ("IndexTeam/IndexTTS-2", "wav2vec2bert_stats.pt", "wav2vec2bert_stats.pt"),
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPCODEC_CONFIG = REPO_ROOT / "configs" / "repcodec_semantic.yaml"


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

    config_target = out_dir / "semantic_codec" / "config.yaml"
    config_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPCODEC_CONFIG, config_target)
    print(f"[copy] {REPCODEC_CONFIG} -> {config_target}")

    print("\nmodel-dir ready:")
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out_dir)}  {path.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

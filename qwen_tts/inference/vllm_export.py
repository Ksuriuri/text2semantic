"""Export the trained speech head/embedding as a text-only Qwen3.5 LM.

The text embedding and speaker encoder are retained separately for prefill.
Training checkpoints are never modified. No tokenizer remapping is involved:
engine input/output IDs are speech IDs and detokenization is disabled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def export_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    config = json.loads((source / "config.json").read_text())
    qwen = dict(config["qwen_config"])
    qwen.update(architectures=["SpeechQwen35ForCausalLM"], model_type="qwen3_5_text",
                vocab_size=config["speech_vocab_size"], tie_word_embeddings=False,
                bos_token_id=config["speech_bos_token_id"],
                eos_token_id=config["speech_eos_token_id"],
                pad_token_id=config["speech_pad_token_id"])
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise ValueError("Expected a local safetensors checkpoint")
    destination.mkdir(parents=True)
    weights, conditioning, index, size = {}, {}, {}, 0
    # Write one engine shard per input shard, avoiding a second full-model copy.
    for number, shard in enumerate(shards):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name).contiguous()
                if name == "backbone.embed_tokens.weight":
                    conditioning["text_embedding.weight"] = tensor
                elif name == "speech_embedding.weight":
                    weights["model.embed_tokens.weight"] = tensor
                    conditioning[name] = tensor
                elif name == "speech_head.weight":
                    weights["lm_head.weight"] = tensor
                elif name.startswith("backbone."):
                    weights["model." + name[len("backbone."):]] = tensor
                elif name.startswith(("speaker_encoder.", "speaker_projection.", "speaker_boundary_embedding.")):
                    conditioning[name] = tensor
                else:
                    raise ValueError(f"Unrecognized checkpoint tensor: {name}")
        if weights:
            filename = f"model-{number:05d}.safetensors"
            save_file(weights, str(destination / filename))
            for key, tensor in weights.items():
                index[key] = filename
                size += tensor.numel() * tensor.element_size()
            weights.clear()
    required = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
    if not required.issubset(index) or "text_embedding.weight" not in conditioning:
        raise ValueError("Incomplete speech checkpoint; export is not ready")
    save_file(conditioning, str(destination / "conditioning.safetensors"))
    (destination / "config.json").write_text(json.dumps(qwen, indent=2))
    (destination / "conditioning_config.json").write_text(json.dumps(config, indent=2))
    (destination / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": size}, "weight_map": index}, indent=2))
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(source).save_pretrained(destination / "text_tokenizer")
    (destination / "READY").write_text("speech IDs; precomputed conditioning; vllm 0.21.0\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    export_checkpoint(args.source, args.destination)

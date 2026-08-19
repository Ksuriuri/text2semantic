"""IndexTTS-2.5 (EnhancedCodec) semantic label path.

What is worth pinning is what MaskGCT's codec never did: codes come out at half
the feature rate, so a code count is not a frame count, and the decode is
context dependent rather than a per-code lookup.
"""

from pathlib import Path

import pytest
import torch
import yaml

from qwen_tts.semantic_codec import (
    DEFAULT_SEMANTIC_CODEC,
    MaskGCTSemanticTokenizer,
    RepCodec,
    canonical_semantic_codec,
    load_codec_checkpoint,
    load_codec_config,
    resolve_codec_assets,
    resolve_float_dtype,
    semantic_codec_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

TINY = {
    "codebook_size": 32,
    "hidden_size": 8,
    "codebook_dim": 4,
    "vocos_dim": 8,
    "vocos_intermediate_dim": 16,
    "vocos_num_layers": 1,
}


def _codec(downsample_scale):
    torch.manual_seed(0)
    return RepCodec(**TINY, downsample_scale=downsample_scale).eval()


def test_codes_are_halved_and_the_decode_restores_the_feature_rate():
    codec = _codec(2)
    with torch.no_grad():
        codes, _ = codec.quantize(torch.randn(1, 20, TINY["hidden_size"]))
        decoded = codec.decode(codes)
    assert codes.shape == (1, 10)
    assert decoded.shape == (1, 20, TINY["hidden_size"])


def test_an_odd_frame_count_rounds_up_to_a_whole_code():
    # kernel=3 stride=2 padding=1 gives ceil(L/2), so the last frame keeps a
    # code instead of being floored away.
    with torch.no_grad():
        codes, _ = _codec(2).quantize(torch.randn(1, 21, TINY["hidden_size"]))
    assert codes.shape == (1, 11)


def test_the_old_codec_stays_one_code_per_frame():
    codec = _codec(1)
    assert not hasattr(codec, "down")
    with torch.no_grad():
        codes, _ = codec.quantize(torch.randn(1, 7, TINY["hidden_size"]))
        decoded = codec.decode(codes)
    assert codes.shape == (1, 7)
    assert decoded.shape == (1, 7, TINY["hidden_size"])


def test_padding_changes_the_decode_so_rows_must_be_unpadded():
    codec = _codec(2)
    codes = torch.randint(1, TINY["codebook_size"], (1, 12))
    with torch.no_grad():
        clean = codec.decode(codes[:, :6])
        padded = codec.decode(
            torch.cat([codes[:, :6], torch.zeros(1, 6, dtype=torch.long)], dim=-1)
        )
    assert not torch.allclose(clean, padded[:, :12], atol=1e-4)


def test_decode_rejects_a_bare_sequence():
    with pytest.raises(ValueError):
        _codec(2).decode(torch.zeros(4, dtype=torch.long))


def test_selector_defaults_to_the_new_codec_and_keeps_the_old_one():
    assert DEFAULT_SEMANTIC_CODEC == "indextts25"
    assert semantic_codec_spec(DEFAULT_SEMANTIC_CODEC)["code_fps"] == 25.0
    assert semantic_codec_spec("maskgct")["code_fps"] == 50.0
    assert semantic_codec_spec("maskgct")["frames_per_code"] == 1
    with pytest.raises(ValueError):
        semantic_codec_spec("nope")


def test_manifest_spellings_normalize():
    for spelling in ("indextts2.5", "IndexTTS-2.5", "EnhancedCodec"):
        assert canonical_semantic_codec(spelling) == "indextts25"
        assert semantic_codec_spec(spelling)["name"] == "indextts25"
    assert canonical_semantic_codec("maskgct") == "maskgct"


def test_asset_layout_is_per_codec():
    config, checkpoint = resolve_codec_assets("/models", "indextts25")
    assert config == Path("/models/enhanced_codec/config.yaml")
    assert checkpoint == Path("/models/enhanced_codec/codec.pth")
    config, checkpoint = resolve_codec_assets("/models", "maskgct")
    assert checkpoint == Path("/models/semantic_codec/model.safetensors")


def test_shipped_configs_match_their_codec():
    enhanced = load_codec_config(
        REPO_ROOT / "configs" / "enhanced_codec_semantic.yaml", "indextts25"
    )
    assert enhanced["downsample_scale"] == 2
    assert enhanced["codebook_size"] == 8192
    maskgct = load_codec_config(
        REPO_ROOT / "configs" / "repcodec_semantic.yaml", "maskgct"
    )
    assert maskgct["downsample_scale"] == 1
    # Both must be constructible with exactly the keys they declare.
    assert set(enhanced) >= set(maskgct)


def test_a_config_without_downsample_scale_is_filled_in_from_the_codec(tmp_path):
    # The published IndexTTS-2.5 config omits it (upstream hard-codes it) and
    # nests the block under "model".
    path = tmp_path / "codec_config.yaml"
    path.write_text(yaml.safe_dump({"model": {"semantic_codec": dict(TINY)}}))
    assert load_codec_config(path, "indextts25")["downsample_scale"] == 2
    assert load_codec_config(path, "maskgct")["downsample_scale"] == 1


def test_a_config_contradicting_the_codec_is_an_error(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"semantic_codec": {**TINY, "downsample_scale": 2}}))
    load_codec_config(path, "indextts25")
    with pytest.raises(ValueError):
        load_codec_config(path, "maskgct")


def test_a_pth_checkpoint_loads_and_a_wrong_one_fails_loudly(tmp_path):
    source = _codec(2)
    path = tmp_path / "codec.pth"
    torch.save({"model": source.state_dict()}, path)
    target = _codec(2)
    load_codec_checkpoint(target, path)
    codes = torch.randint(0, TINY["codebook_size"], (1, 5))
    with torch.no_grad():
        assert torch.equal(source.decode(codes), target.decode(codes))

    wrong = tmp_path / "wrong.pth"
    torch.save(_codec(1).state_dict(), wrong)
    with pytest.raises(RuntimeError):
        load_codec_checkpoint(_codec(2), wrong)


def _stub_tokenizer(frames_per_code, code_length):
    """A tokenizer whose only real parts are the length arithmetic."""

    class FeatureExtractor:
        def encode_files(self, paths, max_audio_seconds):
            # 5 real frames out of 6, i.e. one frame of padding.
            return torch.ones(1, 6, 4), torch.tensor([5])

    class Codec:
        def quantize(self, features):
            assert features.dtype == torch.float32
            return torch.arange(1, code_length + 1).reshape(1, -1), features

    tokenizer = MaskGCTSemanticTokenizer.__new__(MaskGCTSemanticTokenizer)
    tokenizer.device = torch.device("cpu")
    tokenizer.feature_extractor = FeatureExtractor()
    tokenizer.codec = Codec()
    tokenizer.codebook_size = 8192
    tokenizer.frames_per_code = frames_per_code
    return tokenizer


def test_padding_is_trimmed_in_codes_not_in_frames():
    # 5 valid frames at 2 frames per code is 3 codes, not 5: trimming with the
    # raw frame count would keep padding-derived codes.
    assert _stub_tokenizer(2, 3).encode_file("a.wav").tolist() == [1, 2, 3]
    assert _stub_tokenizer(1, 6).encode_file("a.wav").tolist() == [1, 2, 3, 4, 5]


def test_decode_codes_returns_features_at_the_feature_rate():
    tokenizer = MaskGCTSemanticTokenizer.__new__(MaskGCTSemanticTokenizer)
    tokenizer.device = torch.device("cpu")
    tokenizer.codec = _codec(2)
    features = tokenizer.decode_codes([1, 2, 3, 4])
    assert features.shape == (8, TINY["hidden_size"])


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, torch.float32),
        ("fp32", torch.float32),
        ("float32", torch.float32),
        ("bf16", torch.bfloat16),
        ("BFloat16", torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_resolve_float_dtype_accepts_spellings_and_dtypes(given, expected):
    assert resolve_float_dtype(given) is expected


@pytest.mark.parametrize("given", ["float8", "", torch.int64])
def test_resolve_float_dtype_rejects_anything_else(given):
    # A silently ignored dtype would run the frozen extractor in fp32 while the
    # caller believed it had asked for bf16, which is exactly the kind of
    # no-op flag that makes a measured speedup unreproducible.
    with pytest.raises(ValueError):
        resolve_float_dtype(given)

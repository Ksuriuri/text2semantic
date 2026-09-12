import io

import numpy as np
import soundfile as sf

from finetuning.dataset import Text2SemanticDataset
from finetuning.manifest_index import FilterParams


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        return {"input_ids": [2]}


def wav(seconds):
    out = io.BytesIO()
    sf.write(out, np.zeros(round(seconds * 16000)), 16000, format="WAV")
    return out.getvalue()


def test_target_defaults_include_both_boundaries(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(wav(2))
    rows = [dict(id=str(i), text="hello", speaker_id="s", ref_audio=str(ref),
                 audio=f"target{i}", duration=d, semantic_codes=[1])
            for i, d in enumerate([0.49, 0.5, 60.0, 60.01])]
    ds = Text2SemanticDataset(rows, Tokenizer())
    assert [x["duration"] for x in ds.data] == [0.5, 60.0]
    assert ds[0]["speaker_audio"].size == 32000
    assert FilterParams().max_target_seconds == 60.0


def test_reference_floor_and_cap_and_same_speaker_fallback():
    class Store:
        speaker_key_fields = ("speaker_id",)

        def has_usable_ref(self, key, **kwargs):
            return True

        def read_ref(self, key, *, exclude=None, rng=None):
            assert key == ("s",)
            for name, seconds in [("short", 1.99), ("valid", 31.0)]:
                if name not in (exclude or set()):
                    return name, wav(seconds)
            return None

    rows = [dict(id=str(i), text="hello", speaker_id="s", duration=1.0,
                 semantic_codes=[1]) for i in range(2)]
    ds = Text2SemanticDataset(rows, Tokenizer(), ref_store=Store())
    assert ds._decode_audio(io.BytesIO(wav(1.99)), "short.wav") is None
    assert ds._decode_audio(io.BytesIO(wav(2)), "floor.wav").size == 32000
    assert ds[0]["speaker_audio"].size == 480000

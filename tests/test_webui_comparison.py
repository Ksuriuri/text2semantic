from pathlib import Path
from unittest.mock import Mock, call
import sys

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from webui import InferenceApp  # noqa: E402


def test_comparison_reuses_one_semantic_generation_for_both_backends():
    app = InferenceApp.__new__(InferenceApp)
    app.vocoders = {"s2vae": object(), "s2mel": object()}
    app._validate_request = Mock(return_value=("hello", "/tmp/ref.wav"))
    codes = torch.tensor([1, 2, 3])
    prompt_features = torch.ones(1, 4, 8)
    app._generate_semantics = Mock(
        return_value=(codes, prompt_features, 4, 1.25)
    )
    app._vocode = Mock(
        side_effect=[
            ("/tmp/s2vae.wav", "vae status"),
            ("/tmp/s2mel.wav", "mel status"),
        ]
    )

    result = app.generate_comparison("hello", "/tmp/ref.wav", seed=1234)

    assert result == (
        "/tmp/s2vae.wav",
        "vae status",
        "/tmp/s2mel.wav",
        "mel status",
    )
    app._generate_semantics.assert_called_once()
    assert app._generate_semantics.call_args.kwargs["need_prompt_features"] is True
    assert app._vocode.call_args_list == [
        call(
            backend="s2vae",
            codes=codes,
            ref_path="/tmp/ref.wav",
            prompt_features=prompt_features,
            prompt_feature_length=4,
            t2s_elapsed=1.25,
        ),
        call(
            backend="s2mel",
            codes=codes,
            ref_path="/tmp/ref.wav",
            prompt_features=None,
            prompt_feature_length=None,
            t2s_elapsed=1.25,
        ),
    ]

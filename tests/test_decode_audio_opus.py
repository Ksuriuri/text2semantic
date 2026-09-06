import io
import shutil
import subprocess
import numpy as np
import pytest

from finetuning.dataset import Text2SemanticDataset


FFMPEG = shutil.which("ffmpeg")


def _dummy_dataset(ref_max_seconds=20.0):
    ds = Text2SemanticDataset.__new__(Text2SemanticDataset)
    ds.ref_max_seconds = ref_max_seconds
    return ds


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not on PATH")
def test_decode_opus_file_is_16k_mono_and_skips_librosa(tmp_path, monkeypatch):
    wav = tmp_path / "ref.wav"
    opus = tmp_path / "ref.opus"
    proc = subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=1.2:sample_rate=48000",
            "-ac",
            "1",
            str(wav),
        ],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        pytest.skip("ffmpeg cannot write a test wav")
    proc = subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav),
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            str(opus),
        ],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        pytest.skip("ffmpeg cannot encode opus")

    def boom(*_args, **_kwargs):
        raise AssertionError("librosa.load must not run for opus refs")

    monkeypatch.setattr("finetuning.dataset.librosa.load", boom)
    ds = _dummy_dataset(ref_max_seconds=0.5)
    audio = ds._decode_audio(str(opus), str(opus))
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert 7000 <= audio.size <= 9000


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not on PATH")
def test_decode_opus_bytesio_uses_name_suffix(tmp_path, monkeypatch):
    wav = tmp_path / "ref.wav"
    opus = tmp_path / "ref.opus"
    subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.4:sample_rate=48000",
            "-ac",
            "1",
            str(wav),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav),
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            str(opus),
        ],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(
        "finetuning.dataset.librosa.load",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("librosa")),
    )
    ds = _dummy_dataset()
    audio = ds._decode_audio(io.BytesIO(opus.read_bytes()), "speaker/clip.opus")
    assert audio.dtype == np.float32
    assert audio.size > 1000

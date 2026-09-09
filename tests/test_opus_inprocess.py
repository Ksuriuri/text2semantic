import io
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from finetuning.dataset import Text2SemanticDataset
from finetuning.train import loader_worker_options


@pytest.mark.parametrize('channels', [1, 2])
@pytest.mark.parametrize('duration', [0.13, 2.0, None])
def test_opus_parity_and_no_subprocess(tmp_path, monkeypatch, channels, duration):
    path = tmp_path / 'tone.opus'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'sine=frequency=440:duration=0.43:sample_rate=48000',
                    '-ac', str(channels), '-c:a', 'libopus', str(path)], check=True)
    ds = Text2SemanticDataset.__new__(Text2SemanticDataset)
    ds.ref_max_seconds = duration
    expected = ds._decode_opus_ffmpeg(path, path)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('spawned process'))
    actual = ds._decode_audio(path, str(path))
    buffered = ds._decode_audio(io.BytesIO(path.read_bytes()), 'tone.opus')
    np.testing.assert_array_equal(actual, buffered)
    assert actual.dtype == np.float32 and actual.ndim == 1
    assert len(actual) == len(expected)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
    before = len(os.listdir('/proc/self/fd'))
    for _ in range(100):
        ds._decode_audio(path, str(path))
    assert len(os.listdir('/proc/self/fd')) <= before + 1


def test_corrupt_opus_closes_container(tmp_path):
    path = tmp_path / 'bad.opus'
    path.write_bytes(b'not an opus file')
    ds = Text2SemanticDataset.__new__(Text2SemanticDataset)
    ds.ref_max_seconds = 1.0
    before = len(os.listdir('/proc/self/fd'))
    for _ in range(30):
        with pytest.raises(Exception):
            ds._decode_audio(path, str(path))
    assert len(os.listdir('/proc/self/fd')) <= before + 1


def test_mmap_cache_bounded_and_evicted_rows_reopen(tmp_path):
    ds = Text2SemanticDataset.__new__(Text2SemanticDataset)
    ds._semantic_code_cache = {}
    before = len(os.listdir('/proc/self/fd'))
    rows = []
    for i in range(140):
        path = tmp_path / f'{i}.bin'
        np.array([i, i + 1, i + 2], dtype='<u2').tofile(path)
        row = dict(semantic_code_path=str(path), semantic_code_offset=1, semantic_code_length=2)
        rows.append(row)
        assert ds._semantic_codes(row).tolist() == [i + 1, i + 2]
    assert len(ds._semantic_code_cache) == 128
    assert len(os.listdir('/proc/self/fd')) <= before + 129
    assert ds._semantic_codes(rows[0]).tolist() == [1, 2]


def test_worker_options_support_single_process():
    args = SimpleNamespace(num_workers=0, prefetch_factor=4, persistent_workers=True)
    assert loader_worker_options(args) == {'num_workers': 0}
    args.num_workers = 8
    assert loader_worker_options(args) == dict(num_workers=8, prefetch_factor=4, persistent_workers=True)
    args.prefetch_factor = 0
    with pytest.raises(ValueError):
        loader_worker_options(args)

import json
import tarfile
from pathlib import Path

from finetuning.dataset import Text2SemanticDataset
from finetuning.ref_store import (
    SpeakerRefStore,
    default_ref_index_path,
)
from finetuning.train import resolve_ref_store


class _Tok:
    pad_token_id = 0
    eos_token_id = 1


def _write_wav_bytes():
    # 44-byte empty-ish wav header + silence; encode_files is not called here.
    return b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 24


def _build_trainset(tmp_path):
    root = Path(tmp_path)
    shards = root / "refs" / "shards"
    shards.mkdir(parents=True)
    (root / "manifests").mkdir()
    wav = _write_wav_bytes()
    tar_path = shards / "refs-000000.tar"
    with tarfile.open(tar_path, "w") as archive:
        for name in ("spkA/000.flac", "spkA/001.flac"):
            info = tarfile.TarInfo(name)
            info.size = len(wav)
            archive.addfile(info, fileobj=__import__("io").BytesIO(wav))
    index = root / "refs" / "speaker_index.jsonl"
    index.write_text(
        json.dumps(
            {
                "language": "en",
                "speaker_id": "spkA",
                "shard": "refs-000000.tar",
                "members": ["spkA/000.flac", "spkA/001.flac"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train = root / "manifests" / "train.jsonl"
    train.write_text(
        json.dumps(
            {
                "text": "hello",
                "language": "en",
                "speaker_id": "spkA",
                "semantic_codes": [1, 2, 3],
                "duration": 4.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root, train, index


def test_default_ref_index_path_from_manifests(tmp_path):
    root, train, index = _build_trainset(tmp_path)
    assert default_ref_index_path(train) == index.resolve()


def test_ref_store_extracts_and_caps(tmp_path):
    _, _, index = _build_trainset(tmp_path)
    store = SpeakerRefStore(index, refs_per_speaker=1)
    key = ("en", "spkA")
    assert key in store
    assert store.members(key) == ("spkA/000.flac",)
    path = store.pick_path(key)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_dataset_uses_packed_refs_without_loose_audio(tmp_path):
    _, train, index = _build_trainset(tmp_path)
    store = SpeakerRefStore(index)
    data = [json.loads(train.read_text(encoding="utf-8"))]
    dataset = Text2SemanticDataset(
        data,
        _Tok(),
        speaker_counts={("en", "spkA"): 2},
        min_speaker_records=2,
        ref_store=store,
    )
    assert len(dataset) == 1
    path = dataset._speaker_audio_path(dataset.data[0], index=0)
    assert Path(path).is_file()


def test_resolve_ref_store_autodiscovers(tmp_path, monkeypatch):
    _, train, index = _build_trainset(tmp_path)
    args = type("A", (), {})()
    args.loose_refs = False
    args.ref_index = None
    args.ref_root = None
    args.ref_cache = None
    args.refs_per_speaker = 0
    args.train_jsonl = str(train)
    store = resolve_ref_store(args)
    assert store is not None
    assert store.index_path == index.resolve()


def test_loose_refs_disables_store(tmp_path):
    _, train, _ = _build_trainset(tmp_path)
    args = type("A", (), {})()
    args.loose_refs = True
    args.ref_index = None
    args.ref_root = None
    args.ref_cache = None
    args.refs_per_speaker = 0
    args.train_jsonl = str(train)
    assert resolve_ref_store(args) is None

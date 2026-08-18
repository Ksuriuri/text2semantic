import io
import json
import tarfile
from pathlib import Path

import pytest

from finetuning import speaker_index
from finetuning.ref_member_index import member_row_id
from finetuning.source_ref_store import (
    KEY_FIELDS,
    GcsRangeReader,
    LocalRangeReader,
    SourceTarRefStore,
    default_index_path,
    source_refs_row,
)

TAR = "preprocessed/ears/audio/ears-000000.tar"


class FakeReader:
    """Records every ranged read and answers with recognisable bytes."""

    def __init__(self, payloads=None):
        self.reads = []
        self.payloads = payloads or {}

    def read(self, blob_name, offset, size):
        self.reads.append((blob_name, offset, size))
        return self.payloads.get((blob_name, offset), b"x" * size)


def _index(tmp_path, rows):
    path = Path(tmp_path) / "refs" / "source_speaker_index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _row(dataset="ears", language="en", speaker="p001", refs=()):
    return {
        "dataset": dataset,
        "language": language,
        "speaker_id": speaker,
        "refs": [list(ref) for ref in refs],
    }


def _store(tmp_path, rows, reader=None, **kwargs):
    reader = reader or FakeReader()
    return SourceTarRefStore(_index(tmp_path, rows), reader=reader, **kwargs)


def test_a_ref_is_one_ranged_read_of_the_source_tar(tmp_path):
    reader = FakeReader({(TAR, 512): b"fLaC-first"})
    store = _store(
        tmp_path,
        [_row(refs=[(TAR, "ears__p001_a.flac", 512, 10)])],
        reader=reader,
    )

    member, payload = store.read_ref(("ears", "en", "p001"))

    assert member == "ears__p001_a.flac"
    assert payload == b"fLaC-first"
    # Exactly one read, of exactly the member's bytes: no tar is opened and no
    # sidecar is fetched at training time.
    assert reader.reads == [(TAR, 512, 10)]


def test_the_same_speaker_id_in_two_datasets_is_two_speakers(tmp_path):
    store = _store(
        tmp_path,
        [
            _row(dataset="ears", refs=[(TAR, "ears__p001_a.flac", 512, 4)]),
            _row(
                dataset="vctk",
                refs=[("preprocessed/vctk/audio/vctk-000000.tar",
                       "vctk__p001_x.flac", 2048, 4)],
            ),
        ],
    )

    assert store.pick_ref(("ears", "en", "p001"))[1] == "ears__p001_a.flac"
    assert store.pick_ref(("vctk", "en", "p001"))[1] == "vctk__p001_x.flac"
    assert ("Genshin", "en", "p001") not in store


def test_a_two_part_key_is_refused_rather_than_guessed(tmp_path):
    store = _store(tmp_path, [_row(refs=[(TAR, "ears__p001_a.flac", 512, 4)])])
    with pytest.raises(ValueError):
        store.pick_ref(("en", "p001"))


def test_the_target_clip_is_never_its_own_ref(tmp_path):
    store = _store(
        tmp_path,
        [
            _row(
                refs=[
                    (TAR, "ears__p001_a.flac", 512, 4),
                    (TAR, "ears__p001_b.flac", 1024, 4),
                ]
            )
        ],
    )
    key = ("ears", "en", "p001")

    # exclude is a manifest row id, and the member name carries it after the
    # dataset prefix.
    assert store.pick_ref(key, exclude="p001_a")[1] == "ears__p001_b.flac"
    assert store.pick_ref(key, exclude="p001_b")[1] == "ears__p001_a.flac"


def test_a_speaker_whose_only_clip_is_the_target_has_no_ref(tmp_path):
    store = _store(
        tmp_path, [_row(refs=[(TAR, "ears__p001_a.flac", 512, 4)])]
    )
    key = ("ears", "en", "p001")

    assert store.has_usable_ref(key) is True
    assert store.has_usable_ref(key, exclude="p001_a") is False
    assert store.read_ref(key, exclude="p001_a") is None


def test_ref_ids_are_row_ids_so_the_manifest_probe_can_compare_them(tmp_path):
    store = _store(
        tmp_path,
        [
            _row(
                refs=[
                    (TAR, "ears__p001_a.flac", 512, 4),
                    (TAR, "ears__p001_b.flac", 1024, 4),
                ]
            )
        ],
    )
    assert store.ref_ids(("ears", "en", "p001")) == ("p001_a", "p001_b")


def test_refs_per_speaker_caps_the_candidates(tmp_path):
    rows = [
        _row(
            refs=[
                (TAR, f"ears__p001_{index}.flac", 512 * (index + 1), 4)
                for index in range(5)
            ]
        )
    ]
    store = _store(tmp_path, rows, refs_per_speaker=2)
    assert store.members(("ears", "en", "p001")) == (
        "ears__p001_0.flac",
        "ears__p001_1.flac",
    )


def test_a_random_pick_spreads_over_every_clip(tmp_path):
    import random

    rows = [
        _row(
            refs=[
                (TAR, f"ears__p001_{index}.flac", 512 * (index + 1), 4)
                for index in range(20)
            ]
        )
    ]
    store = _store(tmp_path, rows)
    picked = {
        store.pick_ref(("ears", "en", "p001"), rng=random.Random(seed))[1]
        for seed in range(200)
    }
    # The point of reading the source tars is that all of a speaker's clips are
    # candidates, not the handful that got packed.
    assert len(picked) == 20


def test_the_memmapped_backend_answers_like_the_dict(tmp_path):
    rows = [
        _row(refs=[(TAR, "ears__p001_a.flac", 512, 4)]),
        _row(speaker="p002", refs=[(TAR, "ears__p002_a.flac", 1024, 4)]),
    ]
    index_path = _index(tmp_path, rows)
    ram = SourceTarRefStore(
        index_path, reader=FakeReader(), index_backend="ram"
    )
    memmapped = SourceTarRefStore(
        index_path, reader=FakeReader(), index_backend="memmap"
    )

    assert len(memmapped) == len(ram) == 2
    for key in (("ears", "en", "p001"), ("ears", "en", "p002")):
        assert key in memmapped
        assert memmapped.pick_ref(key) == ram.pick_ref(key)
    assert ("ears", "en", "nobody") not in memmapped

    meta = json.loads(
        (speaker_index.index_dir_for(index_path) / speaker_index.META_NAME)
        .read_text()
    )
    assert meta["key_fields"] == list(KEY_FIELDS)


def test_a_table_keyed_on_the_wrong_fields_is_rebuilt(tmp_path):
    rows = [_row(refs=[(TAR, "ears__p001_a.flac", 512, 4)])]
    index_path = _index(tmp_path, rows)
    # Built as if it were a packed index: same file, same mtime, wrong keys.
    speaker_index.build(index_path, log=None)

    store = SourceTarRefStore(
        index_path, reader=FakeReader(), index_backend="memmap"
    )

    assert store.pick_ref(("ears", "en", "p001"))[1] == "ears__p001_a.flac"


def test_a_missing_table_is_not_built_off_the_main_rank(tmp_path):
    index_path = _index(
        tmp_path, [_row(refs=[(TAR, "ears__p001_a.flac", 512, 4)])]
    )
    with pytest.raises(FileNotFoundError, match="main process"):
        SourceTarRefStore(
            index_path,
            reader=FakeReader(),
            index_backend="memmap",
            build_index_if_missing=False,
        )


def test_the_local_reader_returns_the_member_bytes(tmp_path):
    root = Path(tmp_path) / "mirror"
    tar_path = root / TAR
    tar_path.parent.mkdir(parents=True)
    payload = b"fLaC" + b"\x00" * 100
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("ears__p001_a.flac")
        info.size = len(payload)
        archive.addfile(info, fileobj=io.BytesIO(payload))
    with tarfile.open(tar_path) as archive:
        info = archive.getmember("ears__p001_a.flac")

    reader = LocalRangeReader(root)
    assert reader.read(TAR, info.offset_data, info.size) == payload


def test_the_gcs_reader_asks_for_an_inclusive_byte_range(monkeypatch):
    asked = {}

    class FakeBlob:
        def download_as_bytes(self, start=None, end=None):
            asked["range"] = (start, end)
            return b"y" * (end - start + 1)

    class FakeBucket:
        def blob(self, name):
            asked["blob"] = name
            return FakeBlob()

    reader = GcsRangeReader("noiz-taiwan-audio-data")
    monkeypatch.setattr(reader, "_bucket_for_this_process", FakeBucket)

    payload = reader.read(TAR, 512, 10)

    # end is inclusive in the GCS API: asking for offset+size would fetch one
    # byte too many and every ref would carry a stray byte of the next header.
    assert asked == {"blob": TAR, "range": (512, 521)}
    assert len(payload) == 10


def test_the_gcs_reader_refuses_an_empty_read(monkeypatch):
    reader = GcsRangeReader("noiz-taiwan-audio-data")
    with pytest.raises(ValueError):
        reader.read(TAR, 512, 0)


def test_the_local_reader_rejects_a_short_read(tmp_path):
    root = Path(tmp_path) / "mirror"
    (root / TAR).parent.mkdir(parents=True)
    (root / TAR).write_bytes(b"only-a-little")
    reader = LocalRangeReader(root)
    with pytest.raises(OSError, match="short read"):
        reader.read(TAR, 0, 4096)


def test_row_ids_survive_a_member_name_without_a_dataset_prefix():
    assert member_row_id("ears__p001_a.flac", "ears") == "p001_a"
    assert member_row_id("mini/en/spkA/keep-1.flac") == "keep-1"
    assert member_row_id("ears__p001_a.flac") == "ears__p001_a"


def test_the_index_is_discovered_next_to_the_manifests(tmp_path):
    root = Path(tmp_path)
    (root / "manifests").mkdir()
    train = root / "manifests" / "train.jsonl"
    train.write_text("", encoding="utf-8")
    assert default_index_path(train) is None
    _index(tmp_path, [_row(refs=[(TAR, "ears__p001_a.flac", 512, 4)])])
    assert default_index_path(train) == (
        root / "refs" / "source_speaker_index.jsonl"
    )


def test_an_index_row_without_refs_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no refs"):
        _store(tmp_path, [_row(refs=[])])


def test_source_refs_row_coerces_the_json_types():
    entry = source_refs_row({"refs": [[TAR, "ears__p001_a.flac", "512", "10"]]})
    assert entry["refs"] == ((TAR, "ears__p001_a.flac", 512, 10),)


class _Tok:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2] * max(1, len(text.split()))}


def _wav_bytes(seconds=1.0, sample_rate=16000):
    import numpy as np
    import soundfile as sf

    samples = np.linspace(0.0, 1.0, int(sample_rate * seconds), dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _source_tar(root, dataset, members):
    """One source tar, returning {member: (blob, offset, size)}."""
    blob = f"preprocessed/{dataset}/audio/{dataset}-000000.tar"
    path = Path(root) / blob
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _wav_bytes()
    with tarfile.open(path, "w") as archive:
        for name in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, fileobj=io.BytesIO(payload))
    located = {}
    with tarfile.open(path) as archive:
        for info in archive.getmembers():
            located[info.name] = (blob, info.offset_data, info.size)
    return located


def test_two_datasets_sharing_a_speaker_id_never_swap_refs(tmp_path):
    import numpy as np

    from finetuning import manifest_index
    from finetuning.dataset import Text2SemanticDataset

    root = Path(tmp_path)
    mirror = root / "mirror"
    located = {}
    for dataset in ("ears", "vctk"):
        located.update(
            _source_tar(
                mirror,
                dataset,
                [f"{dataset}__p001_a.flac", f"{dataset}__p001_b.flac"],
            )
        )
    (root / "codes" / "mini").mkdir(parents=True)
    (root / "codes" / "mini" / "mini-000000.u2.bin").write_bytes(
        np.arange(64, dtype="<u2").tobytes()
    )
    index_path = _index(
        tmp_path,
        [
            {
                "dataset": dataset,
                "language": "en",
                "speaker_id": "p001",
                "refs": [
                    list(located[f"{dataset}__p001_{suffix}.flac"][:1])
                    + [f"{dataset}__p001_{suffix}.flac"]
                    + list(located[f"{dataset}__p001_{suffix}.flac"][1:])
                    for suffix in ("a", "b")
                ],
            }
            for dataset in ("ears", "vctk")
        ],
    )
    (root / "manifests").mkdir()
    train = root / "manifests" / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"p001_{suffix}",
                    "text": "hello there",
                    "dataset": dataset,
                    "language": "en",
                    "speaker_id": "p001",
                    "duration": 4.0,
                    "semantic_code_path": "../codes/mini/mini-000000.u2.bin",
                    "semantic_code_offset": 0,
                    "semantic_code_length": 16,
                }
            )
            + "\n"
            for dataset in ("ears", "vctk")
            for suffix in ("a", "b")
        ),
        encoding="utf-8",
    )

    reads = []

    class WatchedReader(LocalRangeReader):
        def read(self, blob_name, offset, size):
            reads.append(blob_name)
            return super().read(blob_name, offset, size)

    store = SourceTarRefStore(index_path, reader=WatchedReader(mirror))
    assert store.speaker_key_fields == KEY_FIELDS

    index = manifest_index.load(
        train,
        params=manifest_index.FilterParams(
            min_speaker_records=1,
            ref_index=str(index_path),
            speaker_key=",".join(KEY_FIELDS),
        ),
        ref_store=store,
        log=None,
    )
    dataset = Text2SemanticDataset(index, _Tok(), ref_store=store)

    assert dataset.speaker_key_fields == KEY_FIELDS
    assert len(dataset) == 4
    for position in range(len(dataset)):
        row = index[position]
        sample = dataset[position]
        assert sample["speaker_audio"] is not None
        # The ref came out of the row's own dataset, not the other dataset's
        # p001, who is a different person entirely.
        assert reads[-1] == (
            f"preprocessed/{row['dataset']}/audio/{row['dataset']}-000000.tar"
        )

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

from finetuning import manifest_index, ref_member_index, speaker_index
from finetuning.dataset import Text2SemanticDataset
from finetuning.ref_store import SpeakerRefStore


class _Tok:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2] * max(1, len(text.split()))}


def _wav_bytes(seconds=1.0, sample_rate=16000):
    samples = np.linspace(0.0, 1.0, int(sample_rate * seconds), dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _build_trainset(tmp_path, rows):
    """A miniature packed trainset: manifests/, codes/ and refs/ as packed."""
    root = Path(tmp_path)
    (root / "manifests").mkdir(parents=True)
    (root / "codes" / "mini").mkdir(parents=True)
    shards = root / "refs" / "shards"
    shards.mkdir(parents=True)

    codes = np.arange(64, dtype="<u2")
    (root / "codes" / "mini" / "mini-000000.u2.bin").write_bytes(codes.tobytes())

    payload = _wav_bytes()
    with tarfile.open(shards / "refs-000000.tar", "w") as archive:
        for name in ("spkA/000.wav", "spkA/001.wav", "spkB/000.wav"):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, fileobj=io.BytesIO(payload))
    (root / "refs" / "speaker_index.jsonl").write_text(
        "".join(
            json.dumps(entry) + "\n"
            for entry in (
                {
                    "language": "en",
                    "speaker_id": "spkA",
                    "shard": "refs-000000.tar",
                    "members": ["spkA/000.wav", "spkA/001.wav"],
                },
                {
                    "language": "en",
                    "speaker_id": "spkB",
                    "shard": "refs-000000.tar",
                    "members": ["spkB/000.wav"],
                },
            )
        ),
        encoding="utf-8",
    )
    train = root / "manifests" / "train.jsonl"
    train.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return root, train


def _row(row_id, speaker="spkA", **overrides):
    row = {
        "id": row_id,
        "text": "hello there",
        "dataset": "mini",
        "language": "en",
        "speaker_id": speaker,
        "duration": 4.0,
        "semantic_code_path": "../codes/mini/mini-000000.u2.bin",
        "semantic_code_offset": 0,
        "semantic_code_length": 16,
    }
    row.update(overrides)
    return row


def _store(root):
    return SpeakerRefStore(root / "refs" / "speaker_index.jsonl")


def _params(**overrides):
    values = {
        "min_target_seconds": 0.5,
        "max_target_seconds": 30.0,
        "min_speaker_records": 1,
    }
    values.update(overrides)
    return manifest_index.FilterParams(**values)


def test_index_keeps_only_usable_rows(tmp_path):
    root, train = _build_trainset(
        tmp_path,
        [
            _row("keep-1"),
            _row("keep-short", duration=0.6),
            _row("drop-too-short", duration=0.2),
            _row("drop-too-long", duration=44.0),
            _row("drop-empty-text", text="   "),
            _row("drop-no-ref", speaker="spkC"),
        ],
    )
    index = manifest_index.load(
        train, params=_params(), ref_store=_store(root), log=None
    )
    assert [row["id"] for row in index] == ["keep-1", "keep-short"]
    assert index.raw_size == 6
    assert index.filtered_size == 4


def test_index_resolves_code_paths_against_the_manifest(tmp_path):
    root, train = _build_trainset(tmp_path, [_row("keep-1")])
    index = manifest_index.load(
        train, params=_params(), ref_store=_store(root), log=None
    )
    resolved = Path(index[0]["semantic_code_path"])
    assert resolved.is_absolute()
    assert resolved == (root / "codes" / "mini" / "mini-000000.u2.bin")
    assert resolved.is_file()


def test_min_speaker_records_drops_single_row_speakers(tmp_path):
    root, train = _build_trainset(
        tmp_path, [_row("a-1"), _row("a-2"), _row("b-1", speaker="spkB")]
    )
    index = manifest_index.load(
        train,
        params=_params(min_speaker_records=2),
        ref_store=_store(root),
        log=None,
    )
    assert sorted(row["id"] for row in index) == ["a-1", "a-2"]


def test_index_rebuilds_when_filters_change(tmp_path):
    root, train = _build_trainset(
        tmp_path, [_row("keep-1"), _row("short", duration=0.6)]
    )
    store = _store(root)
    first = manifest_index.load(train, params=_params(), ref_store=store, log=None)
    assert len(first) == 2
    second = manifest_index.load(
        train,
        params=_params(min_target_seconds=3.0),
        ref_store=store,
        log=None,
    )
    assert [row["id"] for row in second] == ["keep-1"]


def test_index_is_reused_when_nothing_changed(tmp_path):
    root, train = _build_trainset(tmp_path, [_row("keep-1")])
    store = _store(root)
    manifest_index.load(train, params=_params(), ref_store=store, log=None)
    meta_path = manifest_index.index_dir_for(train) / manifest_index.META_NAME
    built_at = json.loads(meta_path.read_text())["built_at"]
    stamp = meta_path.stat().st_mtime_ns
    manifest_index.load(train, params=_params(), ref_store=store, log=None)
    assert json.loads(meta_path.read_text())["built_at"] == built_at
    assert meta_path.stat().st_mtime_ns == stamp


def test_ref_member_index_reads_the_same_bytes_as_tarfile(tmp_path):
    root, _ = _build_trainset(tmp_path, [_row("keep-1")])
    shard = root / "refs" / "shards" / "refs-000000.tar"
    store = _store(root)
    payload = store.read_member("refs-000000.tar", "spkA/001.wav")
    with tarfile.open(shard) as archive:
        expected = archive.extractfile("spkA/001.wav").read()
    assert payload == expected
    sidecar = ref_member_index.index_path_for(
        shard, ref_member_index.index_dir_for(root / "refs")
    )
    assert sidecar.is_file()


def test_member_index_refuses_a_half_written_shard(tmp_path):
    root, _ = _build_trainset(tmp_path, [_row("keep-1")])
    shard = root / "refs" / "shards" / "refs-000000.tar"
    index_dir = ref_member_index.index_dir_for(root / "refs")
    whole = shard.read_bytes()
    with tarfile.open(shard) as archive:
        first, second = archive.getmembers()[:2]
    block = tarfile.BLOCKSIZE

    def blocks(size):
        return -(-size // block) * block

    # The dangerous cut is the one tarfile does NOT complain about: on a block
    # boundary, it lists the members it has seen and stops silently. Here that
    # hides the second and third refs of the shard.
    aligned = first.offset_data + blocks(first.size)
    shard.write_bytes(whole[:aligned])
    with tarfile.open(shard) as archive:
        assert [info.name for info in archive] == [first.name]
    for cut in (aligned, second.offset_data + second.size // 2, len(whole) // 2):
        shard.write_bytes(whole[:cut])
        try:
            ref_member_index.build_shard_index(shard, index_dir)
        except (ValueError, tarfile.TarError):
            pass
        else:  # pragma: no cover - the guard is the point of the test
            raise AssertionError(f"expected the shard cut at {cut} to be rejected")
        assert not ref_member_index.index_path_for(shard, index_dir).exists()

    shard.write_bytes(whole)
    members = ref_member_index.read_shard_index(shard, index_dir)
    assert set(members) == {"spkA/000.wav", "spkA/001.wav", "spkB/000.wav"}


def test_member_index_rebuilds_when_the_shard_size_changes(tmp_path):
    root, _ = _build_trainset(tmp_path, [_row("keep-1")])
    shard = root / "refs" / "shards" / "refs-000000.tar"
    index_dir = ref_member_index.index_dir_for(root / "refs")
    ref_member_index.build_shard_index(shard, index_dir)
    sidecar = ref_member_index.index_path_for(shard, index_dir)
    stale = json.loads(sidecar.read_text())
    stale["shard_size"] = 12345
    stale["members"] = {"gone.wav": [0, 0]}
    sidecar.write_text(json.dumps(stale))
    members = ref_member_index.read_shard_index(shard, index_dir)
    assert "gone.wav" not in members
    assert "spkA/000.wav" in members


def test_memmapped_speaker_index_answers_like_the_dict(tmp_path):
    root, _ = _build_trainset(tmp_path, [_row("keep-1")])
    index_jsonl = root / "refs" / "speaker_index.jsonl"
    ram = SpeakerRefStore(index_jsonl, index_backend="ram")
    memmapped = SpeakerRefStore(index_jsonl, index_backend="memmap")

    assert memmapped.index_backend == "memmap"
    assert len(memmapped) == len(ram) == 2
    for key in (("en", "spkA"), ("en", "spkB")):
        assert key in memmapped
        assert memmapped.members(key) == ram.members(key)
        assert memmapped.pick_ref(key) == ram.pick_ref(key)
    assert ("en", "nobody") not in memmapped
    assert memmapped.pick_ref(("en", "nobody")) is None


def test_memmapped_speaker_index_refuses_to_build_off_the_main_rank(tmp_path):
    root, _ = _build_trainset(tmp_path, [_row("keep-1")])
    index_jsonl = root / "refs" / "speaker_index.jsonl"
    try:
        SpeakerRefStore(
            index_jsonl, index_backend="memmap", build_index_if_missing=False
        )
    except FileNotFoundError as error:
        assert "main process" in str(error)
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("expected a missing speaker index table to raise")
    speaker_index.load(index_jsonl, log=None)
    store = SpeakerRefStore(
        index_jsonl, index_backend="memmap", build_index_if_missing=False
    )
    assert store.members(("en", "spkA")) == ("spkA/000.wav", "spkA/001.wav")


def test_dataset_reads_refs_without_writing_them_out(tmp_path):
    root, train = _build_trainset(tmp_path, [_row("keep-1"), _row("keep-2")])
    store = _store(root)
    index = manifest_index.load(train, params=_params(), ref_store=store, log=None)
    dataset = Text2SemanticDataset(index, _Tok(), ref_store=store)

    assert dataset.prefiltered
    assert dataset.ref_audio_in_memory
    sample = dataset[0]
    assert sample["speaker_audio_path"] is None
    assert sample["speaker_audio"].dtype == np.float32
    assert sample["speaker_audio"].size == 16000

    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert "speaker_audio_paths" not in batch
    assert len(batch["speaker_waveforms"]) == 2
    assert not store.cache_dir.exists()


def _row_named_trainset(tmp_path, rows, members_by_speaker):
    """A trainset whose ref members are named after the rows, as the packer does.

    ``pack_trainset`` writes ``<dataset>/<language>/<speaker>/<id>.flac``, so a
    member name carries the id of the row it was packed from.
    """
    root = Path(tmp_path)
    (root / "manifests").mkdir(parents=True)
    (root / "codes" / "mini").mkdir(parents=True)
    shards = root / "refs" / "shards"
    shards.mkdir(parents=True)
    codes = np.arange(64, dtype="<u2")
    (root / "codes" / "mini" / "mini-000000.u2.bin").write_bytes(codes.tobytes())

    payload = _wav_bytes()
    with tarfile.open(shards / "refs-000000.tar", "w") as archive:
        for members in members_by_speaker.values():
            for name in members:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, fileobj=io.BytesIO(payload))
    (root / "refs" / "speaker_index.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "language": "en",
                    "speaker_id": speaker,
                    "shard": "refs-000000.tar",
                    "members": members,
                }
            )
            + "\n"
            for speaker, members in members_by_speaker.items()
        ),
        encoding="utf-8",
    )
    train = root / "manifests" / "train.jsonl"
    train.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return root, train


def test_a_packed_ref_is_never_the_row_it_was_packed_from(tmp_path):
    root, _ = _row_named_trainset(
        tmp_path,
        [_row("keep-1"), _row("keep-2")],
        {
            "spkA": ["mini/en/spkA/keep-1.wav", "mini/en/spkA/keep-2.wav"],
            "spkB": ["mini/en/spkB/alone.wav"],
        },
    )
    store = _store(root)

    # exclude is a row id, and the member is named after that row: comparing it
    # against the whole member path never matches, and the clip would be handed
    # out as its own ref.
    assert store.pick_member(("en", "spkA"), exclude="keep-1") == (
        "mini/en/spkA/keep-2.wav"
    )
    assert store.ref_ids(("en", "spkB")) == ("alone",)
    assert store.has_usable_ref(("en", "spkB"), exclude="alone") is False
    assert store.read_ref(("en", "spkB"), exclude="alone") is None


def test_a_row_whose_only_packed_ref_is_itself_is_filtered_out(tmp_path):
    root, train = _row_named_trainset(
        tmp_path,
        [_row("keep-1"), _row("alone", speaker="spkB")],
        {
            "spkA": ["mini/en/spkA/keep-1.wav", "mini/en/spkA/keep-2.wav"],
            "spkB": ["mini/en/spkB/alone.wav"],
        },
    )
    index = manifest_index.load(
        train, params=_params(), ref_store=_store(root), log=None
    )
    # Keeping "alone" would raise at read time instead: its only ref is itself.
    assert [row["id"] for row in index] == ["keep-1"]

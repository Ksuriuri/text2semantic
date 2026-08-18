import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_source_ref_index as builder  # noqa: E402

from finetuning.source_ref_store import (  # noqa: E402
    KEY_FIELDS,
    SourceTarRefStore,
)


class _Reader:
    def read(self, blob_name, offset, size):  # pragma: no cover - unused
        return b"x" * size


def _sidecar(root, dataset, shard, members):
    """One sidecar in the same shape the header scan writes."""
    path = Path(root) / builder.SIDECAR_PREFIX / dataset / f"{shard}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "shard": f"{shard}.tar",
                "shard_size": 1 << 20,
                "members": {
                    name: [offset, size]
                    for name, (offset, size) in members.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _manifest(root, name, rows):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return name


def _row(dataset, row_id, speaker, *, language="en", duration=6.0):
    return {
        "id": row_id,
        "dataset": dataset,
        "language": language,
        "speaker_id": speaker,
        "duration": duration,
        "text": "hello",
    }


def _build(tmp_path, *, manifests, **kwargs):
    return builder.build(
        builder.LocalBlobs(tmp_path / "bucket"),
        tmp_path / "work",
        manifests=manifests,
        log=None,
        **kwargs,
    )


def test_the_index_points_at_the_right_tar_and_byte_range(tmp_path):
    bucket = tmp_path / "bucket"
    _sidecar(
        bucket,
        "ears",
        "ears-000000",
        {
            "ears__p001_a.flac": (512, 1000),
            "ears__p001_b.flac": (2048, 2000),
        },
    )
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [
            _row("ears", "ears__p001_a", "ears__p001"),
            _row("ears", "ears__p001_b", "ears__p001"),
        ],
    )

    index_path, summary = _build(tmp_path, manifests=[manifest])

    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert rows == [
        {
            "dataset": "ears",
            "language": "en",
            "speaker_id": "ears__p001",
            "refs": [
                ["preprocessed/ears/audio/ears-000000.tar",
                 "ears__p001_a.flac", 512, 1000],
                ["preprocessed/ears/audio/ears-000000.tar",
                 "ears__p001_b.flac", 2048, 2000],
            ],
        }
    ]
    assert summary["speakers"] == 1
    assert summary["refs"] == 2
    # And the store the trainer uses can read it back.
    store = SourceTarRefStore(index_path, reader=_Reader())
    assert store.speaker_key_fields == KEY_FIELDS
    assert store.pick_ref(("ears", "en", "ears__p001"))[2] == 512


def test_the_tar_path_comes_from_the_dataset_dir_not_the_tar_name(tmp_path):
    # laion_emolia-000869.tar lives under preprocessed/laion_emolia_zh/audio/,
    # and its members are prefixed laion_emolia__: neither the path nor the row
    # id can be guessed from the other.
    bucket = tmp_path / "bucket"
    _sidecar(
        bucket,
        "laion_emolia_zh",
        "laion_emolia-000869",
        {"laion_emolia__ZH_B00039_S01897_W000000.flac": (512, 251033)},
    )
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [
            _row(
                "laion_emolia_zh",
                "laion_emolia__ZH_B00039_S01897_W000000",
                "laion_emolia__ZH_B00039_S01897",
                language="zh",
            )
        ],
    )

    index_path, _ = _build(tmp_path, manifests=[manifest])

    row = json.loads(index_path.read_text().splitlines()[0])
    assert row["refs"] == [
        [
            "preprocessed/laion_emolia_zh/audio/laion_emolia-000869.tar",
            "laion_emolia__ZH_B00039_S01897_W000000.flac",
            512,
            251033,
        ]
    ]


def test_a_clip_too_short_to_condition_on_is_not_a_ref(tmp_path):
    bucket = tmp_path / "bucket"
    _sidecar(
        bucket,
        "ears",
        "ears-000000",
        {
            "ears__p001_short.flac": (512, 10),
            "ears__p001_long.flac": (1024, 1000),
            "ears__p001_huge.flac": (4096, 9000),
        },
    )
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [
            # A 1 s clip is a fine target and a useless ref.
            _row("ears", "ears__p001_short", "ears__p001", duration=1.0),
            _row("ears", "ears__p001_long", "ears__p001", duration=6.0),
            _row("ears", "ears__p001_huge", "ears__p001", duration=41.0),
        ],
    )

    index_path, _ = _build(tmp_path, manifests=[manifest])

    row = json.loads(index_path.read_text().splitlines()[0])
    assert [ref[1] for ref in row["refs"]] == ["ears__p001_long.flac"]


def test_a_tar_member_with_no_manifest_row_is_skipped(tmp_path):
    bucket = tmp_path / "bucket"
    _sidecar(
        bucket,
        "ears",
        "ears-000000",
        {
            "ears__p001_a.flac": (512, 1000),
            # Dropped by the trainset filter (bad ASR, no codes, low sample rate).
            "ears__p999_z.flac": (2048, 1000),
        },
    )
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [_row("ears", "ears__p001_a", "ears__p001")],
    )

    index_path, summary = _build(tmp_path, manifests=[manifest])

    row = json.loads(index_path.read_text().splitlines()[0])
    assert [ref[1] for ref in row["refs"]] == ["ears__p001_a.flac"]
    assert summary["per_dataset"]["ears"]["members"] == 2


def test_the_same_speaker_id_in_two_datasets_stays_two_speakers(tmp_path):
    bucket = tmp_path / "bucket"
    _sidecar(bucket, "ears", "ears-000000", {"ears__p001_a.flac": (512, 1000)})
    _sidecar(bucket, "vctk", "vctk-000000", {"vctk__p001_a.flac": (512, 1000)})
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [
            _row("ears", "ears__p001_a", "p001"),
            _row("vctk", "vctk__p001_a", "p001"),
        ],
    )

    index_path, _ = _build(tmp_path, manifests=[manifest])

    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert [(row["dataset"], row["speaker_id"]) for row in rows] == [
        ("ears", "p001"),
        ("vctk", "p001"),
    ]
    store = SourceTarRefStore(index_path, reader=_Reader())
    assert store.pick_ref(("ears", "en", "p001"))[1] == "ears__p001_a.flac"
    assert store.pick_ref(("vctk", "en", "p001"))[1] == "vctk__p001_a.flac"


def test_eval_rows_get_refs_too(tmp_path):
    bucket = tmp_path / "bucket"
    _sidecar(
        bucket,
        "ears",
        "ears-000000",
        {
            "ears__p001_a.flac": (512, 1000),
            "ears__p002_a.flac": (2048, 1000),
        },
    )
    train = _manifest(
        bucket,
        "manifests/train.jsonl",
        [_row("ears", "ears__p001_a", "ears__p001")],
    )
    evalset = _manifest(
        bucket,
        "manifests/eval.jsonl",
        [_row("ears", "ears__p002_a", "ears__p002")],
    )

    index_path, _ = _build(tmp_path, manifests=[train, evalset])

    speakers = [
        json.loads(line)["speaker_id"]
        for line in index_path.read_text().splitlines()
    ]
    assert speakers == ["ears__p001", "ears__p002"]


def test_a_capped_speaker_keeps_a_spread_not_the_first_few(tmp_path):
    bucket = tmp_path / "bucket"
    members = {
        f"ears__p001_{n:03d}.flac": (512 * (n + 1), 1000) for n in range(50)
    }
    _sidecar(bucket, "ears", "ears-000000", members)
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [
            _row("ears", f"ears__p001_{n:03d}", "ears__p001")
            for n in range(50)
        ],
    )

    index_path, _ = _build(
        tmp_path, manifests=[manifest], max_refs_per_speaker=5
    )

    row = json.loads(index_path.read_text().splitlines()[0])
    names = [ref[1] for ref in row["refs"]]
    assert len(names) == 5
    assert names == sorted(names)
    # The first five of a sorted list are one recording session for most
    # speakers, so the cap samples instead of truncating.
    assert names != [f"ears__p001_{n:03d}.flac" for n in range(5)]


def test_matching_nothing_is_an_error_not_an_empty_index(tmp_path):
    bucket = tmp_path / "bucket"
    _sidecar(bucket, "ears", "ears-000000", {"p001_a.flac": (512, 1000)})
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [_row("ears", "ears__p001_a", "ears__p001")],
    )

    with pytest.raises(SystemExit, match="matched none"):
        _build(tmp_path, manifests=[manifest])


def test_a_finished_dataset_is_not_rebuilt_after_a_preemption(tmp_path):
    bucket = tmp_path / "bucket"
    _sidecar(bucket, "ears", "ears-000000", {"ears__p001_a.flac": (512, 1000)})
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [_row("ears", "ears__p001_a", "ears__p001")],
    )
    _build(tmp_path, manifests=[manifest])

    # The sidecars are gone on the second run; the part file must carry it.
    for path in (tmp_path / "bucket" / builder.SIDECAR_PREFIX).rglob("*.json"):
        path.unlink()

    index_path, summary = _build(tmp_path, manifests=[manifest])

    assert summary["refs"] == 1
    assert len(index_path.read_text().splitlines()) == 1


def test_an_unknown_dataset_is_named_rather_than_silently_skipped(tmp_path):
    bucket = tmp_path / "bucket"
    _sidecar(bucket, "ears", "ears-000000", {"ears__p001_a.flac": (512, 1000)})
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [_row("ears", "ears__p001_a", "ears__p001")],
    )

    with pytest.raises(SystemExit, match="no manifest rows for"):
        _build(tmp_path, manifests=[manifest], datasets=["nope"])


def test_changing_the_manifests_does_not_reuse_the_old_parts(tmp_path):
    bucket = tmp_path / "bucket"
    _sidecar(
        bucket,
        "ears",
        "ears-000000",
        {
            "ears__p001_a.flac": (512, 1000),
            "ears__p002_a.flac": (2048, 1000),
        },
    )
    first = _manifest(
        bucket,
        "manifests/train.jsonl",
        [_row("ears", "ears__p001_a", "ears__p001")],
    )
    _build(tmp_path, manifests=[first])

    second = _manifest(
        bucket,
        "manifests/train2.jsonl",
        [_row("ears", "ears__p002_a", "ears__p002")],
    )
    index_path, _ = _build(tmp_path, manifests=[second])

    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert [row["speaker_id"] for row in rows] == ["ears__p002"]

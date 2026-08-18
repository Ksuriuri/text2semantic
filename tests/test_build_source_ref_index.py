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


def _metadata(root, dataset, shard, rows):
    """The pack-time metadata jsonl, which is what maps a row id to a member.

    ``rows`` is ``{row id: member}``; the real rows carry text, speaker and
    duration too, but the join only reads these two fields.
    """
    path = (
        Path(root)
        / builder.METADATA_TEMPLATE.format(dataset=dataset, stem=shard)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {"id": row_id, "audio_path": f"audio/{shard}.tar/{member}"},
                ensure_ascii=False,
            )
            + "\n"
            for row_id, member in rows.items()
        ),
        encoding="utf-8",
    )
    return path


def _packed(root, dataset, shard, members, *, ids=None):
    """A sidecar plus the metadata that names each member's row.

    ``members`` is ``{member: (offset, size)}``, and ``ids`` overrides the row id
    of a member for the datasets whose ids are not the member stem.
    """
    _sidecar(root, dataset, shard, members)
    ids = ids or {}
    _metadata(
        root,
        dataset,
        shard,
        {
            ids.get(member, member.rsplit(".", 1)[0]): member
            for member in members
        },
    )


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
    _packed(
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
                 "ears__p001_a", 512, 1000],
                ["preprocessed/ears/audio/ears-000000.tar",
                 "ears__p001_b", 2048, 2000],
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
    _packed(
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
            "laion_emolia__ZH_B00039_S01897_W000000",
            512,
            251033,
        ]
    ]


def test_a_clip_too_short_to_condition_on_is_not_a_ref(tmp_path):
    bucket = tmp_path / "bucket"
    _packed(
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
    assert [ref[1] for ref in row["refs"]] == ["ears__p001_long"]


def test_a_tar_member_with_no_manifest_row_is_skipped(tmp_path):
    bucket = tmp_path / "bucket"
    _packed(
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
    assert [ref[1] for ref in row["refs"]] == ["ears__p001_a"]
    assert summary["per_dataset"]["ears"]["members"] == 2


def test_the_same_speaker_id_in_two_datasets_stays_two_speakers(tmp_path):
    bucket = tmp_path / "bucket"
    _packed(bucket, "ears", "ears-000000", {"ears__p001_a.flac": (512, 1000)})
    _packed(bucket, "vctk", "vctk-000000", {"vctk__p001_a.flac": (512, 1000)})
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
    assert store.pick_ref(("ears", "en", "p001"))[1] == "ears__p001_a"
    assert store.pick_ref(("vctk", "en", "p001"))[1] == "vctk__p001_a"


def test_eval_rows_get_refs_too(tmp_path):
    bucket = tmp_path / "bucket"
    _packed(
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
    _packed(bucket, "ears", "ears-000000", members)
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
    assert names != [f"ears__p001_{n:03d}" for n in range(5)]


def test_a_sanitized_member_name_still_finds_its_row(tmp_path):
    # The packer turned "Genshin__en/#Unknown/vo_x" into
    # "Genshin__en_Unknown_vo_x.flac", so the member name is not the id and
    # cannot be turned back into it. Deriving one from the other matched 0 of
    # this dataset's 644,716 members; the metadata row states both.
    bucket = tmp_path / "bucket"
    row_id = "Genshin__en/#Unknown/vo_NTAQ008_15_olorun_01"
    _packed(
        bucket,
        "Genshin",
        "Genshin-000000",
        {"Genshin__en_Unknown_vo_NTAQ008_15_olorun_01.flac": (512, 217612)},
        ids={"Genshin__en_Unknown_vo_NTAQ008_15_olorun_01.flac": row_id},
    )
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [_row("Genshin", row_id, "Genshin__#Unknown_en")],
    )

    index_path, summary = _build(tmp_path, manifests=[manifest])

    row = json.loads(index_path.read_text().splitlines()[0])
    assert row["refs"] == [
        ["preprocessed/Genshin/audio/Genshin-000000.tar", row_id, 512, 217612]
    ]
    assert summary["refs"] == 1


def test_a_scanned_tar_with_no_metadata_is_counted_not_guessed_at(tmp_path):
    bucket = tmp_path / "bucket"
    _packed(bucket, "ears", "ears-000000", {"ears__p001_a.flac": (512, 1000)})
    # Scanned, but packed after the last metadata dump: its members cannot be
    # tied to a row id, and inventing that tie is how a ref gets misattributed.
    _sidecar(bucket, "ears", "ears-000001", {"ears__p001_b.flac": (512, 1000)})
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [
            _row("ears", "ears__p001_a", "ears__p001"),
            _row("ears", "ears__p001_b", "ears__p001"),
        ],
    )

    index_path, summary = _build(tmp_path, manifests=[manifest])

    row = json.loads(index_path.read_text().splitlines()[0])
    assert [ref[1] for ref in row["refs"]] == ["ears__p001_a"]
    assert summary["per_dataset"]["ears"]["tars_without_metadata"] == 1


def test_a_row_whose_member_is_not_in_the_tar_is_counted(tmp_path):
    # The metadata names a member the scan never saw: either the tar was
    # repacked or the scan was cut short. Either way the byte range is unknown,
    # so the row cannot become a ref.
    bucket = tmp_path / "bucket"
    _sidecar(bucket, "ears", "ears-000000", {"ears__p001_a.flac": (512, 1000)})
    _metadata(
        bucket,
        "ears",
        "ears-000000",
        {
            "ears__p001_a": "ears__p001_a.flac",
            "ears__p001_gone": "ears__p001_gone.flac",
        },
    )
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [
            _row("ears", "ears__p001_a", "ears__p001"),
            _row("ears", "ears__p001_gone", "ears__p001"),
        ],
    )

    index_path, summary = _build(tmp_path, manifests=[manifest])

    row = json.loads(index_path.read_text().splitlines()[0])
    assert [ref[1] for ref in row["refs"]] == ["ears__p001_a"]
    assert summary["per_dataset"]["ears"]["unlocated"] == 1


def test_matching_nothing_is_an_error_not_an_empty_index(tmp_path):
    bucket = tmp_path / "bucket"
    _packed(bucket, "ears", "ears-000000", {"p001_a.flac": (512, 1000)})
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [_row("ears", "ears__p001_a", "ears__p001")],
    )

    with pytest.raises(SystemExit, match="matched none"):
        _build(tmp_path, manifests=[manifest])


def test_a_finished_dataset_is_not_rebuilt_after_a_preemption(tmp_path):
    bucket = tmp_path / "bucket"
    _packed(bucket, "ears", "ears-000000", {"ears__p001_a.flac": (512, 1000)})
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
    _packed(bucket, "ears", "ears-000000", {"ears__p001_a.flac": (512, 1000)})
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [_row("ears", "ears__p001_a", "ears__p001")],
    )

    with pytest.raises(SystemExit, match="no manifest rows for"):
        _build(tmp_path, manifests=[manifest], datasets=["nope"])


def test_changing_the_manifests_does_not_reuse_the_old_parts(tmp_path):
    bucket = tmp_path / "bucket"
    _packed(
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


def _split(tmp_path, manifests, **kwargs):
    rows_dir, rebuilt = builder.split_manifests(
        builder.LocalBlobs(tmp_path / "bucket"),
        manifests,
        tmp_path / "work",
        log=None,
        **kwargs,
    )
    return rows_dir, rebuilt


def test_a_range_boundary_neither_drops_nor_doubles_a_row(tmp_path):
    # The 54 GB manifest is cut into 135 ranges, and every cut lands mid-line.
    # A range owns the lines that start inside it, so the line straddling the
    # cut belongs to the range before it and to nothing else.
    bucket = tmp_path / "bucket"
    rows = [
        _row("ears", f"ears__p001_{n:04d}", "ears__p001") for n in range(200)
    ]
    manifest = _manifest(bucket, "manifests/train.jsonl", rows)
    size = (bucket / "manifests" / "train.jsonl").stat().st_size

    whole, _ = builder.split_manifests(
        builder.LocalBlobs(bucket),
        [manifest],
        tmp_path / "one",
        part_bytes=size,
        log=None,
    )
    expected = (whole / "ears.tsv").read_bytes()

    for part_bytes in (7, 101, size // 3, size - 1, size + 1):
        rows_dir, _ = builder.split_manifests(
            builder.LocalBlobs(bucket),
            [manifest],
            tmp_path / f"cut{part_bytes}",
            part_bytes=part_bytes,
            workers=4,
            log=None,
        )
        assert (rows_dir / "ears.tsv").read_bytes() == expected
        counts = json.loads((rows_dir / "_COMPLETE").read_text())
        assert counts["rows"] == 200
        assert counts["kept"] == 200


def test_the_ranges_are_concatenated_in_manifest_order(tmp_path):
    # Sorting on the part name is what keeps a rebuild byte-identical, and
    # "10" sorts before "2" as a string.
    bucket = tmp_path / "bucket"
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [_row("ears", f"ears__p001_{n:04d}", "ears__p001") for n in range(60)],
    )

    rows_dir, _ = _split(tmp_path, [manifest], part_bytes=64, workers=2)

    ids = [
        line.split("\t")[0]
        for line in (rows_dir / "ears.tsv").read_text().splitlines()
    ]
    assert ids == [f"ears__p001_{n:04d}" for n in range(60)]


def test_the_split_counts_what_it_dropped(tmp_path):
    bucket = tmp_path / "bucket"
    manifest = _manifest(
        bucket,
        "manifests/train.jsonl",
        [
            _row("ears", "ears__p001_a", "ears__p001", duration=6.0),
            _row("ears", "ears__p001_short", "ears__p001", duration=1.0),
            _row("ears", "ears__p001_huge", "ears__p001", duration=41.0),
            {"id": "ears__p001_x", "dataset": "ears", "duration": 6.0},
        ],
    )

    rows_dir, _ = _split(tmp_path, [manifest], part_bytes=32, workers=3)

    counts = json.loads((rows_dir / "_COMPLETE").read_text())
    assert counts["rows"] == 4
    assert counts["kept"] == 1
    assert counts["bad_duration"] == 2
    assert counts["no_speaker"] == 1
    assert counts["manifests"] == [manifest]


def test_the_split_is_not_redone_but_a_changed_manifest_list_is(tmp_path):
    bucket = tmp_path / "bucket"
    first = _manifest(
        bucket,
        "manifests/train.jsonl",
        [_row("ears", "ears__p001_a", "ears__p001")],
    )
    _split(tmp_path, [first])
    _, rebuilt = _split(tmp_path, [first])
    assert rebuilt is False

    second = _manifest(
        bucket,
        "manifests/eval.jsonl",
        [_row("ears", "ears__p002_a", "ears__p002")],
    )
    rows_dir, rebuilt = _split(tmp_path, [first, second])
    assert rebuilt is True
    ids = [
        line.split("\t")[0]
        for line in (rows_dir / "ears.tsv").read_text().splitlines()
    ]
    assert ids == ["ears__p001_a", "ears__p002_a"]

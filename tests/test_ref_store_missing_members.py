"""What happens when the speaker index and the shards disagree.

Separate from test_ref_store.py on purpose: that module imports finetuning.train
and so needs torch and accelerate, while these tests are about plain file reads
and stay runnable on any box -- which matters, because the bug they cover took a
64-rank job down and the fix had to be checked before the next relaunch.
"""

import io
import json
import random
import tarfile
from pathlib import Path

import pytest

from finetuning import ref_member_index
from finetuning.ref_store import SpeakerRefStore

KEY = ("en", "spkA")


def _wav_bytes():
    return b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 24


def _build(tmp_path, *, packed, indexed):
    """A shard holding `packed` while the speaker index names `indexed`."""
    root = Path(tmp_path)
    shards = root / "refs" / "shards"
    shards.mkdir(parents=True)
    wav = _wav_bytes()
    with tarfile.open(shards / "refs-000000.tar", "w") as archive:
        for name in packed:
            info = tarfile.TarInfo(name)
            info.size = len(wav)
            archive.addfile(info, fileobj=io.BytesIO(wav))
    index = root / "refs" / "speaker_index.jsonl"
    index.write_text(
        json.dumps(
            {
                "language": "en",
                "speaker_id": "spkA",
                "shard": "refs-000000.tar",
                "members": list(indexed),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return index


def test_read_ref_skips_a_member_the_shard_does_not_have(tmp_path):
    # A shard rewritten after a packer restart: the index still names a member
    # the tar no longer holds, and it is the one this row would pick.
    index = _build(
        tmp_path,
        packed=("spkA/000.flac", "spkA/001.flac"),
        indexed=("spkA/gone.flac", "spkA/000.flac", "spkA/001.flac"),
    )
    store = SpeakerRefStore(index)
    assert store.pick_member(KEY) == "spkA/gone.flac"
    member, payload = store.read_ref(KEY)
    assert member == "spkA/000.flac"
    assert payload == _wav_bytes()
    assert store.missing_members == 1


def test_read_ref_is_none_when_the_speaker_has_no_readable_member(tmp_path):
    index = _build(
        tmp_path,
        packed=("spkB/000.flac",),
        indexed=("spkA/gone.flac", "spkA/also-gone.flac"),
    )
    store = SpeakerRefStore(index)
    # None is the contract the dataset already handles: it raises for that row
    # rather than reporting a ref that was never read.
    assert store.read_ref(KEY) is None
    assert store.missing_members == 2


def test_pick_members_begins_with_what_pick_member_chose(tmp_path):
    members = ("spkA/000.flac", "spkA/001.flac", "spkA/002.flac")
    index = _build(tmp_path, packed=members, indexed=members)
    store = SpeakerRefStore(index)
    for seed in range(8):
        # The fallback order must not change which ref a seeded row trains on.
        chosen = store.pick_member(KEY, rng=random.Random(seed))
        order = store.pick_members(KEY, rng=random.Random(seed))
        assert order[0] == chosen
        assert sorted(order) == sorted(members)


def test_exclude_still_applies_to_every_fallback(tmp_path):
    # The row's own clip must never come back as its own ref, not even as the
    # second choice after a missing member.
    index = _build(
        tmp_path,
        packed=("spkA/000.flac", "spkA/001.flac"),
        indexed=("spkA/gone.flac", "spkA/000.flac", "spkA/001.flac"),
    )
    store = SpeakerRefStore(index)
    member, _ = store.read_ref(KEY, exclude="000")
    assert member == "spkA/001.flac"


def test_a_short_read_is_still_fatal(tmp_path):
    members = ("spkA/000.flac", "spkA/001.flac")
    index = _build(tmp_path, packed=members, indexed=members)
    shard = index.parent / "shards" / "refs-000000.tar"
    index_dir = ref_member_index.index_dir_for(index.parent)
    ref_member_index.build_shard_index(shard, index_dir)
    sidecar = ref_member_index.index_path_for(shard, index_dir)
    payload = json.loads(sidecar.read_text())
    offset, size = payload["members"]["spkA/000.flac"]
    # shard_size is left alone so the sidecar is not rebuilt. This is a damaged
    # shard or a bad offset, which must not be mistaken for a stale index and
    # skipped: training on a truncated ref is worse than stopping.
    # Past the end of the archive, allowing for tar's 10240-byte record padding.
    payload["members"]["spkA/000.flac"] = [offset, size + 32768]
    sidecar.write_text(json.dumps(payload))
    store = SpeakerRefStore(index)
    with pytest.raises(OSError) as raised:
        store.read_ref(KEY)
    # FileNotFoundError is an OSError, and it is the one thing read_ref skips.
    assert not isinstance(raised.value, FileNotFoundError)
    assert "short read" in str(raised.value)

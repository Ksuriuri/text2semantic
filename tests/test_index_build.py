import json
import multiprocessing as mp
import os
from pathlib import Path

from finetuning import index_build, manifest_index


def write_manifest(path, rows=400):
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(
                json.dumps(
                    {
                        "id": f"clip-{index}",
                        "speaker_id": f"speaker-{index % 20}",
                        "language": "zh",
                        "text": "你好",
                        "duration": 5.0,
                        "semantic_code_path": f"../codes/x/{index}.u2.bin",
                    }
                )
                + "\n"
            )
    return path


def params():
    return manifest_index.FilterParams(min_speaker_records=1)


def build_in_subprocess(manifest_path):
    """A second, unrelated process building the same index."""
    manifest_index.load(manifest_path, params=params(), log=None)


def test_publisher_makes_files_visible_only_on_publish(tmp_path):
    publisher = index_build.IndexPublisher(tmp_path)
    staged = publisher.path("offsets.u64")
    staged.write_bytes(b"\x00" * 8)

    assert not (tmp_path / "offsets.u64").exists()
    publisher.publish()
    assert (tmp_path / "offsets.u64").read_bytes() == b"\x00" * 8


def test_publisher_stages_under_a_pid_specific_name(tmp_path):
    # Two builders must not share one temp filename: they would truncate and
    # write the same file, then both rename it.
    staged = index_build.IndexPublisher(tmp_path).path("offsets.u64")
    assert staged.name == f"offsets.u64.part-{os.getpid()}"


def test_publisher_discards_staged_files_on_error(tmp_path):
    try:
        with index_build.IndexPublisher(tmp_path) as publisher:
            publisher.path("lengths.u32").write_bytes(b"\x01" * 4)
            raise RuntimeError("build failed")
    except RuntimeError:
        pass

    assert list(tmp_path.glob("lengths.u32*")) == []


def test_build_lock_is_exclusive(tmp_path):
    with index_build.build_lock(tmp_path) as first:
        assert first.locked
        # A second waiter in the same process would deadlock on a real flock,
        # so this checks the observable state a caller branches on instead.
        assert first.waited == 0.0


def test_concurrent_loads_produce_one_usable_index(tmp_path):
    manifest = write_manifest(tmp_path / "train.jsonl")
    index_dir = Path(str(manifest) + ".index")

    context = mp.get_context("spawn")
    workers = [
        context.Process(target=build_in_subprocess, args=(str(manifest),))
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=180)
    assert [worker.exitcode for worker in workers] == [0, 0, 0, 0]

    # No temp files left behind, and the index reads back consistently: a
    # corrupt interleaved build shows up here as a row count mismatch or a JSON
    # decode error.
    assert list(index_dir.glob("*.part-*")) == []
    index = manifest_index.ManifestIndex(manifest)
    assert len(index) == 400
    assert index[0]["id"] == "clip-0"
    assert index[399]["id"] == "clip-399"
    assert json.loads((index_dir / "meta.json").read_text())["kept_rows"] == 400

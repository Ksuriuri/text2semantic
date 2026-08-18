import types

from finetuning import checkpoint_remote


class FakeRun:
    """Records gcloud invocations and replays canned `ls` output."""

    def __init__(self, listing=()):
        self.commands = []
        self.listing = list(listing)

    def __call__(self, command):
        self.commands.append(command)
        if command[:3] == ["gcloud", "storage", "ls"]:
            return types.SimpleNamespace(stdout="\n".join(self.listing) + "\n")
        return types.SimpleNamespace(stdout="")


def listing(*checkpoints):
    return [
        f"{path.rstrip('/')}/{checkpoint_remote.COMPLETE_MARKER}"
        for path in checkpoints
    ]


def test_step_of_reads_both_prefixes():
    assert checkpoint_remote.step_of("gs://b/run/checkpoint-step-1250/") == 1250
    assert (
        checkpoint_remote.step_of("gs://b/run/checkpoint-keep-step-5000") == 5000
    )
    assert checkpoint_remote.step_of("gs://b/run/accelerator_state") is None


def test_latest_complete_orders_by_step_not_lexically():
    # "checkpoint-step-950" sorts after "checkpoint-step-1000" as a string, so a
    # lexical listing order would resume from the older checkpoint.
    run = FakeRun(
        listing(
            "gs://b/run/checkpoint-step-950",
            "gs://b/run/checkpoint-step-1000",
        )
    )
    assert (
        checkpoint_remote.latest_complete("gs://b/run", run=run)
        == "gs://b/run/checkpoint-step-1000"
    )


def test_latest_complete_ignores_checkpoints_without_a_marker():
    run = FakeRun(listing("gs://b/run/checkpoint-step-100"))
    # The listing only ever returns markers, so an interrupted upload simply is
    # not in it; what matters is that a marker-less prefix cannot be selected.
    run.listing.append("gs://b/run/checkpoint-step-200/model.safetensors")
    assert (
        checkpoint_remote.latest_complete("gs://b/run", run=run)
        == "gs://b/run/checkpoint-step-100"
    )


def test_latest_complete_is_none_on_an_empty_prefix():
    run = FakeRun()
    assert checkpoint_remote.latest_complete("gs://b/run", run=run) is None


def test_upload_then_mark_complete_writes_the_marker_last():
    run = FakeRun()
    checkpoint_remote.upload("/out/checkpoint-step-50", "gs://b/run/checkpoint-step-50", run=run)
    checkpoint_remote.mark_complete("gs://b/run/checkpoint-step-50", run=run)
    assert run.commands[0][:4] == ["gcloud", "storage", "rsync", "--recursive"]
    assert run.commands[1] == [
        "gcloud",
        "storage",
        "cp",
        "-",
        f"gs://b/run/checkpoint-step-50/{checkpoint_remote.COMPLETE_MARKER}",
    ]


def test_rotate_keeps_the_newest_rolling_checkpoints():
    run = FakeRun(
        listing(
            "gs://b/run/checkpoint-step-50",
            "gs://b/run/checkpoint-step-100",
            "gs://b/run/checkpoint-step-150",
        )
    )
    deleted = checkpoint_remote.rotate("gs://b/run", 2, run=run)
    assert deleted == ["gs://b/run/checkpoint-step-50"]
    assert run.commands[-1] == [
        "gcloud",
        "storage",
        "rm",
        "--recursive",
        "gs://b/run/checkpoint-step-50",
    ]


def test_rotate_never_deletes_a_persistent_checkpoint():
    run = FakeRun(
        listing(
            "gs://b/run/checkpoint-keep-step-5000",
            "gs://b/run/checkpoint-step-5050",
            "gs://b/run/checkpoint-step-5100",
            "gs://b/run/checkpoint-step-5150",
        )
    )
    deleted = checkpoint_remote.rotate("gs://b/run", 2, run=run)
    assert deleted == ["gs://b/run/checkpoint-step-5050"]


def test_rotate_with_a_zero_limit_deletes_nothing():
    run = FakeRun(listing("gs://b/run/checkpoint-step-50"))
    assert checkpoint_remote.rotate("gs://b/run", 0, run=run) == []
    assert run.commands == []


def test_is_remote_and_join():
    assert checkpoint_remote.is_remote("gs://b/run")
    assert not checkpoint_remote.is_remote("/out/run")
    assert not checkpoint_remote.is_remote(None)
    assert (
        checkpoint_remote.join("gs://b/run/", "/checkpoint-step-50/")
        == "gs://b/run/checkpoint-step-50"
    )

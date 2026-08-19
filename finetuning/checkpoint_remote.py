# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Checkpoints on object storage, because the node's disk goes with the node.

A Spot preemption takes the whole machine: its local SSD, its boot disk, and any
checkpoint that only ever existed there.  With two nodes the job also cannot
survive losing either one, so "the checkpoint is on node 0" is the same as no
checkpoint at all.

So each save is mirrored to a GCS prefix, and a restarted job -- on new machines,
with new local disks -- resumes from there.  Two details make that reliable:

* **Every node uploads its own files.**  ``accelerate.save_state`` writes the
  model and optimizer from the main process but ``random_states_<rank>.pkl`` from
  every rank, so with node-local disks neither node holds the complete set.  The
  union of both nodes' uploads does.
* **A marker object decides completeness.**  A preemption in the middle of an
  upload leaves a partial checkpoint that looks exactly like a whole one from a
  listing; nothing but an explicitly written marker distinguishes them.  Resume
  only ever considers a prefix that has one.

Uploads are synchronous.  A ~14 GB checkpoint takes 10-20 s in-region, which
against a 30-minute save interval is about 1% of the run, and doing it inline
means rotation can never delete a directory that is still being copied.

The transfer shells out to the object store's own CLI -- ``gcloud storage`` for
``gs://`` and ``aws s3`` for ``s3://`` -- because both parallelise across objects
and are already on their platform's images.  ``run`` is injectable so the logic
can be tested without touching the network.

A run on one cloud wants its checkpoints in that cloud: mirroring a ~14 GB save
across clouds every half hour costs egress on every save and adds the one delay
that is inline with training.  The two backends differ only in the four commands
below; everything above them -- the marker, the step ordering, the rotation rule
-- is shared, so resume semantics cannot drift between platforms.
"""

from __future__ import annotations

import re
import subprocess

COMPLETE_MARKER = "_UPLOAD_COMPLETE"
_STEP_PATTERN = re.compile(r"checkpoint-(?:keep-)?step-(\d+)/?$")
GCS = "gs://"
S3 = "s3://"
_SCHEMES = (GCS, S3)


def is_remote(path):
    return bool(path) and str(path).startswith(_SCHEMES)


def scheme_of(uri):
    """Which object store a URI names. Raises rather than guessing."""
    text = str(uri)
    for scheme in _SCHEMES:
        if text.startswith(scheme):
            return scheme
    raise ValueError(f"{uri!r} is not a remote checkpoint URI (gs:// or s3://)")


def join(prefix, *parts):
    joined = str(prefix).rstrip("/")
    for part in parts:
        joined = f"{joined}/{str(part).strip('/')}"
    return joined


def step_of(uri):
    """The global step a checkpoint URI or directory name encodes, else None."""
    match = _STEP_PATTERN.search(str(uri).rstrip("/") + "/")
    return int(match.group(1)) if match else None


def _run(command):
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _sync_command(src, dst):
    if scheme_of(dst if is_remote(dst) else src) == S3:
        # `aws s3 sync` is already recursive and copies only what differs.
        return ["aws", "s3", "sync", src, dst]
    return ["gcloud", "storage", "rsync", "--recursive", src, dst]


def upload(local_dir, remote_dir, *, run=_run):
    """Copy a finished local checkpoint directory to its remote prefix.

    Called by one process per node, each copying what its own node wrote.  The
    copy is a plain recursive rsync: object storage has no partial-write
    visibility problem per object, and the marker covers the set.
    """
    run(_sync_command(str(local_dir).rstrip("/"), str(remote_dir).rstrip("/")))
    return remote_dir


def mark_complete(remote_dir, *, run=_run):
    """Write the marker that makes a remote checkpoint eligible for resume."""
    marker = join(remote_dir, COMPLETE_MARKER)
    if scheme_of(remote_dir) == S3:
        run(["aws", "s3", "cp", "-", marker])
    else:
        run(["gcloud", "storage", "cp", "-", marker])
    return marker


_GCS_NO_MATCH = "matched no objects"


def _list(command, *, run):
    """Run a listing command, treating "the prefix holds nothing" as no output.

    Both CLIs exit non-zero when a prefix matches no object, and at the start of
    a run the checkpoint prefix legitimately matches nothing -- which is exactly
    when `--resume_from_checkpoint auto` has to answer "begin at step 0" instead
    of raising CalledProcessError on every rank.

    What must not be swallowed with it is a real failure: a listing error that
    reads as an empty prefix is how a broken sync hides for hours. The two are
    distinguishable because an empty listing says nothing at all -- `aws s3 ls`
    prints neither stdout nor stderr -- while AccessDenied, a missing bucket or
    expired credentials always explain themselves on stderr. `gcloud storage ls`
    is the one exception, having chosen to phrase "nothing here" as an error, so
    its exact wording is matched and nothing else is.
    """
    try:
        return run(command)
    except subprocess.CalledProcessError as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if stdout.strip():
            raise
        if stderr.strip() and _GCS_NO_MATCH not in stderr:
            raise
        return subprocess.CompletedProcess(command, 0, "", stderr)


def _marker_uris(remote_prefix, *, run):
    """Every COMPLETE_MARKER object under the prefix, as full URIs.

    `gcloud storage ls` takes a `**` glob and prints URIs. `aws s3 ls` has
    neither: it needs `--recursive` and prints `date time size key`, with the key
    relative to the bucket, so the URI has to be put back together.
    """
    prefix = str(remote_prefix).rstrip("/")
    if scheme_of(prefix) == S3:
        bucket = prefix[len(S3) :].split("/", 1)[0]
        result = _list(["aws", "s3", "ls", "--recursive", f"{prefix}/"], run=run)
        uris = []
        for line in result.stdout.splitlines():
            fields = line.split(maxsplit=3)
            if len(fields) == 4:
                uris.append(f"{S3}{bucket}/{fields[3]}")
        return uris
    result = _list(
        ["gcloud", "storage", "ls", join(prefix, f"**/{COMPLETE_MARKER}")], run=run
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def list_complete(remote_prefix, *, run=_run):
    """[(step, uri)] for the remote checkpoints that finished uploading."""
    found = []
    for line in _marker_uris(remote_prefix, run=run):
        if not line.endswith(COMPLETE_MARKER):
            continue
        checkpoint = line[: -len(COMPLETE_MARKER)].rstrip("/")
        step = step_of(checkpoint)
        if step is not None:
            found.append((step, checkpoint))
    found.sort()
    return found


def latest_complete(remote_prefix, *, run=_run):
    """The newest fully uploaded remote checkpoint, or None."""
    found = list_complete(remote_prefix, run=run)
    return found[-1][1] if found else None


def download(remote_dir, local_dir, *, run=_run):
    """Fetch a remote checkpoint onto this node's disk.

    Called by one process per node after a restart: every rank reads the state,
    and each node has its own empty disk.
    """
    run(_sync_command(str(remote_dir).rstrip("/"), str(local_dir).rstrip("/")))
    return local_dir


def rotate(remote_prefix, limit, *, run=_run):
    """Delete all but the newest `limit` rolling checkpoints under the prefix.

    Persistent checkpoints keep their own name (``checkpoint-keep-step-N``) and
    are never considered here -- the point of them is that nothing rotates them
    away.
    """
    if limit <= 0:
        return []
    rolling = [
        (step, uri)
        for step, uri in list_complete(remote_prefix, run=run)
        if "checkpoint-keep-step-" not in uri
    ]
    doomed = [uri for _step, uri in rolling[:-limit]]
    for uri in doomed:
        target = uri.rstrip("/")
        if scheme_of(target) == S3:
            # A trailing slash matters here: without it `rm --recursive` also
            # matches sibling prefixes that merely start with this name, e.g.
            # checkpoint-step-100 would take checkpoint-step-1000 with it.
            run(["aws", "s3", "rm", "--recursive", f"{target}/"])
        else:
            run(["gcloud", "storage", "rm", "--recursive", target])
    return doomed

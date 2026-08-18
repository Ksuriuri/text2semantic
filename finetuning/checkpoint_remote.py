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

The transfer itself shells out to ``gcloud storage``, which parallelises across
objects and is already on every GCP image; ``run`` is injectable so the logic can
be tested without touching the network.
"""

from __future__ import annotations

import re
import subprocess

COMPLETE_MARKER = "_UPLOAD_COMPLETE"
_STEP_PATTERN = re.compile(r"checkpoint-(?:keep-)?step-(\d+)/?$")


def is_remote(path):
    return bool(path) and str(path).startswith("gs://")


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


def upload(local_dir, remote_dir, *, run=_run):
    """Copy a finished local checkpoint directory to its remote prefix.

    Called by one process per node, each copying what its own node wrote.  The
    copy is a plain recursive rsync: object storage has no partial-write
    visibility problem per object, and the marker covers the set.
    """
    run(
        [
            "gcloud",
            "storage",
            "rsync",
            "--recursive",
            str(local_dir).rstrip("/"),
            str(remote_dir).rstrip("/"),
        ]
    )
    return remote_dir


def mark_complete(remote_dir, *, run=_run):
    """Write the marker that makes a remote checkpoint eligible for resume."""
    run(
        [
            "gcloud",
            "storage",
            "cp",
            "-",
            join(remote_dir, COMPLETE_MARKER),
        ]
    )
    return join(remote_dir, COMPLETE_MARKER)


def list_complete(remote_prefix, *, run=_run):
    """[(step, uri)] for the remote checkpoints that finished uploading."""
    result = run(
        [
            "gcloud",
            "storage",
            "ls",
            join(remote_prefix, f"**/{COMPLETE_MARKER}"),
        ]
    )
    found = []
    for line in result.stdout.splitlines():
        line = line.strip()
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
    run(
        [
            "gcloud",
            "storage",
            "rsync",
            "--recursive",
            str(remote_dir).rstrip("/"),
            str(local_dir).rstrip("/"),
        ]
    )
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
        run(["gcloud", "storage", "rm", "--recursive", uri.rstrip("/")])
    return doomed

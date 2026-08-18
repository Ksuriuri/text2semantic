# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Building a shared index from more than one process at a time.

Single-node training could rely on a convention: rank 0 builds the indexes,
everybody waits at a barrier, then all ranks open them.  Two nodes break that.
``accelerator.is_main_process`` is true on global rank 0 only, so with a
node-local index directory the second node never builds anything, and its eight
ranks then all reach ``load`` and start building the same files at once.

The old build wrote ``offsets.bin`` in place, so two builders interleaved their
writes into one file and ``meta.json`` -- written last, by whichever finished
last -- declared the result good.  That is a silently corrupt index: row offsets
point into the middle of other rows.  The failure surfaces much later as a JSON
decode error in a DataLoader worker, if it surfaces at all.

So the invariant moves into the build itself: one builder at a time per index
directory, and files become visible only once they are complete.  That holds for
every combination that matters -- one node or two, a node-local index directory
or a shared filesystem -- instead of holding only as long as callers remember
the convention.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no flock
    fcntl = None

LOCK_NAME = ".build.lock"


class _NullLock:
    """Stand-in for a filesystem that cannot lock.

    Losing the lock is not a reason to refuse to train: it degrades to the old
    behaviour, which is safe as long as one process builds.  It is worth a loud
    line in the log, though, because the reason a concurrent build corrupts an
    index is exactly this.
    """

    waited = 0.0
    locked = False


def build_lock(index_dir, *, log=None):
    """Exclusive build lock for one index directory.

    Returns a context manager.  Entering it blocks until no other process is
    building this index; ``locked`` says whether the lock was really taken, and
    ``waited`` how long the wait was -- a caller re-checks staleness after a
    wait, because the process it waited for has probably just built the index it
    was about to build.
    """
    return _BuildLock(index_dir, log=log)


class _BuildLock:
    def __init__(self, index_dir, *, log=None):
        self.index_dir = Path(index_dir)
        self.log = log
        self.waited = 0.0
        self.locked = False
        self._handle = None

    def __enter__(self):
        if fcntl is None:
            if self.log is not None:
                self.log(
                    "No flock on this platform; index builds are not serialised."
                )
            return self
        self.index_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.index_dir / LOCK_NAME
        try:
            self._handle = open(lock_path, "a+")
            started = time.monotonic()
            try:
                fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                if self.log is not None:
                    self.log(
                        f"Another process is building {self.index_dir}; waiting."
                    )
                fcntl.flock(self._handle, fcntl.LOCK_EX)
                self.waited = time.monotonic() - started
                if self.log is not None:
                    self.log(f"  waited {self.waited:.0f}s for the index build")
            self.locked = True
        except OSError as exc:
            # gcsfuse and some network mounts do not implement flock.
            if self.log is not None:
                self.log(
                    f"Cannot lock {lock_path} ({exc}); index builds are not "
                    "serialised. Build the index once up front if more than "
                    "one process may start together."
                )
            self._close()
        return self

    def __exit__(self, *_exc):
        self._close()
        return False

    def _close(self):
        if self._handle is not None:
            try:
                if self.locked and fcntl is not None:
                    fcntl.flock(self._handle, fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None
        self.locked = False


class IndexPublisher:
    """Write index files under temp names, then move them into place.

    The temp name carries the pid, because a fixed ``.part`` name is itself a
    collision: two builders truncate and write the same temp file and then both
    rename it, which produces a file that is neither build's output.
    """

    def __init__(self, index_dir):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._suffix = f".part-{os.getpid()}"
        self._staged = []

    def path(self, name):
        staged = self.index_dir / (name + self._suffix)
        self._staged.append((staged, self.index_dir / name))
        return staged

    def publish(self):
        """Move every staged file into place.

        ``os.replace`` is atomic per file, so a reader either sees the whole old
        file or the whole new one.  Callers still write ``meta.json`` last: that
        is what makes the *set* of files usable, and a build that dies here
        leaves a directory that ``load`` rejects.
        """
        for staged, final in self._staged:
            os.replace(staged, final)
        self._staged = []

    def discard(self):
        for staged, _final in self._staged:
            try:
                staged.unlink()
            except FileNotFoundError:
                pass
        self._staged = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_rest):
        if exc_type is not None:
            self.discard()
        return False

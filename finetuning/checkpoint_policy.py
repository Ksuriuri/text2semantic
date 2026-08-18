# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""When to write a checkpoint.

Preemptible (Spot) nodes make this a cost decision rather than a bookkeeping
one: every minute of unsaved progress is a minute the whole node may have to
redo.  Saving on a step count alone gets this wrong in both directions -- a
step is seconds at the start of a run and much longer once sequence lengths
grow, so a fixed interval is either too rare to protect the work or so frequent
that the job spends its time writing 28 GB of optimizer state.

So a rolling checkpoint needs *both* conditions to hold:

* at least ``min_interval_seconds`` since the last checkpoint of any kind, and
* the step is a multiple of ``rolling_steps``.

The step multiple keeps the save on a gradient-sync boundary and keeps
checkpoint names predictable; the time gate sets the actual rate.  Separately,
every ``persistent_steps`` steps a checkpoint is kept forever -- those are the
ones a human compares later, and they are never rotated away.

The policy is deliberately clock-injectable and free of torch/accelerate
imports: the decision has to be taken by one rank and broadcast (see
``decide_checkpoint`` in train.py), and a policy that cannot be tested without
a distributed launcher would not be tested at all.
"""

import time


ACTION_NONE = 0
ACTION_ROLLING = 1
ACTION_PERSISTENT = 2

ROLLING_PREFIX = "checkpoint-step-"
PERSISTENT_PREFIX = "checkpoint-keep-step-"


class CheckpointPolicy:
    """Decides whether a step is due a rolling or a persistent checkpoint."""

    def __init__(
        self,
        *,
        rolling_steps,
        persistent_steps,
        min_interval_seconds=0.0,
        clock=time.monotonic,
    ):
        if rolling_steps <= 0:
            raise ValueError("rolling_steps must be positive.")
        if persistent_steps <= 0:
            raise ValueError("persistent_steps must be positive.")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative.")
        self.rolling_steps = rolling_steps
        self.persistent_steps = persistent_steps
        self.min_interval_seconds = float(min_interval_seconds)
        self._clock = clock
        # The run start is the reference for the first interval: "30 minutes
        # since the last save" has to mean something on a fresh run too.
        self._last_save = self._clock()

    def action(self, global_step):
        """ACTION_NONE / ACTION_ROLLING / ACTION_PERSISTENT for this step.

        A persistent step wins when both are due, so a run does not write the
        same weights twice a few seconds apart.  It still resets the interval,
        because what the interval protects is unsaved progress, and a
        persistent checkpoint saves exactly the same state.
        """
        if global_step <= 0:
            return ACTION_NONE
        if global_step % self.persistent_steps == 0:
            return ACTION_PERSISTENT
        if global_step % self.rolling_steps != 0:
            return ACTION_NONE
        if self._clock() - self._last_save < self.min_interval_seconds:
            return ACTION_NONE
        return ACTION_ROLLING

    def record_save(self):
        """Restart the interval.  Call this after a save actually lands."""
        self._last_save = self._clock()

    def seconds_since_save(self):
        return self._clock() - self._last_save


def checkpoint_dir_name(action, global_step):
    if action == ACTION_PERSISTENT:
        return f"{PERSISTENT_PREFIX}{global_step}"
    return f"{ROLLING_PREFIX}{global_step}"

from finetuning.checkpoint_policy import (
    ACTION_NONE,
    ACTION_PERSISTENT,
    ACTION_ROLLING,
    CheckpointPolicy,
    checkpoint_dir_name,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_policy(clock, **overrides):
    params = {
        "rolling_steps": 50,
        "persistent_steps": 5000,
        "min_interval_seconds": 1800.0,
        "clock": clock,
    }
    params.update(overrides)
    return CheckpointPolicy(**params)


def test_rolling_needs_both_the_step_multiple_and_the_interval():
    clock = FakeClock()
    policy = make_policy(clock)

    # Interval not yet elapsed: no save, however many step multiples pass.
    assert policy.action(50) == ACTION_NONE
    assert policy.action(1000) == ACTION_NONE

    clock.advance(1800.0)
    # Interval elapsed but the step is not a multiple of 50.
    assert policy.action(1017) == ACTION_NONE
    assert policy.action(1050) == ACTION_ROLLING


def test_saving_restarts_the_interval():
    clock = FakeClock()
    policy = make_policy(clock)
    clock.advance(1800.0)

    assert policy.action(1050) == ACTION_ROLLING
    policy.record_save()

    clock.advance(600.0)
    assert policy.action(1100) == ACTION_NONE
    clock.advance(1200.0)
    assert policy.action(1150) == ACTION_ROLLING


def test_persistent_steps_ignore_the_interval_and_win_over_rolling():
    clock = FakeClock()
    policy = make_policy(clock)

    # 5000 is also a multiple of 50, and no time has passed at all: the
    # persistent checkpoint still has to be written, and only once.
    assert policy.action(5000) == ACTION_PERSISTENT
    policy.record_save()
    assert policy.action(5050) == ACTION_NONE


def test_persistent_save_also_restarts_the_rolling_interval():
    clock = FakeClock()
    policy = make_policy(clock)
    clock.advance(1800.0)
    assert policy.action(5000) == ACTION_PERSISTENT
    policy.record_save()

    clock.advance(60.0)
    assert policy.action(5050) == ACTION_NONE


def test_zero_interval_saves_on_every_step_multiple():
    clock = FakeClock()
    policy = make_policy(clock, min_interval_seconds=0.0)

    assert policy.action(50) == ACTION_ROLLING
    policy.record_save()
    assert policy.action(100) == ACTION_ROLLING


def test_step_zero_never_saves():
    assert make_policy(FakeClock(), min_interval_seconds=0.0).action(0) == (
        ACTION_NONE
    )


def test_checkpoint_dir_names_keep_the_two_prefixes_apart():
    assert checkpoint_dir_name(ACTION_ROLLING, 1250) == "checkpoint-step-1250"
    assert (
        checkpoint_dir_name(ACTION_PERSISTENT, 5000)
        == "checkpoint-keep-step-5000"
    )

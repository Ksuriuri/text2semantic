import json
import math
import os
import signal
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import (
    Qwen3_5TextConfig,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from finetuning import checkpoint_remote
from finetuning.checkpoint_policy import (
    ACTION_NONE,
    ACTION_PERSISTENT,
    ACTION_ROLLING,
    CheckpointPolicy,
)
from finetuning.train import (
    PreemptionFlag,
    add_speaker_features,
    build_dataset,
    build_optimizer,
    decide_checkpoint,
    evaluate,
    fetch_remote_checkpoint,
    latest_checkpoint,
    mirror_checkpoint,
    load_resume_state,
    learning_rates_by_group,
    parse_args,
    rotate_checkpoints,
    save_checkpoint,
    schedule_steps_for_accelerate,
    speaker_key,
    speaker_statistics,
    sorted_checkpoints,
    steps_for_epochs,
    wandb_init_kwargs,
)
from qwen_tts.core.models import (
    Text2SemanticConfig,
    Text2SemanticForCausalLM,
)

BASE_LR = 4e-5  # the --lr default, so the LR traces read like the real run


class SavingTokenizer:
    def save_pretrained(self, path):
        with open(path / "tokenizer_config.json", "w", encoding="utf-8") as handle:
            json.dump({"test": True}, handle)


def test_parse_args_defaults_match_dataset_limits(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_jsonl",
            "train.jsonl",
            "--eval_jsonl",
            "eval.jsonl",
            "--w2v_bert_path",
            "w2v",
            "--stats_path",
            "stats.pt",
            "--num_epochs",
            "1",
        ],
    )
    args = parse_args()
    assert args.lr == 4e-5
    assert args.new_module_lr == 2e-4
    assert args.max_train_steps is None
    assert args.num_epochs == 1
    assert args.logging_steps == 10
    assert args.eval_steps == 1000
    assert args.checkpointing_steps == 50
    assert args.checkpointing_min_interval_minutes == 30.0
    assert args.checkpoint_total_limit == 2
    assert args.keep_checkpointing_steps == 5000
    assert args.seed == 42
    assert args.max_ref_seconds == 20.0
    assert args.max_target_seconds == 30.0
    assert args.min_speaker_records == 2
    assert args.punctuation_dropout_prob == 0.1
    assert args.punctuation_dropout_keep_word_spaces is True
    assert args.speaker_encoder_dtype == "bfloat16"


def _run_length_argv(*extra):
    return [
        "train.py",
        "--train_jsonl",
        "train.jsonl",
        "--eval_jsonl",
        "eval.jsonl",
        "--w2v_bert_path",
        "w2v",
        "--stats_path",
        "stats.pt",
        *extra,
    ]


def test_parse_args_requires_a_run_length(monkeypatch):
    # Neither flag used to mean "100k steps", which silently ignored a caller
    # who meant one epoch. Refusing is the point of the change.
    monkeypatch.setattr("sys.argv", _run_length_argv())
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_both_run_lengths(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        _run_length_argv("--num_epochs", "1", "--max_train_steps", "1000"),
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_steps_for_epochs_rounds_each_division_up():
    # 1000 batches over 3 ranks is 334 per rank (not 333), and 334 batches at
    # accum 4 is 84 steps (not 83): a partial batch and a partial accumulation
    # window both still cost a step, and rounding either down would end the
    # cosine decay before the epoch does.
    assert steps_for_epochs(1000, 3, 4, 1) == 84
    assert steps_for_epochs(1000, 3, 4, 2) == 168


def test_steps_for_epochs_is_exact_when_nothing_divides_short():
    assert steps_for_epochs(960, 8, 3, 1) == 40
    assert steps_for_epochs(960, 8, 1, 1) == 120


def test_steps_for_epochs_matches_the_t2s_v1_manifest():
    # The number this exists to produce: 99,374,464 rows at batch 32 on 16 GPUs
    # with no accumulation is 512 rows/step, so one epoch is 194,091 steps -- not
    # the 100,000 the old default would have scheduled.
    rows = 99_374_464
    batches = math.ceil(rows / 32)
    assert steps_for_epochs(batches, 16, 1, 1) == 194_091


def test_schedule_steps_for_accelerate_scales_by_processes():
    # The whole point: 194,091 optimizer steps on 16 processes must be built as
    # 3,105,456 scheduler steps, because AcceleratedScheduler advances the
    # wrapped scheduler once per process per optimizer step.
    assert schedule_steps_for_accelerate(194_091, 16, False) == 194_091 * 16
    assert schedule_steps_for_accelerate(100, 1, False) == 100
    # split_batches keeps the dataloader batch size, so one step per step.
    assert schedule_steps_for_accelerate(100, 32, True) == 100
    # --warmup_ratio 0 is legal and must stay 0, not become one process-worth.
    assert schedule_steps_for_accelerate(0, 32, False) == 0
    with pytest.raises(ValueError):
        schedule_steps_for_accelerate(-1, 8, False)
    with pytest.raises(ValueError):
        schedule_steps_for_accelerate(10, 0, False)


def _lr_trace(monkeypatch, num_processes, warmup_scheduler_steps,
              total_scheduler_steps, optimizer_steps, probe_steps,
              schedule="cosine"):
    """LR after each of `probe_steps` optimizer steps, through real accelerate.

    Drives `accelerate.scheduler.AcceleratedScheduler` itself rather than a
    hand-rolled imitation of it, so the test fails if accelerate ever changes
    how a prepared scheduler is advanced.
    """
    from accelerate.scheduler import AcceleratedScheduler

    monkeypatch.setattr(
        "accelerate.scheduler.AcceleratorState",
        lambda: SimpleNamespace(num_processes=num_processes),
    )
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=BASE_LR)
    if schedule == "constant":
        inner = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_scheduler_steps,
        )
    else:
        inner = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_scheduler_steps,
            num_training_steps=total_scheduler_steps,
        )
    wrapped = AcceleratedScheduler(
        inner,
        [SimpleNamespace(step_was_skipped=False)],
        step_with_optimizer=True,
        split_batches=False,
    )
    seen = {}
    for step in range(1, optimizer_steps + 1):
        optimizer.step()  # order the wrapper expects; no grads, so it is a no-op
        wrapped.step()
        if step in probe_steps:
            seen[step] = optimizer.param_groups[0]["lr"]
    return seen


def cosine_lr(step, warmup, total):
    """The LR the run is supposed to be at, straight from the cosine formula."""
    if step < warmup:
        return BASE_LR * step / warmup
    progress = (step - warmup) / (total - warmup)
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def test_prepared_cosine_spans_the_whole_run_on_many_processes(monkeypatch):
    # 4 nodes x 8 H100 = 32 processes, a 1,000-step run, 3% warmup.  With the
    # fix the LR peaks at the end of warmup and still has most of its range left
    # at the halfway point; the bug drove it to zero in the first 2% of the run.
    processes, steps = 32, 1_000
    warmup = int(steps * 0.03)
    trace = _lr_trace(
        monkeypatch,
        processes,
        schedule_steps_for_accelerate(warmup, processes, False),
        schedule_steps_for_accelerate(steps, processes, False),
        steps,
        probe_steps={warmup, steps // 2, steps},
    )
    assert trace[warmup] == pytest.approx(BASE_LR, rel=1e-6)
    assert trace[steps // 2] == pytest.approx(
        cosine_lr(steps // 2, warmup, steps), rel=1e-6
    )
    assert trace[steps // 2] > BASE_LR * 0.4  # still half the range to spend
    assert trace[steps] == pytest.approx(0.0, abs=1e-12)


def test_unscaled_schedule_is_the_bug_this_guards_against(monkeypatch):
    # Pin the defect itself: hand the wrapper the optimizer-step count and the
    # cosine is spent 1/32 of the way into the run.  This is what the 4x5090 run
    # did (LR at 12% of peak by step 5,880 of 30,000) and what 32 processes would
    # have turned into 97% of an epoch on a schedule that no longer means
    # anything -- past its end the clamped cosine oscillates between 0 and peak
    # rather than holding 0, so the tell is non-monotonicity, not smallness.
    processes, steps = 32, 1_000
    warmup = int(steps * 0.03)
    spent = steps // processes + 1
    trace = _lr_trace(
        monkeypatch, processes, warmup, steps, steps,
        probe_steps={spent, steps // 2},
    )
    # The run is 3.2% in and the schedule is already finished, where the
    # intended cosine is still within a whisker of peak.
    assert trace[spent] < cosine_lr(spent, warmup, steps) * 0.01
    # ...and then the LR climbs back up, which a cosine decay never does.
    assert trace[steps // 2] > trace[spent] * 100


def test_constant_schedule_warms_up_then_never_moves(monkeypatch):
    # --lr_schedule constant is what the t2s-v1 run uses: the run length is a
    # budget decision, so the LR of the steps already taken must not depend on
    # where the run happens to stop.  Warmup still needs the num_processes
    # scaling; everything after it is flat at peak.
    processes, steps = 32, 1_000
    warmup = int(steps * 0.03)
    trace = _lr_trace(
        monkeypatch,
        processes,
        schedule_steps_for_accelerate(warmup, processes, False),
        schedule_steps_for_accelerate(steps, processes, False),
        steps,
        probe_steps={warmup // 2, warmup, steps // 2, steps},
        schedule="constant",
    )
    assert trace[warmup // 2] == pytest.approx(BASE_LR * 0.5, rel=1e-3)
    assert trace[warmup] == pytest.approx(BASE_LR, rel=1e-6)
    assert trace[steps // 2] == pytest.approx(BASE_LR, rel=1e-12)
    assert trace[steps] == pytest.approx(BASE_LR, rel=1e-12)


def test_scaled_schedule_reproduces_the_4x5090_measurement(monkeypatch):
    # The unscaled schedule on 4 processes is what actually ran: at optimizer
    # step 5,880 of 30,000 W&B logged lr_backbone 4.6976e-6 against a 4e-5 peak,
    # i.e. 11.74% of peak.  Reproducing that number is what proves the diagnosis
    # rather than merely asserting the fix looks right.
    trace = _lr_trace(monkeypatch, 4, 900, 30_000, 5_880, probe_steps={5_880})
    assert trace[5_880] / BASE_LR == pytest.approx(4.6976e-6 / 4e-5, rel=5e-3)


def test_resume_restores_the_scheduler_position(tmp_path):
    # Trainers that fast-forward the scheduler by hand after a resume have to
    # apply the same num_processes factor, or a preempted run comes back at the
    # wrong LR without any error.  This trainer does not fast-forward: the
    # scheduler goes through `prepare`, so `save_state` records its position and
    # `load_state` restores it.  Pinned here because the day someone drops the
    # scheduler from `prepare`, resume starts silently relearning from LR 0.
    accelerator = Accelerator(project_dir=str(tmp_path))
    model = torch.nn.Linear(4, 4)
    optimizer = AdamW(model.parameters(), lr=BASE_LR)
    model, optimizer, scheduler = accelerator.prepare(
        model,
        optimizer,
        get_cosine_schedule_with_warmup(optimizer, 10, 100),
    )
    for _ in range(40):
        optimizer.step()
        scheduler.step()
    before = optimizer.param_groups[0]["lr"]
    assert before > 0
    accelerator.save_state(str(tmp_path / "state"))

    resumed = Accelerator(project_dir=str(tmp_path))
    fresh_model = torch.nn.Linear(4, 4)
    fresh_optimizer = AdamW(fresh_model.parameters(), lr=BASE_LR)
    fresh_model, fresh_optimizer, fresh_scheduler = resumed.prepare(
        fresh_model,
        fresh_optimizer,
        get_cosine_schedule_with_warmup(fresh_optimizer, 10, 100),
    )
    resumed.load_state(str(tmp_path / "state"))
    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(before, rel=1e-12)


def test_speaker_statistics_are_split_local():
    train_counts, train_paths = speaker_statistics(
        [
            {
                "speaker_id": "speaker-a",
                "language": "en",
                "audio_path": "train-en.wav",
            },
            {
                "speaker_id": "speaker-a",
                "language": "zh",
                "audio_path": "train-zh.wav",
            },
        ]
    )
    eval_counts, eval_paths = speaker_statistics(
        [{"speaker_id": "speaker-a", "language": "en", "audio_path": "eval.wav"}]
    )

    assert speaker_key({"speaker_id": "speaker-a", "language": "en"}) == (
        "en",
        "speaker-a",
    )
    assert train_counts == {("en", "speaker-a"): 1, ("zh", "speaker-a"): 1}
    assert train_paths == {
        ("en", "speaker-a"): ["train-en.wav"],
        ("zh", "speaker-a"): ["train-zh.wav"],
    }
    assert eval_counts == {("en", "speaker-a"): 1}
    assert eval_paths == {("en", "speaker-a"): ["eval.wav"]}


def tiny_model():
    qwen_config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        max_position_embeddings=128,
    )
    return Text2SemanticForCausalLM(
        Text2SemanticConfig(
            qwen_config=qwen_config.to_dict(),
            semantic_vocab_size=16,
            speech_bos_token_id=16,
            speech_eos_token_id=17,
            speech_pad_token_id=18,
            speaker_input_dim=8,
            speaker_conformer_output_size=8,
            speaker_conformer_linear_units=16,
            speaker_conformer_attention_heads=2,
            speaker_conformer_num_blocks=1,
            speaker_conformer_input_layer="linear",
            speaker_num_latents=2,
            speaker_latent_dim=32,
            speaker_perceiver_depth=1,
            speaker_perceiver_ff_mult=2,
        )
    )


def test_accelerator_checkpoint_restores_full_training_state(tmp_path):
    accelerator = Accelerator(cpu=True)
    model = tiny_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )
    original = next(model.parameters()).detach().clone()

    save_checkpoint(
        accelerator,
        model,
        SavingTokenizer(),
        tmp_path,
        epoch=2,
        step_in_epoch=7,
        global_step=19,
    )
    with torch.no_grad():
        next(model.parameters()).add_(1)

    state = load_resume_state(accelerator, tmp_path)
    assert state == (2, 7, 19)
    assert torch.equal(next(model.parameters()).detach(), original)
    assert (tmp_path / "accelerator_state").is_dir()
    assert (tmp_path / "trainer_state.json").is_file()


def test_optimizer_uses_lr_groups_and_no_decay_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", _run_length_argv("--num_epochs", "1"))
    args = parse_args()
    optimizer = build_optimizer(tiny_model(), args)
    group_names = {group["name"] for group in optimizer.param_groups}

    assert group_names == {
        "backbone_decay",
        "backbone_no_decay",
        "new_modules_decay",
        "new_modules_no_decay",
    }
    assert learning_rates_by_group(optimizer) == {
        "backbone": 4e-5,
        "new_modules": 2e-4,
    }
    assert any(group["weight_decay"] == 0 for group in optimizer.param_groups)
    assert any(group["weight_decay"] == 0.01 for group in optimizer.param_groups)


def test_rotate_checkpoints_keeps_latest_regular_steps(tmp_path):
    for step in (1000, 2000, 3000, 4000):
        (tmp_path / f"checkpoint-step-{step}").mkdir()
    (tmp_path / "checkpoint-keep-step-4000").mkdir()

    rotate_checkpoints(tmp_path, limit=2)

    assert [os.path.basename(path) for path in sorted_checkpoints(tmp_path)] == [
        "checkpoint-step-3000",
        "checkpoint-step-4000",
    ]
    assert (tmp_path / "checkpoint-keep-step-4000").is_dir()


def write_checkpoint_dir(root, name, *, finished=True):
    path = root / name
    path.mkdir()
    (path / "accelerator_state").mkdir()
    if finished:
        with open(path / "trainer_state.json", "w", encoding="utf-8") as handle:
            json.dump({"epoch": 0, "step_in_epoch": 1, "global_step": 1}, handle)
    return path


def test_latest_checkpoint_spans_both_prefixes(tmp_path):
    write_checkpoint_dir(tmp_path, "checkpoint-step-4950")
    write_checkpoint_dir(tmp_path, "checkpoint-keep-step-5000")

    assert latest_checkpoint(tmp_path) == str(
        tmp_path / "checkpoint-keep-step-5000"
    )


def test_latest_checkpoint_skips_a_half_written_directory(tmp_path):
    write_checkpoint_dir(tmp_path, "checkpoint-step-100")
    # Killed between save_state and trainer_state.json: resuming from this
    # would raise, so it must not be picked as "latest".
    write_checkpoint_dir(tmp_path, "checkpoint-step-150", finished=False)

    assert latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-step-100")


def test_latest_checkpoint_is_none_on_a_fresh_output_dir(tmp_path):
    assert latest_checkpoint(tmp_path / "missing") is None
    assert latest_checkpoint(tmp_path) is None


def test_decide_checkpoint_follows_the_policy():
    accelerator = Accelerator(cpu=True)
    clock = SimpleNamespace(now=0.0)
    policy = CheckpointPolicy(
        rolling_steps=50,
        persistent_steps=5000,
        min_interval_seconds=1800.0,
        clock=lambda: clock.now,
    )

    assert decide_checkpoint(accelerator, policy, 50) == (ACTION_NONE, False)
    clock.now = 1800.0
    assert decide_checkpoint(accelerator, policy, 1050) == (ACTION_ROLLING, False)
    assert decide_checkpoint(accelerator, policy, 5000) == (
        ACTION_PERSISTENT,
        False,
    )


def test_decide_checkpoint_turns_preemption_into_an_off_schedule_save():
    accelerator = Accelerator(cpu=True)
    policy = CheckpointPolicy(
        rolling_steps=50,
        persistent_steps=5000,
        min_interval_seconds=1800.0,
        clock=lambda: 0.0,
    )

    # Step 17 is neither a multiple of 50 nor past the interval; the signal
    # still has to produce a checkpoint, and the caller has to be told to stop.
    assert decide_checkpoint(accelerator, policy, 17, preempting=True) == (
        ACTION_ROLLING,
        True,
    )


def fake_remote(monkeypatch, *, latest=None):
    """Replace the gcloud calls with a recorder."""
    calls = []

    def upload(local_dir, remote_dir):
        calls.append(("upload", str(local_dir), remote_dir))

    def mark_complete(remote_dir):
        calls.append(("mark_complete", remote_dir))

    def download(remote_dir, local_dir):
        calls.append(("download", remote_dir, str(local_dir)))

    def latest_complete(prefix):
        calls.append(("latest_complete", prefix))
        return latest

    monkeypatch.setattr(checkpoint_remote, "upload", upload)
    monkeypatch.setattr(checkpoint_remote, "mark_complete", mark_complete)
    monkeypatch.setattr(checkpoint_remote, "download", download)
    monkeypatch.setattr(checkpoint_remote, "latest_complete", latest_complete)
    return calls


def test_mirror_checkpoint_marks_complete_after_uploading(monkeypatch):
    calls = fake_remote(monkeypatch)
    accelerator = Accelerator(cpu=True)

    remote_dir = mirror_checkpoint(
        accelerator, "/out/checkpoint-step-50", "gs://b/run"
    )

    assert remote_dir == "gs://b/run/checkpoint-step-50"
    # Order is the whole point: a marker written before the upload finishes
    # would advertise a partial checkpoint as resumable.
    assert calls == [
        ("upload", "/out/checkpoint-step-50", "gs://b/run/checkpoint-step-50"),
        ("mark_complete", "gs://b/run/checkpoint-step-50"),
    ]


def test_mirror_checkpoint_is_a_no_op_without_a_remote_dir(monkeypatch):
    calls = fake_remote(monkeypatch)
    accelerator = Accelerator(cpu=True)

    assert mirror_checkpoint(accelerator, "/out/checkpoint-step-50", None) is None
    assert calls == []


def test_fetch_remote_checkpoint_downloads_the_newest_complete_one(
    monkeypatch, tmp_path
):
    calls = fake_remote(monkeypatch, latest="gs://b/run/checkpoint-step-1000")
    accelerator = Accelerator(cpu=True)

    local = fetch_remote_checkpoint(accelerator, "gs://b/run", tmp_path)

    assert local == str(tmp_path / "checkpoint-step-1000")
    assert (tmp_path / "checkpoint-step-1000").is_dir()
    assert calls[-1] == (
        "download",
        "gs://b/run/checkpoint-step-1000",
        str(tmp_path / "checkpoint-step-1000"),
    )


def test_fetch_remote_checkpoint_ignores_local_checkpoints(monkeypatch, tmp_path):
    # This node survived the preemption and still has step 1100 on disk, but its
    # upload never completed. Resuming from it would leave the two nodes on
    # different weights, so the remote checkpoint wins.
    write_checkpoint_dir(tmp_path, "checkpoint-step-1100")
    fake_remote(monkeypatch, latest="gs://b/run/checkpoint-step-1000")
    accelerator = Accelerator(cpu=True)

    assert fetch_remote_checkpoint(accelerator, "gs://b/run", tmp_path) == str(
        tmp_path / "checkpoint-step-1000"
    )


def test_fetch_remote_checkpoint_starts_fresh_on_an_empty_prefix(
    monkeypatch, tmp_path
):
    calls = fake_remote(monkeypatch, latest=None)
    accelerator = Accelerator(cpu=True)

    assert fetch_remote_checkpoint(accelerator, "gs://b/run", tmp_path) is None
    assert calls == [("latest_complete", "gs://b/run")]


def test_preemption_flag_records_the_signal():
    flag = PreemptionFlag()
    assert flag.triggered is False
    flag._handle(signal.SIGTERM, None)
    assert flag.triggered is True
    assert flag.signal_number == signal.SIGTERM


def test_evaluation_reports_semantic_and_eos_metrics():
    accelerator = Accelerator(cpu=True)
    model = tiny_model()
    samples = [
        {
            "text_input_ids": torch.tensor([2, 3]),
            "text_attention_mask": torch.tensor([1, 1]),
            "speech_input_ids": torch.tensor([16, 4, 5]),
            "speech_attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([4, 5, 17]),
            "speaker_features": torch.randn(5, 8),
            "speaker_feature_lengths": torch.tensor(5),
        }
    ]
    dataloader = DataLoader(samples, batch_size=1)
    model, dataloader = accelerator.prepare(model, dataloader)
    metrics = evaluate(model, dataloader, accelerator)
    assert metrics["eval/loss"] > 0
    assert 0 <= metrics["eval/token_accuracy"] <= 1
    assert 0 <= metrics["eval/eos_accuracy"] <= 1


def test_add_speaker_features_replaces_audio_paths():
    class Extractor:
        def encode_files(self, paths, max_audio_seconds):
            assert paths == ["ref.wav"]
            assert max_audio_seconds == 12.0
            return torch.ones(1, 4, 8), torch.tensor([4])

    batch = {"speaker_audio_paths": ["ref.wav"]}
    result = add_speaker_features(batch, Extractor(), 12.0)
    assert "speaker_audio_paths" not in result
    assert result["speaker_features"].shape == (1, 4, 8)
    assert result["speaker_feature_lengths"].tolist() == [4]


def test_build_dataset_augments_train_text_but_not_eval_text():
    data = [
        {
            "audio": f"clip-{index}.wav",
            "text": "你好，世界！",
            "speaker_id": "speaker-a",
            "semantic_codes": [3, 4],
        }
        for index in range(2)
    ]
    args = SimpleNamespace(
        max_text_tokens=None,
        max_semantic_tokens=None,
        min_speaker_records=2,
        max_target_seconds=30.0,
        punctuation_dropout_keep_word_spaces=True,
        seed=42,
    )
    model_config = SimpleNamespace(
        semantic_vocab_size=8192,
        speech_bos_token_id=8192,
        speech_eos_token_id=8193,
        speech_pad_token_id=8194,
    )

    train_dataset = build_dataset(
        data,
        SimpleNamespace(),
        model_config,
        args,
        None,
        None,
        punctuation_dropout_prob=0.1,
    )
    eval_dataset = build_dataset(
        data, SimpleNamespace(), model_config, args, None, None
    )

    assert train_dataset.punctuation_dropout_prob == 0.1
    assert eval_dataset.punctuation_dropout_prob == 0.0


def test_wandb_init_kwargs_resumes_the_same_run_when_an_id_is_given():
    resumed = wandb_init_kwargs("t2s-v1-aws", "t2s-v1-aws-8node")
    assert resumed["id"] == "t2s-v1-aws-8node"
    # Without resume, a relaunch onto an existing id is an error rather than a
    # continuation, so the id alone would not survive a preemption.
    assert resumed["resume"] == "allow"
    assert resumed["name"] == "t2s-v1-aws"


def test_wandb_init_kwargs_without_an_id_lets_wandb_mint_one():
    fresh = wandb_init_kwargs("t2s-v1-aws", None)
    assert "id" not in fresh and "resume" not in fresh

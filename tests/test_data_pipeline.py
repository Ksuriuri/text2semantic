"""Tests for the GCS-to-manifest data pipeline. No network access required."""

from __future__ import annotations

import gzip
import json

import pytest

from data_pipeline import filters, gcs, lang_stats, pairing
from data_pipeline.filters import FilterConfig, FilterCounters, screen_row
from data_pipeline.scan import speaker_gate


def make_row(**overrides):
    """A metadata row in the exact shape the bucket stores (all values strings)."""
    row = {
        "id": "ds__utt0",
        "audio_path": "audio/ds-000000.tar/ds__utt0.flac",
        "text": "hello world this is a test",
        "speaker_id": "ds__spk0",
        "language": "en",
        "source": "x.wav",
        "duration": "5.0",
        "sample_rate": "44100",
    }
    row.update(overrides)
    return row


# -- text handling -------------------------------------------------------


def test_nullish_covers_exporter_sentinels():
    # The exporter writes the literal string "None" for missing values.
    assert filters.is_nullish("None")
    assert filters.is_nullish("")
    assert filters.is_nullish(None)
    assert not filters.is_nullish("0")
    assert filters.as_text("None") == ""


def test_normalise_absorbs_engine_style_differences():
    cohere = "Why are you beating up my jukebox?"
    granite = "why are you beating up my jukebox"
    assert filters.normalise_text(cohere) == filters.normalise_text(granite)
    assert filters.wer(cohere, granite) == 0.0
    assert filters.cer(cohere, granite) == 0.0


def test_wer_and_cer_values():
    assert filters.wer("a b c d", "a b c d") == 0.0
    assert filters.wer("a b c d", "a b c") == 0.25
    assert filters.cer("abcd", "abc") == 0.25
    # Empty reference is unscoreable and must be reported as maximal error.
    assert filters.wer("", "anything") == 1.0


def test_error_rate_uses_cer_for_cjk():
    metric, _ = filters.error_rate("你好世界", "你好世界", language="zh")
    assert metric == "cer"
    metric, _ = filters.error_rate("hello there", "hello there", language="en")
    assert metric == "wer"


# -- per-row gates -------------------------------------------------------


def screen(row, cohere=None, granite=None, config=None):
    counters = FilterCounters()
    record = screen_row(
        row, cohere, granite, config or FilterConfig(), counters, "ds", 0
    )
    return record, counters


def test_keeps_a_clean_row():
    record, counters = screen(make_row(), cohere="hello world this is a test")
    assert record is not None
    assert counters.kept == 1
    assert record.metric == "wer"
    assert record.error_rate == 0.0
    assert record.speaker_key == ("en", "ds__spk0")


def test_rejects_low_sample_rate():
    record, counters = screen(make_row(sample_rate="16000"), cohere="hello")
    assert record is None
    assert counters.bad_sample_rate == 1


@pytest.mark.parametrize("duration", ["2.9", "3.0", "30.0", "31.0"])
def test_rejects_out_of_range_duration(duration):
    record, counters = screen(make_row(duration=duration), cohere="hello world")
    assert record is None
    assert counters.bad_duration == 1


def test_accepts_duration_inside_the_open_interval():
    for duration in ("3.01", "29.99"):
        record, _ = screen(
            make_row(duration=duration), cohere="hello world this is a test"
        )
        assert record is not None, duration


def test_rejects_missing_speaker_id():
    # noiz-short and worldspeech store the literal "None" for every row.
    record, counters = screen(make_row(speaker_id="None"), cohere="hello world")
    assert record is None
    assert counters.no_speaker == 1


def test_rejects_when_cohere_is_absent():
    record, counters = screen(make_row(), cohere=None)
    assert record is None
    assert counters.no_asr_hypothesis == 1


def test_scores_cohere_against_original_text():
    row = make_row(text="the quick brown fox jumps")
    record, _ = screen(row, cohere="the quick brown fox jumped")
    assert record is not None and record.error_rate == pytest.approx(0.2)

    record, counters = screen(row, cohere="completely different words entirely here")
    assert record is None
    assert counters.asr_above_threshold == 1


def test_falls_back_to_granite_when_no_original_text():
    row = make_row(text="None")
    record, _ = screen(row, cohere="hello world again", granite="hello world again")
    assert record is not None
    assert record.error_rate == 0.0

    record, counters = screen(row, cohere="hello world again", granite="nothing alike ok")
    assert record is None
    assert counters.asr_above_threshold == 1


def test_drops_unscoreable_rows_by_default_and_keeps_them_when_asked():
    row = make_row(text="")
    record, counters = screen(row, cohere="hello world", granite=None)
    assert record is None
    assert counters.no_asr_reference == 1

    record, counters = screen(
        row, cohere="hello world", granite=None,
        config=FilterConfig(require_asr_score=False),
    )
    assert record is not None
    assert record.asr_scored is False
    assert counters.kept == 1


# -- speaker gate --------------------------------------------------------


def index_row(utt_id, speaker, duration, language="en"):
    return {
        "dataset": "ds", "shard": 0, "id": utt_id,
        "audio_path": f"audio/ds-000000.tar/{utt_id}.flac",
        "language": language, "speaker_id": speaker, "duration": duration,
        "sample_rate": 44100, "metric": "wer", "error_rate": 0.0,
        "asr_scored": True,
    }


def write_index(tmp_path, rows):
    path = tmp_path / "index" / "ds" / "ds-000000.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return tmp_path / "index"


def test_speaker_gate_requires_a_peer_and_a_long_utterance(tmp_path):
    rows = [
        # spk_ok: two utterances, one longer than 6s -> both kept.
        index_row("a1", "spk_ok", 4.0),
        index_row("a2", "spk_ok", 7.0),
        # spk_short: two utterances but none longer than 6s -> both dropped.
        index_row("b1", "spk_short", 4.0),
        index_row("b2", "spk_short", 5.0),
        # spk_alone: single utterance, no reference available -> dropped.
        index_row("c1", "spk_alone", 12.0),
    ]
    index_dir = write_index(tmp_path, rows)
    out = tmp_path / "filtered.jsonl"
    stats = speaker_gate(index_dir, FilterConfig(), out)

    kept_ids = [json.loads(l)["id"] for l in out.read_text("utf-8").splitlines()]
    assert sorted(kept_ids) == ["a1", "a2"]
    assert stats["rows_kept"] == 2
    assert stats["dropped_singleton"] == 1
    assert stats["dropped_no_long"] == 2
    assert stats["speakers_kept"] == 1
    assert stats["hours_kept"] == pytest.approx(11.0 / 3600.0, abs=1e-4)


def test_speaker_gate_keys_on_language_and_speaker(tmp_path):
    # The same speaker_id under two languages must not pool into one speaker,
    # matching Text2SemanticDataset._speaker_key.
    rows = [
        index_row("a1", "shared", 7.0, language="en"),
        index_row("b1", "shared", 7.0, language="ja"),
    ]
    index_dir = write_index(tmp_path, rows)
    out = tmp_path / "filtered.jsonl"
    stats = speaker_gate(index_dir, FilterConfig(), out)
    assert stats["rows_kept"] == 0
    assert stats["dropped_singleton"] == 2


def test_apply_speaker_gate_matches_the_streaming_gate():
    records = []
    counters = FilterCounters()
    for utt_id, speaker, duration in (
        ("a1", "s1", 4.0), ("a2", "s1", 7.0), ("b1", "s2", 4.0)
    ):
        record, _ = screen(
            make_row(id=utt_id, speaker_id=speaker, duration=str(duration)),
            cohere="hello world this is a test",
        )
        records.append(record)
    kept = filters.apply_speaker_gate(records, FilterConfig(), counters)
    assert [r.utt_id for r in kept] == ["a1", "a2"]
    assert counters.dropped_speaker_singleton == 1


# -- gcs helpers ---------------------------------------------------------


def test_tar_member_name():
    assert (
        gcs.tar_member_name("audio/ds-000123.tar/ds__utt0.flac") == "ds__utt0.flac"
    )
    assert gcs.tar_member_name("no-tar-here.flac") is None


def test_shard_index():
    assert gcs.shard_index("preprocessed/ds/metadata/ds-000042.jsonl") == 42
    assert gcs.shard_index("preprocessed/ds/metadata/ds-000042.jsonl.gz") == 42
    assert gcs.shard_index("preprocessed/ds/metadata/.keep") is None


# -- pairing constraints -------------------------------------------------


def test_window_index_only_parses_the_w_suffix():
    assert pairing.window_index("laion_emolia__DE_B00000_S00000_W000123") == 123
    # No _W at all: must be None so the caller falls back to the cosine rule.
    # Returning 0 here would silently reject every pair of these datasets.
    assert pairing.window_index("expresso__ex01_confused_00001") is None
    assert pairing.window_index("") is None
    assert pairing.window_index(None) is None


def test_adjacent_clips_are_rejected_but_distant_ones_kept():
    # The failure this exists to stop: neighbouring cuts of one utterance are
    # same-speaker by construction, so they train copying, not timbre transfer.
    assert pairing.pair_is_allowed("g_W000010", "g_W000012")[0] is False
    assert pairing.pair_is_allowed("g_W000010", "g_W000012")[1] == "adjacent_w"
    assert pairing.pair_is_allowed("g_W000010", "g_W000020")[0] is True
    assert pairing.pair_is_allowed("g_W000010", "g_W000010")[1] == "same_clip"


def test_missing_w_index_passes_through_for_the_cosine_fallback():
    allowed, reason = pairing.pair_is_allowed("vctk__p225_001", "vctk__p225_002")
    assert allowed is True
    assert reason == "no_w_index"


def test_reference_clips_are_spread_along_the_source():
    rows = [{"id": f"g_W{i:06d}", "duration": 10 - i * 0.1} for i in range(8)]
    picked = [r["id"] for r in pairing.spread_reference_clips(rows, 2)]
    assert picked == ["g_W000000", "g_W000004"]
    dist = pairing.w_distance(*picked)
    assert dist >= pairing.DEFAULT_MIN_W_DISTANCE


def test_spread_falls_back_rather_than_returning_too_few():
    # A fully consecutive 3-clip group cannot satisfy the spread; returning 1
    # reference would get the speaker silently dropped by _is_usable, so the
    # constraint yields instead and max_cosine is left to catch it.
    rows = [{"id": f"t_W{i:06d}", "duration": 5.0} for i in range(3)]
    assert len(pairing.spread_reference_clips(rows, 2)) == 2


def test_pair_budget_counts_what_the_constraint_costs():
    tight = [{"id": f"t_W{i:06d}", "duration": 5.0} for i in range(3)]
    assert pairing.group_pair_budget(tight)["pairs_allowed"] == 0
    spread = [{"id": f"s_W{i * 10:06d}", "duration": 5.0} for i in range(3)]
    assert pairing.group_pair_budget(spread)["pairs_allowed"] == 3
    summary = pairing.summarize({"tight": tight, "spread": spread})
    assert summary["speakers"] == 2
    assert summary["speakers_without_usable_pair"] == 1


def test_similarity_filter_drops_near_duplicates():
    pairs = [("a", "b"), ("c", "d")]
    kept, dropped = pairing.filter_pairs_by_similarity(
        pairs, lambda x, y: 0.995 if x == "a" else 0.80)
    assert kept == [("c", "d")]
    assert dropped == 1


def test_choose_reference_clips_returns_report_and_spreads():
    from data_pipeline.encode import choose_reference_clips

    grouped = {("laion_emolia", 0): [
        {"id": f"laion_emolia__EN_B0_S0_W{i:06d}", "duration": 10 - i * 0.1,
         "language": "en", "speaker_id": "laion_emolia__EN_B0_S0"}
        for i in range(8)
    ]}
    keep, report = choose_reference_clips(grouped, 2, min_w_distance=4)
    assert len(keep) == 2
    a, b = sorted(keep)
    assert pairing.w_distance(a, b) >= 4
    assert report["speakers"] == 1

    # min_w_distance=0 restores longest-first, which picks adjacent clips.
    keep0, report0 = choose_reference_clips(grouped, 2, min_w_distance=0)
    assert report0 is None
    assert pairing.w_distance(*sorted(keep0)) == 1


def test_is_consecutive_run_needs_w_index():
    run = [{"id": f"laion_emolia__EN_B0_S0_W{i:06d}"} for i in range(5)]
    assert pairing.is_consecutive_run(run)

    gapped = [{"id": "laion_emolia__EN_B0_S0_W000000"},
              {"id": "laion_emolia__EN_B0_S0_W000050"}]
    assert not pairing.is_consecutive_run(gapped)

    # No _W at all must not be mistaken for a consecutive run, otherwise
    # --drop-consecutive-groups would silently delete the 7 trusted datasets.
    assert not pairing.is_consecutive_run(
        [{"id": "vctk__p225_001"}, {"id": "vctk__p225_002"}])


def test_drop_consecutive_groups_drops_instead_of_backfilling():
    """A consecutive slice group loses its speaker rather than take a near-repeat.

    Backfilling is right when a sub-optimal reference beats losing the speaker.
    It inverts for laion's consecutive slice groups, where the backfill is a
    near-repeat of the target -- established by the adjacency structure (85.5% of
    slice groups are one unbroken run), not by a within-group-variance argument.
    """
    from data_pipeline.encode import choose_reference_clips

    # Four adjacent windows: no pair can satisfy min_w_distance=4.
    grouped = {("laion_emolia", 0): [
        {"id": f"laion_emolia__EN_B0_S0_W{i:06d}", "duration": 10 - i,
         "language": "en", "speaker_id": "laion_emolia__EN_B0_S0"}
        for i in range(4)
    ]}
    keep_bf, _ = choose_reference_clips(grouped, 2, min_w_distance=4)
    assert len(keep_bf) == 2, "default must still backfill"

    keep_drop, report = choose_reference_clips(
        grouped, 2, min_w_distance=4, drop_consecutive_groups=True)
    assert len(keep_drop) < 2
    assert report["speakers_dropped_consecutive"] == 1


# -- per-language subset stats -------------------------------------------


def write_subset_index(tmp_path, rows_by_dataset):
    for dataset, rows in rows_by_dataset.items():
        path = tmp_path / "index" / dataset / f"{dataset}-000000.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                row = dict(row, dataset=dataset)
                handle.write(json.dumps(row) + "\n")
    return tmp_path


def test_subset_gate_excludes_speakers_that_only_pass_across_datasets(tmp_path):
    """The gate must be re-run per subset, not reused from the corpus-wide run.

    ``shared`` has one clip in each dataset, so it passes the >=2-clip rule only
    when both datasets are in scope. Selecting one dataset must drop it.
    """
    rows = {
        "vctk": [index_row("a1", "shared", 7.0)],
        "ears": [index_row("b1", "shared", 8.0)],
    }
    root = write_subset_index(tmp_path, rows)

    both = lang_stats.index_shards(root, ["vctk", "ears"])
    keep, seen = lang_stats.subset_speaker_gate(both)
    assert seen == 1 and len(keep) == 1

    one = lang_stats.index_shards(root, ["vctk"])
    keep_one, _ = lang_stats.subset_speaker_gate(one)
    assert keep_one == set()


def test_language_rollup_keeps_speakers_separate_per_language(tmp_path):
    """Hours add up per language and the same speaker_id in two languages counts
    twice, matching the ``(language, speaker_id)`` key the trainer groups on."""
    rows = {
        "ds": [
            index_row("a1", "spk", 4.0, language="en"),
            index_row("a2", "spk", 8.0, language="en"),
            index_row("b1", "spk", 4.0, language="ja"),
            index_row("b2", "spk", 8.0, language="ja"),
        ]
    }
    root = write_subset_index(tmp_path, rows)
    paths = lang_stats.index_shards(root, ["ds"])
    keep, _ = lang_stats.subset_speaker_gate(paths)
    assert len(keep) == 2

    cells = lang_stats.collect(paths, keep, {"ds": 1000.0})
    by_language = lang_stats.roll_up(cells, axis=1)
    assert set(by_language) == {"en", "ja"}
    assert by_language["en"]["seconds"] == pytest.approx(12.0)
    total = lang_stats.totals(by_language)
    assert len(total["speakers"]) == 2
    # 50 Hz x 2 bytes = 100 bytes per audio second.
    assert total["codes_bytes"] == pytest.approx(2400.0)


def test_similarity_filter_is_high_side_only():
    """A low-cosine pair must pass: no min-cosine floor exists, by design.

    Genuine same-speaker pairs go as low as 0.62 (measured vctk/ears p10), which
    overlaps different-speaker pairs, so a floor would cut the most valuable
    high-diversity pairs. Purity is guaranteed upstream instead.
    """
    kept, dropped = pairing.filter_pairs_by_similarity(
        [("a", "b")], lambda x, y: 0.30)
    assert kept == [("a", "b")]
    assert dropped == 0

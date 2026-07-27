"""Same-speaker pairing constraints for laion_emolia.

Why this module exists
----------------------
A speaker reference must be *another* clip of the same speaker -- but "another
clip" is not enough.  laion_emolia's utterance ids end in ``_W<n>``, which
increases monotonically along the source recording, and measurement showed that
**85.5% of the ``B_S`` (slice) groups are one unbroken run of consecutive
``_W``**: their clips are adjacent cuts of a single continuous utterance.

Pairing a clip with its neighbour is same-speaker by construction, so it looks
perfect to any speaker-similarity check -- CAMPPlus-style encoders are trained to
be *invariant* to content and prosody, so they cannot tell "same person, varied
speech" from "same person, near-repeat".  But it teaches the model to **copy the
prompt** rather than to transfer timbre onto new content.  Adjacency is therefore
a constraint the encoder cannot supply; it has to come from the ids.

Two constraints, deliberately independent:

``min_w_distance``
    Structural, free, and applies wherever ids carry ``_W``.  Rejects pairs
    drawn from the same neighbourhood of the source.

``max_cosine``
    Semantic fallback for the rest: reject pairs so similar they are effectively
    the same recording.  Needs embeddings, so it is exposed as a filter for the
    caller that has them (S2mel's pairing) rather than run here.

Both apply to **both** laion halves.  The diarization half is not exempt: 26.2%
of its groups are also consecutive runs.

There is deliberately NO min-cosine floor
-----------------------------------------
The obvious-looking counterpart -- reject pairs whose cosine is too *low*, to
catch two different people sharing one speaker_id -- is not implemented, and must
not be added.  Genuine same-speaker pairs that differ in content and channel run
as low as **0.62** (the measured p10 of vctk/ears same-speaker pairs), which
overlaps the range different-speaker pairs occupy.  A floor would therefore cut
precisely the high-diversity pairs that are the most valuable training signal,
while still passing plenty of mismatched ones.

The consequence is worth stating plainly: **``max_cosine`` filters for diversity,
not for purity.** Under-clustering (two speakers under one id) produces *low*
cosines, so nothing here catches it. Group purity can only be guaranteed upstream,
by the intra-group mixing measurement -- there is no downstream safety net, so
that check cannot be relaxed on the assumption that a filter will catch it.
"""

from __future__ import annotations

import re
from collections import defaultdict

# Trailing window index, e.g. laion_emolia__DE_B00000_S00000_W000123 -> 123.
_W_RE = re.compile(r"_W(\d+)$")

# Default: clips must be at least this far apart in source-window index.  Small
# on purpose -- it only needs to break *adjacency*, and every unit costs
# recall in groups that are short runs.
DEFAULT_MIN_W_DISTANCE = 4

# Default: a pair above this cosine is treated as the same recording rather than
# the same speaker.  Calibrate against the measured same-speaker distribution
# (see docs/gcs-data-pipeline.md 6.1.4) -- do NOT invent a constant.
DEFAULT_MAX_COSINE = 0.98


def window_index(utterance_id):
    """Position of an utterance along its source recording, or None.

    None is normal, not an error: only laion ids carry ``_W``.  Callers must
    treat None as "no adjacency information" and fall back to ``max_cosine``,
    never as distance 0 (which would silently reject every pair) nor as infinite
    distance (which would silently accept every pair).
    """
    match = _W_RE.search(utterance_id or "")
    return int(match.group(1)) if match else None


def w_distance(id_a, id_b):
    """Source-window distance between two utterances, or None if unknown."""
    a, b = window_index(id_a), window_index(id_b)
    if a is None or b is None:
        return None
    return abs(a - b)


def pair_is_allowed(id_a, id_b, min_w_distance=DEFAULT_MIN_W_DISTANCE):
    """Is this same-speaker pair far enough apart to be worth training on?

    Unknown distance is allowed through -- the caller is expected to apply
    ``max_cosine`` to those.  Returns (allowed, reason) so rejections stay
    countable instead of vanishing.
    """
    if id_a == id_b:
        return False, "same_clip"
    dist = w_distance(id_a, id_b)
    if dist is None:
        return True, "no_w_index"
    if dist < min_w_distance:
        return False, "adjacent_w"
    return True, "ok"


def filter_pairs_by_similarity(pairs, cosine_of,
                              max_cosine=DEFAULT_MAX_COSINE):
    """Drop pairs that are near-duplicate recordings.

    ``cosine_of`` is a callable ``(id_a, id_b) -> float``, supplied by whoever
    owns the embeddings.  Kept separate from :func:`pair_is_allowed` so the
    structural constraint stays usable with no model loaded.

    High side only, by design -- see the module docstring on why no min-cosine
    floor exists.  This removes near-duplicate recordings; it does NOT and cannot
    remove pairs of two different speakers.
    """
    kept, dropped = [], 0
    for id_a, id_b in pairs:
        if cosine_of(id_a, id_b) > max_cosine:
            dropped += 1
        else:
            kept.append((id_a, id_b))
    return kept, dropped


def spread_reference_clips(rows, per_speaker,
                           min_w_distance=DEFAULT_MIN_W_DISTANCE,
                           backfill=True):
    """Choose one speaker's reference clips, spread along the source.

    Picking simply the longest clips (the obvious choice, and what this replaces)
    tends to pick neighbours inside a slice group, because a long stretch of
    speech is cut into several long adjacent windows.  Here the longest clip is
    still taken first -- it is the most informative reference -- and each further
    pick must sit ``min_w_distance`` away from every clip already chosen.

    ``backfill`` decides what happens when the constraint cannot be met, and the
    right answer depends on the group, which is why it is a parameter rather than
    a policy:

    True (default)
        Top up with plain longest-first, so a speaker never ends up with *fewer*
        references than requested and silently dropped by ``_is_usable``.  A
        sub-optimal reference beats losing the speaker -- when the speaker is
        worth keeping.

    False
        Return fewer (possibly zero) and let the speaker be dropped.  Correct
        when the backfilled pair would be low value: for laion's consecutive
        slice groups the backfill is a near-repeat of the target and teaches
        copying rather than timbre transfer (see the module docstring).  That
        rests on the adjacency structure -- 85.5% of slice groups are one
        unbroken run, over 4177 groups -- and deliberately NOT on a
        within-group-variance argument: the probe's cross-domain spread ratio has
        a 95% CI of roughly [0.2, 2.1] at smoke size, i.e. it cannot establish a
        direction either way, so it must not be cited here.
    """
    ordered = sorted(rows, key=lambda r: -r["duration"])
    chosen = []
    for row in ordered:
        if len(chosen) >= per_speaker:
            break
        if all(pair_is_allowed(row["id"], picked["id"], min_w_distance)[0]
               for picked in chosen):
            chosen.append(row)
    if backfill and len(chosen) < per_speaker:
        for row in ordered:
            if len(chosen) >= per_speaker:
                break
            if row not in chosen:
                chosen.append(row)
    return chosen


def is_consecutive_run(rows, max_gap=1):
    """Is this group one unbroken run of ``_W`` windows?

    85.5% of laion's slice groups are, and those are the ones where backfilling a
    reference yields a near-repeat.  Returns False when the ids carry no ``_W``,
    so non-laion datasets are never classified as consecutive by accident.
    """
    idx = sorted(i for i in (window_index(r["id"]) for r in rows)
                 if i is not None)
    if len(idx) < 2 or len(idx) != len(rows):
        return False
    return all(b - a <= max_gap for a, b in zip(idx, idx[1:]))


def group_pair_budget(rows, min_w_distance=DEFAULT_MIN_W_DISTANCE):
    """How many usable pairs a speaker group still has under the constraint.

    Diagnostic: a group that drops to 0 contributes nothing but its reference,
    which is what makes the slice half low-value rather than merely noisy.
    """
    ids = [r["id"] for r in rows]
    total = allowed = 0
    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1:]:
            total += 1
            if pair_is_allowed(id_a, id_b, min_w_distance)[0]:
                allowed += 1
    return {"pairs_total": total, "pairs_allowed": allowed,
            "min_w_distance": min_w_distance}


def summarize(rows_by_speaker, min_w_distance=DEFAULT_MIN_W_DISTANCE):
    """Aggregate pair budgets so the cost of the constraint is visible."""
    out = defaultdict(int)
    for rows in rows_by_speaker.values():
        budget = group_pair_budget(rows, min_w_distance)
        out["pairs_total"] += budget["pairs_total"]
        out["pairs_allowed"] += budget["pairs_allowed"]
        out["speakers"] += 1
        if budget["pairs_allowed"] == 0:
            out["speakers_without_usable_pair"] += 1
    result = dict(out)
    result["min_w_distance"] = min_w_distance
    if out["pairs_total"]:
        result["allowed_share"] = round(
            out["pairs_allowed"] / out["pairs_total"], 4)
    return result

"""GCS-to-text2semantic data preparation pipeline.

The pipeline turns the preprocessed corpora stored in
``gs://noiz-taiwan-audio-data/preprocessed/`` into the manifest format that
:class:`finetuning.dataset.Text2SemanticDataset` consumes.

Stages
------
1. :mod:`data_pipeline.gcs` - authenticated read-only access to the bucket.
2. :mod:`data_pipeline.filters` - the row-level and speaker-level quality gates.
3. :mod:`data_pipeline.scan` - stream every metadata/ASR shard and emit a
   compact per-utterance index plus corpus statistics.
4. :mod:`data_pipeline.encode` - materialise audio, run the MaskGCT semantic
   tokenizer and write the packed ``<u2`` code store plus the final manifest.
"""

from data_pipeline.filters import (
    FilterConfig,
    FilterCounters,
    RowRecord,
    cer,
    normalise_text,
    wer,
)

__all__ = [
    "FilterConfig",
    "FilterCounters",
    "RowRecord",
    "cer",
    "normalise_text",
    "wer",
]

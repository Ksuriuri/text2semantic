"""Read-only access helpers for the preprocessed audio bucket.

The bucket uses Uniform Bucket-Level Access, so every call authenticates with
the service-account key referenced by ``GOOGLE_APPLICATION_CREDENTIALS`` (or
``--gcs-key``).  Nothing in this module writes to the bucket.

Layout::

    gs://noiz-taiwan-audio-data/preprocessed/<dataset>/
        metadata/<dataset>-NNNNNN.jsonl        # one row per utterance
        audio/<dataset>-NNNNNN.tar             # flac payloads
        asr/<engine>/<dataset>-NNNNNN.jsonl    # transcription per utterance

``metadata`` and ``asr`` shards share the same ``NNNNNN`` numbering and the same
``id`` values, so a shard can be joined without a global index.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import tarfile
from dataclasses import dataclass

BUCKET = "noiz-taiwan-audio-data"
PROJECT = "noiz-430406"
ROOT = "preprocessed"

COHERE_ENGINE = "cohere-transcribe-03-2026"
GRANITE_ENGINE = "granite-speech-4.1-2b-nar"

#: Datasets published under ``preprocessed/``.
DATASETS = (
    "Genshin",
    "StarRail",
    "WutheringWaves",
    "ears",
    "expresso",
    "hi_fi_tts",
    "laion_emolia",
    "noiz-short",
    "vctk",
    "worldspeech",
)

_SHARD_RE = re.compile(r"-(\d+)\.jsonl(?:\.gz)?$")


def shard_index(blob_name):
    """Return the ``NNNNNN`` shard number encoded in a jsonl blob name."""
    match = _SHARD_RE.search(blob_name)
    if match is None:
        return None
    return int(match.group(1))


def make_client(gcs_key=None, project=PROJECT):
    """Build a ``storage.Client``, honouring an explicit key path."""
    from google.cloud import storage

    if gcs_key:
        return storage.Client.from_service_account_json(str(gcs_key), project=project)
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError(
            "No credentials: pass --gcs-key or export "
            "GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcs-key.json"
        )
    return storage.Client(project=project)


@dataclass(frozen=True)
class Shard:
    """One metadata shard plus the ASR shards that line up with it."""

    dataset: str
    index: int
    metadata_blob: str
    cohere_blob: str = None
    granite_blob: str = None
    metadata_bytes: int = 0


class Corpus:
    """Thin facade over the bucket that yields shards and decoded rows."""

    def __init__(self, client, bucket=BUCKET, root=ROOT):
        self.client = client
        self.bucket = client.bucket(bucket)
        self.root = root.strip("/")

    # -- listing ---------------------------------------------------------
    def _list(self, prefix):
        return list(self.client.list_blobs(self.bucket, prefix=prefix))

    def datasets(self):
        prefix = f"{self.root}/"
        iterator = self.client.list_blobs(self.bucket, prefix=prefix, delimiter="/")
        list(iterator)
        return sorted(p[len(prefix):].strip("/") for p in iterator.prefixes)

    def asr_engines(self, dataset):
        prefix = f"{self.root}/{dataset}/asr/"
        iterator = self.client.list_blobs(self.bucket, prefix=prefix, delimiter="/")
        list(iterator)
        return sorted(p[len(prefix):].strip("/") for p in iterator.prefixes)

    def shards(self, dataset):
        """Join metadata and ASR shards for ``dataset`` on the shard number."""
        meta = {}
        for blob in self._list(f"{self.root}/{dataset}/metadata/"):
            index = shard_index(blob.name)
            if index is None or not blob.size:
                continue  # skips .keep placeholders
            meta[index] = (blob.name, blob.size or 0)

        engines = set(self.asr_engines(dataset))
        asr = {}
        for engine in (COHERE_ENGINE, GRANITE_ENGINE):
            if engine not in engines:
                continue
            table = {}
            for blob in self._list(f"{self.root}/{dataset}/asr/{engine}/"):
                index = shard_index(blob.name)
                if index is None or not blob.size:
                    continue
                table[index] = blob.name
            asr[engine] = table

        shards = []
        for index in sorted(meta):
            name, size = meta[index]
            shards.append(
                Shard(
                    dataset=dataset,
                    index=index,
                    metadata_blob=name,
                    cohere_blob=asr.get(COHERE_ENGINE, {}).get(index),
                    granite_blob=asr.get(GRANITE_ENGINE, {}).get(index),
                    metadata_bytes=size,
                )
            )
        return shards

    # -- reading ---------------------------------------------------------
    def read_bytes(self, blob_name):
        return self.bucket.blob(blob_name).download_as_bytes()

    def read_jsonl(self, blob_name):
        """Yield decoded rows from a (optionally gzipped) jsonl blob."""
        if blob_name is None:
            return
        payload = self.read_bytes(blob_name)
        if blob_name.endswith(".gz"):
            payload = gzip.decompress(payload)
        for line in payload.splitlines():
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

    def audio_blob(self, dataset, shard_index_):
        return f"{self.root}/{dataset}/audio/{dataset}-{shard_index_:06d}.tar"

    def extract_audio(self, dataset, shard_index_, wanted, dest_dir):
        """Extract ``wanted`` member names from one audio tar into ``dest_dir``.

        ``wanted`` maps the tar member name to the output file name.  The tar is
        streamed once, so the caller should batch every utterance it needs from
        a shard into a single call.  Returns the map of member name to the path
        actually written.
        """
        import pathlib

        dest_dir = pathlib.Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        blob = self.bucket.blob(self.audio_blob(dataset, shard_index_))
        written = {}
        with blob.open("rb") as raw:
            with tarfile.open(fileobj=raw, mode="r|*") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    target = wanted.get(member.name)
                    if target is None:
                        target = wanted.get(member.name.lstrip("./"))
                    if target is None:
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    out = dest_dir / target
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with open(out, "wb") as sink:
                        sink.write(handle.read())
                    written[member.name] = out
                    if len(written) == len(wanted):
                        break
        return written


def tar_member_name(audio_path):
    """Split ``audio/<ds>-NNNNNN.tar/<member>`` into the member component.

    Metadata rows store the audio location as a pseudo-path that points inside
    the shard tar.  Everything after ``.tar/`` is the tar member name.
    """
    marker = ".tar/"
    position = audio_path.find(marker)
    if position < 0:
        return None
    return audio_path[position + len(marker):]


def read_text_stream(corpus, blob_name):
    """Return a text stream for a blob, for callers that want line iteration."""
    return io.StringIO(corpus.read_bytes(blob_name).decode("utf-8"))

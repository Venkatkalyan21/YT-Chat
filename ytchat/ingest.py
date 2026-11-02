"""Ingest pipeline: YouTube link -> transcript -> timestamped chunks -> embeddings -> Postgres."""

import logging
import time
from collections.abc import Callable

from . import youtube
from .chunking import chunk_segments
from .config import Settings
from .models import Embedder
from .store import Store

log = logging.getLogger(__name__)

TranscriptFetcher = Callable[[str, list[str], str], youtube.Transcript]
MetadataFetcher = Callable[[str], youtube.VideoMeta]


def index_version(embedder: Embedder, settings: Settings) -> str:
    """Identifies how a video was indexed; changing the embedder or chunking triggers a re-index."""
    return f"{embedder.name}|words={settings.chunk_words}|overlap={settings.chunk_overlap_words}"


def ingest_video(
    url: str,
    *,
    store: Store,
    embedder: Embedder,
    settings: Settings,
    fetch_transcript: TranscriptFetcher = youtube.fetch_transcript,
    fetch_metadata: MetadataFetcher = youtube.fetch_metadata,
    force: bool = False,
) -> dict:
    video_id = youtube.extract_video_id(url)
    version = index_version(embedder, settings)
    existing = store.get_video(video_id)
    if existing and existing["index_version"] == version and not force:
        return existing

    started = time.perf_counter()
    transcript = fetch_transcript(video_id, settings.languages, settings.yt_proxy_url)
    meta = fetch_metadata(video_id)
    chunks = chunk_segments(transcript.segments, settings.chunk_words, settings.chunk_overlap_words)
    vectors = embedder.embed_documents([c.text for c in chunks], title=meta.title)
    store.save_video(transcript, meta, chunks, vectors, version)
    log.info(
        "Indexed %s (%s): %d segments -> %d chunks in %.1fs",
        video_id,
        meta.title,
        len(transcript.segments),
        len(chunks),
        time.perf_counter() - started,
    )
    return store.get_video(video_id)

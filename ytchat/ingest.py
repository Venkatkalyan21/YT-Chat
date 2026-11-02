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


class IngestRejected(RuntimeError):
    """A video this deployment won't index. The message is safe to show; `status` is the HTTP code."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


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
    before_index: Callable[[], None] | None = None,
) -> dict:
    """Return the stored video, indexing it first if needed.

    `before_index` runs only when real (paid) work is about to start, so callers can apply
    rate limits without counting cheap cache hits.
    """
    video_id = youtube.extract_video_id(url)
    version = index_version(embedder, settings)
    existing = store.get_video(video_id)
    # With new videos disabled, an existing index is served even if its settings are outdated.
    if existing and not force and (existing["index_version"] == version or not settings.allow_new_videos):
        return existing
    if not settings.allow_new_videos:
        raise IngestRejected(
            "This demo answers questions about its pre-loaded videos only. Pick one from the list, "
            "or run YtChat locally to add your own.",
            status=403,
        )
    if before_index is not None:
        before_index()

    started = time.perf_counter()
    transcript = fetch_transcript(video_id, settings.languages, settings.yt_proxy_url)
    if settings.max_video_minutes and transcript.duration > settings.max_video_minutes * 60:
        raise IngestRejected(
            f"This video is {transcript.duration / 60:.0f} minutes long; the limit here is {settings.max_video_minutes} minutes.",
            status=422,
        )
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

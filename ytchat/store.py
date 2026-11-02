"""Data access for videos, chunks, conversations and messages."""

import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from pgvector import Vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .chunking import Chunk
from .timestamps import fmt_ts
from .youtube import Segment, Transcript, VideoMeta

_VIDEO_COLUMNS = """
    v.id, v.title, v.channel, v.language, v.is_generated, v.duration_s, v.index_version,
    v.summary IS NOT NULL AS has_summary, v.created_at,
    (SELECT count(*) FROM chunks c WHERE c.video_id = v.id) AS num_chunks
"""


@dataclass(frozen=True)
class RetrievedChunk:
    idx: int
    start: float
    end: float
    text: str
    score: float  # cosine similarity, higher is closer

    @property
    def label(self) -> str:
        return fmt_ts(self.start)

    def to_dict(self) -> dict:
        return {**asdict(self), "label": self.label}


class Store:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    # --- videos -----------------------------------------------------------------------------

    def get_video(self, video_id: str) -> dict | None:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT {_VIDEO_COLUMNS} FROM videos v WHERE v.id = %s", (video_id,))
            return _jsonable(cur.fetchone())

    def list_videos(self, limit: int = 20) -> list[dict]:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT {_VIDEO_COLUMNS} FROM videos v ORDER BY v.created_at DESC LIMIT %s", (limit,))
            return [_jsonable(row) for row in cur.fetchall()]

    def save_video(
        self,
        transcript: Transcript,
        meta: VideoMeta,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        index_version: str,
    ) -> None:
        """Insert or re-index a video. Re-indexing replaces its chunks and clears the cached summary."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        segments = [asdict(seg) for seg in transcript.segments]
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                """
                INSERT INTO videos (id, title, channel, language, is_generated, duration_s, segments, index_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title, channel = EXCLUDED.channel, language = EXCLUDED.language,
                    is_generated = EXCLUDED.is_generated, duration_s = EXCLUDED.duration_s,
                    segments = EXCLUDED.segments, index_version = EXCLUDED.index_version,
                    summary = NULL, summary_model = NULL
                """,
                (
                    transcript.video_id,
                    meta.title,
                    meta.channel,
                    transcript.language_code,
                    transcript.is_generated,
                    transcript.duration,
                    Jsonb(segments),
                    index_version,
                ),
            )
            conn.execute("DELETE FROM chunks WHERE video_id = %s", (transcript.video_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (video_id, idx, start_s, end_s, text, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (transcript.video_id, c.idx, c.start, c.end, c.text, Vector(list(vec)))
                        for c, vec in zip(chunks, vectors)
                    ],
                )

    def get_segments(self, video_id: str) -> list[Segment]:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT segments FROM videos WHERE id = %s", (video_id,)).fetchone()
        return [Segment(**seg) for seg in row[0]] if row else []

    def get_summary(self, video_id: str) -> str | None:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT summary FROM videos WHERE id = %s", (video_id,)).fetchone()
        return row[0] if row else None

    def save_summary(self, video_id: str, summary: str, model: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("UPDATE videos SET summary = %s, summary_model = %s WHERE id = %s", (summary, model, video_id))

    # --- retrieval --------------------------------------------------------------------------

    def search(self, video_id: str, query_vector: Sequence[float], k: int) -> list[RetrievedChunk]:
        """Exact cosine search within one video.

        A single video has tens to a few hundred chunks, so a full scan is fast and has
        perfect recall. An approximate (HNSW) index only pays off when searching across
        many videos, which is Phase 2.
        """
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT idx, start_s, end_s, text, 1 - (embedding <=> %(q)s) AS score
                FROM chunks WHERE video_id = %(video_id)s
                ORDER BY embedding <=> %(q)s, idx
                LIMIT %(k)s
                """,
                {"q": Vector(list(query_vector)), "video_id": video_id, "k": k},
            )
            return [
                RetrievedChunk(idx=r["idx"], start=r["start_s"], end=r["end_s"], text=r["text"], score=float(r["score"]))
                for r in cur.fetchall()
            ]

    # --- conversations ----------------------------------------------------------------------

    def create_conversation(self, video_id: str) -> str:
        with self.pool.connection() as conn:
            row = conn.execute("INSERT INTO conversations (video_id) VALUES (%s) RETURNING id", (video_id,)).fetchone()
        return str(row[0])

    def get_conversation(self, conversation_id: str) -> dict | None:
        conv_uuid = _as_uuid(conversation_id)
        if conv_uuid is None:
            return None
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT id, video_id, created_at FROM conversations WHERE id = %s", (conv_uuid,))
            return _jsonable(cur.fetchone())

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list | None = None,
        sources: list | None = None,
        meta: dict | None = None,
    ) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, citations, sources, meta)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (
                    conversation_id,
                    role,
                    content,
                    Jsonb(citations) if citations is not None else None,
                    Jsonb(sources) if sources is not None else None,
                    Jsonb(meta) if meta is not None else None,
                ),
            ).fetchone()
        return row[0]

    def get_messages(self, conversation_id: str, last: int | None = None) -> list[dict]:
        """Messages in chronological order; `last` keeps only the most recent N."""
        conv_uuid = _as_uuid(conversation_id)
        if conv_uuid is None:
            return []
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM (
                    SELECT id, role, content, citations, sources, meta, created_at FROM messages
                    WHERE conversation_id = %s ORDER BY id DESC LIMIT %s
                ) recent ORDER BY id
                """,
                (conv_uuid, last),
            )
            return [_jsonable(row) for row in cur.fetchall()]


def _as_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None


def _jsonable(row: dict | None) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    for key, value in out.items():
        if hasattr(value, "isoformat"):
            out[key] = value.isoformat()
        elif key == "id" and not isinstance(value, (int, str)):
            out[key] = str(value)  # UUID
    return out

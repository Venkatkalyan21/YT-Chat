"""Postgres connection pool and schema. Falls back to a bundled local Postgres (pgserver) for development."""

import logging
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import Settings

log = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id            TEXT PRIMARY KEY,               -- YouTube video id
    title         TEXT NOT NULL,
    channel       TEXT NOT NULL DEFAULT '',
    language      TEXT,
    is_generated  BOOLEAN,                        -- auto-generated captions are noisier
    duration_s    REAL NOT NULL,
    segments      JSONB NOT NULL,                 -- raw caption lines, kept for re-chunking experiments
    index_version TEXT NOT NULL,                  -- embedder + chunking params used to build the chunks
    summary       TEXT,
    summary_model TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    video_id  TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx       INT NOT NULL,
    start_s   REAL NOT NULL,
    end_s     REAL NOT NULL,
    text      TEXT NOT NULL,
    embedding vector({dim}) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,  -- for hybrid search (Phase 3)
    PRIMARY KEY (video_id, idx)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS conversations (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id   TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    citations       JSONB,
    sources         JSONB,
    meta            JSONB,                        -- latency, token usage, model, rewritten query
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conversation ON messages (conversation_id, id);
"""


class Database:
    """Owns the connection pool (and the bundled local server, when one is used)."""

    def __init__(self, url: str, dim: int, local_server=None):
        self.url = url
        self.dim = dim
        self._local_server = local_server
        _prepare(url, dim)
        self.pool = ConnectionPool(url, min_size=1, max_size=10, configure=register_vector, open=True)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Database":
        if settings.database_url:
            return cls(settings.database_url, settings.embed_dim)
        try:
            import pgserver
        except ImportError as exc:
            raise ConfigError(
                "No DATABASE_URL set. Either point it at Postgres with pgvector, "
                "or install the bundled local database with: pip install -e \".[local]\""
            ) from exc
        data_dir = Path(settings.local_db_dir).resolve()
        log.info("Starting bundled Postgres + pgvector in %s", data_dir)
        server = pgserver.get_server(data_dir, cleanup_mode="stop")
        return cls(server.get_uri(), settings.embed_dim, local_server=server)

    def close(self) -> None:
        self.pool.close()
        if self._local_server is not None:
            self._local_server.cleanup()


def _prepare(url: str, dim: int) -> None:
    """Create the pgvector extension and tables. The extension must exist before vectors can be registered."""
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        existing = conn.execute(
            """
            SELECT a.atttypmod FROM pg_attribute a
            WHERE a.attrelid = to_regclass('chunks') AND a.attname = 'embedding'
            """
        ).fetchone()
        if existing and existing[0] != dim:
            raise ConfigError(
                f"The database stores {existing[0]}-dim embeddings but EMBED_DIM is {dim}. "
                "Use a fresh database or set EMBED_DIM back."
            )
        conn.execute(SCHEMA.format(dim=dim))

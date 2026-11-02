import os

import pytest

from ytchat.db import Database
from ytchat.store import Store

DIM = 768


@pytest.fixture(scope="session")
def pg_url(tmp_path_factory):
    """A Postgres with pgvector: TEST_DATABASE_URL if set, else a throwaway bundled server."""
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        yield url
        return
    pgserver = pytest.importorskip("pgserver", reason="set TEST_DATABASE_URL or install the `local` extra")
    server = pgserver.get_server(tmp_path_factory.mktemp("pgdata"), cleanup_mode="delete")
    yield server.get_uri()
    server.cleanup()


@pytest.fixture(scope="session")
def database(pg_url):
    db = Database(pg_url, DIM)
    yield db
    db.pool.close()


@pytest.fixture
def store(database):
    with database.pool.connection() as conn:
        conn.execute("TRUNCATE videos, chunks, conversations, messages RESTART IDENTITY CASCADE")
    return Store(database.pool)

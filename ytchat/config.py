"""App settings, read from environment variables or a .env file."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database. Leave empty to auto-start a bundled local Postgres + pgvector (needs the `local` extra).
    database_url: str = ""
    local_db_dir: str = ".pgdata"

    # Chat model. Pin an exact model id so eval results stay reproducible.
    llm_provider: str = "google"  # google | anthropic
    llm_model: str = "gemini-3.6-flash"
    llm_temperature: float | None = None  # None = provider default
    llm_timeout_s: float = 60.0  # per request; without one, a slow model leaves the UI waiting forever
    embed_timeout_s: float = 30.0

    # Embeddings. `hash` is an offline, keyless embedder for tests and demos.
    embed_provider: str = "google"  # google | hash
    embed_model: str = "gemini-embedding-2"
    embed_dim: int = 768  # pgvector HNSW indexes cap at 2000 dims; 768 is a Google-recommended size

    google_api_key: str = Field(default="", validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"))
    anthropic_api_key: str = ""

    # Chunking and retrieval. These are the knobs the Phase 3 experiments sweep.
    chunk_words: int = 150
    chunk_overlap_words: int = 30
    top_k: int = 6
    history_turns: int = 6  # previous user/assistant pairs sent with each question

    # Transcripts
    transcript_languages: str = "en,hi,te"  # preferred caption languages, in order
    yt_proxy_url: str = ""  # YouTube often blocks transcript requests from cloud IPs; route through a proxy there

    # Guards for a public deployment. A 0 limit disables that limit.
    allow_new_videos: bool = True  # false: serve only videos already in the database (index them with `ytchat --index`)
    max_video_minutes: int = 120  # longer videos cost more to embed and summarize
    rate_limit_per_minute: int = 10  # costly requests (indexing, summaries, answers) per visitor
    rate_limit_per_day: int = 100
    global_daily_limit: int = 1000  # all visitors combined; protects the API quota
    trust_proxy_headers: bool = False  # set true behind a reverse proxy (Render, Cloud Run) so limits see real visitor IPs

    @property
    def languages(self) -> list[str]:
        return [code.strip() for code in self.transcript_languages.split(",") if code.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

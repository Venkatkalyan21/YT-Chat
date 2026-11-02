"""Pluggable chat models and embedders, so evals can compare providers."""

import hashlib
import logging
import math
import re
import time
from collections.abc import Sequence
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from .config import Settings
from .db import ConfigError

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str], title: str | None = None) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    """Gemini embeddings via the google-genai SDK.

    Two details from Google's docs that a generic wrapper gets wrong:
    - gemini-embedding-2 merges a list of inputs into ONE embedding, so each text is sent
      as its own Content to get one vector per chunk.
    - gemini-embedding-2 takes the retrieval task as a text prefix (no task_type), while
      gemini-embedding-001 takes task_type and needs manual normalization below 3072 dims.
    """

    BATCH_SIZE = 100
    MAX_ATTEMPTS = 5

    def __init__(self, model: str, dim: int, api_key: str):
        from google import genai

        self.model = model
        self.dim = dim
        self.name = f"google/{model}@{dim}"
        self._legacy = model.startswith("gemini-embedding-001")
        self._client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: Sequence[str], title: str | None = None) -> list[list[float]]:
        if not self._legacy:
            texts = [f"title: {title or 'none'} | text: {t}" for t in texts]
        return self._embed(list(texts), task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        if not self._legacy:
            text = f"task: question answering | query: {text}"
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start : start + self.BATCH_SIZE]
            contents = [types.Content(parts=[types.Part(text=t)]) for t in batch]
            config = types.EmbedContentConfig(
                output_dimensionality=self.dim,
                task_type=task_type if self._legacy else None,
            )
            response = self._call_with_retry(contents, config)
            batch_vectors = [list(e.values) for e in response.embeddings or []]
            if len(batch_vectors) != len(batch):
                raise RuntimeError(f"Expected {len(batch)} embeddings, got {len(batch_vectors)}")
            if self._legacy:
                batch_vectors = [_normalize(v) for v in batch_vectors]
            vectors.extend(batch_vectors)
        return vectors

    def _call_with_retry(self, contents, config):
        from google.genai import errors

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                return self._client.models.embed_content(model=self.model, contents=contents, config=config)
            except errors.APIError as exc:
                retryable = exc.code in (429, 500, 503)
                if not retryable or attempt == self.MAX_ATTEMPTS:
                    raise
                wait = 2**attempt
                log.warning("Embedding call failed with %s, retrying in %ss", exc.code, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")


class HashEmbedder:
    """Keyless bag-of-words embedder: deterministic, offline, good enough for tests and wiring checks."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.name = f"hash@{dim}"

    def embed_documents(self, texts: Sequence[str], title: str | None = None) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        if not any(vec):
            vec[0] = 1.0
        return _normalize(vec)


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def get_embedder(settings: Settings) -> Embedder:
    if settings.embed_provider == "hash":
        return HashEmbedder(settings.embed_dim)
    if settings.embed_provider == "google":
        if not settings.google_api_key:
            raise ConfigError("GOOGLE_API_KEY is not set. Add it to .env (get one at https://aistudio.google.com/apikey).")
        return GeminiEmbedder(settings.embed_model, settings.embed_dim, settings.google_api_key)
    raise ConfigError(f"Unknown EMBED_PROVIDER: {settings.embed_provider}")


def get_chat_model(settings: Settings) -> BaseChatModel:
    extra = {"temperature": settings.llm_temperature} if settings.llm_temperature is not None else {}
    if settings.llm_provider == "google":
        if not settings.google_api_key:
            raise ConfigError("GOOGLE_API_KEY is not set. Add it to .env (get one at https://aistudio.google.com/apikey).")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model, google_api_key=settings.google_api_key, max_retries=3, **extra
        )
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is not set.")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ConfigError('Install the Anthropic extra: pip install -e ".[anthropic]"') from exc
        return ChatAnthropic(model=settings.llm_model, api_key=settings.anthropic_api_key, max_retries=3, **extra)
    raise ConfigError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def model_label(settings: Settings) -> str:
    return f"{settings.llm_provider}/{settings.llm_model}"

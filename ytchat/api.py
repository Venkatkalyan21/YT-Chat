"""FastAPI app: index videos, stream summaries and cited answers over server-sent events."""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from . import prompts, youtube
from .config import Settings, get_settings
from .db import Database
from .graph import build_qa_graph, history_from_rows
from .ingest import MetadataFetcher, TranscriptFetcher, ingest_video
from .models import Embedder, get_chat_model, get_embedder, model_label
from .store import Store
from .summary import stream_summary

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class IngestRequest(BaseModel):
    url: str = Field(min_length=1, max_length=500)
    force: bool = False


class ChatRequest(BaseModel):
    video_id: str = Field(min_length=1, max_length=20)
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


@dataclass
class AppContext:
    settings: Settings
    store: Store
    llm: BaseChatModel
    embedder: Embedder
    model_name: str
    fetch_transcript: TranscriptFetcher
    fetch_metadata: MetadataFetcher

    def __post_init__(self):
        self.graph = build_qa_graph(self.llm, self.embedder, self.store, self.settings.top_k)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _causes(exc: BaseException) -> Iterator[BaseException]:
    """The exception and everything it wraps; SDKs often wrap the real error."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def friendly_error(exc: Exception) -> str:
    """Turn provider errors into something a user can act on."""
    chain = list(_causes(exc))
    text = " | ".join(str(e) for e in chain)
    lowered = text.lower()
    if any(isinstance(e, TimeoutError) or "timeout" in type(e).__name__.lower() for e in chain) or "timed out" in lowered:
        return "The model took too long to respond. Try again, or set a faster LLM_MODEL in .env."
    if "api key not valid" in lowered or "api_key_invalid" in lowered or "permission_denied" in lowered:
        return "The model provider rejected the API key. Check GOOGLE_API_KEY in .env."
    if "429" in text or "resource_exhausted" in lowered or "quota" in lowered or "rate limit" in lowered:
        return "Hit the model's rate limit or quota. Wait a minute and try again."
    if "503" in text or "high demand" in lowered or "overloaded" in lowered:
        return "The model is overloaded right now. Try again in a minute, or set a different LLM_MODEL in .env."
    if "404" in text and "model" in lowered:
        return "The configured model wasn't found. Check LLM_MODEL in .env."
    return f"Something went wrong while generating ({type(exc).__name__}). See the server log for details."


def create_app(
    *,
    settings: Settings | None = None,
    store: Store | None = None,
    llm: BaseChatModel | None = None,
    embedder: Embedder | None = None,
    fetch_transcript: TranscriptFetcher | None = None,
    fetch_metadata: MetadataFetcher | None = None,
) -> FastAPI:
    """Build the app. Every dependency can be injected, which is how the tests run without API keys."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = None
        app_store = store
        if app_store is None:
            db = Database.from_settings(settings)
            app_store = Store(db.pool)
        app.state.ctx = AppContext(
            settings=settings,
            store=app_store,
            llm=llm or get_chat_model(settings),
            embedder=embedder or get_embedder(settings),
            model_name=model_label(settings) if llm is None else type(llm).__name__,
            fetch_transcript=fetch_transcript or youtube.fetch_transcript,
            fetch_metadata=fetch_metadata or youtube.fetch_metadata,
        )
        try:
            yield
        finally:
            if db is not None:
                db.close()

    app = FastAPI(title="YtChat", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def ctx(request: Request) -> AppContext:
        return request.app.state.ctx

    def require_video(c: AppContext, video_id: str) -> dict:
        video = c.store.get_video(video_id)
        if video is None:
            raise HTTPException(404, "Video not found. Load it first.")
        return video

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health(request: Request) -> dict:
        c = ctx(request)
        return {
            "status": "ok",
            "model": c.model_name,
            "embedder": c.embedder.name,
            "prompt_version": prompts.PROMPT_VERSION,
        }

    @app.post("/api/videos")
    def add_video(body: IngestRequest, request: Request) -> dict:
        c = ctx(request)
        try:
            return ingest_video(
                body.url,
                store=c.store,
                embedder=c.embedder,
                settings=c.settings,
                fetch_transcript=c.fetch_transcript,
                fetch_metadata=c.fetch_metadata,
                force=body.force,
            )
        except youtube.InvalidVideoUrl as exc:
            raise HTTPException(400, str(exc)) from exc
        except youtube.TranscriptUnavailable as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            log.exception("Indexing failed for %s", body.url)
            raise HTTPException(502, f"Indexing failed. {friendly_error(exc)}") from exc

    @app.get("/api/videos")
    def list_videos(request: Request) -> list[dict]:
        return ctx(request).store.list_videos()

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str, request: Request) -> dict:
        return require_video(ctx(request), video_id)

    @app.post("/api/videos/{video_id}/summary")
    def summarize(video_id: str, request: Request, refresh: bool = False) -> StreamingResponse:
        c = ctx(request)
        video = require_video(c, video_id)
        cached = None if refresh else c.store.get_summary(video_id)

        def events() -> Iterator[str]:
            if cached:
                yield sse("token", {"text": cached})
                yield sse("done", {"summary": cached, "cached": True})
                return
            started = time.perf_counter()
            parts: list[str] = []
            try:
                segments = c.store.get_segments(video_id)
                for text in stream_summary(c.llm, video["title"], video["channel"], segments):
                    parts.append(text)
                    yield sse("token", {"text": text})
                summary = "".join(parts).strip()
                c.store.save_summary(video_id, summary, c.model_name)
                latency_ms = round((time.perf_counter() - started) * 1000)
                yield sse("done", {"summary": summary, "cached": False, "latency_ms": latency_ms})
            except Exception as exc:
                log.exception("Summary failed for %s", video_id)
                yield sse("error", {"message": friendly_error(exc)})

        return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        c = ctx(request)
        video = require_video(c, body.video_id)
        if body.conversation_id:
            conversation = c.store.get_conversation(body.conversation_id)
            if conversation is None or conversation["video_id"] != body.video_id:
                raise HTTPException(404, "Conversation not found for this video.")
            conversation_id = conversation["id"]
        else:
            conversation_id = c.store.create_conversation(body.video_id)

        history_rows = c.store.get_messages(conversation_id, last=c.settings.history_turns * 2)
        c.store.add_message(conversation_id, "user", body.message)

        def events() -> Iterator[str]:
            yield sse("meta", {"conversation_id": conversation_id})
            started = time.perf_counter()
            final: dict = {}
            inputs = {
                "video": {"id": video["id"], "title": video["title"], "channel": video["channel"]},
                "question": body.message,
                "history": history_from_rows(history_rows),
            }
            try:
                for mode, payload in c.graph.stream(inputs, stream_mode=["messages", "updates"]):
                    if mode == "messages":
                        chunk, metadata = payload
                        # Only stream the answer; the condense step's rewrite stays internal.
                        if metadata.get("langgraph_node") == "generate" and (text := chunk.text):
                            yield sse("token", {"text": text})
                        continue
                    for node, update in payload.items():
                        final.update(update or {})
                        if node == "retrieve":
                            yield sse(
                                "sources",
                                {
                                    "search_query": update["search_query"],
                                    "sources": [s.to_dict() for s in update["sources"]],
                                },
                            )

                sources = [s.to_dict() for s in final.get("sources", [])]
                meta = {
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "model": c.model_name,
                    "embedder": c.embedder.name,
                    "prompt_version": prompts.PROMPT_VERSION,
                    "search_query": final.get("search_query"),
                    "usage": final.get("usage") or {},
                }
                message_id = c.store.add_message(
                    conversation_id,
                    "assistant",
                    final["answer"],
                    citations=final["citations"],
                    sources=sources,
                    meta=meta,
                )
                yield sse(
                    "done",
                    {"message_id": message_id, "answer": final["answer"], "citations": final["citations"], "meta": meta},
                )
            except Exception as exc:
                log.exception("Chat failed for video %s", body.video_id)
                yield sse("error", {"message": friendly_error(exc)})

        return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str, request: Request) -> dict:
        c = ctx(request)
        conversation = c.store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(404, "Conversation not found.")
        return {**conversation, "messages": c.store.get_messages(conversation_id)}

    return app

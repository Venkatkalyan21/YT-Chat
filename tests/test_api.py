"""End-to-end API tests: real Postgres + pgvector, scripted fake model, offline embedder, fake transcript."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ytchat.api import create_app
from ytchat.config import Settings
from ytchat.models import HashEmbedder
from ytchat.youtube import Segment, Transcript, TranscriptUnavailable, VideoMeta

pytestmark = pytest.mark.db

VIDEO_ID = "catvideo001"
NO_CAPTIONS_ID = "nocaptions1"

LINES = [
    "welcome to this video about cats and how they spend their day",
    "cats sleep between twelve and sixteen hours every single day",
    "kittens and older cats sleep even more than adult cats",
    "when awake cats spend a lot of time grooming their fur",
    "grooming keeps the coat clean and helps control body temperature",
    "cats hunt at dawn and dusk because their prey is active then",
    "their eyes have a reflective layer that helps them see in low light",
    "domestic cats still keep the hunting instinct even when well fed",
    "playing with toys is a safe way to let indoor cats practise hunting",
    "thanks for watching and give your cat a gentle scratch from me",
] * 3  # 30 lines, 10 seconds apart -> 5 minutes


def fake_transcript(video_id, languages, proxy_url):
    if video_id == NO_CAPTIONS_ID:
        raise TranscriptUnavailable("Captions are turned off for this video, so there's no transcript to read.")
    segments = [Segment(text=line, start=i * 10.0, duration=10.0) for i, line in enumerate(LINES)]
    return Transcript(video_id=video_id, language_code="en", language="English", is_generated=False, segments=segments)


def fake_metadata(video_id):
    return VideoMeta(title="How cats spend their day", channel="Cat Facts")


def scripted_llm(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


def make_client(store, llm, **overrides) -> TestClient:
    base = dict(embed_provider="hash", chunk_words=40, chunk_overlap_words=10, top_k=3)
    settings = Settings(_env_file=None, **{**base, **overrides})
    app = create_app(
        settings=settings,
        store=store,
        llm=llm,
        embedder=HashEmbedder(settings.embed_dim),
        fetch_transcript=fake_transcript,
        fetch_metadata=fake_metadata,
    )
    return TestClient(app)


def read_events(response) -> list[tuple[str, dict]]:
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def tokens(events) -> str:
    return "".join(data["text"] for name, data in events if name == "token")


def test_index_video_and_errors(store):
    with make_client(store, scripted_llm()) as client:
        resp = client.post("/api/videos", json={"url": f"https://youtu.be/{VIDEO_ID}"})
        assert resp.status_code == 200
        video = resp.json()
        assert video["id"] == VIDEO_ID
        assert video["title"] == "How cats spend their day"
        assert video["num_chunks"] > 3
        assert video["index_version"] == "hash@768|words=40|overlap=10"

        again = client.post("/api/videos", json={"url": VIDEO_ID}).json()
        assert again["created_at"] == video["created_at"]  # cached, not re-indexed

        assert client.post("/api/videos", json={"url": "https://example.com"}).status_code == 400
        blocked = client.post("/api/videos", json={"url": NO_CAPTIONS_ID})
        assert blocked.status_code == 422
        assert "Captions are turned off" in blocked.json()["detail"]
        assert client.get("/api/videos/unknown0000").status_code == 404
        assert [v["id"] for v in client.get("/api/videos").json()] == [VIDEO_ID]


def test_summary_streams_then_is_cached(store):
    summary = "## TL;DR\nA video about cat habits [0:10].\n\n## Key points\n- Cats sleep a lot [0:10]"
    with make_client(store, scripted_llm(summary)) as client:
        client.post("/api/videos", json={"url": VIDEO_ID})

        first = read_events(client.post(f"/api/videos/{VIDEO_ID}/summary"))
        assert tokens(first) == summary
        assert first[-1] == ("done", {**first[-1][1], "summary": summary, "cached": False})

        second = read_events(client.post(f"/api/videos/{VIDEO_ID}/summary"))  # the fake has no replies left
        assert second[-1][1]["cached"] is True
        assert second[-1][1]["summary"] == summary
        assert client.get(f"/api/videos/{VIDEO_ID}").json()["has_summary"] is True


def test_chat_streams_cited_answer_and_handles_follow_ups(store):
    answer = "Cats sleep twelve to sixteen hours a day [0:10]. Some say even more [9:59]."
    rewrite = "How much do kittens sleep compared to adult cats?"
    follow_up_answer = "Kittens sleep even more than adult cats [0:20]."
    with make_client(store, scripted_llm(answer, rewrite, follow_up_answer)) as client:
        client.post("/api/videos", json={"url": VIDEO_ID})

        events = read_events(client.post("/api/chat", json={"video_id": VIDEO_ID, "message": "How long do cats sleep?"}))
        names = [name for name, _ in events]
        assert names[0] == "meta" and names[1] == "sources" and names[-1] == "done"
        conversation_id = events[0][1]["conversation_id"]

        sources = events[1][1]["sources"]
        assert len(sources) == 3
        assert any("sixteen hours" in s["text"] for s in sources)  # retrieval found the right passage

        done = events[-1][1]
        assert tokens(events) == answer == done["answer"]
        citations = {c["label"]: c["valid"] for c in done["citations"]}
        assert citations == {"0:10": True, "9:59": False}  # 9:59 is past the end of the video
        assert done["meta"]["search_query"] == "How long do cats sleep?"
        assert done["meta"]["latency_ms"] >= 0

        follow = read_events(
            client.post(
                "/api/chat",
                json={"video_id": VIDEO_ID, "conversation_id": conversation_id, "message": "And kittens?"},
            )
        )
        assert follow[0][1]["conversation_id"] == conversation_id
        assert follow[1][1]["search_query"] == rewrite  # the follow-up was condensed into a standalone query
        assert tokens(follow) == follow_up_answer  # the internal rewrite is not streamed to the user

        stored = client.get(f"/api/conversations/{conversation_id}").json()
        assert [m["role"] for m in stored["messages"]] == ["user", "assistant", "user", "assistant"]
        assert stored["messages"][1]["citations"][0]["label"] == "0:10"
        assert stored["messages"][3]["meta"]["search_query"] == rewrite


def test_chat_rejects_unknown_video_and_foreign_conversation(store):
    with make_client(store, scripted_llm("ok [0:00]")) as client:
        assert client.post("/api/chat", json={"video_id": "unknown0000", "message": "hi"}).status_code == 404
        client.post("/api/videos", json={"url": VIDEO_ID})
        bad = client.post(
            "/api/chat",
            json={"video_id": VIDEO_ID, "conversation_id": "00000000-0000-0000-0000-000000000000", "message": "hi"},
        )
        assert bad.status_code == 404
        assert client.get("/api/conversations/not-a-uuid").status_code == 404


def _wrapped_timeout() -> Exception:
    """A timeout wrapped the way SDKs wrap it, so only the cause chain reveals it."""
    try:
        try:
            raise httpx.ReadTimeout("The read operation timed out")
        except httpx.ReadTimeout as inner:
            raise RuntimeError("Error calling model") from inner
    except RuntimeError as outer:
        return outer


@pytest.mark.parametrize(
    "failure,expected",
    [
        (RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"), "rate limit"),
        (RuntimeError("400 API key not valid. Please pass a valid API key."), "rejected the API key"),
        (RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand."), "overloaded"),
        (_wrapped_timeout(), "took too long"),
    ],
)
def test_model_errors_become_readable_sse_errors(store, failure, expected):
    class Broken(GenericFakeChatModel):
        def _generate(self, *args, **kwargs):
            raise failure

        def _stream(self, *args, **kwargs):
            raise failure

    with make_client(store, Broken(messages=iter([]))) as client:
        client.post("/api/videos", json={"url": VIDEO_ID})
        events = read_events(client.post("/api/chat", json={"video_id": VIDEO_ID, "message": "hi"}))
        assert events[-1][0] == "error"
        assert expected in events[-1][1]["message"]


# ----------------------------------------------------------------------------- deployment guards

NO_LIMITS = dict(rate_limit_per_minute=0, rate_limit_per_day=0, global_daily_limit=0)


def chat(client, message="How long do cats sleep?", **kwargs):
    return client.post("/api/chat", json={"video_id": VIDEO_ID, "message": message}, **kwargs)


def test_demo_mode_serves_only_preloaded_videos(store):
    with make_client(store, scripted_llm()) as client:
        client.post("/api/videos", json={"url": VIDEO_ID})

    # Different chunking than the stored index, to show demo mode never re-indexes.
    with make_client(store, scripted_llm(), allow_new_videos=False, chunk_words=60) as client:
        assert client.get("/api/health").json()["allow_new_videos"] is False
        preloaded = client.post("/api/videos", json={"url": VIDEO_ID})
        assert preloaded.status_code == 200
        assert preloaded.json()["index_version"].endswith("words=40|overlap=10")
        new = client.post("/api/videos", json={"url": "newvideo001"})
        assert new.status_code == 403
        assert "pre-loaded videos" in new.json()["detail"]
        assert client.post("/api/videos", json={"url": VIDEO_ID, "force": True}).status_code == 403


def test_long_videos_are_refused(store):
    with make_client(store, scripted_llm(), max_video_minutes=2) as client:  # the fake video is 5 minutes
        resp = client.post("/api/videos", json={"url": VIDEO_ID})
        assert resp.status_code == 422
        assert "5 minutes long" in resp.json()["detail"]
        assert client.get("/api/videos").json() == []


def test_rate_limits_count_only_costly_requests(store):
    limits = dict(NO_LIMITS, rate_limit_per_minute=3)
    with make_client(store, scripted_llm("## TL;DR\nCats [0:10]", "Twelve hours [0:10]."), **limits) as client:
        assert client.post("/api/videos", json={"url": VIDEO_ID}).status_code == 200  # 1: indexing
        for _ in range(3):
            assert client.post("/api/videos", json={"url": VIDEO_ID}).status_code == 200  # cached: free
        read_events(client.post(f"/api/videos/{VIDEO_ID}/summary"))  # 2: new summary
        read_events(client.post(f"/api/videos/{VIDEO_ID}/summary"))  # cached: free
        read_events(chat(client))  # 3: answer
        limited = chat(client)
        assert limited.status_code == 429
        assert "wait" in limited.json()["detail"]
        assert client.post(f"/api/videos/{VIDEO_ID}/summary").status_code == 200  # cached summary still served


def test_visitor_ip_comes_from_proxy_header_only_when_trusted(store):
    limits = dict(NO_LIMITS, rate_limit_per_minute=1)
    with make_client(store, scripted_llm("a [0:10]", "b [0:10]"), trust_proxy_headers=True, **limits) as client:
        client.post("/api/videos", json={"url": VIDEO_ID})  # counted for the test client's own address
        assert chat(client, headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}).status_code == 200
        assert chat(client, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429
        assert chat(client, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200

    with make_client(store, scripted_llm("c [0:10]"), **limits) as client:  # untrusted: header ignored
        assert chat(client, headers={"X-Forwarded-For": "3.3.3.3"}).status_code == 200
        assert chat(client, headers={"X-Forwarded-For": "4.4.4.4"}).status_code == 429

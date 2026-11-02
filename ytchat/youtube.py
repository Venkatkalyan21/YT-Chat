"""Parse YouTube links and fetch transcripts (with timestamps) and video metadata."""

import html
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import httpx
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.proxies import GenericProxyConfig

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_PATH_PREFIXES = {"embed", "shorts", "live", "v", "e"}
_YOUTUBE_HOSTS = {"youtube.com", "youtube-nocookie.com"}


class InvalidVideoUrl(ValueError):
    pass


class TranscriptUnavailable(RuntimeError):
    """Raised with a message that is safe to show to the user."""


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    duration: float


@dataclass
class Transcript:
    video_id: str
    language_code: str
    language: str
    is_generated: bool
    segments: list[Segment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if not self.segments:
            return 0.0
        last = self.segments[-1]
        return last.start + last.duration


@dataclass(frozen=True)
class VideoMeta:
    title: str
    channel: str


def extract_video_id(url_or_id: str) -> str:
    """Return the 11-character video id from a YouTube link (watch, youtu.be, shorts, embed, live) or a bare id."""
    text = url_or_id.strip()
    if _VIDEO_ID.fullmatch(text):
        return text

    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = (parsed.hostname or "").lower()
    for prefix in ("www.", "m.", "music."):
        host = host.removeprefix(prefix)

    candidate = ""
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in _YOUTUBE_HOSTS:
        query = parse_qs(parsed.query)
        if "v" in query:
            candidate = query["v"][0]
        else:
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
                candidate = parts[1]

    if _VIDEO_ID.fullmatch(candidate):
        return candidate
    raise InvalidVideoUrl("That doesn't look like a YouTube video link.")


def _clean(text: str) -> str:
    return " ".join(html.unescape(text).split())


def fetch_transcript(video_id: str, languages: list[str], proxy_url: str = "") -> Transcript:
    """Fetch captions, preferring manual over auto-generated ones in the given language order."""
    proxy = GenericProxyConfig(http_url=proxy_url, https_url=proxy_url) if proxy_url else None
    api = YouTubeTranscriptApi(proxy_config=proxy)
    try:
        available = api.list(video_id)
        try:
            chosen = available.find_transcript(languages)
        except NoTranscriptFound:
            chosen = next(iter(available))  # fall back to any language the video has
        fetched = chosen.fetch()
    except StopIteration as exc:
        raise TranscriptUnavailable("This video has no captions, so there's no transcript to read.") from exc
    except TranscriptsDisabled as exc:
        raise TranscriptUnavailable("Captions are turned off for this video, so there's no transcript to read.") from exc
    except VideoUnavailable as exc:
        raise TranscriptUnavailable("This video is unavailable (private, deleted or region-locked).") from exc
    except (RequestBlocked, IpBlocked) as exc:
        raise TranscriptUnavailable(
            "YouTube blocked the transcript request from this network. Set YT_PROXY_URL to route through a proxy."
        ) from exc
    except CouldNotRetrieveTranscript as exc:
        raise TranscriptUnavailable("Couldn't get a transcript for this video.") from exc

    segments = [
        Segment(text=text, start=float(snip.start), duration=float(snip.duration))
        for snip in fetched.snippets
        if (text := _clean(snip.text))
    ]
    if not segments:
        raise TranscriptUnavailable("This video's transcript is empty.")
    return Transcript(
        video_id=video_id,
        language_code=fetched.language_code,
        language=fetched.language,
        is_generated=fetched.is_generated,
        segments=segments,
    )


def fetch_metadata(video_id: str) -> VideoMeta:
    """Title and channel via YouTube's public oEmbed endpoint (no API key needed)."""
    try:
        resp = httpx.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return VideoMeta(title=data.get("title") or video_id, channel=data.get("author_name") or "")
    except (httpx.HTTPError, ValueError):
        return VideoMeta(title=video_id, channel="")

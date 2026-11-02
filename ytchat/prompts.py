"""Prompt templates. Kept in one place so prompt changes are easy to version and evaluate."""

from collections.abc import Sequence

from .chunking import Chunk
from .store import RetrievedChunk

PROMPT_VERSION = "v1"

SUMMARY_SYSTEM = """You summarize YouTube videos from their transcript.
The transcript is split into passages; each starts with its timestamp in square brackets, like [12:34].

Write the summary in English using exactly this Markdown structure:

## TL;DR
Two or three sentences on what the video is about and its main conclusion.

## Key points
- Five to eight bullets with the most important ideas, each ending with the timestamp where it is discussed, like [12:34].

## Chapters
- [12:34] Short chapter title: one line on what this part covers.

Rules:
- Use only information from the transcript. Do not add outside facts.
- Copy timestamps exactly as they appear in the passage headers.
- Captions may contain speech-recognition errors; interpret them sensibly and do not quote obvious errors."""

QA_SYSTEM = """You answer questions about one YouTube video, "{title}"{channel_part}, using only the transcript passages below.
Each passage starts with its timestamp in square brackets.

Rules:
- Lead with the direct answer, then supporting detail. Use bullets for lists. Keep it concise.
- Ground every claim in the passages. End each sentence or bullet that uses a passage with that passage's timestamp, copied exactly, like [12:34]. Cite several passages as separate brackets: [1:05] [3:40].
- If the passages do not contain the answer, say you couldn't find that in this video, and do not guess or use outside knowledge. Do not add a citation in that case.
- Answer in the same language the user writes in.
- Captions may contain speech-recognition errors; interpret them sensibly.

Transcript passages:
{passages}"""

CONDENSE_SYSTEM = """You rewrite a follow-up question from a conversation about a YouTube video into a standalone search query.
Resolve pronouns and references using the conversation (for example "it", "that part", "the second one").
Keep the user's language. Return only the rewritten question, nothing else."""


def format_passages(passages: Sequence[Chunk | RetrievedChunk]) -> str:
    ordered = sorted(passages, key=lambda p: p.start)
    return "\n\n".join(f"[{p.label}]\n{p.text}" for p in ordered)


def qa_system_prompt(title: str, channel: str, passages: Sequence[RetrievedChunk]) -> str:
    channel_part = f" by {channel}" if channel else ""
    return QA_SYSTEM.format(title=title, channel_part=channel_part, passages=format_passages(passages))


def summary_user_prompt(title: str, channel: str, windows: Sequence[Chunk]) -> str:
    by = f" by {channel}" if channel else ""
    return f'Summarize the video "{title}"{by}.\n\nTranscript:\n{format_passages(windows)}'

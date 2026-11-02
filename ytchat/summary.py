"""Whole-video summary. The full transcript fits in context, so no retrieval is needed here."""

from collections.abc import Iterator, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from . import prompts
from .chunking import time_windows
from .youtube import Segment


def stream_summary(llm: BaseChatModel, title: str, channel: str, segments: Sequence[Segment]) -> Iterator[str]:
    windows = time_windows(segments, window_s=60.0)
    messages = [
        SystemMessage(prompts.SUMMARY_SYSTEM),
        HumanMessage(prompts.summary_user_prompt(title, channel, windows)),
    ]
    for chunk in llm.stream(messages):
        if text := chunk.text:
            yield text

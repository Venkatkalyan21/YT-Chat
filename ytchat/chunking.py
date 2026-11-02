"""Split a transcript into overlapping chunks that keep their start/end times."""

from collections.abc import Sequence
from dataclasses import dataclass

from .timestamps import fmt_ts
from .youtube import Segment


@dataclass(frozen=True)
class Chunk:
    idx: int
    start: float
    end: float
    text: str

    @property
    def label(self) -> str:
        return fmt_ts(self.start)


def _words(segment: Segment) -> int:
    return len(segment.text.split())


def chunk_segments(segments: Sequence[Segment], target_words: int = 150, overlap_words: int = 30) -> list[Chunk]:
    """Group caption segments into ~target_words chunks.

    Chunks never split a caption segment, so every chunk starts exactly at a caption's
    timestamp. Consecutive chunks share about `overlap_words` words so an answer that
    straddles a boundary is still retrievable from one chunk.
    """
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    if overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("overlap_words must be >= 0 and smaller than target_words")

    chunks: list[Chunk] = []
    i, n = 0, len(segments)
    while i < n:
        j, words = i, 0
        while j < n and (words < target_words or j == i):
            words += _words(segments[j])
            j += 1
        group = segments[i:j]
        end = max(seg.start + seg.duration for seg in group)
        chunks.append(
            Chunk(
                idx=len(chunks),
                start=group[0].start,
                end=max(end, group[0].start),
                text=" ".join(seg.text for seg in group),
            )
        )
        if j >= n:
            break
        # Step back from j to start the next chunk with ~overlap_words of shared context,
        # but always move forward by at least one segment.
        k, back = j, 0
        while k - 1 > i and back < overlap_words:
            k -= 1
            back += _words(segments[k])
        i = max(k, i + 1)
    return chunks


def time_windows(segments: Sequence[Segment], window_s: float = 60.0) -> list[Chunk]:
    """Non-overlapping fixed-time windows, used to show the whole transcript to the summarizer."""
    windows: list[Chunk] = []
    current: list[Segment] = []
    for seg in segments:
        if current and seg.start - current[0].start >= window_s:
            windows.append(_window(len(windows), current))
            current = []
        current.append(seg)
    if current:
        windows.append(_window(len(windows), current))
    return windows


def _window(idx: int, group: list[Segment]) -> Chunk:
    end = max(seg.start + seg.duration for seg in group)
    return Chunk(idx=idx, start=group[0].start, end=end, text=" ".join(seg.text for seg in group))

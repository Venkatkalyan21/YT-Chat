"""Find [mm:ss] citations in an answer and check each one against the passages it was given."""

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .store import RetrievedChunk
from .timestamps import parse_ts

_BRACKET = re.compile(r"\[([0-9:,;\s]+)\]")
_TIMESTAMP = re.compile(r"(?:\d{1,2}:)?\d{1,2}:\d{2}")


@dataclass(frozen=True)
class Citation:
    label: str
    seconds: int
    valid: bool  # True when the time falls inside one of the passages the model was shown
    chunk_idx: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def find_timestamps(text: str) -> list[str]:
    """Timestamp labels in bracket citations, in order, without duplicates. Handles [1:05] and [1:05, 3:40]."""
    seen: dict[str, None] = {}
    for bracket in _BRACKET.finditer(text):
        for part in re.split(r"[,;]", bracket.group(1)):
            label = part.strip()
            if _TIMESTAMP.fullmatch(label):
                seen.setdefault(label, None)
    return list(seen)


def check_citations(answer: str, sources: Sequence[RetrievedChunk], tolerance_s: float = 2.0) -> list[Citation]:
    citations = []
    for label in find_timestamps(answer):
        seconds = parse_ts(label)
        match = next((s for s in sources if s.start - tolerance_s <= seconds <= s.end + tolerance_s), None)
        citations.append(Citation(label, seconds, match is not None, match.idx if match else None))
    return citations

from ytchat.citations import check_citations, find_timestamps
from ytchat.store import RetrievedChunk

SOURCES = [
    RetrievedChunk(idx=0, start=0.0, end=40.0, text="intro", score=0.9),
    RetrievedChunk(idx=3, start=120.0, end=175.5, text="middle", score=0.8),
    RetrievedChunk(idx=9, start=3700.0, end=3760.0, text="late", score=0.7),
]


def test_finds_single_grouped_and_hour_timestamps_in_order_without_duplicates():
    text = "First [0:05]. Then [2:00, 2:30; 2:00] and later [1:01:40]. Again [0:05]."
    assert find_timestamps(text) == ["0:05", "2:00", "2:30", "1:01:40"]


def test_ignores_non_timestamp_brackets():
    assert find_timestamps("See [1] and [note] and [12] and plain 3:45 outside brackets.") == []


def test_citations_are_checked_against_retrieved_passages():
    answer = "Intro [0:10]. Middle [2:55]. Hallucinated [10:00]. Edge [2:57]. Late [1:01:45]."
    result = {c.label: c for c in check_citations(answer, SOURCES)}
    assert result["0:10"].valid and result["0:10"].chunk_idx == 0
    assert result["2:55"].valid and result["2:55"].chunk_idx == 3
    assert result["2:57"].valid  # within the 2 s tolerance after the passage ends at 175.5 s
    assert result["1:01:45"].valid and result["1:01:45"].chunk_idx == 9
    assert not result["10:00"].valid and result["10:00"].chunk_idx is None


def test_answer_without_citations():
    assert check_citations("I couldn't find that in this video.", SOURCES) == []

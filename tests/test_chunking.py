import pytest

from ytchat.chunking import chunk_segments, time_windows
from ytchat.timestamps import fmt_ts, parse_ts
from ytchat.youtube import Segment


def make_segments(n: int, words_each: int = 5, step: float = 3.0) -> list[Segment]:
    return [
        Segment(text=" ".join(f"w{i}_{k}" for k in range(words_each)), start=i * step, duration=step)
        for i in range(n)
    ]


def test_every_segment_lands_in_a_chunk_and_order_is_kept():
    segments = make_segments(200)
    chunks = chunk_segments(segments, target_words=50, overlap_words=10)
    covered = {word for c in chunks for word in c.text.split()}
    assert covered == {word for s in segments for word in s.text.split()}
    assert [c.idx for c in chunks] == list(range(len(chunks)))
    assert all(a.start < b.start for a, b in zip(chunks, chunks[1:]))


def test_chunks_start_on_caption_boundaries_with_correct_times():
    segments = make_segments(100)
    starts = {s.start for s in segments}
    for chunk in chunk_segments(segments, target_words=40, overlap_words=10):
        assert chunk.start in starts
        assert chunk.end > chunk.start


def test_chunk_size_and_overlap():
    segments = make_segments(200, words_each=5)
    chunks = chunk_segments(segments, target_words=50, overlap_words=10)
    for chunk in chunks[:-1]:
        assert 50 <= len(chunk.text.split()) < 55  # stops at the first segment that reaches the target
    for a, b in zip(chunks, chunks[1:]):
        shared = set(a.text.split()) & set(b.text.split())
        assert len(shared) == 10


def test_no_overlap_gives_disjoint_chunks():
    chunks = chunk_segments(make_segments(60), target_words=20, overlap_words=0)
    seen: set[str] = set()
    for chunk in chunks:
        words = set(chunk.text.split())
        assert not words & seen
        seen |= words


def test_long_single_segment_still_progresses():
    segments = [Segment(text="word " * 500, start=0, duration=60), *make_segments(5)]
    chunks = chunk_segments(segments, target_words=50, overlap_words=10)
    assert len(chunks) >= 2


def test_empty_and_invalid_input():
    assert chunk_segments([], 50, 10) == []
    with pytest.raises(ValueError):
        chunk_segments(make_segments(3), target_words=10, overlap_words=10)


def test_time_windows_are_contiguous_and_cover_everything():
    segments = make_segments(100, step=3.0)  # 300 seconds
    windows = time_windows(segments, window_s=60)
    assert len(windows) == 5
    assert sum(len(w.text.split()) for w in windows) == sum(len(s.text.split()) for s in segments)


@pytest.mark.parametrize("seconds,label", [(0, "0:00"), (5.9, "0:05"), (65, "1:05"), (3600, "1:00:00"), (3723, "1:02:03")])
def test_timestamp_round_trip(seconds, label):
    assert fmt_ts(seconds) == label
    assert parse_ts(label) == int(seconds)

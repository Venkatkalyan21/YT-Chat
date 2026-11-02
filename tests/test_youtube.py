import pytest

from ytchat.youtube import InvalidVideoUrl, extract_video_id

VID = "aircAruvnKk"


@pytest.mark.parametrize(
    "url",
    [
        VID,
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&t=42s",
        f"https://m.youtube.com/watch?feature=share&v={VID}",
        f"https://music.youtube.com/watch?v={VID}&list=RDAMVM",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?si=abc123",
        f"youtu.be/{VID}",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/embed/{VID}?start=10",
        f"https://www.youtube.com/live/{VID}?feature=share",
        f"https://www.youtube-nocookie.com/embed/{VID}",
        f"  https://www.youtube.com/watch?v={VID}  ",
    ],
)
def test_extracts_video_id(url):
    assert extract_video_id(url) == VID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "hello world",
        "https://example.com/watch?v=aircAruvnKk",
        "https://www.youtube.com/@3blue1brown",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi",
    ],
)
def test_rejects_non_video_links(url):
    with pytest.raises(InvalidVideoUrl):
        extract_video_id(url)

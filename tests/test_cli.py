from ytchat.__main__ import build_parser


def test_port_and_host_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setenv("HOST", "0.0.0.0")
    args = build_parser().parse_args([])
    assert (args.host, args.port) == ("0.0.0.0", 10000)


def test_defaults_without_environment(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("HOST", raising=False)
    args = build_parser().parse_args([])
    assert (args.host, args.port) == ("127.0.0.1", 8000)


def test_index_takes_several_urls():
    args = build_parser().parse_args(["--index", "https://youtu.be/aircAruvnKk", "IHZwWFHWa-w"])
    assert args.index == ["https://youtu.be/aircAruvnKk", "IHZwWFHWa-w"]

"""Run the app: `ytchat` or `python -m ytchat`."""

import argparse
import getpass
import logging
import os
import sys
from pathlib import Path

import uvicorn

from .api import create_app
from .config import Settings, get_settings
from .db import ConfigError, Database
from .ingest import IngestRejected, ingest_video
from .models import get_chat_model, get_embedder, model_label
from .store import Store
from .summary import stream_summary
from .youtube import InvalidVideoUrl, TranscriptUnavailable


def set_key(env_path: Path = Path(".env"), name: str = "GOOGLE_API_KEY") -> None:
    """Prompt for an API key with hidden input and write it into .env, so keys stay out of chats and shell history."""
    key = getpass.getpass(f"Paste your {name} (input stays hidden), then press Enter: ").strip().strip("\"'")
    if not key:
        print("No key entered; nothing changed.")
        return
    # Hidden input makes a double paste invisible, so catch two keys run together.
    if sum(key.count(prefix) for prefix in ("AIza", "AQ.")) > 1 or any(c.isspace() for c in key):
        print("That looks like more than one key pasted together. Nothing changed; run this again and paste just one key.")
        return
    source = env_path if env_path.exists() else env_path.with_name(".env.example")
    lines = source.read_text(encoding="utf-8").splitlines() if source.exists() else []
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{name}=") or line.strip().startswith(f"# {name}="):
            lines[i] = f"{name}={key}"
            break
    else:
        lines.insert(0, f"{name}={key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {name} ({len(key)} characters) to {env_path.resolve()}")


def index_videos(urls: list[str], settings: Settings) -> int:
    """Index videos and pre-generate their summaries into the configured database (e.g. a hosted one)."""
    db = Database.from_settings(settings)
    store = Store(db.pool)
    embedder, llm = get_embedder(settings), get_chat_model(settings)
    failures = 0
    try:
        for url in urls:
            try:
                video = ingest_video(url, store=store, embedder=embedder, settings=settings)
                if not store.get_summary(video["id"]):
                    summary = "".join(stream_summary(llm, video["title"], video["channel"], store.get_segments(video["id"]))).strip()
                    store.save_summary(video["id"], summary, model_label(settings))
                print(f"OK    {video['id']}  {video['title']}  ({video['num_chunks']} chunks, summary ready)")
            except (InvalidVideoUrl, TranscriptUnavailable, IngestRejected) as exc:
                failures += 1
                print(f"SKIP  {url}  {exc}")
    finally:
        db.close()
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ytchat", description="Chat with YouTube videos.")
    # Hosting platforms (Render, Cloud Run, ...) assign the port through $PORT.
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--set-key", action="store_true", help="save your Gemini API key to .env (hidden input) and exit")
    parser.add_argument(
        "--index",
        nargs="+",
        metavar="URL",
        help="index these videos and pre-generate their summaries into the configured database, then exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.set_key:
        set_key()
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("pgserver", "httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = get_settings()
    try:  # fail fast with a readable message instead of a startup traceback
        get_chat_model(settings)
        get_embedder(settings)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.index:
        sys.exit(index_videos(args.index, settings))

    print(f"YtChat running at http://{args.host}:{args.port}")
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

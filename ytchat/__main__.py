"""Run the app: `ytchat` or `python -m ytchat`."""

import argparse
import logging
import sys

import uvicorn

from .api import create_app
from .config import get_settings
from .db import ConfigError
from .models import get_chat_model, get_embedder


def main() -> None:
    parser = argparse.ArgumentParser(prog="ytchat", description="Chat with YouTube videos.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

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

    print(f"YtChat running at http://{args.host}:{args.port}")
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

"""Run the app: `ytchat` or `python -m ytchat`."""

import argparse
import getpass
import logging
import sys
from pathlib import Path

import uvicorn

from .api import create_app
from .config import get_settings
from .db import ConfigError
from .models import get_chat_model, get_embedder


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


def main() -> None:
    parser = argparse.ArgumentParser(prog="ytchat", description="Chat with YouTube videos.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--set-key", action="store_true", help="save your Gemini API key to .env (hidden input) and exit")
    args = parser.parse_args()

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

    print(f"YtChat running at http://{args.host}:{args.port}")
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

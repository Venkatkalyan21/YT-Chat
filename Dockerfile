FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ytchat ./ytchat
RUN pip install ".[anthropic]"

# The port comes from $PORT when the host sets one (Render, Cloud Run), else 8000.
EXPOSE 8000
CMD ["ytchat", "--host", "0.0.0.0"]

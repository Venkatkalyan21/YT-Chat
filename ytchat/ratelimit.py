"""In-memory request limits per visitor plus a global daily cap.

State lives in one process, which fits a single-instance deployment (e.g. Render's free
tier). Several instances would each keep their own counts; move this to Postgres or Redis then.
"""

import math
import threading
import time
from collections import deque
from collections.abc import Callable

MINUTE = 60.0
DAY = 24 * 3600.0


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int, global_per_day: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute = per_minute
        self.per_day = per_day
        self.global_per_day = global_per_day
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._global: deque[float] = deque()
        self._lock = threading.Lock()
        self._checks = 0

    def check(self, key: str) -> str | None:
        """Record a request for `key` and return None, or return why it's refused (recording nothing)."""
        now = self._clock()
        with self._lock:
            self._checks += 1
            if self._checks % 1000 == 0:
                self._sweep(now)
            hits = self._hits.setdefault(key, deque())
            _drop_older(hits, now - DAY)
            _drop_older(self._global, now - DAY)

            if self.global_per_day and len(self._global) >= self.global_per_day:
                return "This demo has reached its daily limit. Please try again tomorrow."
            if self.per_day and len(hits) >= self.per_day:
                return "You've reached today's limit for this demo. Please try again tomorrow."
            if self.per_minute:
                recent = [t for t in hits if t > now - MINUTE]
                if len(recent) >= self.per_minute:
                    wait = math.ceil(recent[0] + MINUTE - now)
                    return f"Too many requests. Please wait {max(wait, 1)} seconds and try again."

            hits.append(now)
            self._global.append(now)
            return None

    def _sweep(self, now: float) -> None:
        for key in list(self._hits):
            _drop_older(self._hits[key], now - DAY)
            if not self._hits[key]:
                del self._hits[key]


def _drop_older(hits: deque[float], cutoff: float) -> None:
    while hits and hits[0] <= cutoff:
        hits.popleft()

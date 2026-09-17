"""Exact-match response cache (plan 5.1; edge cases O-04, O-05, A-06).

The catalog doesn't change while a process runs, so an identical request earns an identical answer,
and serving it again costs nothing: no Groq call against the account's ~31-calls-a-day token budget.
Bounded by entry count (least recently used goes first) and by age. In-process only: a restart,
which a re-ingest already requires, starts it empty.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from src.core.models import RecommendationResponse


class ResponseCache:
    def __init__(self, *, ttl_s: float, max_entries: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_s, self.max_entries = ttl_s, max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, RecommendationResponse]] = OrderedDict()

    def get(self, key: str) -> RecommendationResponse | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, response = entry
            if self._clock() - stored_at >= self.ttl_s:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return response

    def put(self, key: str, response: RecommendationResponse) -> None:
        with self._lock:
            self._entries[key] = (self._clock(), response)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:  # O-05
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

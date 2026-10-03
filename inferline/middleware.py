"""Latency-recording middleware plus a request-id middleware.

The latency middleware keeps per-endpoint timing stats in memory. Handlers stay
clean: all measurement happens here. Stats are intentionally simple
(in-memory, reset on restart) — documented as a limitation, not a feature.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


def percentile(sorted_values: list[float], pct: float) -> float:
    """Percentile of an already-sorted list (linear interpolation)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


class LatencyStats:
    """Thread-safe in-memory per-endpoint latency store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, list[float]] = defaultdict(list)

    def record(self, endpoint: str, seconds: float) -> None:
        with self._lock:
            self._samples[endpoint].append(seconds)

    def snapshot(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        with self._lock:
            items = {k: sorted(v) for k, v in self._samples.items()}
        for endpoint, values in items.items():
            n = len(values)
            out[endpoint] = {
                "count": n,
                "mean_ms": round(sum(values) / n * 1000, 3) if n else 0.0,
                "p50_ms": round(percentile(values, 50) * 1000, 3),
                "p95_ms": round(percentile(values, 95) * 1000, 3),
                "p99_ms": round(percentile(values, 99) * 1000, 3),
            }
        return out

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()


class LatencyMiddleware(BaseHTTPMiddleware):
    """Record wall-clock latency per endpoint into a LatencyStats store."""

    def __init__(self, app, stats: LatencyStats) -> None:  # noqa: ANN001, ANN202
        super().__init__(app)
        self.stats = stats

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001, ANN202
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        # Normalize: "/predict" stays, trailing slashes stripped.
        endpoint = request.url.path.rstrip("/") or "/"
        self.stats.record(endpoint, elapsed)
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        return response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a request id to every request/response pair."""

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001, ANN202
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

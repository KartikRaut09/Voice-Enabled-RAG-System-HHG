"""Latency instrumentation middleware and pipeline latency tracker."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from backend.app.config import get_logger

logger = get_logger(__name__)


class LatencyTracker:
    """Tracks latency of individual pipeline components.

    Three separately instrumented metrics for engineering benchmarks:
    1. RAG latency:
       query_processing + embedding + retrieval + reranking + generation + guardrails
    2. STT latency:
       Isolated speech-to-text processing time (API/model transcription)
    3. Full E2E latency:
       STT latency + RAG latency (from backend receipt of audio to final response)

    Note: Microphone/user speech duration and browser audio-capture time are
    explicitly excluded from this engineering benchmark.
    """

    RAG_COMPONENTS = [
        "query_processing",
        "embedding",
        "retrieval",
        "reranking",
        "generation",
        "guardrails",
    ]

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    def start(self, component: str) -> None:
        """Start timing a component."""
        self._starts[component] = time.perf_counter()

    def stop(self, component: str) -> float:
        """Stop timing a component and return elapsed ms."""
        if component not in self._starts:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._starts[component]) * 1000
        self.timings[component] = elapsed_ms
        del self._starts[component]
        return elapsed_ms

    def get_stt_latency(self) -> float:
        """Isolated speech-to-text transcription latency."""
        return self.timings.get("stt", 0.0)

    def get_rag_latency(self) -> float:
        """Sum of RAG pipeline component latencies."""
        return sum(self.timings.get(c, 0.0) for c in self.RAG_COMPONENTS)

    def get_e2e_latency(self) -> float:
        """Full end-to-end latency: STT latency + RAG latency.

        Excludes client-side audio recording and speech duration.
        """
        return self.get_stt_latency() + self.get_rag_latency()

    def to_dict(self) -> dict[str, float]:
        """Return all recorded timings with the 3 top-level metrics."""
        result = dict(self.timings)
        result["stt_latency"] = self.get_stt_latency()
        result["rag_latency"] = self.get_rag_latency()
        result["e2e_latency"] = self.get_e2e_latency()
        return result


class LatencyInstrumentationMiddleware(BaseHTTPMiddleware):
    """Middleware that adds request ID and response time headers."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        request.state.request_id = request_id

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

        logger.info(
            "request_completed",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            response_time_ms=round(elapsed_ms, 2),
            request_id=request_id,
        )

        return response

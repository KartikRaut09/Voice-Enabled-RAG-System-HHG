"""API route definitions."""

import uuid

from fastapi import APIRouter

from backend.app.core.logging_config import get_logger
from backend.app.core.middleware import LatencyTracker
from backend.app.models.schemas import (
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Process a query through the RAG pipeline.

    Phase 0: Returns a structured placeholder response.
    """
    request_id = str(uuid.uuid4())
    tracker = LatencyTracker()

    # Phase 0 placeholder: simulate query processing
    tracker.start("query_processing")
    # No actual processing yet
    tracker.stop("query_processing")

    timings = tracker.to_dict()

    latency = LatencyBreakdown(
        stt_ms=timings.get("stt", 0.0),
        query_processing_ms=timings.get("query_processing", 0.0),
        embedding_ms=timings.get("embedding", 0.0),
        retrieval_ms=timings.get("retrieval", 0.0),
        reranking_ms=timings.get("reranking", 0.0),
        generation_ms=timings.get("generation", 0.0),
        guardrails_ms=timings.get("guardrails", 0.0),
        stt_latency_ms=timings.get("stt_latency", 0.0),
        rag_latency_ms=timings.get("rag_latency", 0.0),
        e2e_latency_ms=timings.get("e2e_latency", 0.0),
        total_request_ms=timings.get("query_processing", 0.0),
    )

    logger.info(
        "query_processed",
        request_id=request_id,
        query=request.query,
        language=request.language,
        latency_ms=latency.rag_latency_ms,
    )

    return QueryResponse(
        request_id=request_id,
        query=request.query,
        answer="Phase 0 placeholder — RAG pipeline not yet implemented.",
        sources=[],
        latency=latency,
        status="success",
    )

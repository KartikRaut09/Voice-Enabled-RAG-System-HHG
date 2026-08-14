"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import get_logger, get_settings, setup_logging
from backend.app.middleware import LatencyInstrumentationMiddleware, LatencyTracker
from backend.app.schemas import LatencyBreakdown, QueryRequest, QueryResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = get_logger(__name__)
    logger.info(
        "application_starting",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
    )
    yield
    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    logger = get_logger(__name__)

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(LatencyInstrumentationMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health endpoint
    @app.get("/health")
    async def health_check() -> dict:
        """Return application health status."""
        return {
            "status": "healthy",
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Query endpoint (Phase 0 structured placeholder)
    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        """Process a query through the RAG pipeline."""
        request_id = str(uuid.uuid4())
        tracker = LatencyTracker()

        # Phase 0 placeholder timing
        tracker.start("query_processing")
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

    # Serve frontend static files
    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount(
            "/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend"
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

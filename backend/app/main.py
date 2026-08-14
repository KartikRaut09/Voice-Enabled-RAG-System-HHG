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
from backend.app.schemas import LatencyBreakdown, QueryRequest, QueryResponse, SourcePassage



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

    # Global pipeline instance (lazy/configurable)
    from backend.app.pipeline import RAGPipeline
    pipeline = RAGPipeline(config={"pipeline": {"dense_top_k": 50, "lexical_top_k": 50, "rrf_k": 60, "context_top_k": 5}})

    # Query endpoint — Thin route delegating directly to RAGPipeline (Phase 7)
    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        """Process a query through the orchestrated RAG pipeline."""
        request_id = str(uuid.uuid4())
        return pipeline.orchestrate(
            query=request.query,
            language=request.language,
            request_id=request_id,
            options=request.options,
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

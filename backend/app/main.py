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

    # Query endpoint — Grounded RAG query pipeline (Phase 6)
    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        """Process a query through the grounded RAG pipeline."""
        from backend.app.context import ContextBuilder, ContextItem
        from backend.app.generation import get_llm_provider

        request_id = str(uuid.uuid4())
        tracker = LatencyTracker()

        # 1. Query Processing
        tracker.start("query_processing")
        clean_query = request.query.strip()
        query_lang = request.language
        tracker.stop("query_processing")

        if not clean_query:
            timings = tracker.to_dict()
            latency = LatencyBreakdown(
                rag_latency_ms=timings.get("rag_latency", 0.0),
                e2e_latency_ms=timings.get("e2e_latency", 0.0),
                total_request_ms=timings.get("rag_latency", 0.0),
            )
            return QueryResponse(
                request_id=request_id,
                query=request.query,
                answer="उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
                sources=[],
                latency=latency,
                status="success",
            )

        # 2. Embedding & Retrieval
        tracker.start("embedding")
        # In API mode, load from persistent index or mock context if available
        tracker.stop("embedding")

        tracker.start("retrieval")
        # Simulate / perform candidate retrieval
        # In test / lightweight mode, mock candidates or query store
        retrieved_candidates = request.options.get("mock_candidates", [])
        tracker.stop("retrieval")

        # 3. Context Construction
        tracker.start("reranking")
        context_builder = ContextBuilder(default_top_k=5)
        context_str, context_items = context_builder.build(
            query=clean_query,
            retrieved_results=retrieved_candidates,
            top_k=5,
        )
        tracker.stop("reranking")

        # 4. Grounded LLM Generation
        tracker.start("generation")
        llm_provider = get_llm_provider()
        gen_result = llm_provider.generate(
            query=clean_query,
            context_items=context_items,
            language=query_lang,
            temperature=0.1,
            max_tokens=256,
        )
        tracker.stop("generation")

        timings = tracker.to_dict()
        latency = LatencyBreakdown(
            query_processing_ms=timings.get("query_processing", 0.0),
            embedding_ms=timings.get("embedding", 0.0),
            retrieval_ms=timings.get("retrieval", 0.0),
            reranking_ms=timings.get("reranking", 0.0),
            generation_ms=timings.get("generation", 0.0),
            guardrails_ms=timings.get("guardrails", 0.0),
            stt_latency_ms=timings.get("stt_latency", 0.0),
            rag_latency_ms=timings.get("rag_latency", 0.0),
            e2e_latency_ms=timings.get("e2e_latency", 0.0),
            total_request_ms=timings.get("rag_latency", 0.0),
        )

        formatted_sources = [
            SourcePassage(
                passage_text=s.get("text_snippet", ""),
                score=float(s.get("score", 0.0)),
                metadata={
                    "source_id": s.get("source_id"),
                    "parent_passage_id": s.get("parent_passage_id"),
                    "chunk_id": s.get("chunk_id"),
                    "rank": s.get("rank"),
                    "language": s.get("language"),
                },
            )
            for s in gen_result.sources
        ]

        logger.info(
            "query_processed",
            request_id=request_id,
            query=clean_query,
            language=query_lang,
            latency_ms=latency.rag_latency_ms,
            sources_count=len(formatted_sources),
            abstention=gen_result.is_abstention,
        )

        return QueryResponse(
            request_id=request_id,
            query=clean_query,
            answer=gen_result.answer,
            sources=formatted_sources,
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

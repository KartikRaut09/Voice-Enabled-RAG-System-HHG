"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import get_logger, get_settings, setup_logging
from backend.app.middleware import LatencyInstrumentationMiddleware, LatencyTracker
from backend.app.schemas import (
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
    SourcePassage,
    TranscriptionResponse,
    VoiceQueryResponse,
)





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

    # STT Provider instance (configurable from default.yaml)
    from backend.app.stt import get_stt_provider

    stt_provider = get_stt_provider()


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

    # STT Transcribe endpoint (Phase 9)
    @app.post("/api/transcribe", response_model=TranscriptionResponse)
    async def transcribe_audio(
        file: UploadFile = File(...),
        language: str | None = Form(None),
    ) -> TranscriptionResponse:
        """Transcribe uploaded audio file to text query with language preservation."""
        audio_bytes = await file.read()
        res = stt_provider.transcribe(
            audio_bytes=audio_bytes,
            filename=file.filename,
            language=language,
        )
        return TranscriptionResponse(
            text=res.text,
            language=res.language,
            provider=res.provider,
            model=res.model,
            confidence=res.confidence,
            stt_preprocessing_ms=res.stt_preprocessing_ms,
            stt_inference_ms=res.stt_inference_ms,
            stt_total_ms=res.stt_total_ms,
            latency_ms=res.stt_total_ms,
            status=res.status,
            error=res.error,
        )

    # Full Voice-RAG Endpoint (Phase 10)
    @app.post("/api/voice-query", response_model=VoiceQueryResponse)
    async def voice_query(
        file: UploadFile = File(...),
        language: str | None = Form(None),
    ) -> VoiceQueryResponse:
        """Execute unified Voice-RAG pipeline: Audio -> STT -> RAGPipeline -> Guardrails."""
        req_id = str(uuid.uuid4())
        audio_bytes = await file.read()

        # 1. Validate & Transcribe Audio
        t_start_e2e = time.perf_counter()
        stt_res = stt_provider.transcribe(
            audio_bytes=audio_bytes,
            filename=file.filename,
            language=language,
        )

        # Early termination on STT failure or empty transcription
        if stt_res.status == "empty_transcription" or not stt_res.text.strip():
            t_e2e = (time.perf_counter() - t_start_e2e) * 1000.0
            latency = LatencyBreakdown(
                stt_ms=stt_res.stt_total_ms,
                stt_latency_ms=stt_res.stt_total_ms,
                e2e_latency_ms=t_e2e,
                total_request_ms=t_e2e,
            )
            return VoiceQueryResponse(
                request_id=req_id,
                transcription="",
                language=stt_res.language,
                answer="उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
                sources=[],
                latency=latency,
                status="empty_transcription",
                error=stt_res.error or "No speech detected in audio",
                query_metadata={"original_query": "", "processed_query": "", "language": stt_res.language or "unknown"},
                pipeline_metadata={"retrieval_mode": "none", "reranking_enabled": False},
            )

        if stt_res.status == "error":
            t_e2e = (time.perf_counter() - t_start_e2e) * 1000.0
            latency = LatencyBreakdown(
                stt_ms=stt_res.stt_total_ms,
                stt_latency_ms=stt_res.stt_total_ms,
                e2e_latency_ms=t_e2e,
                total_request_ms=t_e2e,
            )
            return VoiceQueryResponse(
                request_id=req_id,
                transcription="",
                language=stt_res.language,
                answer="उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
                sources=[],
                latency=latency,
                status="error",
                error=stt_res.error,
                query_metadata={"original_query": "", "processed_query": "", "language": stt_res.language or "unknown"},
                pipeline_metadata={"retrieval_mode": "none", "reranking_enabled": False},
            )

        # 2. Delegate to the Single Authoritative Text-RAG Pipeline
        rag_res: QueryResponse = pipeline.orchestrate(
            query=stt_res.text,
            language=stt_res.language,
            request_id=req_id,
        )

        t_integration_overhead = (time.perf_counter() - t_start_e2e) * 1000.0
        e2e_total_ms = stt_res.stt_total_ms + rag_res.latency.rag_latency_ms + t_integration_overhead

        # Construct unified 3-metric Latency Breakdown
        combined_latency = LatencyBreakdown(
            stt_ms=stt_res.stt_total_ms,
            query_processing_ms=rag_res.latency.query_processing_ms,
            embedding_ms=rag_res.latency.embedding_ms,
            retrieval_ms=rag_res.latency.retrieval_ms,
            reranking_ms=rag_res.latency.reranking_ms,
            generation_ms=rag_res.latency.generation_ms,
            guardrails_ms=rag_res.latency.guardrails_ms,
            stt_latency_ms=stt_res.stt_total_ms,
            rag_latency_ms=rag_res.latency.rag_latency_ms,
            e2e_latency_ms=e2e_total_ms,
            total_request_ms=e2e_total_ms,
        )

        return VoiceQueryResponse(
            request_id=req_id,
            transcription=stt_res.text,
            language=rag_res.query_metadata.get("language", stt_res.language),
            answer=rag_res.answer,
            sources=rag_res.sources,
            latency=combined_latency,
            status=rag_res.status,
            error=rag_res.error,
            query_metadata=rag_res.query_metadata,
            pipeline_metadata=rag_res.pipeline_metadata,
            guardrail_flags=rag_res.guardrail_flags,
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

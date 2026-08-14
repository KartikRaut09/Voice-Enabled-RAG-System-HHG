"""Pydantic models for structured request and response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query request."""

    query: str
    language: str = "en"
    audio_base64: str | None = None  # Base64 audio for voice input (Phase 9)
    options: dict = Field(default_factory=dict)


class SourcePassage(BaseModel):
    """A retrieved source passage."""

    passage_text: str
    score: float
    metadata: dict = Field(default_factory=dict)


class LatencyBreakdown(BaseModel):
    """Three separately instrumented latency metrics and component breakdown.

    Primary metrics (engineering benchmark):
    - rag_latency_ms: query_processing + embedding + retrieval + reranking + generation + guardrails
    - stt_latency_ms: isolated speech-to-text processing time
    - e2e_latency_ms: stt_latency_ms + rag_latency_ms (excludes mic/browser audio capture)

    Component timings:
    - individual pipeline stage durations in milliseconds
    """

    # Component timings
    stt_ms: float = 0.0
    query_processing_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    guardrails_ms: float = 0.0

    # Three primary benchmark metrics
    stt_latency_ms: float = 0.0
    rag_latency_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    total_request_ms: float = 0.0


class QueryResponse(BaseModel):
    """Structured query response."""

    request_id: str
    query: str
    answer: str
    sources: list[SourcePassage] = Field(default_factory=list)
    latency: LatencyBreakdown
    status: str = "success"
    error: str | None = None
    query_metadata: dict = Field(default_factory=dict)
    pipeline_metadata: dict = Field(default_factory=dict)
    guardrail_flags: dict = Field(default_factory=dict)


class TranscriptionResponse(BaseModel):
    """Structured response for POST /api/transcribe."""

    text: str
    language: str | None = None
    provider: str
    model: str
    confidence: float | None = None
    stt_preprocessing_ms: float = 0.0
    stt_inference_ms: float = 0.0
    stt_total_ms: float = 0.0
    latency_ms: float = 0.0
    status: str = "success"
    error: str | None = None


class VoiceQueryResponse(BaseModel):
    """Structured response for POST /api/voice-query."""

    request_id: str
    transcription: str
    language: str | None = None
    answer: str
    sources: list[SourcePassage] = Field(default_factory=list)
    latency: LatencyBreakdown
    status: str = "success"
    error: str | None = None
    query_metadata: dict = Field(default_factory=dict)
    pipeline_metadata: dict = Field(default_factory=dict)
    guardrail_flags: dict = Field(default_factory=dict)




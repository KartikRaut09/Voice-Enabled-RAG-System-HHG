# Architecture Documentation

## Offline Pipeline
```
Dataset (MSMARCO-XI) → Streaming → Passage Extraction → Multi-Strategy Chunking → Embedding → Vector Index + BM25 Index → Persistent Storage
```

## Online Pipeline
```
Audio/Text → STT → Input Guardrails → Query Processing → Query Embedding → Dense Retrieval → Lexical Retrieval → Fusion → Reranking → Context Construction → LLM Generation → Grounding Validation → Output Guardrails → Response
```

## Component Interfaces

| Component | Interface | Phase |
|-----------|----------|-------|
| STT | BaseSTTProvider | 9 |
| Query Processing | BaseQueryProcessor | 7 |
| Embeddings | BaseEmbedder | 3 |
| Dense Retrieval | BaseDenseRetriever | 4 |
| Lexical Retrieval | BaseLexicalRetriever | 5 |
| Fusion | BaseRetrieverFusion | 5 |
| Reranker | BaseReranker | 5 |
| Generation | BaseLLMProvider | 6 |
| Guardrails | BaseGuardrail | 8 |
| Orchestrator | BasePipelineOrchestrator | 7 |

## Latency Measurement

Three separately instrumented metrics are tracked and reported as P50 / P70 / P100:

1. **RAG Latency** = query_processing + embedding + retrieval + reranking + generation + guardrails
2. **STT Latency** = isolated speech-to-text processing time (API/model transcription)
3. **Full E2E Latency** = STT Latency + RAG Latency (from backend receipt of audio to final response)

*Engineering Benchmark Note*: Microphone/user speech duration and browser audio-capture time are explicitly excluded from this engineering benchmark.

## Technology Decisions

| Component | Status | Notes |
|-----------|--------|-------|
| Backend | FastAPI | Selected |
| Frontend | Vanilla HTML/CSS/JS | Selected |
| STT | TBD (Sarvam recommended) | Phase 9 |
| Embeddings | TBD | Benchmarked Phase 3 |
| Vector DB | TBD | Benchmarked Phase 3 |
| BM25 | TBD | Benchmarked Phase 5 |
| Reranker | TBD | Benchmarked Phase 5 |
| LLM | TBD | Benchmarked Phase 6 |
| Deployment | Docker | Selected |

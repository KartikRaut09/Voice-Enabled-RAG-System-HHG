"""RAG Pipeline Orchestrator module.

Coordinates query processing, dense & BM25 hybrid retrieval, reciprocal rank fusion,
optional cross-encoder reranking, context building, grounded generation, citation validation,
and detailed latency accounting.
"""

from __future__ import annotations

import time
from typing import Any
import uuid

from backend.app.config import get_logger
from backend.app.context import ContextBuilder
from backend.app.fusion import reciprocal_rank_fusion
from backend.app.generation import LLMProvider, get_llm_provider
from backend.app.middleware import LatencyTracker
from backend.app.query_processor import QueryInput, QueryProcessor
from backend.app.reranking import CrossEncoderReranker, rerank_candidates
from backend.app.schemas import LatencyBreakdown, QueryResponse, SourcePassage

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end Text RAG Pipeline Orchestrator."""

    def __init__(
        self,
        embedder: Any = None,
        vector_store: Any = None,
        bm25_index: Any = None,
        reranker: CrossEncoderReranker | None = None,
        context_builder: ContextBuilder | None = None,
        llm_provider: LLMProvider | None = None,
        query_processor: QueryProcessor | None = None,
        config: dict | None = None,
    ) -> None:
        self.config = config or {}
        p_cfg = self.config.get("pipeline", {})

        self.dense_top_k = p_cfg.get("dense_top_k", 50)
        self.lexical_top_k = p_cfg.get("lexical_top_k", 50)
        self.rrf_k = p_cfg.get("rrf_k", 60)
        self.context_top_k = p_cfg.get("context_top_k", 5)
        self.reranking_enabled = p_cfg.get("reranking_enabled", False)

        self.query_processor = query_processor or QueryProcessor()
        self.context_builder = context_builder or ContextBuilder(default_top_k=self.context_top_k)
        self.llm_provider = llm_provider or get_llm_provider(self.config)

        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker

    def orchestrate(
        self,
        query: str,
        language: str | None = None,
        request_id: str | None = None,
        options: dict | None = None,
    ) -> QueryResponse:
        """Execute the end-to-end grounded RAG pipeline."""
        req_id = request_id or str(uuid.uuid4())
        opts = options or {}
        tracker = LatencyTracker()

        # 1. Query Processing
        tracker.start("query_processing")
        q_input: QueryInput = self.query_processor.process(query, language=language)
        tracker.stop("query_processing")

        if not q_input.is_valid:
            timings = tracker.to_dict()
            latency = self._build_latency(timings)
            return QueryResponse(
                request_id=req_id,
                query=q_input.original_query,
                answer="उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
                sources=[],
                latency=latency,
                status="insufficient_evidence",
                error=q_input.error,
                query_metadata=q_input.to_dict(),
                pipeline_metadata={"retrieval_mode": "none", "reranking": False},
            )

        clean_query = q_input.processed_query
        retrieval_mode = "hybrid"
        retrieved_cands: list[dict[str, Any]] = []

        # 2. Retrieval Stage
        tracker.start("embedding")
        mock_cands = opts.get("mock_candidates")
        if mock_cands is not None:
            # Test / mock mode: bypass physical indexes
            tracker.stop("embedding")
            tracker.start("retrieval")
            retrieved_cands = list(mock_cands)
            retrieval_mode = opts.get("retrieval_mode", "mock")
            tracker.stop("retrieval")
        else:
            dense_cands: list[dict[str, Any]] = []
            bm25_cands: list[dict[str, Any]] = []
            dense_failed = False
            bm25_failed = False

            # Dense search
            try:
                if self.embedder is not None and self.vector_store is not None:
                    q_vec = self.embedder.encode_queries([clean_query], batch_size=1)
                    raw_dense = self.vector_store.search_chunks(q_vec, top_k=self.dense_top_k)
                    for meta, score in raw_dense:
                        item = dict(meta)
                        item["score"] = float(score)
                        dense_cands.append(item)
                else:
                    dense_failed = True
            except Exception as e:
                logger.warn("dense_retrieval_failed", error=str(e), request_id=req_id)
                dense_failed = True

            tracker.stop("embedding")
            tracker.start("retrieval")

            # BM25 search
            try:
                if self.bm25_index is not None:
                    raw_bm25 = self.bm25_index.search_chunks(clean_query, top_k=self.lexical_top_k)
                    for meta, score in raw_bm25:
                        item = dict(meta)
                        item["score"] = float(score)
                        bm25_cands.append(item)
                else:
                    bm25_failed = True
            except Exception as e:
                logger.warn("bm25_retrieval_failed", error=str(e), request_id=req_id)
                bm25_failed = True

            # Fusion & Degraded Mode Resolution
            if not dense_failed and not bm25_failed:
                retrieved_cands = reciprocal_rank_fusion(dense_cands, bm25_cands, rrf_k=self.rrf_k)
                retrieval_mode = "hybrid"
            elif not dense_failed and bm25_failed:
                retrieved_cands = dense_cands
                retrieval_mode = "dense_only"
            elif dense_failed and not bm25_failed:
                retrieved_cands = bm25_cands
                retrieval_mode = "bm25_only"
            elif self.embedder is None and self.vector_store is None and self.bm25_index is None:
                # Unindexed / lightweight mode
                retrieval_mode = "none"
                retrieved_cands = []
            else:
                retrieval_mode = "failed"
                retrieved_cands = []

            tracker.stop("retrieval")

        # 3. Optional Reranking
        tracker.start("reranking")
        rerank_applied = False
        rerank_model_name = None
        if self.reranking_enabled and self.reranker is not None and retrieved_cands:
            try:
                retrieved_cands = rerank_candidates(
                    clean_query,
                    retrieved_cands,
                    reranker=self.reranker,
                    rerank_top_k=opts.get("rerank_top_k", 20),
                )
                rerank_applied = True
                rerank_model_name = getattr(self.reranker, "model_name", "cross-encoder")
            except Exception as e:
                logger.warn("reranking_failed", error=str(e), request_id=req_id)
        tracker.stop("reranking")

        # 4. Context Construction
        context_str, context_items = self.context_builder.build(
            query=clean_query,
            retrieved_results=retrieved_cands,
            top_k=self.context_top_k,
        )

        if not context_items or retrieval_mode == "failed":
            timings = tracker.to_dict()
            latency = self._build_latency(timings)
            status_val = "error" if retrieval_mode == "failed" else "success"
            return QueryResponse(
                request_id=req_id,
                query=q_input.original_query,
                answer="उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
                sources=[],
                latency=latency,
                status=status_val,
                error="Retrieval failed across all channels" if retrieval_mode == "failed" else None,
                query_metadata=q_input.to_dict(),
                pipeline_metadata={
                    "retrieval_mode": retrieval_mode,
                    "reranking_enabled": rerank_applied,
                    "insufficient_evidence": True,
                },
            )


        # 5. LLM Generation
        tracker.start("generation")
        try:
            gen_res = self.llm_provider.generate(
                query=clean_query,
                context_items=context_items,
                language=q_input.language,
                temperature=opts.get("temperature", 0.1),
                max_tokens=opts.get("max_tokens", 256),
            )
        except Exception as e:
            logger.error("generation_failed", error=str(e), request_id=req_id)
            timings = tracker.to_dict()
            latency = self._build_latency(timings)
            return QueryResponse(
                request_id=req_id,
                query=q_input.original_query,
                answer="उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
                sources=[],
                latency=latency,
                status="error",
                error=f"LLM generation failed: {str(e)}",
                query_metadata=q_input.to_dict(),
                pipeline_metadata={
                    "retrieval_mode": retrieval_mode,
                    "reranking_enabled": rerank_applied,
                },
            )
        tracker.stop("generation")

        timings = tracker.to_dict()
        latency = self._build_latency(timings)

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
            for s in gen_res.sources
        ]

        status_flag = "degraded" if retrieval_mode in ("dense_only", "bm25_only") else (
            "insufficient_evidence" if gen_res.is_abstention else "success"
        )

        logger.info(
            "query_completed",
            request_id=req_id,
            language=q_input.language,
            retrieval_mode=retrieval_mode,
            reranking=rerank_applied,
            sources_count=len(formatted_sources),
            model=gen_res.model_name,
            rag_latency_ms=latency.rag_latency_ms,
            status=status_flag,
        )

        return QueryResponse(
            request_id=req_id,
            query=q_input.original_query,
            answer=gen_res.answer,
            sources=formatted_sources,
            latency=latency,
            status=status_flag,
            query_metadata=q_input.to_dict(),
            pipeline_metadata={
                "retrieval_mode": retrieval_mode,
                "reranking_enabled": rerank_applied,
                "reranker_model": rerank_model_name,
                "model_used": gen_res.model_name,
                "provider": gen_res.provider,
                "input_tokens": gen_res.input_tokens,
                "output_tokens": gen_res.output_tokens,
            },
        )

    def _build_latency(self, timings: dict[str, float]) -> LatencyBreakdown:
        """Construct standard 3-metric LatencyBreakdown object."""
        return LatencyBreakdown(
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

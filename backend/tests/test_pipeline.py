"""Unit and integration tests for RAGPipeline orchestrator and failure isolation."""

import pytest

from backend.app.context import ContextBuilder
from backend.app.generation import MockLLMProvider
from backend.app.pipeline import RAGPipeline
from backend.app.query_processor import QueryProcessor
from backend.app.schemas import QueryResponse


class MockEmbedder:
    def encode_queries(self, queries: list[str], batch_size: int = 1):
        return [[0.1] * 384]


class MockVectorStore:
    def search_chunks(self, query_vector, top_k: int = 50):
        return [
            ({"chunk_id": "c1", "parent_passage_id": "p1", "text": "भारत की राजधानी नई दिल्ली है।", "language": "hin_Deva"}, 0.95),
            ({"chunk_id": "c2", "parent_passage_id": "p2", "text": "ताजमहल आगरा में स्थित है।", "language": "hin_Deva"}, 0.85),
        ]


class MockBM25Index:
    def search_chunks(self, query: str, top_k: int = 50):
        return [
            ({"chunk_id": "c2", "parent_passage_id": "p2", "text": "ताजमहल आगरा में स्थित है।", "language": "hin_Deva"}, 12.0),
            ({"chunk_id": "c3", "parent_passage_id": "p3", "text": "गंगा भारत की प्रमुख नदी है।", "language": "hin_Deva"}, 8.5),
        ]


class FailingRetriever:
    def search_chunks(self, *args, **kwargs):
        raise RuntimeError("Database connection timed out")


def test_rag_pipeline_successful_orchestration():
    """Test full successful hybrid RAG orchestration with valid citations and latency breakdown."""
    pipeline = RAGPipeline(
        embedder=MockEmbedder(),
        vector_store=MockVectorStore(),
        bm25_index=MockBM25Index(),
        context_builder=ContextBuilder(default_top_k=5),
        llm_provider=MockLLMProvider(),
        query_processor=QueryProcessor(),
        config={"pipeline": {"reranking_enabled": False}},
    )

    res = pipeline.orchestrate("  भारत की राजधानी क्या है?  ", language="hin_Deva")

    assert isinstance(res, QueryResponse)
    assert res.status == "success"
    assert res.query == "  भारत की राजधानी क्या है?  "
    assert res.query_metadata["processed_query"] == "भारत की राजधानी क्या है?"
    assert res.query_metadata["language"] == "hin_Deva"
    assert res.pipeline_metadata["retrieval_mode"] == "hybrid"
    assert len(res.sources) >= 1
    assert res.latency.rag_latency_ms > 0.0
    assert res.latency.query_processing_ms >= 0.0
    assert res.latency.generation_ms > 0.0


def test_rag_pipeline_reranking_enabled():
    """Test pipeline execution when neural reranker is enabled."""
    class MockReranker:
        model_name = "mock-reranker"
        def rank(self, query: str, documents: list[str], batch_size: int = 16):
            return [0.99] * len(documents)

    pipeline = RAGPipeline(
        embedder=MockEmbedder(),
        vector_store=MockVectorStore(),
        bm25_index=MockBM25Index(),
        reranker=MockReranker(),
        context_builder=ContextBuilder(default_top_k=5),
        llm_provider=MockLLMProvider(),
        config={"pipeline": {"reranking_enabled": True}},
    )

    res = pipeline.orchestrate("ताजमहल", language="hin_Deva")
    assert res.status == "success"
    assert res.pipeline_metadata["reranking_enabled"] is True
    assert res.pipeline_metadata["reranker_model"] == "mock-reranker"
    assert res.latency.reranking_ms >= 0.0


def test_rag_pipeline_dense_failure_degraded_bm25_mode():
    """Test that dense retriever failure gracefully degrades to bm25_only mode without crashing."""
    pipeline = RAGPipeline(
        embedder=MockEmbedder(),
        vector_store=FailingRetriever(),  # dense fails
        bm25_index=MockBM25Index(),      # BM25 works
        context_builder=ContextBuilder(default_top_k=5),
        llm_provider=MockLLMProvider(),
    )

    res = pipeline.orchestrate("गंगा नदी", language="hin_Deva")
    assert res.status in ("success", "degraded")
    assert res.pipeline_metadata["retrieval_mode"] == "bm25_only"
    assert len(res.sources) >= 1


def test_rag_pipeline_bm25_failure_degraded_dense_mode():
    """Test that BM25 failure gracefully degrades to dense_only mode without crashing."""
    pipeline = RAGPipeline(
        embedder=MockEmbedder(),
        vector_store=MockVectorStore(),  # dense works
        bm25_index=FailingRetriever(),   # BM25 fails
        context_builder=ContextBuilder(default_top_k=5),
        llm_provider=MockLLMProvider(),
    )

    res = pipeline.orchestrate("ताजमहल", language="hin_Deva")
    assert res.status in ("success", "degraded")
    assert res.pipeline_metadata["retrieval_mode"] == "dense_only"
    assert len(res.sources) >= 1


def test_rag_pipeline_both_retrieval_failures():
    """Test that simultaneous failure of both retrieval channels produces a controlled error."""
    pipeline = RAGPipeline(
        embedder=MockEmbedder(),
        vector_store=FailingRetriever(),
        bm25_index=FailingRetriever(),
        context_builder=ContextBuilder(default_top_k=5),
        llm_provider=MockLLMProvider(),
    )

    res = pipeline.orchestrate("सवाल", language="hin_Deva")
    assert res.status == "error"
    assert res.pipeline_metadata["retrieval_mode"] == "failed"
    assert len(res.sources) == 0


def test_rag_pipeline_empty_retrieval_abstention():
    """Test that zero retrieved candidates returns safe abstention without calling LLM generation."""
    class EmptyRetriever:
        def search_chunks(self, *args, **kwargs):
            return []

    pipeline = RAGPipeline(
        embedder=MockEmbedder(),
        vector_store=EmptyRetriever(),
        bm25_index=EmptyRetriever(),
        context_builder=ContextBuilder(default_top_k=5),
        llm_provider=MockLLMProvider(),
    )

    res = pipeline.orchestrate("अपरिचित सवाल")
    assert res.status == "insufficient_evidence"
    assert len(res.sources) == 0
    assert "पर्याप्त जानकारी नहीं है" in res.answer


def test_rag_pipeline_llm_failure_isolation():
    """Test that LLM provider exception results in controlled error response."""
    class FailingLLM:
        def generate(self, *args, **kwargs):
            raise TimeoutError("Provider API timeout")

    pipeline = RAGPipeline(
        embedder=MockEmbedder(),
        vector_store=MockVectorStore(),
        bm25_index=MockBM25Index(),
        context_builder=ContextBuilder(default_top_k=5),
        llm_provider=FailingLLM(),
    )

    res = pipeline.orchestrate("भारत", language="hin_Deva")
    assert res.status == "error"
    assert "LLM generation failed" in (res.error or "")

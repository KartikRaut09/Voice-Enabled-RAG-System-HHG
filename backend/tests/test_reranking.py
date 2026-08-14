"""Unit tests for neural cross-encoder reranking and end-to-end hybrid pipeline."""

from typing import Any
import pytest

from backend.app.fusion import aggregate_parent_passages, reciprocal_rank_fusion
from backend.app.reranking import rerank_candidates


class MockReranker:
    """Deterministic mock reranker for unit tests."""

    def rank(self, query: str, documents: list[str], batch_size: int = 16) -> list[float]:
        # Return mock score based on keyword overlap length or fixed mapping
        scores = []
        for doc in documents:
            if "target" in doc.lower():
                scores.append(0.99)
            elif "secondary" in doc.lower():
                scores.append(0.50)
            else:
                scores.append(0.10)
        return scores


def test_rerank_candidates_ordering():
    """Test that reranker re-orders candidates according to neural score."""
    candidates = [
        {"chunk_id": "c1", "parent_passage_id": "p1", "text": "Common irrelevant passage"},
        {"chunk_id": "c2", "parent_passage_id": "p2", "text": "Highly relevant target passage"},
        {"chunk_id": "c3", "parent_passage_id": "p3", "text": "Secondary passage"},
    ]
    reranker = MockReranker()
    reranked = rerank_candidates("find target", candidates, reranker=reranker, rerank_top_k=3)

    assert len(reranked) == 3
    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[0]["rerank_score"] == 0.99
    assert reranked[1]["chunk_id"] == "c3"
    assert reranked[1]["rerank_score"] == 0.50
    assert reranked[2]["chunk_id"] == "c1"
    assert reranked[2]["rerank_score"] == 0.10


def test_rerank_candidates_tail_preservation():
    """Test that candidates beyond rerank_top_k are appended at the tail without truncation."""
    candidates = [
        {"chunk_id": f"c{i}", "parent_passage_id": f"p{i}", "text": f"Passage {i}"}
        for i in range(10)
    ]
    reranker = MockReranker()
    # Rerank only top 3
    reranked = rerank_candidates("query", candidates, reranker=reranker, rerank_top_k=3)

    assert len(reranked) == 10
    # First 3 have rerank_score, remaining 7 do not
    assert all("rerank_score" in c for c in reranked[:3])
    assert all("rerank_score" not in c for c in reranked[3:])


def test_end_to_end_hybrid_pipeline_with_mock():
    """Test full integration: Dense + BM25 results -> RRF -> Reranking -> Parent Aggregation."""
    dense_results = [
        {"chunk_id": "p1_c0", "parent_passage_id": "p1", "text": "Common text", "score": 0.88},
        {"chunk_id": "p2_c0", "parent_passage_id": "p2", "text": "Other text", "score": 0.85},
    ]
    bm25_results = [
        {"chunk_id": "p3_c0", "parent_passage_id": "p3", "text": "Target text found by keyword", "score": 15.0},
        {"chunk_id": "p1_c1", "parent_passage_id": "p1", "text": "Common text secondary", "score": 12.0},
    ]

    # 1. RRF Fusion (4 distinct chunk IDs in pool)
    fused_chunks = reciprocal_rank_fusion(dense_results, bm25_results, rrf_k=60)
    assert len(fused_chunks) == 4

    # 2. Neural Reranking
    reranker = MockReranker()
    reranked_chunks = rerank_candidates("search target", fused_chunks, reranker=reranker, rerank_top_k=10)
    assert len(reranked_chunks) == 4

    # 3. Parent Aggregation (collapses p1_c0 and p1_c1 to parent p1)
    final_parents = aggregate_parent_passages(reranked_chunks, top_k=5)
    assert len(final_parents) == 3
    # Target text should have highest score (0.99) and rank 1st
    assert final_parents[0]["parent_passage_id"] == "p3"
    assert final_parents[0]["score"] == 0.99



def test_rerank_empty_and_whitespace():
    """Test edge cases with empty candidates or blank query."""
    reranker = MockReranker()
    assert rerank_candidates("", [{"chunk_id": "c1", "text": "abc"}], reranker=reranker) == [{"chunk_id": "c1", "text": "abc"}]
    assert rerank_candidates("query", [], reranker=reranker) == []

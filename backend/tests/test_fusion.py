"""Unit tests for hybrid retrieval fusion strategies and parent aggregation."""

from backend.app.fusion import (
    aggregate_parent_passages,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


def test_rrf_both_channels_and_rank_ordering():
    """Test RRF ranking when documents appear in both dense and BM25 channels."""
    dense_results = [
        {"chunk_id": "c1", "parent_passage_id": "p1", "score": 0.95},
        {"chunk_id": "c2", "parent_passage_id": "p2", "score": 0.85},
    ]
    bm25_results = [
        {"chunk_id": "c2", "parent_passage_id": "p2", "score": 12.5},
        {"chunk_id": "c1", "parent_passage_id": "p1", "score": 8.0},
    ]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, rrf_k=60)
    assert len(fused) == 2
    # Both documents have identical sum of reciprocal ranks: 1/(60+1) + 1/(60+2)
    assert fused[0]["fusion_score"] == fused[1]["fusion_score"]
    assert "dense_rank" in fused[0]
    assert "bm25_rank" in fused[0]


def test_rrf_missing_channel_preservation():
    """Test that dense-only and BM25-only documents are preserved with correct RRF scores."""
    dense_results = [
        {"chunk_id": "dense_only", "parent_passage_id": "p_dense", "score": 0.90},
    ]
    bm25_results = [
        {"chunk_id": "bm25_only", "parent_passage_id": "p_bm25", "score": 10.0},
    ]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, rrf_k=60)
    assert len(fused) == 2
    fused_keys = {item["chunk_id"] for item in fused}
    assert "dense_only" in fused_keys
    assert "bm25_only" in fused_keys

    # Both got rank 1 in their respective channel, so fusion score is 1/(60+1) = 0.016393
    assert fused[0]["fusion_score"] == 0.016393
    assert fused[1]["fusion_score"] == 0.016393


def test_weighted_fusion_min_max_normalization():
    """Test min-max normalization and weighted combination."""
    dense_results = [
        {"chunk_id": "c1", "parent_passage_id": "p1", "score": 0.9},
        {"chunk_id": "c2", "parent_passage_id": "p2", "score": 0.5},
    ]
    bm25_results = [
        {"chunk_id": "c1", "parent_passage_id": "p1", "score": 20.0},
        {"chunk_id": "c2", "parent_passage_id": "p2", "score": 0.0},
    ]

    # c1 is top in both (norm=1.0 for both), c2 is bottom in both (norm=0.0 for both)
    fused = weighted_score_fusion(dense_results, bm25_results, dense_weight=0.7)
    assert len(fused) == 2
    assert fused[0]["chunk_id"] == "c1"
    assert fused[0]["fusion_score"] == 1.0
    assert fused[1]["chunk_id"] == "c2"
    assert fused[1]["fusion_score"] == 0.0


def test_weighted_fusion_missing_channel():
    """Test weighted fusion when document is present in only one channel."""
    dense_results = [
        {"chunk_id": "c1", "parent_passage_id": "p1", "score": 0.8},
    ]
    bm25_results = [
        {"chunk_id": "c2", "parent_passage_id": "p2", "score": 15.0},
    ]

    fused = weighted_score_fusion(dense_results, bm25_results, dense_weight=0.8)
    assert len(fused) == 2
    # c1 has dense_norm=1.0, bm25_norm=0.0 -> score = 0.8 * 1.0 = 0.8
    # c2 has dense_norm=0.0, bm25_norm=1.0 -> score = 0.2 * 1.0 = 0.2
    assert fused[0]["chunk_id"] == "c1"
    assert fused[0]["fusion_score"] == 0.8
    assert fused[1]["chunk_id"] == "c2"
    assert fused[1]["fusion_score"] == 0.2


def test_aggregate_parent_passages_max_score():
    """Test parent score aggregation using max to prevent multi-chunk unfair inflation."""
    candidates = [
        {"chunk_id": "p1_c0", "parent_passage_id": "p1", "fusion_score": 0.02},
        {"chunk_id": "p1_c1", "parent_passage_id": "p1", "fusion_score": 0.03},
        {"chunk_id": "p2_c0", "parent_passage_id": "p2", "fusion_score": 0.025},
    ]

    parents = aggregate_parent_passages(candidates, top_k=5, aggregation_method="max")
    assert len(parents) == 2
    # p1 max score is 0.03, p2 max score is 0.025 -> p1 should rank 1st
    assert parents[0]["parent_passage_id"] == "p1"
    assert parents[0]["score"] == 0.03
    assert parents[1]["parent_passage_id"] == "p2"
    assert parents[1]["score"] == 0.025


def test_empty_fusion_inputs():
    """Test edge cases with empty input lists."""
    assert reciprocal_rank_fusion([], []) == []
    assert weighted_score_fusion([], []) == []
    assert aggregate_parent_passages([]) == []

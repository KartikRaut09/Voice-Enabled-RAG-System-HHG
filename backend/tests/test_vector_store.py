"""Unit tests for persistent FAISS vector store and parent deduplication."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from backend.app.vector_store import FAISSVectorStore


@pytest.fixture
def sample_vectors_and_metadata():
    """Create normalized sample vectors and metadata with multi-chunk parent passages."""
    np.random.seed(42)
    dim = 64
    raw_vecs = np.random.randn(5, dim).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norm_vecs = raw_vecs / norms

    metadata = [
        {"chunk_id": "p1_c0", "parent_passage_id": "p1", "query_id": 101, "is_selected": True, "language": "hin_Deva"},
        {"chunk_id": "p1_c1", "parent_passage_id": "p1", "query_id": 101, "is_selected": True, "language": "hin_Deva"},
        {"chunk_id": "p2_c0", "parent_passage_id": "p2", "query_id": 101, "is_selected": False, "language": "hin_Deva"},
        {"chunk_id": "p3_c0", "parent_passage_id": "p3", "query_id": 102, "is_selected": True, "language": "mar_Deva"},
        {"chunk_id": "p3_c1", "parent_passage_id": "p3", "query_id": 102, "is_selected": True, "language": "mar_Deva"},
    ]
    return norm_vecs, metadata, dim


def test_vector_store_add_and_search_chunks(sample_vectors_and_metadata):
    """Test indexing and basic chunk retrieval."""
    vectors, metadata, dim = sample_vectors_and_metadata
    store = FAISSVectorStore(dimension=dim, metric="cosine")
    store.add(vectors, metadata)

    assert store.size == 5

    # Query with vector 0 -> should return chunk p1_c0 as top result with similarity ~1.0
    results = store.search_chunks(vectors[0], top_k=3)
    assert len(results) == 3
    top_meta, top_score = results[0]
    assert top_meta["chunk_id"] == "p1_c0"
    assert pytest.approx(top_score, rel=1e-3) == 1.0


def test_parent_passage_deduplication(sample_vectors_and_metadata):
    """Test that multiple chunks from the same parent passage are collapsed to unique parents."""
    vectors, metadata, dim = sample_vectors_and_metadata
    store = FAISSVectorStore(dimension=dim, metric="cosine")
    store.add(vectors, metadata)

    # Search with vector 0 (p1_c0)
    parents = store.search_parent_passages(vectors[0], top_k=5, fetch_k=10)

    # Although there are 5 chunks, there are only 3 unique parent passages (p1, p2, p3)
    parent_ids = [p["parent_passage_id"] for p in parents]
    assert len(parent_ids) == len(set(parent_ids)), "Parent IDs in results must be strictly unique!"
    assert len(parent_ids) == 3
    assert parent_ids[0] == "p1"
    assert parents[0]["is_selected"] is True


def test_vector_store_save_and_load(sample_vectors_and_metadata):
    """Test saving to disk and reloading vector index."""
    vectors, metadata, dim = sample_vectors_and_metadata
    store = FAISSVectorStore(dimension=dim, metric="cosine")
    store.add(vectors, metadata)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_index"
        store.save(save_path)

        assert (save_path / "index.faiss").exists()
        assert (save_path / "metadata.json").exists()
        assert (save_path / "manifest.json").exists()

        # Reload
        reloaded = FAISSVectorStore.load(save_path)
        assert reloaded.size == store.size
        assert reloaded.dimension == dim

        # Compare search outputs
        orig_res = store.search_chunks(vectors[1], top_k=2)
        reloaded_res = reloaded.search_chunks(vectors[1], top_k=2)

        assert orig_res[0][0]["chunk_id"] == reloaded_res[0][0]["chunk_id"]
        assert pytest.approx(orig_res[0][1], rel=1e-4) == reloaded_res[0][1]


def test_dimension_mismatch_raises_error():
    """Test that adding vectors with wrong dimension raises ValueError."""
    store = FAISSVectorStore(dimension=64)
    bad_vecs = np.zeros((2, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="Embedding dimension 32 does not match index dimension 64"):
        store.add(bad_vecs, [{"id": 1}, {"id": 2}])

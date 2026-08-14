"""Unit tests for embedding model abstraction and encoders."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.embeddings import SentenceTransformerEmbedder, get_embedder


@pytest.fixture(scope="module")
def embedder():
    """Shared lightweight embedder instance for fast testing."""
    return SentenceTransformerEmbedder(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        normalize=True,
    )


def test_embedder_dimension_and_properties(embedder):
    """Test that embedding model has valid dimension and properties."""
    assert embedder.dimension == 384
    assert embedder.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def test_encode_queries_produces_normalized_vectors(embedder):
    """Test that encode_queries returns L2-normalized float32 vectors."""
    queries = ["भारत की राजधानी", "Capital of India"]
    vectors = embedder.encode_queries(queries, batch_size=2)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32

    # Check L2 normalization (norm should be ~1.0)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-4)


def test_encode_documents_batching(embedder):
    """Test document encoding with batching across multiple languages."""
    docs = [
        "नई दिल्ली भारत की राजधानी है।",
        "New Delhi is the capital of India.",
        "मुंबई महाराष्ट्र की राजधानी है।",
        "कोलकाता पश्चिम बंगाल की राजधानी है।",
        "சென்னை தமிழ்நாட்டின் தலைநகரம்.",
    ]
    vectors = embedder.encode_documents(docs, batch_size=2)

    assert vectors.shape == (5, 384)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, np.ones(5), atol=1e-4)


def test_empty_input_handling(embedder):
    """Test that empty query and document lists return empty arrays without error."""
    empty_q = embedder.encode_queries([])
    assert empty_q.shape == (0, 384)

    empty_d = embedder.encode_documents([])
    assert empty_d.shape == (0, 384)


def test_get_embedder_factory():
    """Test get_embedder factory resolution."""
    config = {
        "embedding": {
            "device": "cpu",
            "normalize": True,
        }
    }
    instance = get_embedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", config)
    assert isinstance(instance, SentenceTransformerEmbedder)
    assert instance.dimension == 384

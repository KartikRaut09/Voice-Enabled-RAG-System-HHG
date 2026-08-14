"""Embedding models and abstraction layer for multilingual dense retrieval.

Provides a unified Embedder protocol and implementations supporting query/document
prefixes (e.g., E5-style), batching, L2 normalization for cosine similarity, and CPU inference.
"""

from __future__ import annotations

from typing import Any, Protocol
import numpy as np


class Embedder(Protocol):
    """Unified protocol for dense embedding models."""

    dimension: int
    model_name: str

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of search queries into normalized dense vectors."""
        ...

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of document passages/chunks into normalized dense vectors."""
        ...


class SentenceTransformerEmbedder:
    """Embedder implementation powered by SentenceTransformers.

    Automatically handles L2 normalization, batching, and model-specific prefixes
    (e.g., 'query: ' and 'passage: ' for intfloat/multilingual-e5-* models).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        query_prefix: str = "",
        document_prefix: str = "",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize

        # Automatic prefix detection for known model families
        if "multilingual-e5" in model_name.lower():
            self.query_prefix = query_prefix or "query: "
            self.document_prefix = document_prefix or "passage: "
        elif "bge" in model_name.lower() and "bge-m3" not in model_name.lower():
            self.query_prefix = query_prefix or "Represent this sentence for searching relevant passages: "
            self.document_prefix = document_prefix or ""
        else:
            self.query_prefix = query_prefix
            self.document_prefix = document_prefix

        # Lazy load model
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode queries with query prefix and L2 normalization."""
        if not queries:
            return np.empty((0, self.dimension), dtype=np.float32)

        prefixed_queries = [f"{self.query_prefix}{q}".strip() for q in queries]
        embeddings = self.model.encode(
            prefixed_queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode documents with document prefix and L2 normalization."""
        if not documents:
            return np.empty((0, self.dimension), dtype=np.float32)

        prefixed_docs = [f"{self.document_prefix}{d}".strip() for d in documents]
        embeddings = self.model.encode(
            prefixed_docs,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


def get_embedder(model_name: str, config: dict[str, Any] | None = None) -> Embedder:
    """Instantiate and return an Embedder instance."""
    cfg = config or {}
    emb_cfg = cfg.get("embedding", {})
    device = emb_cfg.get("device", "cpu")
    normalize = emb_cfg.get("normalize", True)

    return SentenceTransformerEmbedder(
        model_name=model_name,
        device=device,
        normalize=normalize,
    )

"""
Retrieval utilities for FAISS vector search and BM25 lexical retrieval.
Handles index creation, vector search, hybrid fusion, and metadata mapping.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np


@dataclass
class RetrievalResult:
    """A single retrieval result with passage text, score, and metadata."""

    passage_text: str
    score: float
    chunk_id: str = ""
    passage_index: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


# ── FAISS Index Manager ──


class FAISSIndex:
    """Manages a FAISS vector index with metadata mapping.

    Stores passage texts and metadata alongside the vector index
    so that retrieval results include full provenance.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "IndexFlatIP",
        nprobe: int | None = None,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index(dimension, index_type)
        if nprobe and hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        self.texts: list[str] = []
        self.metadata: list[dict[str, Any]] = []

    def _create_index(self, dim: int, index_type: str) -> faiss.Index:
        """Create a FAISS index of the specified type."""
        if index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(dim)
        elif index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(dim)
        elif index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatIP(dim)
            return faiss.IndexIVFFlat(quantizer, dim, 100)
        elif index_type == "IndexHNSWFlat":
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
            return index
        else:
            raise ValueError(f"Unknown FAISS index type: {index_type}")

    def add(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors with associated texts and metadata.

        Args:
            embeddings: Numpy array of shape (N, dimension).
            texts: List of N passage texts.
            metadata: Optional list of N metadata dicts.
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        # Train if needed (IVF indexes)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(embeddings.astype(np.float32))

        self.index.add(embeddings.astype(np.float32))
        self.texts.extend(texts)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Search for the most similar vectors.

        Args:
            query_embedding: Query vector of shape (1, dimension) or (dimension,).
            top_k: Number of results to return.

        Returns:
            Sorted list of RetrievalResult objects.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                RetrievalResult(
                    passage_text=self.texts[idx],
                    score=float(score),
                    passage_index=idx,
                    metadata=self.metadata[idx] if idx < len(self.metadata) else {},
                )
            )

        return results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> list[list[RetrievalResult]]:
        """Batch search for multiple queries.

        Args:
            query_embeddings: Array of shape (Q, dimension).
            top_k: Number of results per query.

        Returns:
            List of lists of RetrievalResult.
        """
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embeddings.astype(np.float32), k)

        all_results = []
        for q_scores, q_indices in zip(scores, indices):
            results = []
            for score, idx in zip(q_scores, q_indices):
                if idx < 0:
                    continue
                results.append(
                    RetrievalResult(
                        passage_text=self.texts[idx],
                        score=float(score),
                        passage_index=idx,
                        metadata=self.metadata[idx] if idx < len(self.metadata) else {},
                    )
                )
            all_results.append(results)

        return all_results

    @property
    def size(self) -> int:
        return self.index.ntotal

    def save(self, directory: str | Path) -> None:
        """Save the index, texts, and metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss_index.bin"))
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSIndex":
        """Load a saved index from disk."""
        directory = Path(directory)
        index = faiss.read_index(str(directory / "faiss_index.bin"))
        with open(directory / "metadata.pkl", "rb") as f:
            data = pickle.load(f)

        obj = cls.__new__(cls)
        obj.index = index
        obj.dimension = index.d
        obj.index_type = type(index).__name__
        obj.texts = data["texts"]
        obj.metadata = data["metadata"]
        return obj


# ── BM25 Retrieval ──


class BM25Index:
    """Lightweight BM25 retrieval using rank_bm25."""

    def __init__(self, texts: list[str], metadata: list[dict[str, Any]] | None = None):
        from rank_bm25 import BM25Okapi

        self.texts = texts
        self.metadata = metadata or [{}] * len(texts)
        tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                RetrievalResult(
                    passage_text=self.texts[idx],
                    score=float(scores[idx]),
                    passage_index=int(idx),
                    metadata=self.metadata[idx] if idx < len(self.metadata) else {},
                )
            )
        return results


# ── Hybrid Fusion (RRF) ──


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievalResult]:
    """Fuse dense and BM25 results using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i) for each list where the document appears.
    """
    scores: dict[int, float] = {}
    result_map: dict[int, RetrievalResult] = {}

    for rank, r in enumerate(dense_results):
        idx = r.passage_index
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        result_map[idx] = r

    for rank, r in enumerate(bm25_results):
        idx = r.passage_index
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        if idx not in result_map:
            result_map[idx] = r

    sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for idx in sorted_indices[:top_k]:
        r = result_map[idx]
        results.append(
            RetrievalResult(
                passage_text=r.passage_text,
                score=scores[idx],
                passage_index=r.passage_index,
                metadata=r.metadata,
            )
        )
    return results


# ── Timed Retrieval ──


def timed_search(
    index: FAISSIndex,
    query_embedding: np.ndarray,
    top_k: int = 10,
) -> tuple[list[RetrievalResult], float]:
    """Search with timing.

    Returns:
        Tuple of (results, latency_ms).
    """
    t0 = time.perf_counter()
    results = index.search(query_embedding, top_k=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms

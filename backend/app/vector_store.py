"""Persistent vector store implementation based on FAISS.

Provides fast cosine similarity search with exact parent-passage deduplication,
metadata mapping, and serialization to disk.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import faiss
import numpy as np


class FAISSVectorStore:
    """Lightweight persistent vector store powered by FAISS IndexFlatIP."""

    def __init__(self, dimension: int, metric: str = "cosine") -> None:
        self.dimension = dimension
        self.metric = metric
        # IndexFlatIP calculates inner product. With L2-normalized vectors, inner product == cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        """Return total number of vectors indexed."""
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, metadata_list: list[dict[str, Any]]) -> None:
        """Add dense vectors and corresponding metadata to the index."""
        if embeddings.shape[0] != len(metadata_list):
            raise ValueError(f"Mismatch between embeddings count ({embeddings.shape[0]}) and metadata count ({len(metadata_list)})")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")

        # Ensure float32 format
        vectors = embeddings.astype(np.float32)
        self.index.add(vectors)
        self.metadata.extend(metadata_list)

    def search_chunks(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[dict[str, Any], float]]:
        """Search top-k chunks by cosine similarity."""
        if self.index.ntotal == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype(np.float32)
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, top_k)
        results: list[tuple[dict[str, Any], float]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))

        return results

    def search_parent_passages(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        fetch_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Search nearest chunks and deduplicate by parent_passage_id.

        Collapses multiple chunks derived from the same parent passage into the single
        highest-scoring entry to prevent chunking fragmentation from skewing parent retrieval.
        """
        if self.index.ntotal == 0:
            return []

        # Retrieve a wider candidate pool of chunks for deduplication
        candidate_k = min(max(fetch_k, top_k * 5), self.index.ntotal)
        chunk_results = self.search_chunks(query_vector, top_k=candidate_k)

        seen_parent_ids: set[str] = set()
        unique_parent_results: list[dict[str, Any]] = []

        for meta, score in chunk_results:
            parent_id = meta.get("parent_passage_id") or meta.get("passage_id") or meta.get("chunk_id")
            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                unique_parent_results.append(
                    {
                        "parent_passage_id": parent_id,
                        "score": score,
                        "chunk_id": meta.get("chunk_id"),
                        "query_id": meta.get("query_id"),
                        "language": meta.get("language"),
                        "is_selected": meta.get("is_selected", False),
                        "chunk_strategy": meta.get("chunk_strategy"),
                    }
                )
                if len(unique_parent_results) >= top_k:
                    break

        return unique_parent_results

    def save(self, directory: Path | str) -> None:
        """Persist index, metadata, and manifest to directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / "index.faiss"
        metadata_file = dir_path / "metadata.json"
        manifest_file = dir_path / "manifest.json"

        faiss.write_index(self.index, str(index_file))

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        manifest = {
            "dimension": self.dimension,
            "metric": self.metric,
            "total_vectors": self.index.ntotal,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, directory: Path | str) -> FAISSVectorStore:
        """Load a persisted FAISS index and metadata from directory."""
        dir_path = Path(directory)
        index_file = dir_path / "index.faiss"
        metadata_file = dir_path / "metadata.json"
        manifest_file = dir_path / "manifest.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Missing index or metadata files in {dir_path}")

        index = faiss.read_index(str(index_file))
        dimension = index.d

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        metric = "cosine"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                metric = manifest.get("metric", "cosine")

        store = cls(dimension=dimension, metric=metric)
        store.index = index
        store.metadata = metadata
        return store

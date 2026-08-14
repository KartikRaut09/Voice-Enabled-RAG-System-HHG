"""Multilingual cross-encoder reranking module for hybrid retrieval candidates.

Implements Reranker protocol, CrossEncoderReranker wrapper, and candidate reranking pipeline.
"""

from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    """Protocol for neural cross-encoder rerankers."""

    def rank(self, query: str, documents: list[str], batch_size: int = 16) -> list[float]:
        """Compute relevance scores for a list of candidate document strings given a query."""
        ...


class CrossEncoderReranker:
    """Multilingual neural cross-encoder reranker."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-load the underlying CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rank(self, query: str, documents: list[str], batch_size: int = 16) -> list[float]:
        """Compute relevance scores for query-document pairs in batches."""
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        raw_scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # Ensure float outputs (handle scalar or numpy array output)
        if hasattr(raw_scores, "tolist"):
            scores_list = raw_scores.tolist()
        else:
            scores_list = list(raw_scores)

        return [float(s) for s in scores_list]


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    reranker: Reranker,
    rerank_top_k: int = 20,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Rerank top-N candidate items using a neural cross-encoder.

    Reranks the top `rerank_top_k` candidates and appends any remaining candidates
    at the tail to preserve overall candidate pool depth.
    """
    if not candidates or not query.strip():
        return list(candidates)

    rerank_slice = candidates[:rerank_top_k]
    tail_slice = candidates[rerank_top_k:]

    doc_texts = [str(c.get("text", "") or c.get("english_text", "") or "") for c in rerank_slice]
    scores = reranker.rank(query, doc_texts, batch_size=batch_size)

    reranked_items = []
    for item, score in zip(rerank_slice, scores):
        updated = dict(item)
        updated["rerank_score"] = round(score, 6)
        reranked_items.append(updated)

    # Sort reranked items descending by rerank score
    reranked_items.sort(key=lambda x: x["rerank_score"], reverse=True)

    return reranked_items + tail_slice

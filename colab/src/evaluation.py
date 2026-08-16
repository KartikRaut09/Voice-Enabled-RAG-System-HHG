"""
Evaluation metrics for retrieval quality.
Computes Recall@K, MRR, and generates comparison tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RetrievalEvalResult:
    """Result of evaluating retrieval quality on a set of queries."""

    config_name: str
    num_queries: int
    recall_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    mean_num_results: float = 0.0
    details: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "config": self.config_name,
            "num_queries": self.num_queries,
            "mrr": round(self.mrr, 4),
        }
        for k, v in sorted(self.recall_at.items()):
            result[f"recall@{k}"] = round(v, 4)
        return result


def compute_recall_at_k(
    retrieved_indices: list[int],
    relevant_indices: set[int],
    k: int,
) -> float:
    """Compute Recall@K for a single query.

    Args:
        retrieved_indices: Ordered list of retrieved passage indices.
        relevant_indices: Set of ground-truth relevant passage indices.
        k: Cutoff rank.

    Returns:
        Recall@K value in [0, 1].
    """
    if not relevant_indices:
        return 0.0
    retrieved_at_k = set(retrieved_indices[:k])
    hits = len(retrieved_at_k & relevant_indices)
    return hits / len(relevant_indices)


def compute_mrr(
    retrieved_indices: list[int],
    relevant_indices: set[int],
) -> float:
    """Compute Mean Reciprocal Rank for a single query.

    Args:
        retrieved_indices: Ordered list of retrieved passage indices.
        relevant_indices: Set of ground-truth relevant passage indices.

    Returns:
        Reciprocal rank (1/rank of first relevant result), or 0 if none found.
    """
    for rank, idx in enumerate(retrieved_indices, 1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    queries: list[str],
    query_relevant_indices: list[set[int]],
    search_fn,
    recall_at_k: list[int] | None = None,
    config_name: str = "default",
) -> RetrievalEvalResult:
    """Evaluate retrieval quality over a set of queries.

    Args:
        queries: List of query strings.
        query_relevant_indices: For each query, the set of passage indices
                                that are relevant (ground truth).
        search_fn: Callable that takes a query string and returns a list
                   of (passage_index, score) tuples or RetrievalResult objects.
        recall_at_k: List of K values for Recall@K. Default: [1, 5, 10].
        config_name: Name for this evaluation configuration.

    Returns:
        RetrievalEvalResult with aggregated metrics.
    """
    if recall_at_k is None:
        recall_at_k = [1, 5, 10]

    max_k = max(recall_at_k)
    all_recalls: dict[int, list[float]] = {k: [] for k in recall_at_k}
    all_mrr: list[float] = []
    details: list[dict[str, Any]] = []

    for q_idx, (query, relevant) in enumerate(zip(queries, query_relevant_indices)):
        # Get retrieval results
        results = search_fn(query)

        # Extract passage indices from results
        retrieved_indices = []
        for r in results:
            if hasattr(r, "passage_index"):
                retrieved_indices.append(r.passage_index)
            elif isinstance(r, tuple):
                retrieved_indices.append(r[0])
            elif isinstance(r, dict):
                retrieved_indices.append(r.get("passage_index", -1))

        # Compute metrics
        mrr_val = compute_mrr(retrieved_indices, relevant)
        all_mrr.append(mrr_val)

        query_detail: dict[str, Any] = {
            "query_index": q_idx,
            "num_relevant": len(relevant),
            "num_retrieved": len(retrieved_indices),
            "mrr": mrr_val,
        }

        for k in recall_at_k:
            r_at_k = compute_recall_at_k(retrieved_indices, relevant, k)
            all_recalls[k].append(r_at_k)
            query_detail[f"recall@{k}"] = r_at_k

        details.append(query_detail)

    # Aggregate
    recall_means = {k: float(np.mean(vals)) if vals else 0.0 for k, vals in all_recalls.items()}
    mrr_mean = float(np.mean(all_mrr)) if all_mrr else 0.0

    return RetrievalEvalResult(
        config_name=config_name,
        num_queries=len(queries),
        recall_at=recall_means,
        mrr=mrr_mean,
        mean_num_results=float(np.mean([d["num_retrieved"] for d in details])),
        details=details,
    )


def compare_configurations(results: list[RetrievalEvalResult]) -> list[dict[str, Any]]:
    """Generate a comparison table from multiple evaluation results.

    Args:
        results: List of RetrievalEvalResult objects.

    Returns:
        List of summary dicts suitable for DataFrame or table display.
    """
    return [r.summary() for r in results]

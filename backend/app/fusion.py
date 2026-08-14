"""Hybrid retrieval fusion strategies for Dense and BM25 search.

Implements Reciprocal Rank Fusion (RRF), Min-Max Normalized Weighted Score Fusion,
and parent-passage score aggregation.
"""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Combine Dense and BM25 ranked lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF(d) = sum_{c in channels} 1.0 / (rrf_k + rank_c(d))
    where rank_c(d) is 1-indexed rank within channel c. Preserves items appearing
    in dense-only, BM25-only, or both channels.
    """
    scores: dict[str, float] = {}
    doc_lookup: dict[str, dict[str, Any]] = {}
    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    # Dense channel
    for rank_idx, item in enumerate(dense_results, start=1):
        doc_key = item.get("chunk_id") or item.get("parent_passage_id") or str(id(item))
        dense_ranks[doc_key] = rank_idx
        scores[doc_key] = scores.get(doc_key, 0.0) + (1.0 / (rrf_k + rank_idx))
        if doc_key not in doc_lookup:
            doc_lookup[doc_key] = dict(item)

    # BM25 channel
    for rank_idx, item in enumerate(bm25_results, start=1):
        doc_key = item.get("chunk_id") or item.get("parent_passage_id") or str(id(item))
        bm25_ranks[doc_key] = rank_idx
        scores[doc_key] = scores.get(doc_key, 0.0) + (1.0 / (rrf_k + rank_idx))
        if doc_key not in doc_lookup:
            doc_lookup[doc_key] = dict(item)

    if not scores:
        return []

    # Sort descending by RRF score
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    fused_results: list[dict[str, Any]] = []
    for k in sorted_keys:
        fused_item = dict(doc_lookup[k])
        fused_item["fusion_score"] = round(scores[k], 6)
        fused_item["dense_rank"] = dense_ranks.get(k)
        fused_item["bm25_rank"] = bm25_ranks.get(k)
        fused_results.append(fused_item)

    return fused_results


def _min_max_normalize(results: list[dict[str, Any]], score_key: str = "score") -> dict[str, float]:
    """Min-max normalize scores within a single retrieval channel to [0.0, 1.0]."""
    if not results:
        return {}

    raw_scores = [float(r.get(score_key, 0.0)) for r in results]
    min_s = min(raw_scores)
    max_s = max(raw_scores)
    spread = max_s - min_s

    normalized: dict[str, float] = {}
    for r in results:
        doc_key = r.get("chunk_id") or r.get("parent_passage_id") or str(id(r))
        raw = float(r.get(score_key, 0.0))
        norm_val = 1.0 if spread == 0.0 else (raw - min_s) / spread
        normalized[doc_key] = norm_val

    return normalized


def weighted_score_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    dense_weight: float = 0.7,
) -> list[dict[str, Any]]:
    """Combine Dense and BM25 results using Min-Max Normalized Weighted Score Fusion.

    Formula: score(d) = alpha * norm_dense(d) + (1.0 - alpha) * norm_bm25(d)
    where alpha is dense_weight. Missing channels contribute 0.0 normalized score.
    """
    alpha = max(0.0, min(1.0, float(dense_weight)))
    beta = 1.0 - alpha

    norm_dense = _min_max_normalize(dense_results)
    norm_bm25 = _min_max_normalize(bm25_results)

    all_keys = set(norm_dense.keys()).union(set(norm_bm25.keys()))
    doc_lookup: dict[str, dict[str, Any]] = {}

    for item in dense_results:
        k = item.get("chunk_id") or item.get("parent_passage_id") or str(id(item))
        if k not in doc_lookup:
            doc_lookup[k] = dict(item)

    for item in bm25_results:
        k = item.get("chunk_id") or item.get("parent_passage_id") or str(id(item))
        if k not in doc_lookup:
            doc_lookup[k] = dict(item)

    scores: dict[str, float] = {}
    for k in all_keys:
        s_dense = norm_dense.get(k, 0.0)
        s_bm25 = norm_bm25.get(k, 0.0)
        scores[k] = (alpha * s_dense) + (beta * s_bm25)

    if not scores:
        return []

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    fused_results: list[dict[str, Any]] = []
    for k in sorted_keys:
        fused_item = dict(doc_lookup[k])
        fused_item["fusion_score"] = round(scores[k], 6)
        fused_item["norm_dense_score"] = round(norm_dense.get(k, 0.0), 4)
        fused_item["norm_bm25_score"] = round(norm_bm25.get(k, 0.0), 4)
        fused_results.append(fused_item)

    return fused_results


def aggregate_parent_passages(
    candidate_results: list[dict[str, Any]],
    top_k: int = 10,
    aggregation_method: str = "max",
) -> list[dict[str, Any]]:
    """Aggregate chunk-level candidate scores into deduplicated parent passages.

    Default aggregation is 'max' score across all chunks belonging to the same parent_passage_id,
    preventing passages with more chunks from receiving an unfair cumulative advantage.
    """
    if not candidate_results:
        return []

    # Map parent_passage_id -> list of chunk results
    parent_groups: dict[str, list[dict[str, Any]]] = {}
    for item in candidate_results:
        pid = item.get("parent_passage_id") or item.get("passage_id") or item.get("chunk_id")
        if pid not in parent_groups:
            parent_groups[pid] = []
        parent_groups[pid].append(item)

    parent_results: list[dict[str, Any]] = []
    for pid, chunks in parent_groups.items():
        score_keys = ["rerank_score", "fusion_score", "score"]
        active_score_key = next((k for k in score_keys if k in chunks[0]), "score")

        scores = [float(c.get(active_score_key, 0.0)) for c in chunks]
        if aggregation_method == "mean":
            agg_score = sum(scores) / len(scores) if scores else 0.0
        else:  # default "max"
            agg_score = max(scores) if scores else 0.0

        best_chunk = max(chunks, key=lambda c: float(c.get(active_score_key, 0.0)))
        parent_item = dict(best_chunk)
        parent_item["parent_passage_id"] = pid
        parent_item["score"] = round(agg_score, 6)
        parent_results.append(parent_item)

    # Sort descending by aggregated score
    parent_results.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
    return parent_results[:top_k]

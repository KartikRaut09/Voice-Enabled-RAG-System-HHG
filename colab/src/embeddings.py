"""
Embedding utilities for multilingual retrieval experiments.
Handles model loading, batch encoding, normalization, and benchmarking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EmbeddingBenchmarkResult:
    """Result of benchmarking a single embedding model."""

    model_name: str
    dimension: int
    num_texts: int
    total_time_s: float
    throughput_texts_per_s: float
    avg_latency_ms: float
    batch_size: int
    normalize: bool
    model_size_mb: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "throughput": round(self.throughput_texts_per_s, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_time_s": round(self.total_time_s, 3),
            "model_size_mb": self.model_size_mb,
        }


class EmbeddingModel:
    """Wrapper around a sentence-transformer embedding model.

    Provides consistent batch encoding, optional normalization,
    query/passage prefix handling (for E5-family models), and
    benchmarking utilities.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        batch_size: int = 64,
        normalize: bool = True,
        max_seq_length: int = 512,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length

        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode_queries(self, queries: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode query texts with the query prefix."""
        prefixed = [self.query_prefix + q for q in queries]
        return self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )

    def encode_passages(self, passages: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode passage texts with the passage prefix."""
        prefixed = [self.passage_prefix + p for p in passages]
        return self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )

    def encode_raw(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode texts without any prefix."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )

    def get_model_size_mb(self) -> float:
        """Estimate model size in megabytes from parameter count."""
        try:
            total_params = sum(
                p.numel() for p in self.model[0].auto_model.parameters()
            )
            return round(total_params * 4 / (1024 * 1024), 1)  # float32
        except Exception:
            return -1.0

    def benchmark(
        self,
        texts: list[str],
        mode: str = "passage",
    ) -> EmbeddingBenchmarkResult:
        """Benchmark encoding throughput and latency.

        Args:
            texts: List of texts to encode.
            mode: ``"passage"`` or ``"query"`` — determines prefix.

        Returns:
            EmbeddingBenchmarkResult with timing data.
        """
        encode_fn = self.encode_passages if mode == "passage" else self.encode_queries

        # Warmup
        _ = encode_fn(texts[:min(8, len(texts))])

        # Timed run
        t0 = time.perf_counter()
        _ = encode_fn(texts)
        elapsed = time.perf_counter() - t0

        n = len(texts)
        return EmbeddingBenchmarkResult(
            model_name=self.model_name,
            dimension=self.dimension,
            num_texts=n,
            total_time_s=elapsed,
            throughput_texts_per_s=n / elapsed if elapsed > 0 else 0,
            avg_latency_ms=(elapsed / n * 1000) if n > 0 else 0,
            batch_size=self.batch_size,
            normalize=self.normalize,
            model_size_mb=self.get_model_size_mb(),
        )


def benchmark_models(
    model_names: list[str],
    texts: list[str],
    batch_size: int = 64,
    normalize: bool = True,
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
) -> list[dict[str, Any]]:
    """Benchmark multiple embedding models on the same texts.

    Args:
        model_names: List of Hugging Face model identifiers.
        texts: Sample texts to encode.
        batch_size: Batch size for all models.
        normalize: Whether to L2-normalize embeddings.
        query_prefix: Prefix for query encoding.
        passage_prefix: Prefix for passage encoding.

    Returns:
        List of summary dictionaries for comparison.
    """
    results = []
    for name in model_names:
        try:
            print(f"  Benchmarking: {name}...")
            model = EmbeddingModel(
                model_name=name,
                batch_size=batch_size,
                normalize=normalize,
                query_prefix=query_prefix,
                passage_prefix=passage_prefix,
            )
            result = model.benchmark(texts, mode="passage")
            results.append(result.summary())
            # Free memory
            del model
            import gc; gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            results.append({"model": name, "error": str(e)})

    return results

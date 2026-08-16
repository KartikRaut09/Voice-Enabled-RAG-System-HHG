"""
Latency measurement utilities for RAG pipeline benchmarking.
Handles timing, percentile calculations, and benchmark orchestration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class LatencyMeasurement:
    """A single latency measurement with component breakdown."""

    label: str
    total_ms: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class LatencyBenchmarkResult:
    """Result of a latency benchmark run."""

    name: str
    num_measurements: int
    measurements_ms: list[float]
    percentiles: dict[str, float] = field(default_factory=dict)
    component_percentiles: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "benchmark": self.name,
            "n": self.num_measurements,
            "mean_ms": round(float(np.mean(self.measurements_ms)), 2),
            "min_ms": round(float(np.min(self.measurements_ms)), 2),
            "max_ms": round(float(np.max(self.measurements_ms)), 2),
        }
        result.update(self.percentiles)
        return result


class LatencyTimer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000


def compute_percentiles(
    values: list[float],
    percentiles: list[int] | None = None,
) -> dict[str, float]:
    """Compute named percentiles from a list of values.

    Args:
        values: List of measured values (e.g., latencies in ms).
        percentiles: Percentile values to compute. Default: [50, 70, 90, 95, 99, 100].

    Returns:
        Dictionary mapping ``"P50"`` etc. to computed values.
    """
    if not values:
        return {}
    if percentiles is None:
        percentiles = [50, 70, 90, 95, 99, 100]

    result = {}
    for p in percentiles:
        if p == 100:
            result[f"P{p}"] = round(float(np.max(values)), 2)
        else:
            result[f"P{p}"] = round(float(np.percentile(values, p)), 2)
    return result


def run_latency_benchmark(
    name: str,
    fn: Callable[[], float | dict[str, float]],
    num_warmup: int = 10,
    num_runs: int = 100,
    percentiles: list[int] | None = None,
) -> LatencyBenchmarkResult:
    """Run a latency benchmark by executing a callable repeatedly.

    Args:
        name: Human-readable benchmark name.
        fn: Callable that performs the operation to benchmark.
            Should return either:
            - A float (total latency in ms), or
            - A dict mapping component names to latencies in ms,
              with a ``"total"`` key.
        num_warmup: Number of warmup iterations (not measured).
        num_runs: Number of measured iterations.
        percentiles: Percentile values to report.

    Returns:
        LatencyBenchmarkResult with timing statistics.
    """
    # Warmup
    for _ in range(num_warmup):
        fn()

    # Measured runs
    all_totals: list[float] = []
    all_components: dict[str, list[float]] = {}

    for _ in range(num_runs):
        result = fn()

        if isinstance(result, dict):
            total = result.get("total", sum(result.values()))
            all_totals.append(total)
            for k, v in result.items():
                if k not in all_components:
                    all_components[k] = []
                all_components[k].append(v)
        else:
            all_totals.append(float(result))

    # Compute percentiles
    pcts = compute_percentiles(all_totals, percentiles)

    component_pcts = {}
    for comp_name, comp_values in all_components.items():
        component_pcts[comp_name] = compute_percentiles(comp_values, percentiles)

    return LatencyBenchmarkResult(
        name=name,
        num_measurements=len(all_totals),
        measurements_ms=all_totals,
        percentiles=pcts,
        component_percentiles=component_pcts,
    )


def benchmark_rag_pipeline(
    queries: list[str],
    embed_fn: Callable[[str], np.ndarray],
    search_fn: Callable[[np.ndarray], list],
    num_warmup: int = 5,
) -> LatencyBenchmarkResult:
    """Benchmark the core RAG retrieval pipeline (embedding + search).

    This measures **online latency only** — the time to go from a raw
    query string to retrieved passages.  Corpus preprocessing (chunking,
    embedding, index creation) is NOT included.

    Args:
        queries: List of query strings to benchmark.
        embed_fn: Callable that encodes a query string to a numpy vector.
        search_fn: Callable that takes an embedding and returns results.
        num_warmup: Number of warmup queries.

    Returns:
        LatencyBenchmarkResult with per-component timing.
    """
    # Warmup
    for q in queries[:num_warmup]:
        emb = embed_fn(q)
        _ = search_fn(emb)

    all_totals: list[float] = []
    embed_times: list[float] = []
    search_times: list[float] = []

    for query in queries:
        # Time embedding
        t0 = time.perf_counter()
        emb = embed_fn(query)
        embed_ms = (time.perf_counter() - t0) * 1000

        # Time search
        t1 = time.perf_counter()
        _ = search_fn(emb)
        search_ms = (time.perf_counter() - t1) * 1000

        total_ms = embed_ms + search_ms
        all_totals.append(total_ms)
        embed_times.append(embed_ms)
        search_times.append(search_ms)

    pcts = compute_percentiles(all_totals)
    component_pcts = {
        "embedding": compute_percentiles(embed_times),
        "search": compute_percentiles(search_times),
    }

    return LatencyBenchmarkResult(
        name="RAG Pipeline (Embedding + Search)",
        num_measurements=len(all_totals),
        measurements_ms=all_totals,
        percentiles=pcts,
        component_percentiles=component_pcts,
    )

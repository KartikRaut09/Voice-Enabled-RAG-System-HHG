"""
Chunking strategies for MSMARCO-XI passages.
Implements original, fixed-size, sentence-aware, semantic, and metadata-aware chunking.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single chunk with text and metadata."""

    text: str
    chunk_id: str = ""
    strategy: str = ""
    source_passage_id: str = ""
    query_id: str = ""
    language: str = ""
    is_selected: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_length(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class ChunkingResult:
    """Result of a chunking strategy applied to a corpus."""

    strategy_name: str
    chunks: list[Chunk]
    processing_time_s: float
    config: dict[str, Any]

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def avg_chunk_size(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(c.char_length for c in self.chunks) / len(self.chunks)

    @property
    def median_chunk_size(self) -> float:
        if not self.chunks:
            return 0.0
        sizes = sorted(c.char_length for c in self.chunks)
        n = len(sizes)
        if n % 2 == 1:
            return float(sizes[n // 2])
        return (sizes[n // 2 - 1] + sizes[n // 2]) / 2.0

    @property
    def min_chunk_size(self) -> int:
        return min((c.char_length for c in self.chunks), default=0)

    @property
    def max_chunk_size(self) -> int:
        return max((c.char_length for c in self.chunks), default=0)

    def summary(self) -> dict[str, Any]:
        """Generate a summary dictionary for comparison tables."""
        return {
            "strategy": self.strategy_name,
            "num_chunks": self.num_chunks,
            "avg_size": round(self.avg_chunk_size, 1),
            "median_size": round(self.median_chunk_size, 1),
            "min_size": self.min_chunk_size,
            "max_size": self.max_chunk_size,
            "processing_time_s": round(self.processing_time_s, 3),
        }


# ── Sentence Segmentation ──

# Regex-based sentence splitter that handles Indic scripts, abbreviations,
# decimal numbers, and common edge cases.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[।!?\.\?!])\s+|(?<=\n)\s*"
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex boundary detection.

    Handles Devanagari purna viram (।), Latin punctuation, and newlines.

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings (stripped, non-empty).
    """
    parts = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in parts if s.strip()]


# ── Strategy 1: Original Passage ──


def chunk_original(
    passages: list[dict[str, Any]],
    min_size: int = 50,
) -> ChunkingResult:
    """Use dataset-provided passage boundaries as chunks.

    Each passage becomes one chunk.  Passages shorter than ``min_size``
    characters are discarded.
    """
    t0 = time.perf_counter()
    chunks = []
    for i, p in enumerate(passages):
        text = p.get("text", "").strip()
        if len(text) < min_size:
            continue
        chunks.append(
            Chunk(
                text=text,
                chunk_id=f"orig_{i}",
                strategy="original",
                source_passage_id=str(p.get("passage_index", i)),
                query_id=str(p.get("query_id", "")),
                language=p.get("language", ""),
                is_selected=p.get("is_selected", 0),
                metadata={"url": p.get("url", "")},
            )
        )
    elapsed = time.perf_counter() - t0
    return ChunkingResult("original", chunks, elapsed, {"min_size": min_size})


# ── Strategy 2: Fixed-Size Chunking ──


def chunk_fixed(
    passages: list[dict[str, Any]],
    chunk_size: int = 700,
    overlap: int = 100,
    min_size: int = 50,
) -> ChunkingResult:
    """Split passages into fixed-size character windows with overlap."""
    t0 = time.perf_counter()
    chunks = []
    chunk_idx = 0
    for p in passages:
        text = p.get("text", "").strip()
        if len(text) < min_size:
            continue
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= min_size:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=f"fixed_{chunk_idx}",
                        strategy="fixed",
                        source_passage_id=str(p.get("passage_index", "")),
                        query_id=str(p.get("query_id", "")),
                        language=p.get("language", ""),
                        is_selected=p.get("is_selected", 0),
                    )
                )
                chunk_idx += 1
            step = chunk_size - overlap
            if step <= 0:
                step = max(1, chunk_size // 2)
            start += step
    elapsed = time.perf_counter() - t0
    return ChunkingResult(
        "fixed",
        chunks,
        elapsed,
        {"chunk_size": chunk_size, "overlap": overlap, "min_size": min_size},
    )


# ── Strategy 3: Sentence-Aware Chunking ──


def chunk_sentence(
    passages: list[dict[str, Any]],
    chunk_size: int = 700,
    overlap: int = 100,
    min_size: int = 50,
) -> ChunkingResult:
    """Pack sentences into chunks up to a character limit.

    Sentences are never split mid-sentence.  A new chunk begins when
    adding the next sentence would exceed ``chunk_size``.
    """
    t0 = time.perf_counter()
    chunks = []
    chunk_idx = 0

    for p in passages:
        text = p.get("text", "").strip()
        if len(text) < min_size:
            continue

        sentences = split_sentences(text)
        if not sentences:
            continue

        current_sents: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > chunk_size and current_sents:
                # Flush current chunk
                chunk_text = " ".join(current_sents).strip()
                if len(chunk_text) >= min_size:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id=f"sent_{chunk_idx}",
                            strategy="sentence",
                            source_passage_id=str(p.get("passage_index", "")),
                            query_id=str(p.get("query_id", "")),
                            language=p.get("language", ""),
                            is_selected=p.get("is_selected", 0),
                        )
                    )
                    chunk_idx += 1

                # Keep overlap sentences for context continuity
                overlap_sents: list[str] = []
                overlap_len = 0
                for s in reversed(current_sents):
                    if overlap_len + len(s) > overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                current_sents = overlap_sents
                current_len = overlap_len

            current_sents.append(sent)
            current_len += sent_len

        # Flush remainder
        if current_sents:
            chunk_text = " ".join(current_sents).strip()
            if len(chunk_text) >= min_size:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=f"sent_{chunk_idx}",
                        strategy="sentence",
                        source_passage_id=str(p.get("passage_index", "")),
                        query_id=str(p.get("query_id", "")),
                        language=p.get("language", ""),
                        is_selected=p.get("is_selected", 0),
                    )
                )
                chunk_idx += 1

    elapsed = time.perf_counter() - t0
    return ChunkingResult(
        "sentence",
        chunks,
        elapsed,
        {"chunk_size": chunk_size, "overlap": overlap, "min_size": min_size},
    )


# ── Strategy 4: Semantic Chunking ──


def chunk_semantic(
    passages: list[dict[str, Any]],
    encode_fn: Any = None,
    similarity_threshold: float = 0.5,
    max_chunk_size: int = 1200,
    min_size: int = 50,
) -> ChunkingResult:
    """Split at semantic boundaries detected by sentence embedding similarity.

    Args:
        passages: List of passage dicts.
        encode_fn: Callable that takes a list of strings and returns a
                   numpy array of embeddings.  If None, falls back to
                   sentence-aware chunking.
        similarity_threshold: Cosine similarity below this value triggers
                              a chunk boundary.
        max_chunk_size: Maximum chunk size in characters.
        min_size: Minimum chunk size.

    Returns:
        ChunkingResult with semantic chunks.
    """
    import numpy as np

    t0 = time.perf_counter()

    if encode_fn is None:
        # Fallback to sentence chunking if no encoder is provided
        result = chunk_sentence(passages, chunk_size=700, min_size=min_size)
        result.strategy_name = "semantic (fallback: sentence)"
        return result

    chunks = []
    chunk_idx = 0

    for p in passages:
        text = p.get("text", "").strip()
        if len(text) < min_size:
            continue

        sentences = split_sentences(text)
        if len(sentences) <= 1:
            chunks.append(
                Chunk(
                    text=text,
                    chunk_id=f"sem_{chunk_idx}",
                    strategy="semantic",
                    source_passage_id=str(p.get("passage_index", "")),
                    query_id=str(p.get("query_id", "")),
                    language=p.get("language", ""),
                    is_selected=p.get("is_selected", 0),
                )
            )
            chunk_idx += 1
            continue

        # Encode sentences
        embeddings = encode_fn(sentences)
        if embeddings is None or len(embeddings) == 0:
            continue

        # Compute adjacent cosine similarities
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = embeddings / norms
        sims = np.sum(normed[:-1] * normed[1:], axis=1)

        # Detect boundaries where similarity drops
        current_sents = [sentences[0]]
        current_len = len(sentences[0])

        for i, sim in enumerate(sims):
            next_sent = sentences[i + 1]
            next_len = len(next_sent)

            # Split if similarity is below threshold or chunk would be too large
            if (sim < similarity_threshold or current_len + next_len > max_chunk_size) and current_sents:
                chunk_text = " ".join(current_sents).strip()
                if len(chunk_text) >= min_size:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id=f"sem_{chunk_idx}",
                            strategy="semantic",
                            source_passage_id=str(p.get("passage_index", "")),
                            query_id=str(p.get("query_id", "")),
                            language=p.get("language", ""),
                            is_selected=p.get("is_selected", 0),
                            metadata={"boundary_similarity": float(sim)},
                        )
                    )
                    chunk_idx += 1
                current_sents = []
                current_len = 0

            current_sents.append(next_sent)
            current_len += next_len

        # Flush remainder
        if current_sents:
            chunk_text = " ".join(current_sents).strip()
            if len(chunk_text) >= min_size:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=f"sem_{chunk_idx}",
                        strategy="semantic",
                        source_passage_id=str(p.get("passage_index", "")),
                        query_id=str(p.get("query_id", "")),
                        language=p.get("language", ""),
                        is_selected=p.get("is_selected", 0),
                    )
                )
                chunk_idx += 1

    elapsed = time.perf_counter() - t0
    return ChunkingResult(
        "semantic",
        chunks,
        elapsed,
        {
            "similarity_threshold": similarity_threshold,
            "max_chunk_size": max_chunk_size,
            "min_size": min_size,
        },
    )


# ── Strategy 5: Metadata-Aware Chunking ──


def chunk_metadata_aware(
    passages: list[dict[str, Any]],
    chunk_size: int = 700,
    overlap: int = 100,
    min_size: int = 50,
) -> ChunkingResult:
    """Sentence-aware chunking that preserves and embeds passage metadata.

    Each chunk carries full provenance: query_id, query_type, language,
    passage_id, is_selected, and source URL.  The metadata is stored in
    the Chunk object but NOT injected into the text.
    """
    t0 = time.perf_counter()

    # Use sentence chunking as the base strategy
    base = chunk_sentence(passages, chunk_size=chunk_size, overlap=overlap, min_size=min_size)

    # Enrich each chunk with full metadata from the source passage
    passage_map = {}
    for p in passages:
        key = str(p.get("passage_index", ""))
        passage_map[key] = p

    for chunk in base.chunks:
        chunk.strategy = "metadata_aware"
        chunk.chunk_id = chunk.chunk_id.replace("sent_", "meta_")
        source = passage_map.get(chunk.source_passage_id, {})
        chunk.metadata = {
            "query_type": source.get("query_type", ""),
            "url": source.get("url", ""),
            "is_selected": source.get("is_selected", 0),
            "language": source.get("language", ""),
        }

    elapsed = time.perf_counter() - t0
    return ChunkingResult(
        "metadata_aware",
        base.chunks,
        elapsed,
        {"chunk_size": chunk_size, "overlap": overlap, "min_size": min_size},
    )


# ── Strategy Runner ──


def run_chunking(
    strategy: str,
    passages: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    encode_fn: Any = None,
) -> ChunkingResult:
    """Run a named chunking strategy.

    Args:
        strategy: One of ``"original"``, ``"fixed"``, ``"sentence"``,
                  ``"semantic"``, ``"metadata_aware"``.
        passages: List of passage dicts from ``dataset_utils.extract_passages``.
        config: Configuration dict (chunk_size, overlap, etc.).
        encode_fn: Encoder function for semantic chunking.

    Returns:
        ChunkingResult from the selected strategy.
    """
    cfg = config or {}
    chunk_size = cfg.get("chunk_size", 700)
    overlap = cfg.get("overlap", 100)
    min_size = cfg.get("min_chunk_size", 50)

    if strategy == "original":
        return chunk_original(passages, min_size=min_size)
    elif strategy == "fixed":
        return chunk_fixed(passages, chunk_size=chunk_size, overlap=overlap, min_size=min_size)
    elif strategy == "sentence":
        return chunk_sentence(passages, chunk_size=chunk_size, overlap=overlap, min_size=min_size)
    elif strategy == "semantic":
        return chunk_semantic(
            passages,
            encode_fn=encode_fn,
            similarity_threshold=cfg.get("similarity_threshold", 0.5),
            max_chunk_size=cfg.get("max_chunk_size", 1200),
            min_size=min_size,
        )
    elif strategy == "metadata_aware":
        return chunk_metadata_aware(passages, chunk_size=chunk_size, overlap=overlap, min_size=min_size)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def compare_strategies(
    passages: list[dict[str, Any]],
    strategies: list[str] | None = None,
    config: dict[str, Any] | None = None,
    encode_fn: Any = None,
) -> list[dict[str, Any]]:
    """Run multiple chunking strategies and return a comparison table.

    Returns:
        List of summary dicts, one per strategy.
    """
    if strategies is None:
        strategies = ["original", "fixed", "sentence", "metadata_aware"]

    results = []
    for strat in strategies:
        try:
            result = run_chunking(strat, passages, config=config, encode_fn=encode_fn)
            results.append(result.summary())
        except Exception as e:
            results.append({"strategy": strat, "error": str(e)})

    return results

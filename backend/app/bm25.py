"""BM25 lexical retrieval engine with multilingual Indic Unicode tokenization.

Implements standard Okapi BM25, inverted index, parent-passage deduplication,
and index serialization for Indic and multilingual text.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Protocol
import unicodedata


def tokenize_indic(text: str) -> list[str]:
    """Tokenize multilingual Indic and Latin text.

    Applies NFKC Unicode normalization, lowercasing, and extracts complete word
    tokens across Devanagari, Bengali, Tamil, Telugu, and other Indic scripts
    by including Indic vowel signs (matras), virama/halant, and joiners (\u0900-\u0D7F\u200C\u200D).
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).lower()
    # \u0900-\u0D7F covers all major Indic scripts and combining marks; \u200C\u200D are ZWNJ/ZWJ
    tokens = re.findall(r"[\w\u0900-\u0D7F\u200C\u200D]+", normalized, re.UNICODE)
    return tokens



class LexicalRetriever(Protocol):
    """Protocol for lexical retrieval engines."""

    def search_parent_passages(self, query: str, top_k: int = 10, fetch_k: int = 50) -> list[dict[str, Any]]:
        """Search and return deduplicated parent passages with BM25 scores."""
        ...


class BM25Index:
    """Okapi BM25 inverted index with parent-passage deduplication."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths: list[int] = []
        self.doc_frequencies: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self.metadata: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        """Return total number of indexed documents/chunks."""
        return self.corpus_size

    def build(self, documents: list[str], metadata: list[dict[str, Any]]) -> None:
        """Build the BM25 inverted index from a list of document strings and metadata."""
        if len(documents) != len(metadata):
            raise ValueError(f"Mismatch between documents count ({len(documents)}) and metadata count ({len(metadata)})")

        self.corpus_size = len(documents)
        self.metadata = list(metadata)
        self.doc_lengths = []
        self.doc_frequencies = {}
        self.inverted_index = {}

        total_length = 0

        # Build term frequencies and inverted index
        for doc_id, doc_text in enumerate(documents):
            tokens = tokenize_indic(doc_text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_id, count))

        self.avgdl = (total_length / self.corpus_size) if self.corpus_size > 0 else 0.0

        # Precompute standard Okapi BM25 IDF for all terms
        # IDF formula: ln((N - df + 0.5) / (df + 0.5) + 1.0)
        self.idf = {}
        for term, df in self.doc_frequencies.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def search_chunks(self, query: str, top_k: int = 10) -> list[tuple[dict[str, Any], float]]:
        """Search top-k chunks using Okapi BM25 scoring."""
        if self.corpus_size == 0 or not query.strip():
            return []

        query_tokens = tokenize_indic(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = {}

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            idf = self.idf.get(term, 0.0)
            if idf <= 0.0:
                continue

            postings = self.inverted_index[term]
            for doc_id, tf in postings:
                doc_len = self.doc_lengths[doc_id]
                # Okapi BM25 term weight
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl if self.avgdl > 0 else 1.0))
                term_score = idf * (numerator / denominator)
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score

        if not scores:
            return []

        # Sort descending by score
        sorted_doc_ids = sorted(scores.keys(), key=lambda did: scores[did], reverse=True)[:top_k]
        return [(self.metadata[did], scores[did]) for did in sorted_doc_ids]

    def search_parent_passages(
        self,
        query: str,
        top_k: int = 10,
        fetch_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Search nearest chunks and deduplicate by parent_passage_id.

        Collapses multiple chunks from the same parent passage into the single highest-scoring entry.
        """
        if self.corpus_size == 0 or not query.strip():
            return []

        candidate_k = min(max(fetch_k, top_k * 5), self.corpus_size)
        chunk_results = self.search_chunks(query, top_k=candidate_k)

        seen_parent_ids: set[str] = set()
        unique_parent_results: list[dict[str, Any]] = []

        for meta, score in chunk_results:
            parent_id = meta.get("parent_passage_id") or meta.get("passage_id") or meta.get("chunk_id")
            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                unique_parent_results.append(
                    {
                        "parent_passage_id": parent_id,
                        "score": round(score, 4),
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
        """Persist BM25 index, metadata, and manifest to directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_data = {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
            "doc_lengths": self.doc_lengths,
            "doc_frequencies": self.doc_frequencies,
            "idf": self.idf,
            "inverted_index": self.inverted_index,
        }

        with open(dir_path / "index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False)

        with open(dir_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        manifest = {
            "retrieval_type": "bm25",
            "k1": self.k1,
            "b": self.b,
            "tokenizer": "indic_unicode",
            "num_documents": self.corpus_size,
            "avg_doc_length": round(self.avgdl, 2),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(dir_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, directory: Path | str) -> BM25Index:
        """Load a persisted BM25 index and metadata from directory."""
        dir_path = Path(directory)
        index_file = dir_path / "index.json"
        metadata_file = dir_path / "metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Missing BM25 index or metadata in {dir_path}")

        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        store = cls(k1=index_data.get("k1", 1.5), b=index_data.get("b", 0.75))
        store.corpus_size = index_data["corpus_size"]
        store.avgdl = index_data["avgdl"]
        store.doc_lengths = index_data["doc_lengths"]
        store.doc_frequencies = index_data["doc_frequencies"]
        store.idf = index_data["idf"]
        store.inverted_index = {k: [tuple(x) for x in v] for k, v in index_data["inverted_index"].items()}
        store.metadata = metadata
        return store

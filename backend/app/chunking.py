"""Multi-strategy chunking implementations for Indic and multilingual text.

Strategies:
1. PassageChunker: Original passage as one intact chunk (baseline).
2. FixedChunker: Word-bounded fixed-size chunks for long passages.
3. OverlapChunker: Sliding-window word-bounded chunks with configurable overlap.
4. StructureAwareChunker: Paragraph and sentence boundary-aware splitting (supports Indic danda '।' and punctuation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Protocol


@dataclass
class Chunk:
    """Normalized internal representation of a document chunk."""

    chunk_id: str
    parent_passage_id: str
    text: str
    english_text: str
    chunk_index: int
    chunk_strategy: str
    query_id: int | None
    query_type: str | None
    language: str | None
    source_lang: str | None
    target_lang: str | None
    is_selected: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk dataclass to dictionary."""
        return asdict(self)


class Chunker(Protocol):
    """Protocol for chunking strategies."""

    def chunk_passage(self, passage: dict[str, Any], record_meta: dict[str, Any]) -> list[Chunk]:
        """Chunk a single passage into one or more Chunk objects."""
        ...


class PassageChunker:
    """Strategy 1: Passage-level baseline chunker.

    Leaves every source passage intact as exactly one chunk.
    Preserves all provenance and relevance labels.
    """

    strategy_name = "passage"

    def chunk_passage(self, passage: dict[str, Any], record_meta: dict[str, Any]) -> list[Chunk]:
        text = str(passage.get("text", "") or "").strip()
        eng_text = str(passage.get("english_text", "") or "").strip()
        if not text and not eng_text:
            return []

        passage_id = str(passage.get("passage_id", f"{record_meta.get('query_id')}_p0"))
        is_selected = bool(passage.get("is_selected", False))

        return [
            Chunk(
                chunk_id=f"{passage_id}_c0",
                parent_passage_id=passage_id,
                text=text,
                english_text=eng_text,
                chunk_index=0,
                chunk_strategy=self.strategy_name,
                query_id=record_meta.get("query_id"),
                query_type=record_meta.get("query_type"),
                language=passage.get("language") or record_meta.get("target_lang"),
                source_lang=record_meta.get("source_lang"),
                target_lang=record_meta.get("target_lang"),
                is_selected=is_selected,
            )
        ]


class FixedChunker:
    """Strategy 2: Word-bounded non-overlapping chunker.

    Splits long passages into consecutive non-overlapping windows of `max_words`.
    Passages <= max_words remain intact as a single chunk.
    """

    strategy_name = "fixed"

    def __init__(self, max_words: int = 80) -> None:
        self.max_words = max(10, int(max_words))

    def chunk_passage(self, passage: dict[str, Any], record_meta: dict[str, Any]) -> list[Chunk]:
        text = str(passage.get("text", "") or "").strip()
        eng_text = str(passage.get("english_text", "") or "").strip()
        if not text and not eng_text:
            return []

        passage_id = str(passage.get("passage_id", f"{record_meta.get('query_id')}_p0"))
        is_selected = bool(passage.get("is_selected", False))

        words = text.split()
        if len(words) <= self.max_words:
            return [
                Chunk(
                    chunk_id=f"{passage_id}_c0",
                    parent_passage_id=passage_id,
                    text=text,
                    english_text=eng_text,
                    chunk_index=0,
                    chunk_strategy=self.strategy_name,
                    query_id=record_meta.get("query_id"),
                    query_type=record_meta.get("query_type"),
                    language=passage.get("language") or record_meta.get("target_lang"),
                    source_lang=record_meta.get("source_lang"),
                    target_lang=record_meta.get("target_lang"),
                    is_selected=is_selected,
                )
            ]

        chunks: list[Chunk] = []
        chunk_idx = 0
        for i in range(0, len(words), self.max_words):
            chunk_words = words[i : i + self.max_words]
            chunk_text = " ".join(chunk_words)
            chunks.append(
                Chunk(
                    chunk_id=f"{passage_id}_c{chunk_idx}",
                    parent_passage_id=passage_id,
                    text=chunk_text,
                    english_text=eng_text,  # Keep passage-level English text as context
                    chunk_index=chunk_idx,
                    chunk_strategy=self.strategy_name,
                    query_id=record_meta.get("query_id"),
                    query_type=record_meta.get("query_type"),
                    language=passage.get("language") or record_meta.get("target_lang"),
                    source_lang=record_meta.get("source_lang"),
                    target_lang=record_meta.get("target_lang"),
                    is_selected=is_selected,
                )
            )
            chunk_idx += 1

        return chunks


class OverlapChunker:
    """Strategy 3: Word-bounded sliding window chunker with overlap.

    Creates overlapping chunk windows of size `max_words` with step `max_words - overlap_words`.
    Passages <= max_words remain intact as a single chunk without duplication.
    """

    strategy_name = "overlap"

    def __init__(self, max_words: int = 80, overlap_words: int = 20) -> None:
        self.max_words = max(10, int(max_words))
        # Ensure overlap is strictly less than max_words to prevent infinite loops
        self.overlap_words = min(int(overlap_words), self.max_words - 1)
        self.step = max(1, self.max_words - self.overlap_words)

    def chunk_passage(self, passage: dict[str, Any], record_meta: dict[str, Any]) -> list[Chunk]:
        text = str(passage.get("text", "") or "").strip()
        eng_text = str(passage.get("english_text", "") or "").strip()
        if not text and not eng_text:
            return []

        passage_id = str(passage.get("passage_id", f"{record_meta.get('query_id')}_p0"))
        is_selected = bool(passage.get("is_selected", False))

        words = text.split()
        if len(words) <= self.max_words:
            return [
                Chunk(
                    chunk_id=f"{passage_id}_c0",
                    parent_passage_id=passage_id,
                    text=text,
                    english_text=eng_text,
                    chunk_index=0,
                    chunk_strategy=self.strategy_name,
                    query_id=record_meta.get("query_id"),
                    query_type=record_meta.get("query_type"),
                    language=passage.get("language") or record_meta.get("target_lang"),
                    source_lang=record_meta.get("source_lang"),
                    target_lang=record_meta.get("target_lang"),
                    is_selected=is_selected,
                )
            ]

        chunks: list[Chunk] = []
        chunk_idx = 0
        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.max_words]
            chunk_text = " ".join(chunk_words)
            chunks.append(
                Chunk(
                    chunk_id=f"{passage_id}_c{chunk_idx}",
                    parent_passage_id=passage_id,
                    text=chunk_text,
                    english_text=eng_text,
                    chunk_index=chunk_idx,
                    chunk_strategy=self.strategy_name,
                    query_id=record_meta.get("query_id"),
                    query_type=record_meta.get("query_type"),
                    language=passage.get("language") or record_meta.get("target_lang"),
                    source_lang=record_meta.get("source_lang"),
                    target_lang=record_meta.get("target_lang"),
                    is_selected=is_selected,
                )
            )
            chunk_idx += 1
            if i + self.max_words >= len(words):
                break
            i += self.step

        return chunks


class StructureAwareChunker:
    """Strategy 4: Structure-aware chunker.

    Preserves paragraph and sentence boundaries (including Indic sentence terminators '।', '.', '?', '!').
    Accumulates full sentences up to `max_words`.
    Falls back to word-level splitting only if an individual sentence exceeds `max_words`.
    """

    strategy_name = "structure_aware"

    # Regex splits sentences while preserving punctuation attached:
    # Matches Indic danda (।), English period (.), question mark (?), exclamation (!), semicolon (;), or newlines
    SENTENCE_SPLIT_REGEX = re.compile(r"([^।\n.?!;]+[।\n.?!;]*|\n+)")

    def __init__(self, max_words: int = 80) -> None:
        self.max_words = max(10, int(max_words))

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences honoring Indic and Latin sentence boundaries."""
        matches = self.SENTENCE_SPLIT_REGEX.findall(text)
        sentences = [m.strip() for m in matches if m and m.strip()]
        return sentences if sentences else [text]

    def chunk_passage(self, passage: dict[str, Any], record_meta: dict[str, Any]) -> list[Chunk]:
        text = str(passage.get("text", "") or "").strip()
        eng_text = str(passage.get("english_text", "") or "").strip()
        if not text and not eng_text:
            return []

        passage_id = str(passage.get("passage_id", f"{record_meta.get('query_id')}_p0"))
        is_selected = bool(passage.get("is_selected", False))

        words = text.split()
        if len(words) <= self.max_words:
            return [
                Chunk(
                    chunk_id=f"{passage_id}_c0",
                    parent_passage_id=passage_id,
                    text=text,
                    english_text=eng_text,
                    chunk_index=0,
                    chunk_strategy=self.strategy_name,
                    query_id=record_meta.get("query_id"),
                    query_type=record_meta.get("query_type"),
                    language=passage.get("language") or record_meta.get("target_lang"),
                    source_lang=record_meta.get("source_lang"),
                    target_lang=record_meta.get("target_lang"),
                    is_selected=is_selected,
                )
            ]

        # Multi-sentence passage splitting
        sentences = self._split_into_sentences(text)
        chunks: list[Chunk] = []
        current_chunk_sentences: list[str] = []
        current_word_count = 0
        chunk_idx = 0

        for sent in sentences:
            sent_word_count = len(sent.split())

            # If a single sentence exceeds max_words, split it by words
            if sent_word_count > self.max_words:
                # Flush pending buffer first
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences).strip()
                    if chunk_text:
                        chunks.append(
                            Chunk(
                                chunk_id=f"{passage_id}_c{chunk_idx}",
                                parent_passage_id=passage_id,
                                text=chunk_text,
                                english_text=eng_text,
                                chunk_index=chunk_idx,
                                chunk_strategy=self.strategy_name,
                                query_id=record_meta.get("query_id"),
                                query_type=record_meta.get("query_type"),
                                language=passage.get("language") or record_meta.get("target_lang"),
                                source_lang=record_meta.get("source_lang"),
                                target_lang=record_meta.get("target_lang"),
                                is_selected=is_selected,
                            )
                        )
                        chunk_idx += 1
                        current_chunk_sentences = []
                        current_word_count = 0

                # Split the long sentence
                sent_words = sent.split()
                for i in range(0, len(sent_words), self.max_words):
                    sub_text = " ".join(sent_words[i : i + self.max_words]).strip()
                    if sub_text:
                        chunks.append(
                            Chunk(
                                chunk_id=f"{passage_id}_c{chunk_idx}",
                                parent_passage_id=passage_id,
                                text=sub_text,
                                english_text=eng_text,
                                chunk_index=chunk_idx,
                                chunk_strategy=self.strategy_name,
                                query_id=record_meta.get("query_id"),
                                query_type=record_meta.get("query_type"),
                                language=passage.get("language") or record_meta.get("target_lang"),
                                source_lang=record_meta.get("source_lang"),
                                target_lang=record_meta.get("target_lang"),
                                is_selected=is_selected,
                            )
                        )
                        chunk_idx += 1
                continue

            # If adding this sentence exceeds max_words, flush current buffer
            if current_word_count + sent_word_count > self.max_words and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences).strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{passage_id}_c{chunk_idx}",
                            parent_passage_id=passage_id,
                            text=chunk_text,
                            english_text=eng_text,
                            chunk_index=chunk_idx,
                            chunk_strategy=self.strategy_name,
                            query_id=record_meta.get("query_id"),
                            query_type=record_meta.get("query_type"),
                            language=passage.get("language") or record_meta.get("target_lang"),
                            source_lang=record_meta.get("source_lang"),
                            target_lang=record_meta.get("target_lang"),
                            is_selected=is_selected,
                        )
                    )
                    chunk_idx += 1
                current_chunk_sentences = [sent]
                current_word_count = sent_word_count
            else:
                current_chunk_sentences.append(sent)
                current_word_count += sent_word_count

        # Flush remaining buffer
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{passage_id}_c{chunk_idx}",
                        parent_passage_id=passage_id,
                        text=chunk_text,
                        english_text=eng_text,
                        chunk_index=chunk_idx,
                        chunk_strategy=self.strategy_name,
                        query_id=record_meta.get("query_id"),
                        query_type=record_meta.get("query_type"),
                        language=passage.get("language") or record_meta.get("target_lang"),
                        source_lang=record_meta.get("source_lang"),
                        target_lang=record_meta.get("target_lang"),
                        is_selected=is_selected,
                    )
                )

        return chunks


def get_chunker(strategy: str, config: dict[str, Any] | None = None) -> Chunker:
    """Instantiate a chunker based on strategy name and configuration."""
    cfg = config or {}
    chunking_cfg = cfg.get("chunking", {})

    if strategy == "passage":
        return PassageChunker()
    elif strategy == "fixed":
        fixed_cfg = chunking_cfg.get("fixed", {})
        return FixedChunker(max_words=fixed_cfg.get("max_words", 80))
    elif strategy == "overlap":
        overlap_cfg = chunking_cfg.get("overlap", {})
        return OverlapChunker(
            max_words=overlap_cfg.get("max_words", 80),
            overlap_words=overlap_cfg.get("overlap_words", 20),
        )
    elif strategy == "structure_aware":
        struct_cfg = chunking_cfg.get("structure_aware", {})
        return StructureAwareChunker(max_words=struct_cfg.get("max_words", 80))
    else:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'. Supported: passage, fixed, overlap, structure_aware")


def chunk_record(record: dict[str, Any], chunker: Chunker) -> list[Chunk]:
    """Chunk all passages within a single normalized dataset record."""
    passages = record.get("passages", [])
    record_meta = {
        "query_id": record.get("query_id"),
        "query_type": record.get("query_type"),
        "source_lang": record.get("source_lang"),
        "target_lang": record.get("target_lang"),
    }

    all_chunks: list[Chunk] = []
    for passage in passages:
        p_chunks = chunker.chunk_passage(passage, record_meta)
        all_chunks.extend(p_chunks)

    return all_chunks

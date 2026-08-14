"""Structured context construction module for grounded LLM generation.

Constructs structured, parent-deduplicated evidence context with explicit source IDs
and strict isolation from ground-truth or evaluation metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ContextItem:
    """Structured evidence item included in the LLM generation prompt."""

    source_id: int
    parent_passage_id: str
    chunk_id: str
    text: str
    language: str
    retrieval_rank: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextBuilder:
    """Builds formatted evidence context for LLM generation from retrieval results."""

    def __init__(self, default_top_k: int = 5, max_context_chars: int = 3000) -> None:
        self.default_top_k = default_top_k
        self.max_context_chars = max_context_chars

    def build(
        self,
        query: str,
        retrieved_results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> tuple[str, list[ContextItem]]:
        """Construct structured evidence string and return source items.

        Deduplicates parent passages, limits to top_k, and formats clean text blocks.
        """
        k = top_k or self.default_top_k
        if not retrieved_results:
            return "", []

        seen_parents: set[str] = set()
        context_items: list[ContextItem] = []
        source_counter = 1

        for rank_idx, item in enumerate(retrieved_results, start=1):
            pid = str(item.get("parent_passage_id") or item.get("passage_id") or item.get("chunk_id"))
            if pid in seen_parents:
                continue

            text = str(item.get("text", "")).strip()
            if not text:
                continue

            seen_parents.add(pid)
            c_item = ContextItem(
                source_id=source_counter,
                parent_passage_id=pid,
                chunk_id=str(item.get("chunk_id", f"{pid}_c0")),
                text=text,
                language=str(item.get("language") or item.get("target_lang") or "unknown"),
                retrieval_rank=rank_idx,
                score=float(item.get("score") or item.get("fusion_score") or 0.0),
            )
            context_items.append(c_item)
            source_counter += 1

            if len(context_items) >= k:
                break

        if not context_items:
            return "", []

        # Format structured prompt text
        formatted_blocks = []
        total_chars = 0

        for item in context_items:
            block = (
                f"[Source {item.source_id}]\n"
                f"Source ID: {item.source_id}\n"
                f"Parent Passage ID: {item.parent_passage_id}\n"
                f"Language: {item.language}\n"
                f"Content: {item.text}\n"
            )
            if total_chars + len(block) > self.max_context_chars and formatted_blocks:
                break
            formatted_blocks.append(block)
            total_chars += len(block)

        context_string = "\n".join(formatted_blocks).strip()
        active_items = context_items[: len(formatted_blocks)]

        return context_string, active_items

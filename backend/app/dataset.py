"""MSMARCO-XI dataset normalization and streaming utilities."""

from __future__ import annotations

from typing import Any


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw MSMARCO-XI record into a clean, consistent schema.

    Preserves:
    - Query information (translated and English)
    - Answer information (translated and English)
    - All passage texts with alignment, language tags, and `is_selected` labels
    - Query type and translation metadata
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict, got {type(raw).__name__}")

    query_id = raw.get("query_id")
    if query_id is None:
        raise ValueError("Record missing required 'query_id'")

    query = str(raw.get("query", "") or "").strip()
    eng_query = str(raw.get("Eng_Query", "") or "").strip()
    answer = str(raw.get("Answer", "") or "").strip()
    eng_answer = str(raw.get("Eng_Answer", "") or "").strip()
    query_type = str(raw.get("query_type", "UNKNOWN") or "UNKNOWN").strip()
    source_lang = str(raw.get("source_lang", "eng_Latn") or "eng_Latn").strip()
    target_lang = str(raw.get("target_lang", "") or "").strip()
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}

    raw_passages = raw.get("passages") or {}
    english_passages = raw_passages.get("English_passages") or []
    translated_passages = raw_passages.get("Translated_passages") or []
    is_selected_flags = raw_passages.get("is_selected") or []

    # Number of passages
    num_passages = max(len(english_passages), len(translated_passages), len(is_selected_flags))

    passages: list[dict[str, Any]] = []
    for idx in range(num_passages):
        eng_text = english_passages[idx] if idx < len(english_passages) else ""
        trans_text = translated_passages[idx] if idx < len(translated_passages) else ""
        is_sel = bool(is_selected_flags[idx]) if idx < len(is_selected_flags) else False

        passages.append(
            {
                "passage_id": f"{query_id}_p{idx}",
                "text": str(trans_text or "").strip(),
                "english_text": str(eng_text or "").strip(),
                "language": target_lang,
                "is_selected": is_sel,
            }
        )

    return {
        "query_id": query_id,
        "query": query,
        "english_query": eng_query,
        "answer": answer,
        "english_answer": eng_answer,
        "query_type": query_type,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "passages": passages,
        "meta": meta,
    }

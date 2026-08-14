"""Query processor module for input validation, normalization, and language propagation.

Preserves original user query and derives clean retrieval query without destructive rewrites.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


@dataclass
class QueryInput:
    """Structured query input maintaining original and processed forms."""

    original_query: str
    processed_query: str
    language: str
    is_valid: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_indic_script(text: str) -> str | None:
    """Lightweight deterministic Unicode range script detection for Indic languages.

    Returns generic 'und_Deva' for Devanagari text when specific language (Hindi vs Marathi)
    is not provided, avoiding false linguistic assumptions.
    """
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return "und_Deva"  # Generic Devanagari (Hindi / Marathi)
        elif 0x0980 <= code <= 0x09FF:
            return "ben_Beng"  # Bengali
        elif 0x0B80 <= code <= 0x0BFF:
            return "tam_Taml"  # Tamil
        elif 0x0C00 <= code <= 0x0C7F:
            return "tel_Telu"  # Telugu
    return None



class QueryProcessor:
    """Processes, validates, and normalizes user query inputs."""

    def __init__(self, default_language: str = "en") -> None:
        self.default_language = default_language

    def process(self, query: str, language: str | None = None) -> QueryInput:
        """Validate, normalize whitespace, and preserve original query."""
        if not isinstance(query, str):
            return QueryInput(
                original_query=str(query or ""),
                processed_query="",
                language=language or self.default_language,
                is_valid=False,
                error="Query must be a non-empty string",
            )

        trimmed = query.strip()
        if not trimmed:
            return QueryInput(
                original_query=query,
                processed_query="",
                language=language or self.default_language,
                is_valid=False,
                error="Query cannot be empty or whitespace only",
            )

        # Normalize whitespace (single spaces, strip control characters)
        normalized = re.sub(r"\s+", " ", trimmed)

        # Retain explicit language if given; otherwise detect Indic script or default
        lang = language
        if not lang or lang in ("en", "unknown", "auto"):
            detected = detect_indic_script(normalized)
            lang = detected or lang or self.default_language

        return QueryInput(
            original_query=query,
            processed_query=normalized,
            language=lang,
            is_valid=True,
            error=None,
        )

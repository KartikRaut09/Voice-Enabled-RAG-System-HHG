"""Unit tests for context construction and parent-deduplicated evidence formatting."""

from backend.app.context import ContextBuilder, ContextItem


def test_context_builder_parent_deduplication():
    """Test that multiple chunks from the same parent passage are collapsed to a single source."""
    builder = ContextBuilder(default_top_k=5)
    candidates = [
        {"parent_passage_id": "p1", "chunk_id": "p1_c0", "text": "First chunk of passage 1.", "language": "hin_Deva", "score": 0.95},
        {"parent_passage_id": "p1", "chunk_id": "p1_c1", "text": "Second chunk of passage 1.", "language": "hin_Deva", "score": 0.85},
        {"parent_passage_id": "p2", "chunk_id": "p2_c0", "text": "First chunk of passage 2.", "language": "hin_Deva", "score": 0.80},
    ]

    context_str, items = builder.build("query", candidates, top_k=5)
    assert len(items) == 2
    assert items[0].source_id == 1
    assert items[0].parent_passage_id == "p1"
    assert items[1].source_id == 2
    assert items[1].parent_passage_id == "p2"
    assert "[Source 1]" in context_str
    assert "[Source 2]" in context_str
    assert "[Source 3]" not in context_str


def test_context_builder_top_k_limiting():
    """Test that context builder respects top_k budget."""
    builder = ContextBuilder(default_top_k=5)
    candidates = [
        {"parent_passage_id": f"p{i}", "chunk_id": f"p{i}_c0", "text": f"Text {i}.", "language": "hin_Deva", "score": 0.9 - i * 0.05}
        for i in range(10)
    ]

    context_str, items = builder.build("query", candidates, top_k=3)
    assert len(items) == 3
    assert [item.source_id for item in items] == [1, 2, 3]
    assert "Text 0" in context_str
    assert "Text 2" in context_str
    assert "Text 3" not in context_str


def test_context_builder_max_chars_truncation():
    """Test that context builder limits total character count."""
    builder = ContextBuilder(default_top_k=5, max_context_chars=120)
    candidates = [
        {"parent_passage_id": "p1", "chunk_id": "p1_c0", "text": "A" * 60, "language": "hin_Deva", "score": 0.9},
        {"parent_passage_id": "p2", "chunk_id": "p2_c0", "text": "B" * 60, "language": "hin_Deva", "score": 0.8},
        {"parent_passage_id": "p3", "chunk_id": "p3_c0", "text": "C" * 60, "language": "hin_Deva", "score": 0.7},
    ]

    context_str, items = builder.build("query", candidates, top_k=5)
    assert len(items) < 3
    assert len(context_str) <= 200


def test_context_builder_empty_results():
    """Test that empty results return empty context string and empty list."""
    builder = ContextBuilder()
    context_str, items = builder.build("query", [])
    assert context_str == ""
    assert items == []

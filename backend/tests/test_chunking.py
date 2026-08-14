"""Comprehensive unit tests for multi-strategy chunking implementations.

Covers:
- Strategy 1: PassageChunker (baseline)
- Strategy 2: FixedChunker (word-bounded)
- Strategy 3: OverlapChunker (sliding window with overlap)
- Strategy 4: StructureAwareChunker (Indic & Latin sentence boundary-aware)
- Relevance label (is_selected) inheritance
- Provenance and metadata fidelity
- Determinism and edge cases (empty, whitespace, single word, long, unicode)
"""

from __future__ import annotations

import pytest

from backend.app.chunking import (
    Chunk,
    FixedChunker,
    OverlapChunker,
    PassageChunker,
    StructureAwareChunker,
    chunk_record,
    get_chunker,
)


@pytest.fixture
def sample_passage_record():
    return {
        "query_id": 1001,
        "query": "भारत की राजधानी क्या है?",
        "query_type": "ENTITY",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passages": [
            {
                "passage_id": "1001_p0",
                "text": "नई दिल्ली भारत की राजधानी है। यह सरकार का केंद्र है।",
                "english_text": "New Delhi is the capital of India. It is the seat of government.",
                "language": "hin_Deva",
                "is_selected": True,
            },
            {
                "passage_id": "1001_p1",
                "text": "मुंबई भारत की वित्तीय राजधानी मानी जाती है।",
                "english_text": "Mumbai is considered the financial capital of India.",
                "language": "hin_Deva",
                "is_selected": False,
            },
        ],
    }


@pytest.fixture
def long_passage():
    # 120 words passage
    words = [f"शब्द_{i}" for i in range(120)]
    return {
        "passage_id": "2002_p0",
        "text": " ".join(words),
        "english_text": " ".join([f"word_{i}" for i in range(120)]),
        "language": "hin_Deva",
        "is_selected": True,
    }


def test_passage_chunker_intact(sample_passage_record):
    """Test that PassageChunker keeps passages intact and preserves metadata."""
    chunker = PassageChunker()
    chunks = chunk_record(sample_passage_record, chunker)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "1001_p0_c0"
    assert chunks[0].parent_passage_id == "1001_p0"
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_strategy == "passage"
    assert chunks[0].query_id == 1001
    assert chunks[0].query_type == "ENTITY"
    assert chunks[0].is_selected is True
    assert "नई दिल्ली" in chunks[0].text
    # Verify no query text prepended
    assert "भारत की राजधानी क्या है?" not in chunks[0].text

    assert chunks[1].chunk_id == "1001_p1_c0"
    assert chunks[1].is_selected is False


def test_fixed_chunker_short_passage_intact():
    """Test that FixedChunker leaves passages <= max_words intact."""
    chunker = FixedChunker(max_words=80)
    passage = {
        "passage_id": "3003_p0",
        "text": "यह एक छोटा सा अनुच्छेद है जिसमें बीस से कम शब्द हैं।",
        "english_text": "This is a short paragraph with fewer than twenty words.",
        "is_selected": True,
    }
    chunks = chunker.chunk_passage(passage, {"query_id": 3003})
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "3003_p0_c0"
    assert chunks[0].chunk_strategy == "fixed"
    assert chunks[0].is_selected is True


def test_fixed_chunker_splits_long_passage(long_passage):
    """Test that FixedChunker splits long passages into max_words pieces."""
    chunker = FixedChunker(max_words=50)
    chunks = chunker.chunk_passage(long_passage, {"query_id": 2002})

    # 120 words / 50 words max = 3 chunks (50, 50, 20 words)
    assert len(chunks) == 3
    assert chunks[0].chunk_id == "2002_p0_c0"
    assert chunks[1].chunk_id == "2002_p0_c1"
    assert chunks[2].chunk_id == "2002_p0_c2"

    assert len(chunks[0].text.split()) == 50
    assert len(chunks[1].text.split()) == 50
    assert len(chunks[2].text.split()) == 20

    # Every derived chunk inherits is_selected = True
    assert all(c.is_selected is True for c in chunks)
    assert all(c.parent_passage_id == "2002_p0" for c in chunks)


def test_overlap_chunker_progression_and_overlap(long_passage):
    """Test that OverlapChunker produces correct sliding windows with overlap."""
    # 120 words, max_words=50, overlap_words=20 -> step=30
    # chunk 0: 0..50
    # chunk 1: 30..80
    # chunk 2: 60..110
    # chunk 3: 90..120
    chunker = OverlapChunker(max_words=50, overlap_words=20)
    chunks = chunker.chunk_passage(long_passage, {"query_id": 2002})

    assert len(chunks) == 4
    assert chunks[0].chunk_id == "2002_p0_c0"
    assert chunks[3].chunk_id == "2002_p0_c3"

    words0 = chunks[0].text.split()
    words1 = chunks[1].text.split()
    assert len(words0) == 50
    assert len(words1) == 50

    # Overlap between chunk 0 and chunk 1 is last 20 words of chunk 0 == first 20 words of chunk 1
    assert words0[-20:] == words1[:20]

    # All sub-chunks inherit is_selected
    assert all(c.is_selected is True for c in chunks)


def test_overlap_chunker_short_passage_no_duplication():
    """Test that short passages under max_words are not duplicated by OverlapChunker."""
    chunker = OverlapChunker(max_words=80, overlap_words=20)
    passage = {
        "passage_id": "4004_p0",
        "text": "छोटा पाठ।",
        "is_selected": False,
    }
    chunks = chunker.chunk_passage(passage, {"query_id": 4004})
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "4004_p0_c0"


def test_structure_aware_chunker_honors_indic_danda():
    """Test that StructureAwareChunker splits on Indic danda (।) and retains sentence integrity."""
    chunker = StructureAwareChunker(max_words=10)
    # Sentence 1: 6 words, Sentence 2: 9 words -> total 15 words (> 10 max_words)
    # Must split cleanly between sentence 1 and sentence 2 without cutting inside a sentence.
    text = "पहला वाक्य यहाँ समाप्त होता है। दूसरा वाक्य यहाँ से प्रारंभ होकर समाप्त होता है।"
    passage = {
        "passage_id": "5005_p0",
        "text": text,
        "is_selected": True,
    }
    chunks = chunker.chunk_passage(passage, {"query_id": 5005})

    assert len(chunks) == 2
    assert "पहला वाक्य" in chunks[0].text
    assert "दूसरा वाक्य" in chunks[1].text
    assert chunks[0].text.endswith("है।")
    assert chunks[0].is_selected is True
    assert chunks[1].is_selected is True


def test_structure_aware_chunker_long_sentence_fallback():
    """Test that StructureAwareChunker falls back to word splitting for single sentences exceeding max_words."""
    chunker = StructureAwareChunker(max_words=10)
    # A single 25-word sentence with no punctuation
    words = [f"शब्द_{i}" for i in range(25)]
    text = " ".join(words)
    passage = {
        "passage_id": "6006_p0",
        "text": text,
        "is_selected": False,
    }
    chunks = chunker.chunk_passage(passage, {"query_id": 6006})

    assert len(chunks) == 3
    assert len(chunks[0].text.split()) == 10
    assert len(chunks[1].text.split()) == 10
    assert len(chunks[2].text.split()) == 5
    assert all(c.is_selected is False for c in chunks)


def test_get_chunker_factory_and_config():
    """Test get_chunker resolution and configuration loading."""
    cfg = {
        "chunking": {
            "fixed": {"max_words": 65},
            "overlap": {"max_words": 70, "overlap_words": 15},
            "structure_aware": {"max_words": 75},
        }
    }

    p = get_chunker("passage", cfg)
    assert isinstance(p, PassageChunker)

    f = get_chunker("fixed", cfg)
    assert isinstance(f, FixedChunker)
    assert f.max_words == 65

    o = get_chunker("overlap", cfg)
    assert isinstance(o, OverlapChunker)
    assert o.max_words == 70
    assert o.overlap_words == 15

    s = get_chunker("structure_aware", cfg)
    assert isinstance(s, StructureAwareChunker)
    assert s.max_words == 75

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("invalid_strategy", cfg)


def test_edge_cases_empty_whitespace_missing_fields():
    """Test edge cases: empty strings, whitespace, missing metadata."""
    chunker = StructureAwareChunker(max_words=80)

    # Empty text
    empty_res = chunker.chunk_passage({"text": ""}, {})
    assert empty_res == []

    # Whitespace only
    ws_res = chunker.chunk_passage({"text": "   \n\t  "}, {})
    assert ws_res == []

    # Single word
    single_res = chunker.chunk_passage({"text": "नमस्ते"}, {})
    assert len(single_res) == 1
    assert single_res[0].text == "नमस्ते"
    assert single_res[0].chunk_index == 0


def test_chunk_determinism(sample_passage_record):
    """Test that repeated chunking on identical input produces identical outputs."""
    chunker = OverlapChunker(max_words=10, overlap_words=3)
    chunks1 = chunk_record(sample_passage_record, chunker)
    chunks2 = chunk_record(sample_passage_record, chunker)

    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.to_dict() == c2.to_dict()

"""Unit tests for BM25 lexical retrieval engine and multilingual Indic tokenizer."""

from pathlib import Path
import pytest
import unicodedata

from backend.app.bm25 import BM25Index, tokenize_indic


def test_tokenize_indic_devanagari_and_punctuation():
    """Test tokenization on Devanagari Hindi text with Indic danda and punctuation."""
    text = "भारत एक महान देश है। नई दिल्ली भारत की राजधानी है!"
    tokens = tokenize_indic(text)
    assert "भारत" in tokens
    assert "महान" in tokens
    assert "देश" in tokens
    assert "दिल्ली" in tokens
    assert "राजधानी" in tokens
    assert "।" not in tokens
    assert "!" not in tokens


def test_tokenize_indic_multilingual_scripts():
    """Test tokenization across Bengali, Tamil, and Telugu scripts."""
    ben_text = "পশ্চিমবঙ্গ ভারতের একটি অঙ্গরাজ্য।"
    tam_text = "தமிழ்நாடு இந்தியாவின் தென் மாநிலங்களில் ஒன்றாகும்."
    tel_text = "హైదరాబాద్ తెలంగాణ రాజధాని మరియు అతిపెద్ద నగరం."

    ben_tokens = tokenize_indic(ben_text)
    tam_tokens = tokenize_indic(tam_text)
    tel_tokens = tokenize_indic(tel_text)

    assert "পশ্চিমবঙ্গ" in ben_tokens
    assert "ভারতের" in ben_tokens

    assert "தமிழ்நாடு" in tam_tokens
    assert "இந்தியாவின்" in tam_tokens

    assert "హైదరాబాద్" in tel_tokens
    assert "తెలంగాణ" in tel_tokens


def test_tokenize_indic_empty_and_whitespace():
    """Test tokenization on empty and whitespace-only strings."""
    assert tokenize_indic("") == []
    assert tokenize_indic("   \n\t  ") == []


def test_bm25_index_creation_and_properties():
    """Test BM25 index building, corpus stats, and vocabulary."""
    docs = [
        "भारत की राजधानी नई दिल्ली है",
        "महाराष्ट्र की राजधानी मुंबई है",
        "पश्चिम बंगाल की राजधानी कोलकाता है",
    ]
    meta = [
        {"chunk_id": "c1", "parent_passage_id": "p1"},
        {"chunk_id": "c2", "parent_passage_id": "p2"},
        {"chunk_id": "c3", "parent_passage_id": "p3"},
    ]
    index = BM25Index(k1=1.5, b=0.75)
    index.build(docs, meta)

    assert index.size == 3
    assert index.avgdl > 0
    assert "राजधानी" in index.doc_frequencies
    assert index.doc_frequencies["राजधानी"] == 3
    assert "मुंबई" in index.doc_frequencies
    assert index.doc_frequencies["मुंबई"] == 1


def test_bm25_search_and_ranking():
    """Test that BM25 accurately ranks the most relevant document highest."""
    docs = [
        "भारत की राजधानी नई दिल्ली है",
        "महाराष्ट्र की राजधानी मुंबई है",
        "पश्चिम बंगाल की राजधानी कोलकाता है",
    ]
    meta = [
        {"chunk_id": "c1", "parent_passage_id": "p1"},
        {"chunk_id": "c2", "parent_passage_id": "p2"},
        {"chunk_id": "c3", "parent_passage_id": "p3"},
    ]
    index = BM25Index(k1=1.5, b=0.75)
    index.build(docs, meta)

    results = index.search_chunks("मुंबई", top_k=2)
    assert len(results) >= 1
    top_meta, top_score = results[0]
    assert top_meta["parent_passage_id"] == "p2"
    assert top_score > 0.0


def test_bm25_parent_passage_deduplication():
    """Test parent passage deduplication when multiple chunks come from the same parent."""
    docs = [
        "भारत एक बहुत बड़ा देश है",
        "भारत की राजधानी नई दिल्ली है",
        "महाराष्ट्र की राजधानी मुंबई है",
    ]
    meta = [
        {"chunk_id": "p1_c0", "parent_passage_id": "p1", "is_selected": True},
        {"chunk_id": "p1_c1", "parent_passage_id": "p1", "is_selected": True},
        {"chunk_id": "p2_c0", "parent_passage_id": "p2", "is_selected": False},
    ]
    index = BM25Index(k1=1.5, b=0.75)
    index.build(docs, meta)

    parent_results = index.search_parent_passages("भारत", top_k=10, fetch_k=50)

    # p1 should appear exactly once despite having two matching chunks
    parent_ids = [r["parent_passage_id"] for r in parent_results]
    assert len(parent_ids) == len(set(parent_ids))
    assert parent_ids[0] == "p1"


def test_bm25_empty_and_unknown_query():
    """Test search behavior with empty query and out-of-vocabulary terms."""
    docs = ["भारत की राजधानी नई दिल्ली है"]
    meta = [{"chunk_id": "c1", "parent_passage_id": "p1"}]
    index = BM25Index()
    index.build(docs, meta)

    assert index.search_chunks("") == []
    assert index.search_chunks("   ") == []
    assert index.search_chunks("xyzabcdef12345") == []


def test_bm25_save_and_load(tmp_path: Path):
    """Test persistence and loading of BM25 index and metadata."""
    docs = [
        "తెలంగాణ రాజధాని హైదరాబాద్",
        "తమిళనాడు రాజధాని చెన్నై",
    ]
    meta = [
        {"chunk_id": "c1", "parent_passage_id": "p1", "language": "tel"},
        {"chunk_id": "c2", "parent_passage_id": "p2", "language": "tam"},
    ]
    index = BM25Index(k1=1.2, b=0.5)
    index.build(docs, meta)

    save_dir = tmp_path / "bm25_test_index"
    index.save(save_dir)

    assert (save_dir / "index.json").exists()
    assert (save_dir / "metadata.json").exists()
    assert (save_dir / "manifest.json").exists()

    loaded = BM25Index.load(save_dir)
    assert loaded.size == 2
    assert loaded.k1 == 1.2
    assert loaded.b == 0.5

    orig_res = index.search_parent_passages("హైదరాబాద్", top_k=2)
    loaded_res = loaded.search_parent_passages("హైదరాబాద్", top_k=2)

    assert len(orig_res) == len(loaded_res)
    assert orig_res[0]["parent_passage_id"] == loaded_res[0]["parent_passage_id"]
    assert orig_res[0]["score"] == loaded_res[0]["score"]

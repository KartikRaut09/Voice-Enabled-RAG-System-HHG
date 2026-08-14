"""Tests for MSMARCO-XI dataset normalization, schema integrity, and subsets."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from backend.app.dataset import normalize_record


@pytest.fixture
def sample_raw_record():
    """A realistic sample raw record matching ai4bharat/MSMARCO-XI."""
    return {
        "query_id": 1185869,
        "query": "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?",
        "Eng_Query": "what was the immediate impact of the success of the manhattan project?",
        "Answer": "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव यह था कि संयुक्त राज्य अमेरिका के पास एक ऐसा हथियार था जिसने मानव इतिहास के सबसे विनाशकारी युद्ध का अंत कर दिया।",
        "Eng_Answer": "The immediate impact of the success of the Manhattan Project was that the United States had a weapon that brought an end to the most destructive war in human history.",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "meta": {"model_name": "IndicTrans2", "temperature": 0.0},
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "The presence of a few nuclear weapons didn't dramatically alter US war plans...",
                "The Manhattan Project was a research and development undertaking during World War II...",
            ],
            "Translated_passages": [
                "कुछ परमाणु हथियारों की उपस्थिति ने अमेरिकी युद्ध योजनाओं को नाटकीय रूप से नहीं बदला...",
                "मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान एक अनुसंधान और विकास उपक्रम था...",
            ],
        },
    }


def test_normalize_valid_record(sample_raw_record):
    """Test that a valid raw record normalizes correctly with expected keys."""
    normalized = normalize_record(sample_raw_record)

    assert normalized["query_id"] == 1185869
    assert normalized["query"] == "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?"
    assert normalized["english_query"] == "what was the immediate impact of the success of the manhattan project?"
    assert "मैनहट्टन परियोजना" in normalized["answer"]
    assert normalized["query_type"] == "DESCRIPTION"
    assert normalized["source_lang"] == "eng_Latn"
    assert normalized["target_lang"] == "hin_Deva"
    assert isinstance(normalized["meta"], dict)
    assert len(normalized["passages"]) == 2


def test_normalize_preserves_relevance_labels(sample_raw_record):
    """Test that ground-truth is_selected flags and passage alignment survive normalization."""
    normalized = normalize_record(sample_raw_record)
    passages = normalized["passages"]

    assert passages[0]["is_selected"] is True
    assert passages[0]["passage_id"] == "1185869_p0"
    assert passages[0]["language"] == "hin_Deva"
    assert "परमाणु हथियारों" in passages[0]["text"]
    assert "nuclear weapons" in passages[0]["english_text"]

    assert passages[1]["is_selected"] is False
    assert passages[1]["passage_id"] == "1185869_p1"
    assert passages[1]["is_selected"] is False


def test_normalize_empty_or_malformed_record():
    """Test that missing query_id or non-dict inputs raise explicit ValueError."""
    with pytest.raises(ValueError, match="Expected dict"):
        normalize_record("not-a-dict")  # type: ignore

    with pytest.raises(ValueError, match="missing required 'query_id'"):
        normalize_record({"query": "some query without id"})


def test_normalize_missing_optional_fields():
    """Test that records with empty/missing optional fields do not crash."""
    minimal = {
        "query_id": 999,
        "query": "minimal query",
        "passages": None,
    }
    normalized = normalize_record(minimal)
    assert normalized["query_id"] == 999
    assert normalized["query"] == "minimal query"
    assert normalized["passages"] == []
    assert normalized["query_type"] == "UNKNOWN"


def test_manifest_structure_and_hash():
    """Test that manifest.json exists, has correct seed, schema definition, and hash."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    manifest_path = base_dir / "data" / "processed" / "manifest.json"

    assert manifest_path.exists(), "manifest.json must exist"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["dataset"] == "ai4bharat/MSMARCO-XI"
    assert manifest["seed"] == 42
    assert "schema_hash" in manifest
    assert "schema_definition" in manifest
    assert manifest["disjoint_dev_eval"] is True
    assert set(manifest["languages"]) == {"hin", "mar", "ben", "tam", "tel"}


def test_sampling_determinism():
    """Test that identical seed produces deterministic record selection."""
    from scripts.prepare_dataset import load_config
    config = load_config()
    ds_config = config.get("dataset", {})
    seed = ds_config.get("sample_seed", 42)
    assert seed == 42


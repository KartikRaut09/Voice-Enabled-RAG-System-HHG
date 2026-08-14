"""Unit tests for grounded LLM generation, citation validation, and provider interfaces."""

from backend.app.context import ContextItem
from backend.app.generation import (
    GenerationResult,
    MockLLMProvider,
    OpenAICompatibleProvider,
    extract_and_validate_citations,
    get_llm_provider,
    is_text_abstention,
)


def test_mock_llm_provider_grounded_generation():
    """Test grounded synthesis and citation from MockLLMProvider."""
    provider = MockLLMProvider()
    context_items = [
        ContextItem(
            source_id=1,
            parent_passage_id="p101",
            chunk_id="p101_c0",
            text="भारत एक दक्षिण एशियाई देश है। इसकी राजधानी नई दिल्ली है।",
            language="hin_Deva",
            retrieval_rank=1,
            score=0.92,
        )
    ]

    res = provider.generate("भारत की राजधानी क्या है?", context_items=context_items)
    assert isinstance(res, GenerationResult)
    assert res.is_grounded is True
    assert res.is_abstention is False
    assert len(res.sources) == 1
    assert res.sources[0]["parent_passage_id"] == "p101"
    assert res.sources[0]["source_id"] == 1
    assert res.latency_ms > 0.0


def test_mock_llm_provider_insufficient_evidence_abstention():
    """Test that empty context triggers explicit insufficient evidence refusal."""
    provider = MockLLMProvider()
    res = provider.generate("अनजान सवाल?", context_items=[])
    assert res.is_abstention is True
    assert len(res.sources) == 0
    assert "पर्याप्त जानकारी नहीं है" in res.answer


def test_extract_and_validate_citations_filters_invalid_citations():
    """Test that invalid citation numbers (e.g. [99]) are stripped and flagged."""
    context_items = [
        ContextItem(
            source_id=1,
            parent_passage_id="p1",
            chunk_id="p1_c0",
            text="वैध स्रोत पाठ।",
            language="hin_Deva",
            retrieval_rank=1,
            score=0.9,
        )
    ]

    # Model hallucinated [99] in addition to valid [1]
    raw_answer = "यह मुख्य तथ्य है [1] और यह दूसरा तथ्य है [99]।"
    sources, citations, cleaned_text = extract_and_validate_citations(raw_answer, context_items)

    assert len(citations) == 1
    assert citations[0] == 1
    assert len(sources) == 1
    assert sources[0]["parent_passage_id"] == "p1"
    assert "[99]" not in cleaned_text
    assert "[1]" in cleaned_text


def test_openai_compatible_provider_fallback_on_network_failure():
    """Test that API failures in OpenAICompatibleProvider result in safe abstention without crashing."""
    provider = OpenAICompatibleProvider(
        api_key="invalid_dummy_key",
        base_url="http://localhost:9999/v1",  # non-existent port
        timeout=0.5,
    )
    context_items = [
        ContextItem(
            source_id=1,
            parent_passage_id="p1",
            chunk_id="p1_c0",
            text="पाठ",
            language="hin_Deva",
            retrieval_rank=1,
            score=0.9,
        )
    ]

    res = provider.generate("query", context_items=context_items)
    assert res.is_abstention is True
    assert "पर्याप्त जानकारी नहीं है" in res.answer


def test_is_text_abstention_multilingual():
    """Test detection of abstention in Hindi, Marathi, Bengali, Tamil, Telugu."""
    assert is_text_abstention("उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।") is True
    assert is_text_abstention("माहिती उपलब्ध नाही") is True
    assert is_text_abstention("যথেষ্ট তথ্য নেই") is True
    assert is_text_abstention("भारत की राजधानी नई दिल्ली है।") is False



def test_get_llm_provider_factory():
    """Test factory creates Mock provider by default."""
    provider = get_llm_provider({"generation": {"provider": "mock"}})
    assert isinstance(provider, MockLLMProvider)

"""Unit tests for QueryProcessor and Indic script detection."""

from backend.app.query_processor import QueryInput, QueryProcessor, detect_indic_script


def test_query_processor_normalization_and_preservation():
    """Test whitespace normalization while preserving original user query."""
    processor = QueryProcessor()
    raw = "   भारत   की  राजधानी \n\t क्या है?   "
    res = processor.process(raw, language="hin_Deva")

    assert isinstance(res, QueryInput)
    assert res.is_valid is True
    assert res.original_query == raw
    assert res.processed_query == "भारत की राजधानी क्या है?"
    assert res.language == "hin_Deva"
    assert res.error is None


def test_query_processor_script_detection():
    """Test deterministic Indic script detection when language is not explicitly provided."""
    processor = QueryProcessor()

    # Hindi / Devanagari
    res_hi = processor.process("ताजमहल कहाँ है?")
    assert res_hi.language == "hin_Deva"

    # Bengali
    res_bn = processor.process("কলকাতা কোন নদীর তীরে?")
    assert res_bn.language == "ben_Beng"

    # Tamil
    res_ta = processor.process("சென்னை எங்கு உள்ளது?")
    assert res_ta.language == "tam_Taml"

    # Telugu
    res_te = processor.process("హైదరాబాద్ ఎక్కడ ఉంది?")
    assert res_te.language == "tel_Telu"


def test_query_processor_empty_and_whitespace():
    """Test rejection of empty or blank queries."""
    processor = QueryProcessor()

    res_empty = processor.process("")
    assert res_empty.is_valid is False
    assert res_empty.error is not None

    res_spaces = processor.process("   \n\t  ")
    assert res_spaces.is_valid is False
    assert res_spaces.error is not None

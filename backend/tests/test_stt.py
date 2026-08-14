"""Unit and integration tests for Speech-to-Text (STT) and voice input layer."""

import io
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.pipeline import RAGPipeline
from backend.app.query_processor import QueryProcessor
from backend.app.schemas import TranscriptionResponse
from backend.app.stt import (
    MockSTTProvider,
    TranscriptionResult,
    get_stt_provider,
    normalize_stt_language,
    validate_audio_input,
)


def create_dummy_wav_bytes(content: str = "RIFF....WAVEfmt ....data....") -> bytes:
    """Create simulated WAV header bytes for testing."""
    header = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
    header += (1).to_bytes(2, "little") + (1).to_bytes(2, "little")  # PCM, 1 channel
    header += (16000).to_bytes(4, "little") + (32000).to_bytes(4, "little")  # sample rate
    header += (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
    header += b"data" + (len(content.encode("utf-8"))).to_bytes(4, "little")
    return header + content.encode("utf-8")


def test_validate_audio_input_valid_wav():
    """Test audio validation on valid WAV bytes."""
    wav_bytes = create_dummy_wav_bytes()
    valid, err = validate_audio_input(wav_bytes, max_size_mb=10, filename="test.wav")
    assert valid is True
    assert err is None


def test_validate_audio_input_empty_and_oversized():
    """Test rejection of empty and oversized audio inputs."""
    # Empty
    valid, err = validate_audio_input(b"")
    assert valid is False
    assert "empty" in (err or "").lower()

    # Oversized (>1 MB limit in test)
    large_bytes = b"RIFF" + b"WAVE" + (b"0" * (2 * 1024 * 1024))
    valid, err = validate_audio_input(large_bytes, max_size_mb=1)
    assert valid is False
    assert "maximum size limit" in (err or "")


def test_validate_audio_input_unsupported_corrupt_format():
    """Test rejection of non-audio corrupt payloads."""
    corrupt_bytes = b"NOT_AN_AUDIO_FILE_DATA_CORRUPT"
    valid, err = validate_audio_input(corrupt_bytes, filename="invalid.exe")
    assert valid is False
    assert "unsupported" in (err or "").lower()


def test_mock_stt_provider_transcribe_and_timings():
    """Test MockSTTProvider transcription output, language, and latency fields."""
    provider = MockSTTProvider(default_transcript="भारत की राजधानी क्या है?", default_language="und_Deva")
    wav_bytes = create_dummy_wav_bytes()

    res = provider.transcribe(wav_bytes, filename="test.wav")
    assert isinstance(res, TranscriptionResult)
    assert res.status == "success"
    assert res.text == "भारत की राजधानी क्या है?"
    assert res.language == "und_Deva"
    assert res.stt_preprocessing_ms > 0
    assert res.stt_inference_ms > 0
    assert res.stt_total_ms >= res.stt_preprocessing_ms + res.stt_inference_ms


def test_mock_stt_provider_empty_and_failure_states():
    """Test provider empty transcription and error handling."""
    provider = MockSTTProvider()
    wav_bytes = create_dummy_wav_bytes()

    # Failure simulation
    provider.should_fail = True
    res_fail = provider.transcribe(wav_bytes)
    assert res_fail.status == "error"
    assert "failure" in (res_fail.error or "").lower()

    # Empty speech simulation
    provider.should_fail = False
    provider.should_return_empty = True
    res_empty = provider.transcribe(wav_bytes)
    assert res_empty.status == "empty_transcription"
    assert res_empty.text == ""


def test_stt_multilingual_indic_language_preservation():
    """Test language normalization and preservation across Hindi, Marathi, Bengali, Tamil, Telugu."""
    # Hindi explicit
    assert normalize_stt_language("hi", "भारत की राजधानी क्या है?", explicit_lang="hin_Deva") == "hin_Deva"
    # Marathi explicit
    assert normalize_stt_language("mr", "महाराष्ट्राची राजधानी कोणती?", explicit_lang="mar_Deva") == "mar_Deva"
    # Unannotated Devanagari -> und_Deva
    assert normalize_stt_language("hi", "भारत की राजधानी क्या है?") == "und_Deva"
    # Bengali
    assert normalize_stt_language("bn", "কলকাতা কোন নদীর তীরে?") == "ben_Beng"
    # Tamil
    assert normalize_stt_language("ta", "சென்னையின் முக்கிய இடம் எது?") == "tam_Taml"
    # Telugu
    assert normalize_stt_language("te", "హైదరాబాద్ ఎక్కడ ఉంది?") == "tel_Telu"


def test_get_stt_provider_factory():
    """Test STT provider factory instantiating mock and sarvam providers."""
    mock_p = get_stt_provider({"stt": {"provider": "mock", "model_name": "mock-stt-v1"}})
    assert isinstance(mock_p, MockSTTProvider)

    sarvam_p = get_stt_provider({"stt": {"provider": "sarvam", "model_name": "saaras:v3"}})
    from backend.app.stt import SarvamSTTProvider
    assert isinstance(sarvam_p, SarvamSTTProvider)
    assert sarvam_p.model_name == "saaras:v3"


def test_sarvam_provider_missing_api_key(monkeypatch):
    """Test Sarvam provider handles missing API key cleanly."""
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    from backend.app.stt import SarvamSTTProvider
    provider = SarvamSTTProvider(api_key=None)
    wav_bytes = create_dummy_wav_bytes()

    res = provider.transcribe(wav_bytes)
    assert res.status == "error"
    assert "SARVAM_API_KEY" in (res.error or "")


def test_sarvam_provider_successful_transcription(monkeypatch):
    """Test Sarvam provider parses valid JSON response from API."""
    from backend.app.stt import SarvamSTTProvider

    class MockResponse:
        status_code = 200
        text = '{"transcript": "भारत की राजधानी क्या है?", "language_code": "hi-IN"}'
        def json(self):
            return {"transcript": "भारत की राजधानी क्या है?", "language_code": "hi-IN"}

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, headers=None, files=None, data=None):
            assert headers.get("api-subscription-key") == "dummy_sarvam_key"
            assert data.get("model") == "saaras:v3"
            assert data.get("language_code") == "hi-IN"
            return MockResponse()

    monkeypatch.setattr("httpx.Client", MockClient)
    provider = SarvamSTTProvider(api_key="dummy_sarvam_key")
    wav_bytes = create_dummy_wav_bytes()

    res = provider.transcribe(wav_bytes, language="hin_Deva")
    assert res.status == "success"
    assert res.text == "भारत की राजधानी क्या है?"
    assert res.language == "hin_Deva"
    assert res.provider == "sarvam"
    assert res.model == "saaras:v3"
    assert res.stt_total_ms > 0


def test_sarvam_provider_api_error_handling(monkeypatch):
    """Test Sarvam provider handles non-200 HTTP response codes."""
    from backend.app.stt import SarvamSTTProvider

    class MockErrorResponse:
        status_code = 401
        text = '{"detail": "Invalid subscription key"}'
        def json(self):
            return {"detail": "Invalid subscription key"}

    class MockErrorClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            return MockErrorResponse()

    monkeypatch.setattr("httpx.Client", MockErrorClient)
    provider = SarvamSTTProvider(api_key="bad_key")
    wav_bytes = create_dummy_wav_bytes()

    res = provider.transcribe(wav_bytes)
    assert res.status == "error"
    assert "401" in (res.error or "")


def test_sarvam_provider_timeout_handling(monkeypatch):
    """Test Sarvam provider handles network timeout exceptions."""
    import httpx
    from backend.app.stt import SarvamSTTProvider

    class MockTimeoutClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            raise httpx.ReadTimeout("Connection timed out")

    monkeypatch.setattr("httpx.Client", MockTimeoutClient)
    provider = SarvamSTTProvider(api_key="dummy_key")
    wav_bytes = create_dummy_wav_bytes()

    res = provider.transcribe(wav_bytes)
    assert res.status == "error"
    assert "timed out" in (res.error or "").lower()


def test_sarvam_provider_empty_transcription_handling(monkeypatch):
    """Test Sarvam provider returns empty_transcription status when transcript is empty."""
    from backend.app.stt import SarvamSTTProvider

    class MockEmptyResponse:
        status_code = 200
        text = '{"transcript": ""}'
        def json(self):
            return {"transcript": ""}

    class MockEmptyClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            return MockEmptyResponse()

    monkeypatch.setattr("httpx.Client", MockEmptyClient)
    provider = SarvamSTTProvider(api_key="dummy_key")
    wav_bytes = create_dummy_wav_bytes()

    res = provider.transcribe(wav_bytes)
    assert res.status == "empty_transcription"
    assert res.text == ""


def test_sarvam_provider_language_mapping(monkeypatch):
    """Test language mapping from internal Indic representations to Sarvam BCP-47 codes."""
    from backend.app.stt import SarvamSTTProvider

    captured_lang_codes = []

    class MockLangCaptureClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, headers=None, files=None, data=None):
            captured_lang_codes.append(data.get("language_code"))
            class Resp:
                status_code = 200
                text = '{"transcript": "test"}'
                def json(self):
                    return {"transcript": "test"}
            return Resp()

    monkeypatch.setattr("httpx.Client", MockLangCaptureClient)
    provider = SarvamSTTProvider(api_key="dummy_key")
    wav_bytes = create_dummy_wav_bytes()

    # Test all 5 benchmarked Indic languages
    langs = [("hin_Deva", "hi-IN"), ("mar_Deva", "mr-IN"), ("ben_Beng", "bn-IN"), ("tam_Taml", "ta-IN"), ("tel_Telu", "te-IN")]
    for int_lang, expected_sarvam in langs:
        provider.transcribe(wav_bytes, language=int_lang)

    assert captured_lang_codes == ["hi-IN", "mr-IN", "bn-IN", "ta-IN", "te-IN"]


def test_api_transcribe_endpoint():
    """Test POST /api/transcribe endpoint with multipart audio upload."""
    mock_stt = MockSTTProvider(default_transcript="भारत की राजधानी क्या है?", default_language="hin_Deva")
    app.state.stt_provider = mock_stt

    client = TestClient(app)
    wav_bytes = create_dummy_wav_bytes()

    files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data = {"language": "hin_Deva"}
    response = client.post("/api/transcribe", files=files, data=data)

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"
    assert "text" in res_json
    assert res_json["language"] == "hin_Deva"
    assert res_json["latency_ms"] > 0
    assert res_json["provider"] == "mock"


def test_api_transcribe_empty_audio_rejection():
    """Test POST /api/transcribe with empty file."""
    mock_stt = MockSTTProvider()
    app.state.stt_provider = mock_stt

    client = TestClient(app)
    files = {"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    response = client.post("/api/transcribe", files=files)

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "error"
    assert "empty" in res_json["error"].lower()



def test_stt_to_rag_pipeline_compatibility():
    """Integration Test: Audio -> STT -> QueryProcessor -> RAGPipeline text synthesis."""
    stt = MockSTTProvider(default_transcript="भारत की राजधानी क्या है?", default_language="hin_Deva")
    qp = QueryProcessor()
    pipeline = RAGPipeline(query_processor=qp)

    # 1. Transcribe audio
    wav_bytes = create_dummy_wav_bytes()
    stt_res = stt.transcribe(wav_bytes, language="hin_Deva")
    assert stt_res.status == "success"

    # 2. Feed into QueryProcessor
    q_input = qp.process(stt_res.text, language=stt_res.language)
    assert q_input.is_valid is True
    assert q_input.language == "hin_Deva"

    # 3. Feed into RAGPipeline
    mock_candidates = [
        {
            "text": "भारत की राजधानी नई दिल्ली है।",
            "score": 0.95,
            "parent_passage_id": "p_india",
            "chunk_id": "c1",
            "language": "hin_Deva",
        }
    ]
    rag_resp = pipeline.orchestrate(
        query=stt_res.text,
        language=stt_res.language,
        options={"mock_candidates": mock_candidates},
    )

    assert rag_resp.status == "success"
    assert len(rag_resp.sources) == 1
    assert "नई दिल्ली" in rag_resp.answer


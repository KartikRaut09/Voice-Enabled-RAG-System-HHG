"""Unit and integration tests for POST /api/voice-query full voice-RAG endpoint."""

import io
import pytest
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.schemas import VoiceQueryResponse
from backend.tests.test_stt import create_dummy_wav_bytes


def test_voice_query_endpoint_success():
    """Test successful voice query execution through the integrated STT-RAG pipeline."""
    client = TestClient(app)
    wav_bytes = create_dummy_wav_bytes("भारत की राजधानी क्या है?")

    files = {"file": ("voice.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data = {"language": "hin_Deva"}
    response = client.post("/api/voice-query", files=files, data=data)

    assert response.status_code == 200
    res_json = response.json()

    assert "transcription" in res_json
    assert res_json["transcription"] == "भारत की राजधानी क्या है?"
    assert res_json["language"] == "hin_Deva"
    assert "answer" in res_json
    assert "latency" in res_json

    # Latency structure validation (3 separately instrumented metrics)
    lat = res_json["latency"]
    assert "stt_latency_ms" in lat
    assert "rag_latency_ms" in lat
    assert "e2e_latency_ms" in lat
    assert lat["stt_latency_ms"] >= 0
    assert lat["rag_latency_ms"] >= 0
    assert lat["e2e_latency_ms"] >= lat["stt_latency_ms"]


def test_voice_query_empty_audio_rejection():
    """Test rejection of empty audio bytes on /api/voice-query."""
    client = TestClient(app)
    files = {"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    response = client.post("/api/voice-query", files=files)

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] in ("empty_transcription", "error")
    assert "empty" in (res_json.get("error") or "").lower()



def test_voice_query_multilingual_language_propagation():
    """Test language propagation through voice endpoint for Marathi, Bengali, Tamil, Telugu."""
    client = TestClient(app)
    languages = [
        ("mar_Deva", "महाराष्ट्राची राजधानी कोणती?"),
        ("ben_Beng", "পশ্চিমবঙ্গের রাজধানী কোথায়?"),
        ("tam_Taml", "சென்னையின் முக்கிய இடம் எது?"),
        ("tel_Telu", "హైదరాబాద్ యొక్క ప్రసిద్ధ ప్రదేశం ఏది?"),
    ]

    for lang, query_text in languages:
        wav_bytes = create_dummy_wav_bytes(query_text)
        files = {"file": ("query.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {"language": lang}
        response = client.post("/api/voice-query", files=files, data=data)

        assert response.status_code == 200
        res_json = response.json()
        assert res_json["language"] == lang
        assert res_json["query_metadata"]["language"] == lang


def test_voice_query_and_text_query_coexistence():
    """Test that text endpoint /api/query and voice endpoint /api/voice-query both work cleanly."""
    client = TestClient(app)

    # 1. Text query
    text_res = client.post("/api/query", json={"query": "भारत की राजधानी क्या है?", "language": "hin_Deva"})
    assert text_res.status_code == 200
    assert "answer" in text_res.json()

    # 2. Voice query
    wav_bytes = create_dummy_wav_bytes("भारत की राजधानी क्या है?")
    files = {"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
    voice_res = client.post("/api/voice-query", files=files, data={"language": "hin_Deva"})
    assert voice_res.status_code == 200
    assert "answer" in voice_res.json()

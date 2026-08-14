import time

from backend.app.middleware import LatencyTracker


def test_latency_tracker_start_stop():
    tracker = LatencyTracker()
    tracker.start("query_processing")
    time.sleep(0.01)  # 10ms
    elapsed = tracker.stop("query_processing")
    assert elapsed > 0
    assert "query_processing" in tracker.timings


def test_latency_tracker_stt_latency():
    tracker = LatencyTracker()
    tracker.timings = {"stt": 120.0}
    stt = tracker.get_stt_latency()
    assert stt == 120.0


def test_latency_tracker_rag_latency():
    tracker = LatencyTracker()
    tracker.timings = {
        "query_processing": 5.0,
        "embedding": 10.0,
        "retrieval": 20.0,
        "reranking": 15.0,
        "generation": 50.0,
        "guardrails": 5.0,
    }
    rag = tracker.get_rag_latency()
    assert rag == 105.0


def test_latency_tracker_e2e_latency():
    tracker = LatencyTracker()
    tracker.timings = {
        "stt": 200.0,
        "query_processing": 5.0,
        "embedding": 10.0,
        "retrieval": 20.0,
        "reranking": 15.0,
        "generation": 50.0,
        "guardrails": 5.0,
    }
    # E2E is strictly STT latency + RAG latency
    e2e = tracker.get_e2e_latency()
    assert e2e == 305.0


def test_latency_tracker_missing_components():
    tracker = LatencyTracker()
    assert tracker.get_stt_latency() == 0.0
    assert tracker.get_rag_latency() == 0.0
    assert tracker.get_e2e_latency() == 0.0


def test_latency_tracker_to_dict():
    tracker = LatencyTracker()
    tracker.timings = {"stt": 50.0, "query_processing": 5.0}
    d = tracker.to_dict()
    assert isinstance(d, dict)
    assert d["stt"] == 50.0
    assert d["query_processing"] == 5.0
    assert "stt_latency" in d
    assert d["stt_latency"] == 50.0
    assert "rag_latency" in d
    assert d["rag_latency"] == 5.0
    assert "e2e_latency" in d
    assert d["e2e_latency"] == 55.0


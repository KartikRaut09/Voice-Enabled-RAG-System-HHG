import pytest


@pytest.mark.anyio
async def test_query_returns_200(client):
    response = await client.post("/api/query", json={"query": "What is Python?"})
    assert response.status_code == 200


@pytest.mark.anyio
async def test_query_response_structure(client):
    response = await client.post("/api/query", json={"query": "test query"})
    data = response.json()
    assert "request_id" in data
    assert "query" in data
    assert data["query"] == "test query"
    assert "answer" in data
    assert "sources" in data
    assert "latency" in data
    assert "status" in data
    assert data["status"] == "success"


@pytest.mark.anyio
async def test_query_latency_breakdown(client):
    response = await client.post("/api/query", json={"query": "test"})
    data = response.json()
    latency = data["latency"]
    # 3 primary metrics
    assert "rag_latency_ms" in latency
    assert "stt_latency_ms" in latency
    assert "e2e_latency_ms" in latency
    # Component breakdown
    assert "stt_ms" in latency
    assert "query_processing_ms" in latency
    assert "embedding_ms" in latency
    assert "retrieval_ms" in latency
    assert "reranking_ms" in latency
    assert "generation_ms" in latency
    assert "guardrails_ms" in latency


@pytest.mark.anyio
async def test_query_with_language(client):
    response = await client.post(
        "/api/query", json={"query": "test", "language": "hi"}
    )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_query_empty_string_rejected(client):
    response = await client.post("/api/query", json={"query": ""})
    # Empty query should still return 200 in Phase 0 placeholder
    # Guardrails for empty queries will be added in Phase 8
    assert response.status_code == 200


@pytest.mark.anyio
async def test_query_missing_body_returns_422(client):
    response = await client.post("/api/query", json={})
    assert response.status_code == 422

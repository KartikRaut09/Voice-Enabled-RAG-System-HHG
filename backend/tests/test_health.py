import pytest


@pytest.mark.anyio
async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_health_response_structure(client):
    response = await client.get("/health")
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "version" in data
    assert "timestamp" in data


@pytest.mark.anyio
async def test_health_has_request_id_header(client):
    response = await client.get("/health")
    assert "x-request-id" in response.headers


@pytest.mark.anyio
async def test_health_has_response_time_header(client):
    response = await client.get("/health")
    assert "x-response-time-ms" in response.headers
    time_ms = float(response.headers["x-response-time-ms"])
    assert time_ms >= 0


@pytest.mark.anyio
async def test_root_serves_frontend(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "HHGoa RAG" in response.text
    assert "<form id=\"query-form\">" in response.text


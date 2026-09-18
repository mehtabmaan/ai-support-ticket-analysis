"""
Tests for FastAPI REST API endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from main import app
from src.data_layer.loader import initialize_database


@pytest.fixture(scope="module", autouse=True)
def client():
    initialize_database()
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_records"] == 500
    assert data["reference_now"] == "2024-04-04 12:23"
    assert "TKT-108" in data["reference_now_source"]


def test_query_endpoint(client):
    payload = {"question": "How many tickets are currently open?"}
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["row_count"] == 1
    assert data["results"][0]["open_tickets_count"] == 111
    assert "111" in data["answer"]


def test_anomalies_endpoint(client):
    response = client.get("/api/anomalies")
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["total_anomalies"] == 102
    assert data["summary"]["resolution_time_outliers"] == 22
    assert data["summary"]["sla_breaches"] == 80
    assert data["total_count"] == 102
    assert len(data["anomalies"]) == 102


def test_anomalies_filtered(client):
    response = client.get("/api/anomalies?category=Billing")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 30
    for item in data["anomalies"]:
        assert item["category"] == "Billing"


def test_metrics_endpoint(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_tickets"] == 500
    assert data["resolved_tickets"] == 327
    assert data["open_tickets"] == 111
    assert data["escalated_tickets"] == 62
    assert data["active_agents_count"] == 12


def test_query_validation_error(client):
    # Too short question
    response = client.post("/api/query", json={"question": "hi"})
    assert response.status_code == 422

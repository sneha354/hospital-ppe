"""
Integration tests for FastAPI endpoints: /api/stats, /api/zones, /api/incidents.
"""

import pytest
from fastapi.testclient import TestClient
from web.server import app

client = TestClient(app)


def test_get_stats_endpoint():
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "compliance_rate_percent" in data
    assert "total_evaluations" in data
    assert "active_tracks" in data
    assert "current_source" in data


def test_zones_crud_endpoints():
    # 1. List zones
    res_list = client.get("/api/zones")
    assert res_list.status_code == 200
    zones = res_list.json()
    assert isinstance(zones, list)
    assert len(zones) >= 3

    # 2. Create new zone
    new_zone_payload = {
        "id": "zone_api_test",
        "name": "API Test Room",
        "code": "API-01",
        "risk_level": "LOW",
        "color": [100, 150, 200],
        "polygon": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],
        "required_ppe": ["face_mask"],
        "optional_ppe": [],
        "grace_period_seconds": 4.0,
        "alert_severity": "INFO",
        "description": "API Test description"
    }
    res_create = client.post("/api/zones", json=new_zone_payload)
    assert res_create.status_code == 200
    assert res_create.json()["status"] == "success"

    # 3. Delete the zone
    res_del = client.delete("/api/zones/zone_api_test")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "deleted"


def test_incidents_endpoint():
    response = client.get("/api/incidents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


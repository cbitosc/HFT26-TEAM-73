import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import io
import json
import responses
import pytest
from PIL import Image

from harness.api_client import VisionGuardAPIClient, APIResult
from harness.selftest import SelfTestRunner, CheckResult


@pytest.fixture
def client():
    return VisionGuardAPIClient(base_url="http://testbackend:8000", default_timeout=5.0)


@responses.activate
def test_get_health_success(client):
    responses.add(
        responses.GET,
        "http://testbackend:8000/health",
        json={"status": "ok", "backend_mode": "stub", "wrappers": {}},
        status=200
    )
    res = client.get_health()
    assert res.ok is True
    assert res.status_code == 200
    assert res.data["status"] == "ok"
    assert res.elapsed_ms >= 0.0


@responses.activate
def test_get_vision_status_fallback(client):
    # /vision/status returns 404, fallback to /health
    responses.add(
        responses.GET,
        "http://testbackend:8000/vision/status",
        status=404
    )
    responses.add(
        responses.GET,
        "http://testbackend:8000/health",
        json={
            "status": "ok",
            "backend_mode": "stub",
            "wrappers": {"board_detector": {"available": True, "status": "Ready"}}
        },
        status=200
    )
    res = client.get_vision_status()
    assert res.ok is True
    assert "models" in res.data
    assert "board_detector" in res.data["models"]


@responses.activate
def test_network_error_graceful_handling(client):
    # Server completely unreachable (ConnectionRefused)
    responses.add(
        responses.GET,
        "http://testbackend:8000/health",
        body=Exception("Connection refused")
    )
    res = client.get_health()
    assert res.ok is False
    assert res.status_code == 0
    assert "Network error" in res.error
    assert res.data is None


@responses.activate
def test_create_and_get_circuit(client):
    mock_circuit = {
        "version": 1,
        "id": "c123",
        "board": {"template": "arduino-uno"},
        "parts": [],
        "connections": [],
        "inspection": []
    }
    responses.add(
        responses.POST,
        "http://testbackend:8000/circuits",
        json=mock_circuit,
        status=201
    )
    responses.add(
        responses.GET,
        "http://testbackend:8000/circuits/c123",
        json=mock_circuit,
        status=200
    )

    create_res = client.create_circuit()
    assert create_res.ok is True
    assert create_res.status_code == 201
    assert create_res.data["id"] == "c123"

    get_res = client.get_circuit("c123")
    assert get_res.ok is True
    assert get_res.data["id"] == "c123"


@responses.activate
def test_update_circuit_user_flags(client):
    mock_updated = {
        "circuit": {
            "version": 1,
            "id": "c123",
            "board": {"template": "arduino-uno"},
            "parts": [],
            "connections": [{"from": "uno:13", "to": "r1:1", "source": "user", "conf": 1.0, "confirmed": True}],
            "inspection": []
        },
        "validation": {"status": "pass", "findings": []}
    }
    responses.add(
        responses.PUT,
        "http://testbackend:8000/circuits/c123",
        json=mock_updated,
        status=200
    )

    res = client.update_circuit("c123", {"connections": [{"from": "uno:13", "to": "r1:1"}]})
    assert res.ok is True
    conn = res.data["circuit"]["connections"][0]
    assert conn["source"] == "user"
    assert conn["confirmed"] is True
    assert conn["conf"] == 1.0


@responses.activate
def test_validate_circuit(client):
    responses.add(
        responses.POST,
        "http://testbackend:8000/circuits/c123/validate",
        json={"status": "warn", "findings": [{"rule_id": "R010", "severity": "info", "message": "Unconfirmed wires"}]},
        status=200
    )
    res = client.validate_circuit("c123")
    assert res.ok is True
    assert res.data["status"] == "warn"
    assert len(res.data["findings"]) == 1


@responses.activate
def test_exports(client):
    responses.add(
        responses.POST,
        "http://testbackend:8000/circuits/c123/export",
        body="<svg><rect/></svg>",
        content_type="image/svg+xml",
        status=200
    )
    responses.add(
        responses.POST,
        "http://testbackend:8000/circuits/c123/export/wokwi",
        json={
            "diagram": {
                "version": 1,
                "editor": "wokwi",
                "parts": [{"id": "uno", "type": "wokwi-arduino-uno"}],
                "connections": []
            },
            "unsupported_parts": [],
            "proposed_wires": []
        },
        status=200
    )

    svg_res = client.export_circuit("c123", "svg")
    assert svg_res.ok is True
    assert "<svg>" in svg_res.raw_text

    wokwi_res = client.export_wokwi("c123")
    assert wokwi_res.ok is True
    assert "diagram" in wokwi_res.data
    assert wokwi_res.data["diagram"]["editor"] == "wokwi"


@responses.activate
def test_analyze_image_fallback(client):
    # First /analyze returns 404, then fallback to /circuits/analyze succeeds
    responses.add(
        responses.POST,
        "http://testbackend:8000/analyze",
        status=404
    )
    responses.add(
        responses.POST,
        "http://testbackend:8000/circuits/analyze",
        json={
            "circuit": {"id": "c_analyzed", "parts": [], "connections": []},
            "validation": {"status": "pass", "findings": []}
        },
        status=200
    )

    img = Image.new("RGB", (100, 100), color=(50, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    res = client.analyze_image(buf.getvalue())
    assert res.ok is True
    assert res.data["circuit"]["id"] == "c_analyzed"


@responses.activate
def test_selftest_runner_mocked(client):
    # Set up basic responses for selftest checks
    responses.add(
        responses.GET,
        "http://testbackend:8000/health",
        json={
            "status": "ok",
            "backend_mode": "stub",
            "wrappers": {
                "board_detector": {"available": True, "status": "Ready"},
                "component_detector": {"available": True, "status": "Ready"},
                "slot_inspector": {"available": True, "status": "Ready"}
            }
        },
        status=200
    )
    responses.add(
        responses.GET,
        "http://testbackend:8000/vision/status",
        json={
            "models": {
                "board_detector": {"available": True},
                "component_detector": {"available": True},
                "slot_inspector": {"available": True}
            }
        },
        status=200
    )

    runner = SelfTestRunner(client=client)
    runner.check_01_health()
    runner.check_02_vision_status()

    summary = runner.summary()
    assert summary["total"] == 2
    assert summary["passed"] == 2
    assert summary["failed"] == 0


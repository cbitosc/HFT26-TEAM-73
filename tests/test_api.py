import io
from fastapi.testclient import TestClient
from PIL import Image

from visionguard.app import app
from visionguard.repository.memory import default_repository

client = TestClient(app)

def create_test_image_bytes():
    img = Image.new("RGB", (640, 480), color=(30, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_root_and_health():
    res = client.get("/")
    assert res.status_code == 200
    assert "version" in res.json()

    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "wrappers" in data

def test_schema_and_templates():
    res = client.get("/circuits/schema")
    assert res.status_code == 200
    schema = res.json()
    assert schema["title"] == "Circuit"

    res = client.get("/templates")
    assert res.status_code == 200
    tmpl = res.json()
    assert "arduino-uno" in tmpl["boards"]
    assert "led" in tmpl["parts"]

def test_circuit_crud():
    # 1. Create circuit
    res = client.post("/circuits")
    assert res.status_code == 201
    circuit = res.json()
    cid = circuit["id"]
    assert circuit["version"] == 1
    assert circuit["board"]["template"] == "arduino-uno"

    # 2. Get circuit
    res = client.get(f"/circuits/{cid}")
    assert res.status_code == 200
    assert res.json()["id"] == cid

    # 3. Update circuit (user edit)
    update_payload = {
        "id": cid,
        "version": 1,
        "board": {"template": "arduino-uno"},
        "parts": [
            {"id": "r1", "type": "resistor", "value": "220"},
            {"id": "led1", "type": "led", "color": "red"}
        ],
        "connections": [
            {"from": "uno:13", "to": "r1:1"},
            {"from": "r1:2", "to": "led1:A"},
            {"from": "led1:C", "to": "uno:GND.1"}
        ]
    }
    res = client.put(f"/circuits/{cid}", json=update_payload)
    assert res.status_code == 200
    data = res.json()
    updated_circuit = data["circuit"]
    assert updated_circuit["version"] == 2
    # Rule check: user edit sets source="user", conf=1.0, confirmed=True
    for p in updated_circuit["parts"]:
        assert p["source"] == "user"
        assert p["confirmed"] is True
        assert p["conf"] == 1.0
    for c in updated_circuit["connections"]:
        assert c["source"] == "user"
        assert c["confirmed"] is True
        assert c["conf"] == 1.0

    # 4. Validation endpoint
    res = client.get(f"/circuits/{cid}/validate?target=led-blink-pin13")
    assert res.status_code == 200
    val_data = res.json()
    assert val_data["status"] == "pass"

    # 5. Explain endpoint
    res = client.post(f"/circuits/{cid}/explain")
    assert res.status_code == 200
    exp_data = res.json()
    assert "verdict" in exp_data
    assert "explanation" in exp_data

    # 6. Exports
    res = client.get(f"/circuits/{cid}/export/json")
    assert res.status_code == 200
    assert res.json()["id"] == cid

    res = client.get(f"/circuits/{cid}/export/svg")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/svg+xml")

    res = client.get(f"/circuits/{cid}/export/wokwi")
    assert res.status_code == 200
    wokwi_data = res.json()
    assert "diagram" in wokwi_data
    assert len(wokwi_data["diagram"]["connections"]) == 3

def test_analyze_image():
    from visionguard.config import settings
    old_mode = settings.VISION_BACKEND
    try:
        settings.VISION_BACKEND = "stub"
        img_bytes = create_test_image_bytes()
        files = {"file": ("pcb.jpg", img_bytes, "image/jpeg")}
        res = client.post("/circuits/analyze?connections=proximity,template", files=files)
        assert res.status_code == 200
        assert res.headers.get("x-vision-mode") == "stub"
        data = res.json()
        assert "circuit" in data
        assert "validation" in data

        circuit = data["circuit"]
        assert len(circuit["parts"]) > 0
        assert len(circuit["detections"]) > 0
        # Parts start unconfirmed and labelled source="stub" in stub mode!
        for p in circuit["parts"]:
            assert p["confirmed"] is False
            assert p["source"] == "stub"
        # STUB_MODE_ACTIVE finding must be present
        findings = data["validation"]["findings"]
        f_ids = [f["rule_id"] for f in findings]
        assert "STUB_MODE_ACTIVE" in f_ids
    finally:
        settings.VISION_BACKEND = old_mode

def test_live_frame():
    # First create an empty circuit
    res = client.post("/circuits")
    cid = res.json()["id"]

    img_bytes = create_test_image_bytes()
    # Feed multiple frames to test promotion
    for _ in range(6):
        files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
        res = client.post(f"/circuits/{cid}/live-frame", files=files)
        assert res.status_code == 200

    data = res.json()
    assert "circuit" in data
    assert "active_tracks" in data
    assert "promoted_count" in data


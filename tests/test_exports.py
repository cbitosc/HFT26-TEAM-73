import json
from visionguard.models.circuit import Circuit, Part, Connection
from visionguard.exports.svg import generate_circuit_svg
from visionguard.exports.png import render_circuit_png
from visionguard.exports.wokwi import export_wokwi

def build_sample_circuit():
    return Circuit(
        id="test-export-circuit",
        board={"template": "arduino-uno"},
        parts=[
            Part(id="r1", type="resistor", value="220", conf=1.0, confirmed=True, source="user"),
            Part(id="led1", type="led", color="red", conf=1.0, confirmed=True, source="user"),
            Part(id="btn1", type="pushbutton", conf=0.7, confirmed=False, source="vision"),
            Part(id="mystery", type="sensor_xyz", conf=0.5, confirmed=False, source="vision"),
        ],
        connections=[
            Connection(**{"from": "uno:13", "to": "r1:1", "source": "user", "conf": 1.0, "confirmed": True}),
            Connection(**{"from": "r1:2", "to": "led1:A", "source": "user", "conf": 1.0, "confirmed": True}),
            Connection(**{"from": "led1:C", "to": "uno:GND.1", "source": "user", "conf": 1.0, "confirmed": True}),
            # Unconfirmed proposed wire
            Connection(**{"from": "uno:5V", "to": "btn1:1.l", "source": "vision", "conf": 0.45, "confirmed": False}),
        ]
    )

def test_svg_export():
    circuit = build_sample_circuit()
    svg = generate_circuit_svg(circuit)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "r1" in svg
    assert "led1" in svg
    assert "btn1" in svg
    assert "ARDUINO UNO" in svg
    # Confirmed wire solid, unconfirmed dashed
    assert 'stroke-dasharray="none"' in svg
    assert 'stroke-dasharray="6,4"' in svg

def test_png_export_graceful_degradation():
    circuit = build_sample_circuit()
    # Should either return valid png bytes or a graceful error string, never crash
    png_bytes, err = render_circuit_png(circuit)
    if png_bytes is not None:
        assert len(png_bytes) > 0
        assert png_bytes.startswith(b"\x89PNG")
    else:
        assert err is not None
        assert "cairo" in err.lower()

def test_wokwi_export():
    circuit = build_sample_circuit()
    wokwi_out = export_wokwi(circuit)

    assert "diagram" in wokwi_out
    assert "netlist_text" in wokwi_out
    assert "unsupported_parts" in wokwi_out
    assert "proposed_wires" in wokwi_out

    diagram = wokwi_out["diagram"]
    assert diagram["version"] == 1
    assert diagram["author"] == "VisionGuard"

    part_ids = [p["id"] for p in diagram["parts"]]
    assert "uno" in part_ids
    assert "r1" in part_ids
    assert "led1" in part_ids

    # Check attrs
    r1_part = next(p for p in diagram["parts"] if p["id"] == "r1")
    assert r1_part["attrs"]["value"] == "220"

    led_part = next(p for p in diagram["parts"] if p["id"] == "led1")
    assert led_part["attrs"]["color"] == "red"

    # Unsupported part "mystery" must NOT be silently dropped!
    unsupported_ids = [u["id"] for u in wokwi_out["unsupported_parts"]]
    assert "mystery" in unsupported_ids

    # In diagram["connections"], ONLY CONFIRMED wires must be exported!
    conn_list = diagram["connections"]
    assert len(conn_list) == 3
    for c in conn_list:
        assert len(c) == 4
        # Verify wire colors (ground is black, signal is green)
        if "GND" in c[0] or "GND" in c[1]:
            assert c[2] == "black"
        else:
            assert c[2] == "green"

    # Unconfirmed wire must be in proposed_wires
    assert len(wokwi_out["proposed_wires"]) == 1
    assert wokwi_out["proposed_wires"][0]["from"] == "uno:5V"

    # Netlist text contains sections
    assert "# CONFIRMED" in wokwi_out["netlist_text"]
    assert "# PROPOSED (unconfirmed)" in wokwi_out["netlist_text"]


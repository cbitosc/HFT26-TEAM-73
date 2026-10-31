from visionguard.models.circuit import Circuit, Part, Connection, InspectionSlot
from visionguard.validation.engine import validate_circuit

def test_r001_unsupported_pin():
    circuit = Circuit(
        id="test-r001",
        board={"template": "arduino-uno"},
        parts=[Part(id="r1", type="resistor", conf=0.9, confirmed=True)],
        connections=[
            # Invalid pin 99 on Uno
            Connection(**{"from": "uno:99", "to": "r1:1", "source": "user", "conf": 1.0, "confirmed": True}),
            # Invalid pin 5 on Resistor
            Connection(**{"from": "r1:5", "to": "uno:GND.1", "source": "user", "conf": 1.0, "confirmed": True}),
        ]
    )
    result = validate_circuit(circuit)
    rule_ids = [f.rule_id for f in result.findings]
    assert "R001" in rule_ids
    assert result.status == "fail"

def test_r002_unknown_part_ref():
    circuit = Circuit(
        id="test-r002",
        board={"template": "arduino-uno"},
        parts=[Part(id="led1", type="led", conf=0.9, confirmed=True)],
        connections=[
            # References non-existent part 'r99'
            Connection(**{"from": "uno:13", "to": "r99:1", "source": "user", "conf": 1.0, "confirmed": True}),
        ]
    )
    result = validate_circuit(circuit)
    rule_ids = [f.rule_id for f in result.findings]
    assert "R002" in rule_ids
    assert result.status == "fail"

def test_r003_duplicate_connection():
    circuit = Circuit(
        id="test-r003",
        board={"template": "arduino-uno"},
        parts=[Part(id="r1", type="resistor", conf=0.9, confirmed=True)],
        connections=[
            Connection(**{"from": "uno:13", "to": "r1:1", "source": "user", "conf": 1.0, "confirmed": True}),
            # Reverse order of identical connection
            Connection(**{"from": "r1:1", "to": "uno:13", "source": "user", "conf": 1.0, "confirmed": True}),
        ]
    )
    result = validate_circuit(circuit)
    rule_ids = [f.rule_id for f in result.findings]
    assert "R003" in rule_ids

def test_r004_self_connection():
    circuit = Circuit(
        id="test-r004",
        board={"template": "arduino-uno"},
        parts=[Part(id="r1", type="resistor", conf=0.9, confirmed=True)],
        connections=[
            # Resistor pin 1 connected to its own pin 2
            Connection(**{"from": "r1:1", "to": "r1:2", "source": "user", "conf": 1.0, "confirmed": True}),
        ]
    )
    result = validate_circuit(circuit)
    rule_ids = [f.rule_id for f in result.findings]
    assert "R004" in rule_ids
    assert result.status == "fail"

def test_r005_required_pins_unconnected():
    # Resistor has pin 1 connected, but pin 2 left floating
    circuit = Circuit(
        id="test-r005",
        board={"template": "arduino-uno"},
        parts=[Part(id="r1", type="resistor", conf=0.9, confirmed=True)],
        connections=[
            Connection(**{"from": "uno:13", "to": "r1:1", "source": "user", "conf": 1.0, "confirmed": True}),
        ]
    )
    result = validate_circuit(circuit)
    rule_ids = [f.rule_id for f in result.findings]
    assert "R005" in rule_ids

def test_r006_required_parts_missing():
    # Target led-blink-pin13 requires both led and resistor
    circuit = Circuit(
        id="test-r006",
        board={"template": "arduino-uno"},
        parts=[Part(id="led1", type="led", conf=0.9, confirmed=True)],
        connections=[]
    )
    result = validate_circuit(circuit, target_name="led-blink-pin13")
    rule_ids = [f.rule_id for f in result.findings]
    assert "R006" in rule_ids
    assert result.status == "fail"

def test_r007_slot_reported_missing():
    circuit = Circuit(
        id="test-r007",
        board={"template": "arduino-uno"},
        parts=[],
        connections=[],
        inspection=[
            InspectionSlot(slot="slot_mcu", state="missing", conf=0.95),
            InspectionSlot(slot="slot_usb", state="skipped", conf=0.0),
        ]
    )
    result = validate_circuit(circuit)
    rule_ids = [f.rule_id for f in result.findings]
    assert "R007" in rule_ids
    # missing is warning, so status should be warn
    assert result.status == "warn"

def test_r008_voltage_limit_mismatch():
    # Direct short from 5V to GND
    circuit = Circuit(
        id="test-r008",
        board={"template": "arduino-uno"},
        parts=[],
        connections=[
            Connection(**{"from": "uno:5V", "to": "uno:GND.1", "source": "user", "conf": 1.0, "confirmed": True}),
        ]
    )
    result = validate_circuit(circuit)
    rule_ids = [f.rule_id for f in result.findings]
    assert "R008" in rule_ids
    assert result.status == "fail"

    # High voltage VIN to unverified component
    circuit_vin = Circuit(
        id="test-r008-vin",
        board={"template": "arduino-uno"},
        parts=[Part(id="btn1", type="pushbutton", conf=1.0, confirmed=True, source="user")],
        connections=[
            Connection(**{"from": "uno:VIN", "to": "btn1:1.l", "source": "user", "conf": 1.0, "confirmed": True}),
        ]
    )
    res_vin = validate_circuit(circuit_vin)
    rule_ids_vin = [f.rule_id for f in res_vin.findings]
    assert "R008" in rule_ids_vin
    assert res_vin.status == "warn"

def test_r009_led_needs_series_resistor_triggered():
    # LED directly between pin 13 and GND with NO resistor
    circuit = Circuit(
        id="test-r009-bad",
        board={"template": "arduino-uno"},
        parts=[Part(id="led1", type="led", conf=0.9, confirmed=True)],
        connections=[
            Connection(**{"from": "uno:13", "to": "led1:A", "source": "user", "conf": 1.0, "confirmed": True}),
            Connection(**{"from": "led1:C", "to": "uno:GND.1", "source": "user", "conf": 1.0, "confirmed": True}),
        ]
    )
    result = validate_circuit(circuit)
    rule_ids = [f.rule_id for f in result.findings]
    assert "R009" in rule_ids
    assert result.status == "fail"

def test_r009_led_with_series_resistor_passes():
    # LED with resistor in series: uno:13 -> r1:1, r1:2 -> led1:A, led1:C -> uno:GND.1
    circuit = Circuit(
        id="test-r009-good",
        board={"template": "arduino-uno"},
        parts=[
            Part(id="r1", type="resistor", value="220", conf=1.0, confirmed=True, source="user"),
            Part(id="led1", type="led", color="red", conf=1.0, confirmed=True, source="user"),
        ],
        connections=[
            Connection(**{"from": "uno:13", "to": "r1:1", "source": "user", "conf": 1.0, "confirmed": True}),
            Connection(**{"from": "r1:2", "to": "led1:A", "source": "user", "conf": 1.0, "confirmed": True}),
            Connection(**{"from": "led1:C", "to": "uno:GND.1", "source": "user", "conf": 1.0, "confirmed": True}),
        ]
    )
    result = validate_circuit(circuit, target_name="led-blink-pin13")
    rule_ids = [f.rule_id for f in result.findings]
    assert "R009" not in rule_ids
    assert result.status == "pass"
    assert result.target_comparison["matched"] is True

def test_r010_unconfirmed_wires():
    circuit = Circuit(
        id="test-r010",
        board={"template": "arduino-uno"},
        parts=[
            Part(id="r1", type="resistor", conf=0.9, confirmed=False, source="vision"),
        ],
        connections=[
            Connection(**{"from": "uno:13", "to": "r1:1", "source": "vision", "conf": 0.5, "confirmed": False}),
        ]
    )
    result = validate_circuit(circuit)
    rule_ids = [f.rule_id for f in result.findings]
    assert "R010" in rule_ids
    # Unconfirmed items must NEVER count as pass
    assert result.status != "pass"

def test_r011_unsupported_part():
    circuit = Circuit(
        id="test-r011",
        board={"template": "arduino-uno"},
        parts=[
            Part(id="mystery1", type="unknown", conf=0.7, confirmed=True, source="user"),
        ],
        connections=[]
    )
    result = validate_circuit(circuit)
    rule_ids = [f.rule_id for f in result.findings]
    assert "R011" in rule_ids
    assert result.status == "warn"

def test_r012_weak_class_detections():
    circuit = Circuit(
        id="test-r012",
        board={"template": "arduino-uno"},
        parts=[
            # LED detection with conf 0.35 (below 0.60 threshold)
            Part(id="led1", type="led", conf=0.35, confirmed=False, source="vision"),
        ],
        connections=[]
    )
    result = validate_circuit(circuit)
    rule_ids = [f.rule_id for f in result.findings]
    assert "R012" in rule_ids
    assert result.status == "warn"

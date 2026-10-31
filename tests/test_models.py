import pytest
from visionguard.models.circuit import Circuit, Part, Connection, InspectionSlot, create_empty_circuit

def test_create_empty_circuit():
    circuit = create_empty_circuit(board_template="arduino-uno")
    assert circuit.version == 1
    assert circuit.board.template == "arduino-uno"
    assert len(circuit.parts) == 0
    assert len(circuit.connections) == 0
    assert len(circuit.inspection) == 0
    assert len(circuit.id) > 0

def test_unique_part_ids_validation():
    # Duplicate part ids should raise ValueError
    with pytest.raises(ValueError, match="Duplicate part id"):
        Circuit(
            id="test-1",
            board={"template": "arduino-uno"},
            parts=[
                Part(id="led1", type="led", conf=0.9, confirmed=False),
                Part(id="led1", type="led", conf=0.8, confirmed=False),
            ]
        )

def test_pin_normalization_on_circuit():
    circuit = Circuit(
        id="test-norm",
        board={"template": "arduino-uno"},
        parts=[
            Part(id="r1", type="resistor", conf=0.9, confirmed=False),
            Part(id="led1", type="led", conf=0.9, confirmed=False),
        ],
        connections=[
            Connection(**{"from": "uno:d13", "to": "r1:pin1", "source": "vision", "conf": 0.8, "confirmed": False}),
            Connection(**{"from": "r1:p2", "to": "led1:anode", "source": "vision", "conf": 0.8, "confirmed": False}),
            Connection(**{"from": "led1:cathode", "to": "uno:gnd", "source": "vision", "conf": 0.8, "confirmed": False}),
        ]
    )

    # Verify pins were normalized: d13 -> 13, pin1 -> 1, p2 -> 2, anode -> A, cathode -> C, gnd -> GND.1
    c1, c2, c3 = circuit.connections
    assert c1.from_ == "uno:13"
    assert c1.to == "r1:1"
    assert c2.from_ == "r1:2"
    assert c2.to == "led1:A"
    assert c3.from_ == "led1:C"
    assert c3.to == "uno:GND.1"

def test_part_and_connection_confirm_user():
    part = Part(id="led1", type="led", source="vision", conf=0.4, confirmed=False)
    part.confirm_user()
    assert part.source == "user"
    assert part.conf == 1.0
    assert part.confirmed is True

    conn = Connection(**{"from": "uno:13", "to": "r1:1", "source": "vision", "conf": 0.5, "confirmed": False})
    conn.confirm_user()
    assert conn.source == "user"
    assert conn.conf == 1.0
    assert conn.confirmed is True

def test_model_json_schema():
    schema = Circuit.model_json_schema()
    assert "properties" in schema
    assert "version" in schema["properties"]
    assert "board" in schema["properties"]
    assert "parts" in schema["properties"]
    assert "connections" in schema["properties"]
    assert "inspection" in schema["properties"]


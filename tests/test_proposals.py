from visionguard.models.circuit import Circuit, Part, Connection
from visionguard.proposals.template import propose_from_template
from visionguard.proposals.proximity import propose_from_proximity
from visionguard.proposals.manager import apply_proposals, apply_user_updates

def test_propose_from_template():
    circuit = Circuit(
        id="test-tmpl",
        board={"template": "arduino-uno"},
        parts=[
            Part(id="led1", type="led", conf=0.8, confirmed=False),
        ],
        connections=[]
    )
    proposals = propose_from_template(circuit)
    assert len(proposals) >= 1
    # Proposes cathode to ground rail
    p = proposals[0]
    assert p.from_ == "led1:C"
    assert p.to == "uno:GND.1"
    assert p.source == "template"
    assert p.confirmed is False

def test_propose_from_proximity():
    # Place a resistor very close to header pin 13 (around x=0.38, y=0.05)
    circuit = Circuit(
        id="test-prox",
        board={"template": "arduino-uno"},
        parts=[
            Part(id="r1", type="resistor", bbox=[0.36, 0.04, 0.04, 0.04], conf=0.8, confirmed=False),
        ],
        connections=[]
    )
    proposals = propose_from_proximity(circuit)
    assert len(proposals) >= 1
    p = proposals[0]
    assert "uno:13" in (p.from_, p.to)
    assert "r1:1" in (p.from_, p.to)
    assert p.source == "vision"
    assert p.confirmed is False
    assert 0.35 <= p.conf <= 0.60

def test_apply_proposals():
    circuit = Circuit(
        id="test-apply",
        board={"template": "arduino-uno"},
        parts=[
            Part(id="led1", type="led", bbox=[0.38, 0.08, 0.05, 0.05], conf=0.8, confirmed=False),
        ],
        connections=[]
    )
    circuit, findings = apply_proposals(circuit, use_template=True, use_proximity=True, use_llm=False)
    assert len(circuit.connections) >= 1
    # Ensure all proposals start confirmed=False
    for c in circuit.connections:
        assert c.confirmed is False

def test_apply_user_updates():
    saved = Circuit(
        id="test-user-up",
        version=1,
        board={"template": "arduino-uno"},
        parts=[
            Part(id="r1", type="resistor", conf=0.4, confirmed=False, source="vision"),
        ],
        connections=[
            Connection(**{"from": "uno:13", "to": "r1:1", "source": "vision", "conf": 0.45, "confirmed": False})
        ]
    )

    update_payload = {
        "id": "test-user-up",
        "version": 1,
        "board": {"template": "arduino-uno"},
        "parts": [
            {"id": "r1", "type": "resistor", "value": "220"}
        ],
        "connections": [
            {"from": "uno:13", "to": "r1:1"}
        ]
    }

    updated = apply_user_updates(saved, update_payload)
    assert updated.version == 2
    # User edit rules: source="user", conf=1.0, confirmed=True
    assert updated.parts[0].source == "user"
    assert updated.parts[0].conf == 1.0
    assert updated.parts[0].confirmed is True

    assert updated.connections[0].source == "user"
    assert updated.connections[0].conf == 1.0
    assert updated.connections[0].confirmed is True

def test_apply_user_updates_confirm_all():
    saved = Circuit(
        id="test-confirm-all",
        version=1,
        board={"template": "arduino-uno"},
        parts=[
            Part(id="r1", type="resistor", conf=0.4, confirmed=False, source="vision"),
            Part(id="led1", type="led", conf=0.5, confirmed=False, source="vision"),
        ],
        connections=[
            Connection(**{"from": "uno:13", "to": "r1:1", "source": "vision", "conf": 0.45, "confirmed": False})
        ]
    )

    updated = apply_user_updates(saved, {"confirm_all": True})
    assert updated.version == 2
    for p in updated.parts:
        assert p.confirmed is True
        assert p.conf == 1.0
        assert p.source == "user"
    for c in updated.connections:
        assert c.confirmed is True
        assert c.conf == 1.0
        assert c.source == "user"

def test_propose_from_vision_llm_result_unpacking():
    from visionguard.proposals.llm import ProposalResult, propose_from_vision_llm

    # Test ProposalResult directly
    dummy_parts = [Part(id="r1", type="resistor", conf=0.8, confirmed=False)]
    dummy_conns = [Connection(**{"from": "uno:13", "to": "r1:1", "conf": 0.7, "confirmed": False})]
    dummy_findings = []
    res = ProposalResult(dummy_parts, dummy_conns, dummy_findings)

    # 2-tuple unpacking
    proposal, findings = res
    assert "parts" in proposal
    assert "connections" in proposal
    assert len(proposal["parts"]) == 1

    # 3-tuple unpacking
    parts, conns, finds = res
    assert len(parts) == 1
    assert len(conns) == 1

    # Dict access
    assert len(res["parts"]) == 1
    assert len(res["connections"]) == 1

    # Real call without API key returns graceful empty proposal
    res_call = propose_from_vision_llm(b"fake_image_bytes")
    p_call, f_call = res_call
    assert len(p_call["parts"]) == 0
    assert len(p_call["connections"]) == 0
    assert len(f_call) >= 1

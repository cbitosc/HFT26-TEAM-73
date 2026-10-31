from typing import List, Optional, Set, Tuple
from visionguard.models.circuit import Circuit, Connection, Part
from visionguard.models.finding import Finding
from visionguard.proposals.template import propose_from_template
from visionguard.proposals.proximity import propose_from_proximity
from visionguard.proposals.llm import propose_from_vision_llm

def apply_proposals(
    circuit: Circuit,
    use_template: bool = True,
    use_proximity: bool = True,
    use_llm: bool = False,
    image_bytes: Optional[bytes] = None
) -> Tuple[Circuit, List[Finding]]:
    """
    Combines connection proposals from template rules, proximity heuristics,
    and optional multimodal Vision LLM.
    Ensures all proposals start confirmed=False.
    """
    findings: List[Finding] = []
    new_connections: List[Connection] = list(circuit.connections)
    new_parts: List[Part] = list(circuit.parts)

    existing_wires = {(c.from_, c.to) for c in new_connections}
    existing_wires.update({(c.to, c.from_) for c in new_connections})

    # 1. Proximity proposals
    if use_proximity:
        prox_conns = propose_from_proximity(circuit)
        for c in prox_conns:
            if (c.from_, c.to) not in existing_wires and (c.to, c.from_) not in existing_wires:
                new_connections.append(c)
                existing_wires.add((c.from_, c.to))

    # 2. Template proposals
    if use_template:
        tmpl_conns = propose_from_template(circuit)
        for c in tmpl_conns:
            if (c.from_, c.to) not in existing_wires and (c.to, c.from_) not in existing_wires:
                new_connections.append(c)
                existing_wires.add((c.from_, c.to))

    # 3. Multimodal LLM proposals (optional)
    if use_llm and image_bytes:
        llm_parts, llm_conns, llm_findings = propose_from_vision_llm(image_bytes)
        findings.extend(llm_findings)

        # Merge new parts from LLM if not already detected
        existing_part_ids = {p.id for p in new_parts}
        for lp in llm_parts:
            if lp.id not in existing_part_ids:
                new_parts.append(lp)
                existing_part_ids.add(lp.id)

        # Merge connections
        for lc in llm_conns:
            if (lc.from_, lc.to) not in existing_wires and (lc.to, lc.from_) not in existing_wires:
                new_connections.append(lc)
                existing_wires.add((lc.from_, lc.to))

    circuit.parts = new_parts
    circuit.connections = new_connections
    return circuit, findings

def apply_user_updates(saved_circuit: Circuit, update_payload: dict) -> Circuit:
    """
    Applies user edits to a saved circuit.
    Non-negotiable rule: Any user edit sets source='user', conf=1.0, confirmed=True.
    """
    payload = dict(update_payload)

    # Support shorthand confirmation: {"confirm_all": True}
    if payload.get("confirm_all"):
        confirmed_parts = [
            p.model_copy(update={"source": "user", "conf": 1.0, "confirmed": True})
            for p in saved_circuit.parts
        ]
        confirmed_conns = [
            c.model_copy(update={"source": "user", "conf": 1.0, "confirmed": True})
            for c in saved_circuit.connections
        ]
        return Circuit(
            version=saved_circuit.version + 1,
            id=saved_circuit.id,
            image_id=saved_circuit.image_id,
            board=saved_circuit.board,
            parts=confirmed_parts,
            connections=confirmed_conns,
            inspection=saved_circuit.inspection,
            detections=saved_circuit.detections,
            defects=saved_circuit.defects,
            health=saved_circuit.health
        )

    # Fill defaults from saved circuit for partial update payloads
    if "id" not in payload:
        payload["id"] = saved_circuit.id
    if "board" not in payload:
        payload["board"] = saved_circuit.board.model_dump()
    if "version" not in payload:
        payload["version"] = saved_circuit.version
    if "parts" not in payload:
        payload["parts"] = [p.model_dump() for p in saved_circuit.parts]
    if "connections" not in payload:
        payload["connections"] = [c.model_dump(by_alias=True) for c in saved_circuit.connections]

    incoming_circuit = Circuit.model_validate(payload)

    # Any parts/connections in incoming payload get confirmed=True, source="user", conf=1.0
    confirmed_parts: List[Part] = []
    for p in incoming_circuit.parts:
        p_copy = p.model_copy()
        p_copy.source = "user"
        p_copy.conf = 1.0
        p_copy.confirmed = True
        confirmed_parts.append(p_copy)

    confirmed_connections: List[Connection] = []
    for c in incoming_circuit.connections:
        c_copy = c.model_copy()
        c_copy.source = "user"
        c_copy.conf = 1.0
        c_copy.confirmed = True
        confirmed_connections.append(c_copy)

    # Return updated circuit
    return Circuit(
        version=saved_circuit.version + 1,
        id=saved_circuit.id,
        image_id=saved_circuit.image_id,
        board=incoming_circuit.board,
        parts=confirmed_parts,
        connections=confirmed_connections,
        inspection=saved_circuit.inspection,
        detections=saved_circuit.detections,
        defects=saved_circuit.defects,
        health=saved_circuit.health
    )


from typing import Any, Dict, List, Tuple
from visionguard.models.circuit import Circuit, Connection
from visionguard.templates.boards import PART_SPECS, get_wokwi_pin

USER_DOCUMENTATION_NOTE = (
    "NOTE: The Wokwi export format is generated based on standard Wokwi diagram.json conventions. "
    "To simulate, paste the returned 'diagram' object directly into the diagram.json tab of a new "
    "Wokwi project (https://wokwi.com). If Wokwi rejects any specific pin name, check the pin table "
    "in visionguard/templates/boards.py."
)

def determine_wire_color(endpoint1: str, endpoint2: str, explicit_color: str = None) -> str:
    """Wire colors: black for GND, red for 5V/3.3V/VIN, otherwise green."""
    if explicit_color and explicit_color in ("black", "red", "green", "blue", "yellow", "orange"):
        return explicit_color
    eps = f"{endpoint1} {endpoint2}".upper()
    if "GND" in eps:
        return "black"
    if "5V" in eps or "3.3V" in eps or "VIN" in eps:
        return "red"
    return "green"

def format_wokwi_endpoint(raw_endpoint: str, part_type_map: Dict[str, str]) -> str:
    """Translates 'partId:pin' to Wokwi endpoint format with mapped pin names."""
    if ":" not in raw_endpoint:
        return raw_endpoint
    part_id, pin = raw_endpoint.split(":", 1)
    part_id = part_id.strip()
    pin = pin.strip()

    if part_id.lower() in ("uno", "board", "arduino", "arduino-uno"):
        wokwi_pin = get_wokwi_pin("arduino-uno", pin)
        return f"uno:{wokwi_pin}"

    ptype = part_type_map.get(part_id, "unknown")
    wokwi_pin = get_wokwi_pin(ptype, pin)
    return f"{part_id}:{wokwi_pin}"

def export_wokwi(circuit: Circuit) -> Dict[str, Any]:
    """
    Exports circuit to Wokwi diagram.json structure.
    - Only CONFIRMED wires are included in diagram.json
    - Parts with no mapping are routed to unsupported_parts (never dropped silently)
    - Auto-layouts parts left-to-right
    - Returns {diagram, netlist_text, unsupported_parts, proposed_wires, note}
    """
    wokwi_parts: List[Dict[str, Any]] = []
    unsupported_parts: List[Dict[str, Any]] = []

    # 1. Arduino Uno base part
    wokwi_parts.append({
        "type": "wokwi-arduino-uno",
        "id": "uno",
        "top": 0,
        "left": 0,
        "attrs": {}
    })

    part_type_map: Dict[str, str] = {"uno": "arduino-uno"}

    # 2. Discrete parts auto-layout left-to-right
    discrete_parts = [p for p in circuit.parts if p.type not in ("arduino-uno", "board")]
    for idx, part in enumerate(discrete_parts):
        part_type_map[part.id] = part.type
        spec = PART_SPECS.get(part.type)
        wokwi_type = part.wokwi_type or (spec.wokwi_type if spec else None)

        if not wokwi_type:
            unsupported_parts.append({
                "id": part.id,
                "type": part.type,
                "reason": f"No Wokwi component model available for type '{part.type}'"
            })
            continue

        attrs: Dict[str, Any] = {}
        if part.type == "led":
            attrs["color"] = part.color or spec.default_color or "red"
        elif part.type == "resistor":
            attrs["value"] = part.value or spec.default_value or "220"

        # Position layout: left to right with staggered vertical spacing
        left_pos = 320 + idx * 160
        top_pos = 60 if (idx % 2 == 0) else 180

        wokwi_parts.append({
            "type": wokwi_type,
            "id": part.id,
            "top": top_pos,
            "left": left_pos,
            "attrs": attrs
        })

    # 3. Connections: ONLY confirmed wires are exported in diagram.json
    confirmed_connections: List[List[Any]] = []
    proposed_wires: List[Dict[str, Any]] = []

    netlist_confirmed_lines: List[str] = ["# CONFIRMED"]
    netlist_proposed_lines: List[str] = ["# PROPOSED (unconfirmed)"]

    for conn in circuit.connections:
        w_from = format_wokwi_endpoint(conn.from_, part_type_map)
        w_to = format_wokwi_endpoint(conn.to, part_type_map)
        color = determine_wire_color(w_from, w_to, conn.wire_color)

        if conn.confirmed:
            # Wokwi connection format: [ from, to, color, [ routing_instructions ] ]
            confirmed_connections.append([w_from, w_to, color, []])
            netlist_confirmed_lines.append(f"{w_from} <-> {w_to} (color: {color})")
        else:
            proposed_wires.append({
                "from": w_from,
                "to": w_to,
                "color": color,
                "conf": conn.conf,
                "source": conn.source,
                "note": conn.note or "unconfirmed proposal"
            })
            netlist_proposed_lines.append(f"{w_from} <-> {w_to} (conf: {conn.conf:.2f}, note: {conn.note or 'proposal'})")

    if len(netlist_confirmed_lines) == 1:
        netlist_confirmed_lines.append("(No confirmed wires)")
    if len(netlist_proposed_lines) == 1:
        netlist_proposed_lines.append("(No proposed wires)")

    netlist_text = "\n".join(netlist_confirmed_lines) + "\n\n" + "\n".join(netlist_proposed_lines)

    diagram = {
        "version": 1,
        "author": "VisionGuard",
        "editor": "wokwi",
        "parts": wokwi_parts,
        "connections": confirmed_connections,
        "dependencies": {}
    }

    return {
        "diagram": diagram,
        "netlist_text": netlist_text,
        "unsupported_parts": unsupported_parts,
        "proposed_wires": proposed_wires,
        "note": USER_DOCUMENTATION_NOTE
    }


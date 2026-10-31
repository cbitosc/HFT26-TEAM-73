from typing import Dict, List, Set, Tuple
from collections import defaultdict

from visionguard.models.circuit import Circuit, Connection, Part
from visionguard.models.finding import Finding
from visionguard.templates.boards import (
    PART_SPECS,
    ARDUINO_UNO_PINS,
    ARDUINO_UNO_LIMITS,
)
from visionguard.config import settings

def check_r001_unsupported_pins(circuit: Circuit) -> List[Finding]:
    """R001 UNSUPPORTED_PIN: Pin not supported by board template or part spec."""
    findings: List[Finding] = []
    part_type_map = {p.id: p.type for p in circuit.parts}

    for idx, conn in enumerate(circuit.connections):
        conn_ref = f"conn[{idx}]"
        for endpoint in (conn.from_, conn.to):
            if ":" not in endpoint:
                findings.append(Finding(
                    rule_id="R001",
                    severity="error",
                    message=f"Malformed endpoint '{endpoint}', expected 'partId:pin'",
                    refs=[conn_ref, endpoint]
                ))
                continue

            part_id, pin = endpoint.split(":", 1)
            part_id = part_id.strip()
            pin = pin.strip()

            if part_id.lower() in ("uno", "board", "arduino", "arduino-uno"):
                if pin not in ARDUINO_UNO_PINS:
                    findings.append(Finding(
                        rule_id="R001",
                        severity="error",
                        message=f"Pin '{pin}' is unsupported on board template '{circuit.board.template}'",
                        refs=[conn_ref, endpoint]
                    ))
            elif part_id in part_type_map:
                p_type = part_type_map[part_id]
                spec = PART_SPECS.get(p_type)
                if spec and pin not in spec.allowed_pins:
                    findings.append(Finding(
                        rule_id="R001",
                        severity="error",
                        message=f"Pin '{pin}' is unsupported on part '{part_id}' of type '{p_type}'",
                        refs=[conn_ref, endpoint, part_id]
                    ))
    return findings

def check_r002_unknown_part_ref(circuit: Circuit) -> List[Finding]:
    """R002 UNKNOWN_PART_REF: Connection references a partId that does not exist."""
    findings: List[Finding] = []
    known_parts = {p.id for p in circuit.parts}
    known_parts.update({"uno", "board", "arduino", "arduino-uno"})

    for idx, conn in enumerate(circuit.connections):
        conn_ref = f"conn[{idx}]"
        for endpoint in (conn.from_, conn.to):
            if ":" in endpoint:
                part_id = endpoint.split(":", 1)[0].strip()
                if part_id not in known_parts:
                    findings.append(Finding(
                        rule_id="R002",
                        severity="error",
                        message=f"Connection references unknown part '{part_id}'",
                        refs=[conn_ref, part_id]
                    ))
    return findings

def check_r003_duplicate_connections(circuit: Circuit) -> List[Finding]:
    """R003 DUPLICATE_CONNECTION: Same endpoints connected multiple times (order-independent)."""
    findings: List[Finding] = []
    seen_wires: Set[Tuple[str, str]] = set()

    for idx, conn in enumerate(circuit.connections):
        a = conn.from_.strip()
        b = conn.to.strip()
        wire_key = (min(a, b), max(a, b))
        if wire_key in seen_wires:
            findings.append(Finding(
                rule_id="R003",
                severity="warning",
                message=f"Duplicate connection between '{a}' and '{b}'",
                refs=[f"conn[{idx}]", a, b]
            ))
        else:
            seen_wires.add(wire_key)
    return findings

def check_r004_self_connection(circuit: Circuit) -> List[Finding]:
    """R004 SELF_CONNECTION: Pin connected to itself or pins on the exact same component."""
    findings: List[Finding] = []

    for idx, conn in enumerate(circuit.connections):
        a = conn.from_.strip()
        b = conn.to.strip()
        conn_ref = f"conn[{idx}]"

        if a == b:
            findings.append(Finding(
                rule_id="R004",
                severity="error",
                message=f"Self-connection: pin '{a}' is connected to itself",
                refs=[conn_ref, a]
            ))
        elif ":" in a and ":" in b:
            p1 = a.split(":", 1)[0].strip()
            p2 = b.split(":", 1)[0].strip()
            # If not the board itself (board has many independent pins), shorting pins on a discrete component is self-connection
            if p1 == p2 and p1.lower() not in ("uno", "board", "arduino", "arduino-uno"):
                findings.append(Finding(
                    rule_id="R004",
                    severity="error",
                    message=f"Component '{p1}' has its own pins shorted together: '{a}' to '{b}'",
                    refs=[conn_ref, p1, a, b]
                ))
    return findings

def check_r005_required_pins_unconnected(circuit: Circuit) -> List[Finding]:
    """R005 REQUIRED_PIN_UNCONNECTED: Per template or target, essential pins are left unconnected."""
    findings: List[Finding] = []
    # Only confirmed connections count towards connectivity
    confirmed_endpoints = set()
    for conn in circuit.connections:
        if conn.confirmed:
            confirmed_endpoints.add(conn.from_.strip())
            confirmed_endpoints.add(conn.to.strip())

    # For parts in the circuit, every 2-pin part (like LED or resistor) should have both pins connected
    for part in circuit.parts:
        spec = PART_SPECS.get(part.type)
        if spec and spec.type_name in ("led", "resistor"):
            for pin in spec.allowed_pins:
                endpoint = f"{part.id}:{pin}"
                if endpoint not in confirmed_endpoints:
                    findings.append(Finding(
                        rule_id="R005",
                        severity="warning",
                        message=f"Required pin '{endpoint}' on '{part.id}' ({part.type}) is unconnected in confirmed circuit",
                        refs=[part.id, endpoint]
                    ))
    return findings

def check_r006_required_parts_missing(circuit: Circuit, required_part_types: List[str]) -> List[Finding]:
    """R006 REQUIRED_PART_MISSING: Circuit lacks required part types for target project."""
    findings: List[Finding] = []
    present_types = {p.type for p in circuit.parts}

    for req in required_part_types:
        if req not in present_types:
            findings.append(Finding(
                rule_id="R006",
                severity="error",
                message=f"Required component type '{req}' is missing from the circuit",
                refs=[req]
            ))
    return findings

def check_r007_slot_reported_missing(circuit: Circuit) -> List[Finding]:
    """R007 SLOT_REPORTED_MISSING: Generated only if slot inspection ran."""
    findings: List[Finding] = []
    if not circuit.inspection:
        return findings

    for slot in circuit.inspection:
        if slot.state == "missing":
            findings.append(Finding(
                rule_id="R007",
                severity="warning",
                message=f"Inspection reported slot '{slot.slot}' is missing (confidence: {slot.conf:.2f})",
                refs=[slot.slot]
            ))
        elif slot.state == "skipped":
            findings.append(Finding(
                rule_id="R007",
                severity="info",
                message=f"Inspection skipped slot '{slot.slot}'",
                refs=[slot.slot]
            ))
    return findings

def check_r008_voltage_limits(circuit: Circuit) -> List[Finding]:
    """
    R008 VOLTAGE_LIMIT_MISMATCH:
    Only evaluated with verified specs; otherwise a warning saying 'unverified'.
    """
    findings: List[Finding] = []

    # Check for direct short between verified power pins and ground pins in confirmed connections
    for idx, conn in enumerate(circuit.connections):
        if not conn.confirmed:
            continue
        p1_ep = conn.from_.strip()
        p2_ep = conn.to.strip()

        # Check board pins
        limit1 = get_pin_limit(p1_ep, circuit)
        limit2 = get_pin_limit(p2_ep, circuit)

        if limit1 and limit2:
            if (limit1.is_power and limit2.is_ground) or (limit2.is_power and limit1.is_ground):
                findings.append(Finding(
                    rule_id="R008",
                    severity="error",
                    message=f"Direct short circuit detected between power pin '{p1_ep}' and ground pin '{p2_ep}'",
                    refs=[f"conn[{idx}]", p1_ep, p2_ep]
                ))
            elif limit1.nominal_voltage is not None and limit2.nominal_voltage is not None:
                if abs(limit1.nominal_voltage - limit2.nominal_voltage) > 1.5:
                    findings.append(Finding(
                        rule_id="R008",
                        severity="error",
                        message=f"Voltage mismatch between '{p1_ep}' ({limit1.nominal_voltage}V) and '{p2_ep}' ({limit2.nominal_voltage}V)",
                        refs=[f"conn[{idx}]", p1_ep, p2_ep]
                    ))
        elif limit1 and (limit1.pin == "VIN" or (limit1.nominal_voltage and limit1.nominal_voltage > 5.5)):
            findings.append(Finding(
                rule_id="R008",
                severity="warning",
                message=f"High voltage pin '{p1_ep}' ({limit1.nominal_voltage}V) connected to '{p2_ep}' with unverified voltage limits",
                refs=[f"conn[{idx}]", p1_ep, p2_ep]
            ))
        elif limit2 and (limit2.pin == "VIN" or (limit2.nominal_voltage and limit2.nominal_voltage > 5.5)):
            findings.append(Finding(
                rule_id="R008",
                severity="warning",
                message=f"High voltage pin '{p2_ep}' ({limit2.nominal_voltage}V) connected to '{p1_ep}' with unverified voltage limits",
                refs=[f"conn[{idx}]", p1_ep, p2_ep]
            ))

    return findings

def get_pin_limit(endpoint: str, circuit: Circuit):
    if ":" not in endpoint:
        return None
    pid, pin = endpoint.split(":", 1)
    pid = pid.strip().lower()
    pin = pin.strip()
    if pid in ("uno", "board", "arduino", "arduino-uno"):
        return ARDUINO_UNO_LIMITS.get(pin)
    return None

def check_r009_led_needs_series_resistor(circuit: Circuit) -> List[Finding]:
    """
    R009 LED_NEEDS_SERIES_RESISTOR:
    Checks if an LED is connected between power/GPIO and GND without a series resistor.
    Operates on CONFIRMED connections only.
    """
    findings: List[Finding] = []
    led_parts = [p for p in circuit.parts if p.type == "led"]
    if not led_parts:
        return findings

    # Build adjacency graph of confirmed connections
    adj: Dict[str, Set[str]] = defaultdict(set)
    # Also record component internal pin connectivity (e.g. resistor connects pin 1 to pin 2)
    resistor_parts = {p.id for p in circuit.parts if p.type == "resistor"}

    for conn in circuit.connections:
        if conn.confirmed:
            u = conn.from_.strip()
            v = conn.to.strip()
            adj[u].add(v)
            adj[v].add(u)

    # For each resistor, internal connection between pin 1 and 2
    for r_id in resistor_parts:
        p1 = f"{r_id}:1"
        p2 = f"{r_id}:2"
        adj[p1].add(p2)
        adj[p2].add(p1)

    # Define power/GPIO pins on Uno
    power_or_gpio_pins = {f"uno:{p}" for p in ARDUINO_UNO_PINS if p not in ("GND.1", "GND.2", "GND.3", "RESET", "IOREF", "AREF")}
    # Add alias variants like board:13
    for uno_alias in ("board", "arduino", "arduino-uno"):
        power_or_gpio_pins.update({f"{uno_alias}:{p}" for p in ARDUINO_UNO_PINS if p not in ("GND.1", "GND.2", "GND.3", "RESET", "IOREF", "AREF")})

    ground_pins = {"uno:GND.1", "uno:GND.2", "uno:GND.3"}
    for uno_alias in ("board", "arduino", "arduino-uno"):
        ground_pins.update({f"{uno_alias}:GND.1", f"{uno_alias}:GND.2", f"{uno_alias}:GND.3"})

    for led in led_parts:
        anode = f"{led.id}:A"
        cathode = f"{led.id}:C"

        # Check if anode directly reaches power/GPIO WITHOUT traversing a resistor
        # and cathode directly reaches ground WITHOUT traversing a resistor (or vice-versa)
        def reaches_without_resistor(start_node: str, target_nodes: Set[str]) -> bool:
            visited = set()
            queue = [start_node]
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                if curr in target_nodes:
                    return True
                for neighbor in adj.get(curr, set()):
                    if neighbor in visited:
                        continue
                    # Check if edge traverses through a resistor
                    if ":" in neighbor:
                        part_id = neighbor.split(":", 1)[0].strip()
                        if part_id in resistor_parts:
                            # Traversing through resistor! Stop this branch
                            continue
                    queue.append(neighbor)
            return False

        anode_direct_power = reaches_without_resistor(anode, power_or_gpio_pins)
        cathode_direct_gnd = reaches_without_resistor(cathode, ground_pins)

        # Also check reverse or direct short
        cathode_direct_power = reaches_without_resistor(cathode, power_or_gpio_pins)
        anode_direct_gnd = reaches_without_resistor(anode, ground_pins)

        if (anode_direct_power and cathode_direct_gnd) or (cathode_direct_power and anode_direct_gnd):
            findings.append(Finding(
                rule_id="R009",
                severity="error",
                message=f"LED '{led.id}' is connected to power/GPIO without a current-limiting series resistor",
                refs=[led.id, anode, cathode]
            ))

    return findings

def check_r010_unconfirmed_wires(circuit: Circuit) -> List[Finding]:
    """R010 UNCONFIRMED_WIRES: Info findings for unconfirmed wire proposals."""
    findings: List[Finding] = []
    for idx, conn in enumerate(circuit.connections):
        if not conn.confirmed:
            findings.append(Finding(
                rule_id="R010",
                severity="info",
                message=f"Unconfirmed wire proposal from '{conn.from_}' to '{conn.to}' (source: {conn.source}, conf: {conn.conf:.2f})",
                refs=[f"conn[{idx}]", conn.from_, conn.to]
            ))
    return findings

def check_r011_unsupported_parts(circuit: Circuit) -> List[Finding]:
    """R011 UNSUPPORTED_PART: Part type is unknown or not supported in PART_SPECS."""
    findings: List[Finding] = []
    for part in circuit.parts:
        if part.type == "unknown" or part.type not in PART_SPECS:
            findings.append(Finding(
                rule_id="R011",
                severity="warning",
                message=f"Part '{part.id}' has unsupported type '{part.type}'",
                refs=[part.id]
            ))
    return findings

def check_r012_weak_class_detections(circuit: Circuit) -> List[Finding]:
    """R012 WEAK_CLASS_CONFIRMATION_REQUIRED: Low-confidence vision detections."""
    findings: List[Finding] = []
    for part in circuit.parts:
        if part.source in ("vision", "live", "stub"):
            threshold = settings.CONFIDENCE_THRESHOLDS.get(part.type, settings.CONFIDENCE_THRESHOLDS["default"])
            if part.conf < threshold:
                findings.append(Finding(
                    rule_id="R012",
                    severity="warning",
                    message=f"Part '{part.id}' ({part.type}) has low detection confidence ({part.conf:.2f} < {threshold:.2f}); user confirmation required",
                    refs=[part.id]
                ))
    return findings

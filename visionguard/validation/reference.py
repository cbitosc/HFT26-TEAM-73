from typing import Dict, List, Set, Tuple
from visionguard.models.circuit import Circuit
from visionguard.models.finding import Finding
from visionguard.templates.boards import REFERENCE_PROJECTS

def normalize_wire(u: str, v: str) -> Tuple[str, str]:
    """Returns order-independent canonical representation of a wire."""
    u_norm = u.strip()
    v_norm = v.strip()
    # Normalize board prefix
    for alias in ("board", "arduino", "arduino-uno"):
        if u_norm.lower().startswith(f"{alias}:"):
            u_norm = "uno:" + u_norm.split(":", 1)[1]
        if v_norm.lower().startswith(f"{alias}:"):
            v_norm = "uno:" + v_norm.split(":", 1)[1]
    return (min(u_norm, v_norm), max(u_norm, v_norm))

def compare_reference_circuit(circuit: Circuit, target_name: str) -> Tuple[List[Finding], Dict[str, any]]:
    """
    Compares the confirmed connections in circuit against a named reference target (e.g. 'led-blink-pin13').
    Ignores wire direction and reports missing and extra wires.
    """
    findings: List[Finding] = []
    comparison_details = {
        "target": target_name,
        "matched": False,
        "missing_wires": [],
        "extra_wires": [],
    }

    project = REFERENCE_PROJECTS.get(target_name)
    if not project:
        findings.append(Finding(
            rule_id="R006",
            severity="warning",
            message=f"Target reference project '{target_name}' is not recognized",
            refs=[target_name]
        ))
        return findings, comparison_details

    # Gather confirmed wires in circuit
    confirmed_wires: Set[Tuple[str, str]] = set()
    for conn in circuit.connections:
        if conn.confirmed:
            confirmed_wires.add(normalize_wire(conn.from_, conn.to))

    # Map circuit parts to generic roles
    # E.g. first resistor -> r1, first led -> led1
    leds = [p.id for p in circuit.parts if p.type == "led"]
    resistors = [p.id for p in circuit.parts if p.type == "resistor"]

    if not leds or not resistors:
        findings.append(Finding(
            rule_id="R006",
            severity="error",
            message=f"Target '{target_name}' requires both an LED and a Resistor",
            refs=[target_name]
        ))
        return findings, comparison_details

    valid_ground_pins = ["uno:GND.1", "uno:GND.2", "uno:GND.3"]

    best_match_missing = None
    best_target_wires = None

    for led_id in leds:
        for res_id in resistors:
            for r_in, r_out in [("1", "2"), ("2", "1")]:
                for gnd_pin in valid_ground_pins:
                    # Candidate 1: Resistor on anode
                    cand1 = {
                        normalize_wire("uno:13", f"{res_id}:{r_in}"),
                        normalize_wire(f"{res_id}:{r_out}", f"{led_id}:A"),
                        normalize_wire(f"{led_id}:C", gnd_pin),
                    }
                    # Candidate 2: Resistor on cathode
                    cand2 = {
                        normalize_wire("uno:13", f"{led_id}:A"),
                        normalize_wire(f"{led_id}:C", f"{res_id}:{r_in}"),
                        normalize_wire(f"{res_id}:{r_out}", gnd_pin),
                    }

                    for cand in (cand1, cand2):
                        missing = cand - confirmed_wires
                        if best_match_missing is None or len(missing) < len(best_match_missing):
                            best_match_missing = missing
                            best_target_wires = cand
                        if not missing:
                            break
                    if best_match_missing is not None and len(best_match_missing) == 0:
                        break
                if best_match_missing is not None and len(best_match_missing) == 0:
                    break
            if best_match_missing is not None and len(best_match_missing) == 0:
                break
        if best_match_missing is not None and len(best_match_missing) == 0:
            break

    missing_list = [f"{u} <-> {v}" for u, v in (best_match_missing or [])]
    extra_wires = confirmed_wires - (best_target_wires or set())
    extra_list = [f"{u} <-> {v}" for u, v in extra_wires]

    comparison_details["missing_wires"] = missing_list
    comparison_details["extra_wires"] = extra_list

    if not missing_list:
        comparison_details["matched"] = True
    else:
        for m in missing_list:
            findings.append(Finding(
                rule_id="R005",
                severity="error",
                message=f"Reference target '{target_name}' missing expected connection: {m}",
                refs=[target_name, m]
            ))

    for e in extra_list:
        findings.append(Finding(
            rule_id="R005",
            severity="warning",
            message=f"Reference target '{target_name}' has extra connection not in reference: {e}",
            refs=[target_name, e]
        ))

    return findings, comparison_details

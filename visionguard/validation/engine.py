from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field

from visionguard.models.circuit import Circuit
from visionguard.models.finding import Finding
from visionguard.templates.boards import REFERENCE_PROJECTS
from visionguard.validation.rules import (
    check_r001_unsupported_pins,
    check_r002_unknown_part_ref,
    check_r003_duplicate_connections,
    check_r004_self_connection,
    check_r005_required_pins_unconnected,
    check_r006_required_parts_missing,
    check_r007_slot_reported_missing,
    check_r008_voltage_limits,
    check_r009_led_needs_series_resistor,
    check_r010_unconfirmed_wires,
    check_r011_unsupported_parts,
    check_r012_weak_class_detections,
)
from visionguard.validation.reference import compare_reference_circuit

class ValidationResult(BaseModel):
    status: Literal["pass", "warn", "fail"]
    findings: List[Finding] = Field(default_factory=list)
    target_comparison: Optional[dict] = None

def validate_circuit(circuit: Circuit, target_name: Optional[str] = None) -> ValidationResult:
    """
    Deterministic validation engine (pure function, no I/O).
    Runs rules R001 to R012 and optional target comparison.
    Validation runs on CONFIRMED connections; unconfirmed ones only generate warnings/info
    and NEVER count as a pass.
    """
    findings: List[Finding] = []

    # Rule checks
    findings.extend(check_r001_unsupported_pins(circuit))
    findings.extend(check_r002_unknown_part_ref(circuit))
    findings.extend(check_r003_duplicate_connections(circuit))
    findings.extend(check_r004_self_connection(circuit))
    findings.extend(check_r005_required_pins_unconnected(circuit))

    # Required parts check (per target or sample project if given)
    if target_name and target_name in REFERENCE_PROJECTS:
        req_parts = REFERENCE_PROJECTS[target_name].required_parts
        findings.extend(check_r006_required_parts_missing(circuit, req_parts))

    findings.extend(check_r007_slot_reported_missing(circuit))
    findings.extend(check_r008_voltage_limits(circuit))
    findings.extend(check_r009_led_needs_series_resistor(circuit))
    findings.extend(check_r010_unconfirmed_wires(circuit))
    findings.extend(check_r011_unsupported_parts(circuit))
    findings.extend(check_r012_weak_class_detections(circuit))

    # Reference circuit comparison if target specified
    target_comp = None
    if target_name:
        ref_findings, target_comp = compare_reference_circuit(circuit, target_name)
        findings.extend(ref_findings)

    # Check for any unconfirmed parts or connections
    has_unconfirmed = any(not c.confirmed for c in circuit.connections) or any(not p.confirmed for p in circuit.parts)

    has_error = any(f.severity == "error" for f in findings)
    has_warning = any(f.severity == "warning" for f in findings)

    # If unconfirmed items exist and no errors, it must never be "pass"
    if has_error:
        status = "fail"
    elif has_warning or has_unconfirmed:
        if has_unconfirmed and not has_warning:
            # Emit a warning indicating unconfirmed elements prevent a "pass" verdict
            findings.append(Finding(
                rule_id="R010",
                severity="warning",
                message="Circuit contains unconfirmed proposals; connections must be confirmed by user to pass",
                refs=["unconfirmed"]
            ))
        status = "warn"
    else:
        status = "pass"

    return ValidationResult(
        status=status,
        findings=findings,
        target_comparison=target_comp
    )


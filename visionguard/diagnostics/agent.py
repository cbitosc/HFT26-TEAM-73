import json
import logging
from typing import Any, Dict, List, Optional
import httpx

from visionguard.config import settings
from visionguard.models.circuit import Circuit
from visionguard.knowledge.pinouts import default_pinout_kb
from visionguard.validation.engine import ValidationResult

logger = logging.getLogger("visionguard.diagnostics")

DISCLAIMER_TEXT = (
    "Diagnostic analysis and step-by-step guidance are generated for prototyping inspection only. "
    "Hardware cannot be certified safe without laboratory electrical verification. Always test "
    "power rails with a multimeter before connecting power sources."
)

def assemble_diagnostic_context(circuit: Circuit, val_result: Optional[ValidationResult] = None) -> Dict[str, Any]:
    """
    Assembles structured diagnostic data payload for the LLM.
    Strictly structured metadata only—no raw images are sent.
    """
    # 1. Detections
    dets_data = []
    matched_pinouts = []
    seen_pinout_keys = set()

    for d in circuit.detections:
        d_dict = {
            "id": d.id,
            "name": d.name,
            "category": d.category,
            "bbox_norm": d.bbox_norm,
            "conf": d.conf,
            "ocr_text": d.ocr_text,
            "part_number": d.part_number,
            "weak": d.weak
        }
        dets_data.append(d_dict)

        # Pinout lookup
        query = d.part_number or d.name
        pinout_data, _ = default_pinout_kb.lookup(query)
        if pinout_data and pinout_data.get("name") not in seen_pinout_keys:
            seen_pinout_keys.add(pinout_data.get("name"))
            matched_pinouts.append({
                "part_query": query,
                "name": pinout_data.get("name"),
                "verified": pinout_data.get("verified", False),
                "voltage_limits": pinout_data.get("voltage_limits", {}),
                "pins": [
                    {"name": p["name"], "pin_number": p.get("pin_number"), "function": p.get("function"), "max_voltage": p.get("max_voltage")}
                    for p in pinout_data.get("pins", [])
                ]
            })

    # Also check board template pinout
    board_template = circuit.board.template
    board_pinout, _ = default_pinout_kb.lookup(board_template)
    if board_pinout and board_pinout.get("name") not in seen_pinout_keys:
        matched_pinouts.append({
            "part_query": board_template,
            "name": board_pinout.get("name"),
            "verified": board_pinout.get("verified", False),
            "voltage_limits": board_pinout.get("voltage_limits", {}),
            "pins": board_pinout.get("pins", [])
        })

    # 2. Connections
    conns_data = [
        {"from": c.from_, "to": c.to, "confirmed": c.confirmed, "source": c.source, "wire_color": c.wire_color}
        for c in circuit.connections
    ]

    # 3. Rule findings
    findings_data = []
    rule_status = "unknown"
    if val_result:
        rule_status = val_result.status
        findings_data = [
            {"rule_id": f.rule_id, "severity": f.severity, "message": f.message, "refs": f.refs}
            for f in val_result.findings
        ]

    # 4. Defects
    defects_data = [
        {"id": df.id, "type": df.type, "bbox_px": df.bbox_px, "severity": df.severity, "conf": df.conf, "note": df.note}
        for df in circuit.defects
    ]

    # 5. Health summary
    health_data = circuit.health.model_dump() if circuit.health else {}
    skipped_checks = health_data.get("checks_not_run", [])

    return {
        "circuit_id": circuit.id,
        "board_template": circuit.board.template,
        "detections": dets_data,
        "matched_pinouts": matched_pinouts,
        "connections": conns_data,
        "rule_verdict": rule_status,
        "rule_findings": findings_data,
        "defects": defects_data,
        "health_summary": health_data,
        "checks_skipped": skipped_checks
    }

def generate_deterministic_fallback_diagnose(context: Dict[str, Any], question: Optional[str] = None) -> Dict[str, Any]:
    """Generates guaranteed schema-valid diagnostic output when LLM is unconfigured or unavailable."""
    detections = context.get("detections", [])
    defects = context.get("defects", [])
    findings = context.get("rule_findings", [])
    rule_verdict = context.get("rule_verdict", "unknown")

    # Plain language summary
    det_count = len(detections)
    defect_count = len(defects)
    err_count = sum(1 for f in findings if f.get("severity") == "error")
    warn_count = sum(1 for f in findings if f.get("severity") == "warning")

    summary = (
        f"Inspected circuit with {det_count} detected component(s). "
        f"Validation status is '{rule_verdict.upper()}' with {err_count} design error(s) and {warn_count} warning(s). "
        f"Identified {defect_count} potential physical/visual defect(s)."
    )
    if question:
        summary += f" Query addressed: '{question}'."

    # Component overview
    comp_overview = [
        {"id": d.get("id"), "name": d.get("name"), "description": f"Detected {d.get('category')} (confidence: {d.get('conf')})"}
        for d in detections[:10]
    ]

    # Pinout notes
    pinout_notes = []
    for p in context.get("matched_pinouts", []):
        pinout_notes.append({
            "part": p.get("name"),
            "verified": p.get("verified", False),
            "notes": f"Verified pinout with {len(p.get('pins', []))} documented terminals."
        })
    if not pinout_notes:
        pinout_notes.append({
            "part": "Generic",
            "verified": False,
            "notes": "no verified pinout available"
        })

    # Suspected issues
    suspected_issues = []
    for f in findings:
        suspected_issues.append({
            "issue_id": f.get("rule_id"),
            "tied_ref_id": ", ".join(f.get("refs", [])),
            "description": f.get("message"),
            "severity": f.get("severity")
        })
    for d in defects:
        suspected_issues.append({
            "issue_id": d.get("id"),
            "tied_ref_id": d.get("id"),
            "description": d.get("note") or f"Detected defect of type {d.get('type')}",
            "severity": d.get("severity")
        })

    # Step-by-step troubleshooting/assembly guide
    step_guide = []
    step_idx = 1
    for f in findings:
        target_id = f.get("refs")[0] if f.get("refs") else (detections[0].get("id") if detections else "board")
        bbox = detections[0].get("bbox_norm") if detections else [0.1, 0.1, 0.8, 0.8]
        step_guide.append({
            "step_number": step_idx,
            "instruction": f"Resolve rule finding [{f.get('rule_id')}]: {f.get('message')}.",
            "target_component_id": target_id,
            "bbox": bbox
        })
        step_idx += 1

    for d in defects:
        step_guide.append({
            "step_number": step_idx,
            "instruction": f"Manually verify potential defect '{d.get('type')}': {d.get('note')}.",
            "target_component_id": d.get("id"),
            "bbox": d.get("bbox_px")
        })
        step_idx += 1

    if not step_guide:
        step_guide.append({
            "step_number": 1,
            "instruction": "All automated checks passed. Verify supply voltages with a multimeter prior to powering.",
            "target_component_id": detections[0].get("id") if detections else "board",
            "bbox": detections[0].get("bbox_norm") if detections else [0.0, 0.0, 1.0, 1.0]
        })

    return {
        "summary_plain_language": summary,
        "component_overview": comp_overview,
        "pinout_notes": pinout_notes,
        "suspected_issues": suspected_issues,
        "step_by_step_guide": step_guide,
        "assumptions": ["Assumed standard 5V/3.3V operating levels for common microcontrollers"],
        "not_checked": context.get("checks_skipped", ["pcb_manufacturing_defects", "solder_joint_xray"]),
        "disclaimer": DISCLAIMER_TEXT,
        "llm_available": False
    }

async def run_diagnostic_agent(
    circuit: Circuit,
    val_result: Optional[ValidationResult] = None,
    question: Optional[str] = None,
    mode: str = "troubleshoot"
) -> Dict[str, Any]:
    """
    Executes Featherless diagnostic agent with strict JSON prompting and 20s timeout.
    Returns structured output conforming to the required diagnostic schema.
    """
    context = assemble_diagnostic_context(circuit, val_result)

    if not settings.FEATHERLESS_API_KEY:
        res = generate_deterministic_fallback_diagnose(context, question)
        res["llm_available"] = False
        return res

    system_prompt = (
        "You are an expert electronic hardware diagnostic AI assistant.\n"
        "STRICT NON-NEGOTIABLE OPERATING RULES:\n"
        "1. Use ONLY the provided structured hardware metadata.\n"
        "2. NEVER invent pinouts, part numbers, operating voltages, or datasheet facts.\n"
        "3. NEVER state or claim that the hardware is 'safe'.\n"
        "4. NEVER override or contradict deterministic rule verdicts or health status.\n"
        "5. Recommend manual multimeter/microscope inspection for all 'possible' defects.\n"
        "6. Return ONLY valid JSON adhering exactly to the specified JSON schema."
    )

    user_prompt = (
        f"Mode: {mode}\n"
        f"User Query: {question or 'Provide comprehensive hardware diagnostic summary and step-by-step verification guide.'}\n\n"
        f"Hardware Data JSON:\n{json.dumps(context, indent=2)}\n\n"
        "Respond with a JSON object containing keys:\n"
        "- summary_plain_language: str\n"
        "- component_overview: list of {id: str, name: str, description: str}\n"
        "- pinout_notes: list of {part: str, verified: bool, notes: str}\n"
        "- suspected_issues: list of {issue_id: str, tied_ref_id: str, description: str, severity: str}\n"
        "- step_by_step_guide: list of {step_number: int, instruction: str, target_component_id: str, bbox: list}\n"
        "- assumptions: list of str\n"
        "- not_checked: list of str\n"
        "- disclaimer: str"
    )

    try:
        url = f"{settings.FEATHERLESS_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.FEATHERLESS_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.FEATHERLESS_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                parsed["disclaimer"] = DISCLAIMER_TEXT
                parsed["llm_available"] = True
                return parsed
            else:
                logger.warning(f"Featherless API returned HTTP {resp.status_code}: {resp.text}")

    except Exception as exc:
        logger.warning(f"Featherless diagnostic call error: {exc}")

    # Fallback if call fails or times out
    fallback = generate_deterministic_fallback_diagnose(context, question)
    fallback["llm_available"] = False
    return fallback


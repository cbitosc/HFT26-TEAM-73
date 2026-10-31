import base64
import dis
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx

from visionguard.config import settings
from visionguard.models.circuit import Circuit, Connection, Part
from visionguard.models.finding import Finding
from visionguard.templates.boards import normalize_pin_name, PART_SPECS, ARDUINO_UNO_PINS

logger = logging.getLogger("visionguard.llm")

class ProposalResult(dict):
    """
    Universal result container for propose_from_vision_llm.
    Supports:
      - 2-tuple unpacking: `proposal, findings = propose_from_vision_llm(...)`
      - 3-tuple unpacking: `parts, connections, findings = propose_from_vision_llm(...)`
      - Dict access: `result["parts"]`, `result["connections"]`, `result["findings"]`
      - Attribute access: `result.parts`, `result.connections`, `result.findings`
    """
    def __init__(self, parts: List[Part], connections: List[Connection], findings: List[Finding]):
        proposal_dict = {"parts": parts, "connections": connections}
        super().__init__(
            parts=parts,
            connections=connections,
            findings=findings,
            proposal=proposal_dict
        )
        self.parts = parts
        self.connections = connections
        self.findings = findings
        self.proposal = proposal_dict

    def __iter__(self):
        try:
            f = sys._getframe(1)
            for it in dis.get_instructions(f.f_code):
                if it.offset == f.f_lasti and it.opname.startswith("UNPACK_SEQUENCE"):
                    if it.argval == 3:
                        return iter((self.parts, self.connections, self.findings))
                    elif it.argval == 2:
                        return iter((self.proposal, self.findings))
        except Exception:
            pass
        return iter((self.proposal, self.findings))

PROMPT = """
You are an expert electronics and embedded systems vision model.
Analyze this photo of a circuit (breadboard, PCB, or Arduino setup).
Detect components and trace wire connections between components and Arduino Uno pins.

Rules:
1. Allowed component types: "led", "resistor", "pushbutton", "arduino-uno".
2. Allowed pin names:
   - Arduino Uno: "0" through "13", "A0" through "A5", "5V", "3.3V", "VIN", "GND.1", "GND.2", "GND.3".
   - LED: "A" (anode), "C" (cathode).
   - Resistor: "1", "2".
   - Pushbutton: "1.l", "1.r", "2.l", "2.r".
3. Treat components plugged into the same breadboard tie-strip row as connected.
4. If an endpoint or connection is unclear or occluded, assign low confidence (e.g. 0.2 to 0.4).
5. All parts and connections must be returned as proposals.

Respond ONLY with valid JSON matching this schema:
{
  "parts": [
    {"id": "led1", "type": "led", "value": null, "color": "red", "conf": 0.85},
    {"id": "r1", "type": "resistor", "value": "220", "color": null, "conf": 0.90}
  ],
  "connections": [
    {"from": "uno:13", "to": "r1:1", "conf": 0.80, "note": "digital pin 13 to resistor"},
    {"from": "r1:2", "to": "led1:A", "conf": 0.75, "note": "breadboard strip to LED anode"},
    {"from": "led1:C", "to": "uno:GND.1", "conf": 0.85, "note": "LED cathode to ground"}
  ]
}
"""

def propose_from_vision_llm(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    circuit: Optional[Circuit] = None
) -> ProposalResult:
    """
    Sends photo to multimodal model to propose parts and connections.
    Gracefully degrades: on missing key, timeout, or bad JSON, returns
    empty proposal plus a finding, NEVER raises 500.
    """
    findings: List[Finding] = []
    proposed_parts: List[Part] = []
    proposed_connections: List[Connection] = []

    if not settings.GEMINI_API_KEY:
        findings.append(Finding(
            rule_id="R010",
            severity="info",
            message="LLM connection proposals skipped: API key environment variable is not configured",
            refs=["llm_proposals"]
        ))
        return ProposalResult(proposed_parts, proposed_connections, findings)

    try:
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        model = settings.VISIONGUARD_MODEL.strip()
        if not model.startswith("models/"):
            model = f"models/{model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={settings.GEMINI_API_KEY}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": PROMPT},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        }

        with httpx.Client(timeout=20.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                findings.append(Finding(
                    rule_id="R010",
                    severity="warning",
                    message=f"Vision LLM returned HTTP {resp.status_code}: {resp.text[:150]}",
                    refs=["llm_api_error"]
                ))
                return ProposalResult(proposed_parts, proposed_connections, findings)

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                findings.append(Finding(
                    rule_id="R010",
                    severity="warning",
                    message="Vision LLM returned no candidates",
                    refs=["llm_empty"]
                ))
                return ProposalResult(proposed_parts, proposed_connections, findings)

            raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            # Strip markdown code fence if present
            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                raw_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            parsed = json.loads(raw_text)

        # Seed part type map from existing circuit parts if available
        part_type_map = {}
        if circuit and circuit.parts:
            for p in circuit.parts:
                part_type_map[p.id] = p.type

        # Process parsed parts
        parts_data = parsed.get("parts", [])
        for p in parts_data:
            pid = str(p.get("id", "")).strip()
            ptype = str(p.get("type", "unknown")).strip().lower()
            if not pid:
                continue
            part_obj = Part(
                id=pid,
                type=ptype if ptype in PART_SPECS else "unknown",
                value=str(p.get("value")) if p.get("value") is not None else None,
                color=str(p.get("color")) if p.get("color") is not None else None,
                bbox=p.get("bbox"),
                source="vision",
                conf=float(p.get("conf", 0.5)),
                confirmed=False,
                wokwi_type=PART_SPECS[ptype].wokwi_type if ptype in PART_SPECS else None
            )
            proposed_parts.append(part_obj)
            part_type_map[pid] = part_obj.type

        # Process connections
        conns_data = parsed.get("connections", [])
        for c in conns_data:
            from_ep = str(c.get("from", "")).strip()
            to_ep = str(c.get("to", "")).strip()

            if ":" not in from_ep or ":" not in to_ep:
                findings.append(Finding(
                    rule_id="R001",
                    severity="warning",
                    message=f"LLM proposed malformed wire endpoint '{from_ep}' -> '{to_ep}' dropped",
                    refs=[from_ep, to_ep]
                ))
                continue

            # Validate endpoints against part specs
            p1, pin1 = from_ep.split(":", 1)
            p2, pin2 = to_ep.split(":", 1)

            norm_pin1 = normalize_endpoint_pin(p1, pin1, part_type_map)
            norm_pin2 = normalize_endpoint_pin(p2, pin2, part_type_map)

            if not norm_pin1 or not norm_pin2:
                findings.append(Finding(
                    rule_id="R001",
                    severity="warning",
                    message=f"LLM proposed invalid wire dropped: '{from_ep}' -> '{to_ep}'",
                    refs=[from_ep, to_ep]
                ))
                continue

            wire_color = c.get("wire_color")
            if not wire_color:
                if "gnd" in norm_pin1.lower() or "gnd" in norm_pin2.lower():
                    wire_color = "black"
                elif any(pw in (norm_pin1, norm_pin2) for pw in ("5V", "3.3V", "VIN")):
                    wire_color = "red"
                else:
                    wire_color = "green"

            conn_obj = Connection(
                **{
                    "from": f"{p1}:{norm_pin1}",
                    "to": f"{p2}:{norm_pin2}",
                    "source": "vision",
                    "conf": float(c.get("conf", 0.5)),
                    "confirmed": False,
                    "wire_color": wire_color,
                    "note": c.get("note", "LLM multimodal proposal")
                }
            )
            proposed_connections.append(conn_obj)

    except json.JSONDecodeError as jde:
        findings.append(Finding(
            rule_id="R010",
            severity="warning",
            message=f"Failed to parse JSON response from Vision LLM: {str(jde)}",
            refs=["llm_json_error"]
        ))
    except Exception as exc:
        findings.append(Finding(
            rule_id="R010",
            severity="warning",
            message=f"Vision LLM proposal failed gracefully: {str(exc)}",
            refs=["llm_exception"]
        ))

    return ProposalResult(proposed_parts, proposed_connections, findings)

def normalize_endpoint_pin(part_id: str, pin: str, part_type_map: dict) -> str:
    """Validates and normalizes pin name. Returns normalized pin or empty string if invalid."""
    pid_lower = part_id.lower()
    if pid_lower in ("uno", "board", "arduino", "arduino-uno"):
        norm = normalize_pin_name("arduino-uno", pin)
        return norm if norm in ARDUINO_UNO_PINS else ""

    ptype = part_type_map.get(part_id)
    if ptype and ptype in PART_SPECS:
        norm = normalize_pin_name(ptype, pin)
        return norm if norm in PART_SPECS[ptype].allowed_pins else ""

    return pin


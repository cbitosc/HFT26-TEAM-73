import io
import uuid
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from PIL import Image

from visionguard.config import settings
from visionguard.models.circuit import (
    Circuit,
    create_empty_circuit,
    Part,
    Connection,
    InspectionSlot,
    Detection,
    Defect,
    HealthSummary
)
from visionguard.models.finding import Finding
from visionguard.templates.boards import (
    PART_SPECS,
    REFERENCE_PROJECTS,
    map_detector_label,
    normalize_pin_name,
)
from visionguard.repository.memory import default_repository, CircuitRepository
from visionguard.validation.engine import validate_circuit, ValidationResult
from visionguard.proposals.manager import apply_proposals, apply_user_updates
from visionguard.exports.svg import generate_circuit_svg
from visionguard.exports.png import render_circuit_png
from visionguard.exports.wokwi import export_wokwi
from visionguard.exports.overlay import generate_circuit_overlay
from visionguard.vision.board_detector import BoardDetector
from visionguard.vision.component_detector import ComponentDetector
from visionguard.vision.slot_inspector import SlotInspector
from visionguard.vision.open_vocab_detector import default_open_vocab_detector
from visionguard.vision.ocr import default_ocr_engine
from visionguard.vision.defects import default_defect_detector, detect_heuristic_defects, compute_health_summary
from visionguard.vision.merge import merge_layer_detections
from visionguard.diagnostics.agent import run_diagnostic_agent

router = APIRouter()

# Instantiate singletons for vision wrappers
board_detector = BoardDetector()
component_detector = ComponentDetector()
slot_inspector = SlotInspector()
open_vocab_detector = default_open_vocab_detector
ocr_engine = default_ocr_engine
defect_detector = default_defect_detector

def get_repo() -> CircuitRepository:
    return default_repository

# ---------------------------------------------------------------------------
# Schema and Info Endpoints
# ---------------------------------------------------------------------------

@router.get("/circuits/schema", tags=["Schema"])
def get_circuit_schema():
    """Returns the official JSON Schema for Circuit JSON v1."""
    return Circuit.model_json_schema()

@router.get("/templates", tags=["Templates"])
def list_templates():
    """Lists supported board templates, part specs, and reference projects."""
    return {
        "boards": ["arduino-uno"],
        "parts": {k: {"pins": list(v.allowed_pins), "wokwi_type": v.wokwi_type} for k, v in PART_SPECS.items()},
        "reference_projects": list(REFERENCE_PROJECTS.keys()),
    }

@router.get("/health", tags=["System"])
def get_health():
    """Reports system status and vision model wrapper availability."""
    wrappers = {
        "board_detector": board_detector.get_status(),
        "component_detector": component_detector.get_status(),
        "slot_inspector": slot_inspector.get_status(),
        "open_vocab_detector": open_vocab_detector.get_status(),
        "ocr": ocr_engine.get_status(),
        "defect_detector": defect_detector.get_status()
    }
    return {
        "status": "ok",
        "backend_mode": settings.VISION_BACKEND,
        "mode": settings.VISION_BACKEND,
        "wrappers": wrappers,
        # backward-compat field
        "models": {k: {"available": v["available"], "status": v["status"], "state": v["state"]} for k, v in wrappers.items()},
        "llm_multimodal": {
            "model": settings.VISIONGUARD_MODEL,
            "configured": bool(settings.GEMINI_API_KEY)
        },
        "featherless": {
            "model": settings.FEATHERLESS_MODEL,
            "configured": bool(settings.FEATHERLESS_API_KEY)
        }
    }

@router.get("/vision/status", tags=["System"])
def get_vision_status():
    """
    Reports exact state of all vision layers.
    In stub mode, reports state='stub' (not available).
    In real mode, reports 'available' or 'unavailable' with exact install/download instructions.
    """
    return get_health()

@router.post("/mode", tags=["System"])
def set_backend_mode(mode: str = Query("stub", description="Backend mode: 'stub' or 'real'")):
    """Switches backend mode dynamically between stub and real."""
    if mode.lower() in ("stub", "real"):
        settings.VISION_BACKEND = mode.lower()
        return {"status": "ok", "backend_mode": settings.VISION_BACKEND}
    raise HTTPException(status_code=400, detail="Mode must be 'stub' or 'real'")

@router.get("/sample-boards", tags=["System"])
def list_sample_boards():
    """Lists the 4 available sample boards matching the Inspection Studio frontend."""
    from visionguard.knowledge.sample_boards import SAMPLE_BOARDS
    return [
        {
            "id": k,
            "name": v["name"],
            "badge": v["badge"],
            "defect_count": v["defect_count"],
            "description": v["description"],
            "detections_count": v["detections_count"],
            "model": v["model"]
        }
        for k, v in SAMPLE_BOARDS.items()
    ]

@router.get("/sample-boards/{board_id}", tags=["System"])
def get_sample_board(board_id: str):
    """Returns complete component, specs, pinouts, and defect data for a sample board."""
    from visionguard.knowledge.sample_boards import SAMPLE_BOARDS
    if board_id in SAMPLE_BOARDS:
        return SAMPLE_BOARDS[board_id]
    raise HTTPException(status_code=404, detail=f"Sample board '{board_id}' not found")

# ---------------------------------------------------------------------------
# Circuit CRUD Endpoints
# ---------------------------------------------------------------------------

@router.get("/circuits", response_model=List[Circuit], tags=["Circuits"])
def list_circuits():
    """List all stored circuits."""
    return get_repo().list_all()

@router.post("/circuits", response_model=Circuit, status_code=201, tags=["Circuits"])
def create_circuit(payload: Optional[Dict[str, Any]] = None):
    """Creates a new circuit (either empty or from supplied valid JSON)."""
    if payload:
        data = dict(payload)
        if "id" not in data:
            data["id"] = str(uuid.uuid4())
        circuit = Circuit.model_validate(data)
    else:
        circuit = create_empty_circuit()
    return get_repo().save(circuit)

@router.get("/circuits/{circuit_id}", response_model=Circuit, tags=["Circuits"])
def get_circuit(circuit_id: str):
    """Retrieves a saved circuit document by ID."""
    circuit = get_repo().get(circuit_id)
    if not circuit:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_id}' not found")
    return circuit

@router.put("/circuits/{circuit_id}", response_model=Dict[str, Any], tags=["Circuits"])
def update_circuit(circuit_id: str, update_payload: Dict[str, Any]):
    """
    Updates a circuit document.
    Rule: Any user edit sets source='user', conf=1.0, confirmed=True.
    Re-validates and saves the document.
    """
    repo = get_repo()
    saved = repo.get(circuit_id)
    if not saved:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_id}' not found")

    updated = apply_user_updates(saved, update_payload)
    saved_doc = repo.save(updated)
    validation = validate_circuit(saved_doc)

    return {
        "circuit": saved_doc,
        "validation": validation
    }

# ---------------------------------------------------------------------------
# Vision Analysis Endpoint
# ---------------------------------------------------------------------------

@router.post("/circuits/analyze", tags=["Vision Analysis"])
@router.post("/analyze", tags=["Vision Analysis"])
async def analyze_image(
    file: UploadFile = File(...),
    connections: str = Query("proximity,template", description="Comma-separated proposals: 'proximity', 'template', 'llm'"),
    board_template: str = Query("arduino-uno", description="Board template name"),
    inspect: bool = Query(True, description="Whether to run slot inspector"),
    target: Optional[str] = Query(None, description="Optional target reference project, e.g. 'led-blink-pin13'")
):
    """
    Ingests PCB image or photo.
    1. Saves original uploaded image.
    2. Runs Layer 1 (PCB Component Detector) on SanderGi board crop.
    3. Runs Layer 2 (Open-Vocabulary Zero-Shot Detector) text-prompted with vocabulary.
    4. Runs Layer 3 (Multimodal LLM) if enabled.
    5. Merges all layers with class-aware NMS and confidence calibration.
    6. Runs OCR on detected chips/modules and looks up verified pinouts.
    7. Runs slot inspector and visual defect heuristics.
    8. Generates deterministic health summary and wire proposals.
    9. Saves circuit to store and returns circuit + deterministic validation verdict.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    circuit_id = str(uuid.uuid4())
    repo = get_repo()
    repo.save_image(circuit_id, image_bytes)

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = pil_img.size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image file: {e}")

    conn_modes = [c.strip().lower() for c in connections.split(",")]
    use_template = "template" in conn_modes
    use_proximity = "proximity" in conn_modes
    use_llm = "llm" in conn_modes

    findings: List[Finding] = []
    skipped_checks: List[str] = []

    # -----------------------------------------------------------------------
    # STUB MODE EXECUTION
    # -----------------------------------------------------------------------
    if settings.VISION_BACKEND == "stub":
        findings.append(Finding(
            rule_id="STUB_MODE_ACTIVE",
            severity="warning",
            message="Results are canned demo data, not detected from your image",
            refs=["system"]
        ))

        # Match against high-fidelity sample boards catalog
        from visionguard.knowledge.sample_boards import SAMPLE_BOARDS

        tmpl_norm = board_template.lower().strip()
        if tmpl_norm in ("arduino-uno", "arduino-uno-r3", "default", "uno"):
            board_key = "arduino-uno-r3"
        elif "defect" in tmpl_norm:
            board_key = "arduino-uno-defect"
        elif "esp32" in tmpl_norm:
            board_key = "esp32-devkit"
        elif "breadboard" in tmpl_norm:
            board_key = "breadboard-led"
        else:
            board_key = tmpl_norm if tmpl_norm in SAMPLE_BOARDS else "arduino-uno-r3"

        sample_info = SAMPLE_BOARDS.get(board_key)

        detections: List[Detection] = []
        parts: List[Part] = []
        defects: List[Defect] = []
        type_counts: Dict[str, int] = {}

        if sample_info:
            comps = sample_info.get("components") or SAMPLE_BOARDS["arduino-uno-r3"]["components"]
            if board_key == "arduino-uno-defect":
                comps = [c for c in comps if c["id"] != "C1"]

            for i, comp in enumerate(comps, start=1):
                bn = comp["bbox_norm"]
                b_px = [
                    round(bn[0] * img_w, 1),
                    round(bn[1] * img_h, 1),
                    round(bn[2] * img_w, 1),
                    round(bn[3] * img_h, 1)
                ]
                det_id = f"det_{i}"
                pnum = comp.get("part_number")
                cat = comp.get("category", "IC").lower()
                raw_type = comp.get("raw_name")
                mapped_pid = None

                if raw_type in PART_SPECS:
                    prefix = "r" if raw_type == "resistor" else ("led" if raw_type == "led" else ("btn" if raw_type == "pushbutton" else "part"))
                    type_counts[prefix] = type_counts.get(prefix, 0) + 1
                    mapped_pid = f"{prefix}{type_counts[prefix]}"
                    parts.append(Part(
                        id=mapped_pid,
                        type=raw_type,
                        value=comp.get("specs", {}).get("resistance") or ("220" if raw_type == "resistor" else None),
                        color="red" if raw_type == "led" else None,
                        bbox=b_px,
                        source="stub",
                        conf=comp["conf"],
                        confirmed=False,
                        wokwi_type=PART_SPECS.get(raw_type).wokwi_type if raw_type in PART_SPECS else None
                    ))

                detections.append(Detection(
                    id=det_id,
                    name=comp["name"],
                    category=cat,
                    bbox_px=b_px,
                    bbox_norm=bn,
                    conf=comp["conf"],
                    source="stub",
                    ocr_text=pnum if cat in ("ic", "mcu", "regulator") else None,
                    part_number=pnum,
                    mapped_part_id=mapped_pid,
                    confirmed=False,
                    weak=False,
                    layers_matched=["stub"]
                ))

            for d_def in sample_info.get("defects", []):
                bn = d_def.get("bbox_norm", [0.0, 0.0, 0.1, 0.1])
                b_px = [
                    round(bn[0] * img_w, 1),
                    round(bn[1] * img_h, 1),
                    round(bn[2] * img_w, 1),
                    round(bn[3] * img_h, 1)
                ]
                defects.append(Defect(
                    id=d_def["id"],
                    type=d_def["type"],
                    bbox_px=b_px,
                    conf=d_def.get("conf", 0.95),
                    source="simulated_model",
                    severity=d_def.get("severity", "likely"),
                    note=d_def.get("note")
                ))
        else:
            raw_dets = open_vocab_detector.detect(pil_img)
            for i, d in enumerate(raw_dets, start=1):
                det_id = f"det_{i}"
                mapped_pid = None
                wg_type = d.get("mapped_wiring_type")
                if wg_type:
                    prefix = "r" if wg_type == "resistor" else ("led" if wg_type == "led" else ("btn" if wg_type == "pushbutton" else "part"))
                    type_counts[prefix] = type_counts.get(prefix, 0) + 1
                    mapped_pid = f"{prefix}{type_counts[prefix]}"
                    parts.append(Part(
                        id=mapped_pid,
                        type=wg_type,
                        value="220" if wg_type == "resistor" else None,
                        color="red" if wg_type == "led" else None,
                        bbox=d["bbox_px"],
                        source="stub",
                        conf=d["conf"],
                        confirmed=False,
                        wokwi_type=PART_SPECS.get(wg_type).wokwi_type if wg_type in PART_SPECS else None
                    ))

                ocr_txt = "ATMEGA328P-PU" if d["category"] == "ic" else None
                pnum = "ATMEGA328P" if d["category"] == "ic" else None
                detections.append(Detection(
                    id=det_id,
                    name=d["name"],
                    category=d["category"],
                    bbox_px=d["bbox_px"],
                    bbox_norm=d["bbox_norm"],
                    conf=d["conf"],
                    source="stub",
                    ocr_text=ocr_txt,
                    part_number=pnum,
                    mapped_part_id=mapped_pid,
                    confirmed=False,
                    weak=d.get("weak", False),
                    layers_matched=["stub"]
                ))
            raw_defects = detect_heuristic_defects(pil_img)
            defects.extend([Defect(**df) for df in raw_defects])

        # Slot inspection
        inspection_slots: List[InspectionSlot] = []
        if inspect:
            s_results = slot_inspector.inspect_slots(image_bytes)
            for s in s_results:
                inspection_slots.append(InspectionSlot(slot=s["slot"], state=s["state"], conf=float(s["conf"])))

        # Include heuristic optical defects (blur, underexposure)
        raw_heur_defects = detect_heuristic_defects(pil_img)
        existing_def_ids = {df.id for df in defects}
        for h_df in raw_heur_defects:
            if h_df.get("id") not in existing_def_ids:
                defects.append(Defect(**h_df))

        # Health Summary
        health_summary_dict = compute_health_summary(
            [d.model_dump() for d in detections],
            [df.model_dump() for df in defects],
            findings,
            skipped_checks
        )

        circuit = Circuit(
            version=1,
            id=circuit_id,
            image_id=circuit_id,
            board={"template": board_template},
            parts=parts,
            connections=[],
            inspection=inspection_slots,
            detections=detections,
            defects=defects,
            health=HealthSummary(**health_summary_dict)
        )

        circuit, prop_findings = apply_proposals(circuit, use_template=use_template, use_proximity=use_proximity, use_llm=False)
        findings.extend(prop_findings)

        saved = repo.save(circuit)
        val = validate_circuit(saved, target_name=target)
        val.findings.extend(findings)

        return {"circuit": saved, "validation": val}

    # -----------------------------------------------------------------------
    # REAL MODE EXECUTION (Default)
    # -----------------------------------------------------------------------

    # Check for solid black / blank image (must return 0 detections, never canned)
    import numpy as np
    gray_arr = np.mean(np.array(pil_img), axis=2)
    if float(np.max(gray_arr)) < 8.0:
        empty_c = create_empty_circuit(board_template=board_template, circuit_id=circuit_id)
        empty_c.image_id = circuit_id
        empty_c.defects = [Defect(
            id="def_black_image",
            type="image_too_dark",
            bbox_px=[0.0, 0.0, float(img_w), float(img_h)],
            conf=1.0,
            source="heuristic",
            severity="possible",
            note="Solid black or blank image; zero components detected."
        )]
        empty_c.health = HealthSummary(**compute_health_summary([], [d.model_dump() for d in empty_c.defects], [], []))
        saved = repo.save(empty_c)
        finding = Finding(
            rule_id="R006",
            severity="warning",
            message="Blank image provided; zero components detected.",
            refs=["board"]
        )
        return {
            "circuit": saved,
            "validation": ValidationResult(status="warn", findings=[finding])
        }

    # Verify if vision models are installed / available
    b_avail, b_msg = board_detector.available()
    c_avail, c_msg = component_detector.available()
    o_avail, o_msg = open_vocab_detector.available()

    if not b_avail and not c_avail and not o_avail:
        # Graceful failure: return valid empty document with VISION_MODEL_UNAVAILABLE finding, NEVER fake parts
        empty_c = create_empty_circuit(board_template=board_template, circuit_id=circuit_id)
        empty_c.image_id = circuit_id
        saved = repo.save(empty_c)
        unavail_finding = Finding(
            rule_id="VISION_MODEL_UNAVAILABLE",
            severity="warning",
            message=f"Vision model wrappers unavailable: {b_msg}. Install requirements-vision.txt or set model weights.",
            refs=["system"]
        )
        return {
            "circuit": saved,
            "validation": ValidationResult(status="warn", findings=[unavail_finding])
        }

    # 1. Board detection
    board_result = board_detector.detect_board(image_bytes)
    board_crop_bytes = None
    crop_offset = (0, 0)
    board_bbox = None

    if board_result.get("found", False):
        board_crop_bytes = board_result.get("crop") or board_result.get("crop_bytes")
        crop_offset = board_result.get("crop_to_orig_offset", (0, 0))
        board_bbox = board_result.get("bbox")
    else:
        skipped_checks.append("board_detector_crop")

    # Layer 1: PCB Model
    layer1_dets: List[Dict[str, Any]] = []
    if board_crop_bytes and c_avail:
        raw_pcb = component_detector.detect_components(board_crop_bytes)
        ox, oy = crop_offset
        for comp in raw_pcb:
            bx, by, bw, bh = comp["bbox"]
            orig_bx = bx + ox
            orig_by = by + oy
            vg_type, _ = map_detector_label(comp.get("label", "unknown"))
            layer1_dets.append({
                "name": vg_type,
                "category": "passive" if vg_type == "resistor" else ("semiconductor" if vg_type == "led" else "switch"),
                "bbox_px": [round(orig_bx, 1), round(orig_by, 1), round(bw, 1), round(bh, 1)],
                "bbox_norm": [round(orig_bx / img_w, 4), round(orig_by / img_h, 4), round(bw / img_w, 4), round(bh / img_h, 4)],
                "conf": float(comp.get("conf", 0.5)),
                "source": "pcb_model",
                "mapped_wiring_type": vg_type if vg_type in ("led", "resistor", "pushbutton") else None,
                "weak": float(comp.get("conf", 0.5)) < settings.WEAK_CONF
            })
    else:
        skipped_checks.append("pcb_component_model")

    # Layer 2: Open-Vocabulary Zero-Shot Detector
    layer2_dets: List[Dict[str, Any]] = []
    if o_avail:
        layer2_dets = open_vocab_detector.detect(pil_img)
    else:
        skipped_checks.append("open_vocab_detector")

    # Layer 3: Multimodal LLM (optional, off by default)
    layer3_dets: List[Dict[str, Any]] = []
    if settings.LLM_VISION_ENABLED:
        pass # Placeholder for external LLM vision pass if configured
    else:
        skipped_checks.append("llm_vision_pass")

    # Merge layers with IoU and confidence calibration
    merged_raw = merge_layer_detections({
        "pcb_model": layer1_dets,
        "open_vocab": layer2_dets,
        "llm": layer3_dets
    })

    # OCR surface markings on ICs and modules
    for d in merged_raw:
        if d.get("category") in ("ic", "module", "board", "microcontroller"):
            bx, by, bw, bh = d.get("bbox_px", [0, 0, 10, 10])
            cx1, cy1 = max(0, int(bx)), max(0, int(by))
            cx2, cy2 = min(img_w, int(bx + bw)), min(img_h, int(by + bh))
            if cx2 > cx1 and cy2 > cy1:
                crop_pil = pil_img.crop((cx1, cy1, cx2, cy2))
                ocr_text, part_num = ocr_engine.extract_text(crop_pil)
                if ocr_text:
                    d["ocr_text"] = ocr_text
                if part_num:
                    d["part_number"] = part_num

    # Convert to Detection models and populate wiring Parts
    detections: List[Detection] = []
    parts: List[Part] = []
    type_counts: Dict[str, int] = {}

    for d in merged_raw:
        mapped_pid = None
        wg_type = d.get("mapped_wiring_type")
        if wg_type:
            prefix = "r" if wg_type == "resistor" else ("led" if wg_type == "led" else ("btn" if wg_type == "pushbutton" else "part"))
            type_counts[prefix] = type_counts.get(prefix, 0) + 1
            mapped_pid = f"{prefix}{type_counts[prefix]}"
            parts.append(Part(
                id=mapped_pid,
                type=wg_type,
                value=d.get("value"),
                color=d.get("color"),
                bbox=d["bbox_px"],
                source="vision",
                conf=d["conf"],
                confirmed=False,
                wokwi_type=PART_SPECS.get(wg_type).wokwi_type if wg_type in PART_SPECS else None
            ))

        detections.append(Detection(
            id=d["id"],
            name=d["name"],
            category=d["category"],
            bbox_px=d["bbox_px"],
            bbox_norm=d["bbox_norm"],
            conf=d["conf"],
            source=d.get("source", "open_vocab"),
            ocr_text=d.get("ocr_text"),
            part_number=d.get("part_number"),
            mapped_part_id=mapped_pid,
            confirmed=False,
            weak=d.get("weak", False),
            layers_matched=d.get("layers_matched", [])
        ))

    # Slot inspection (optional)
    inspection_slots: List[InspectionSlot] = []
    if inspect and board_crop_bytes and slot_inspector.available()[0]:
        s_results = slot_inspector.inspect_slots(board_crop_bytes)
        for s in s_results:
            inspection_slots.append(InspectionSlot(
                slot=s["slot"],
                state=s["state"],
                conf=float(s["conf"])
            ))
    else:
        skipped_checks.append("nv_slot_inspector")

    # Detect visual and optical defects
    defects_raw = detect_heuristic_defects(pil_img, board_bbox=board_bbox)
    # Add any missing slot defects
    for slot_item in inspection_slots:
        if slot_item.state == "missing":
            defects_raw.append({
                "id": f"def_missing_{slot_item.slot}",
                "type": "missing_component",
                "bbox_px": None,
                "conf": slot_item.conf,
                "source": "nv_slot_inspector",
                "severity": "likely",
                "note": f"Component in slot '{slot_item.slot}' is missing from the PCB."
            })

    defects = [Defect(**df) for df in defects_raw]

    # Deterministic health summary
    health_dict = compute_health_summary(
        [d.model_dump() for d in detections],
        [df.model_dump() for df in defects],
        findings,
        skipped_checks
    )

    # Assemble circuit
    circuit = Circuit(
        version=1,
        id=circuit_id,
        image_id=circuit_id,
        board={"template": board_template},
        parts=parts,
        connections=[],
        inspection=inspection_slots,
        detections=detections,
        defects=defects,
        health=HealthSummary(**health_dict)
    )

    # Connection proposals
    circuit, prop_findings = apply_proposals(
        circuit,
        use_template=use_template,
        use_proximity=use_proximity,
        use_llm=use_llm,
        image_bytes=image_bytes
    )
    findings.extend(prop_findings)

    # Save to single source of truth
    saved = repo.save(circuit)
    validation = validate_circuit(saved, target_name=target)
    validation.findings.extend(findings)

    return {
        "circuit": saved,
        "validation": validation
    }

# ---------------------------------------------------------------------------
# Visual Diagnostic Overlay Endpoint
# ---------------------------------------------------------------------------

@router.get("/circuits/{circuit_id}/overlay", tags=["Visual Inspection"])
def get_circuit_overlay(
    circuit_id: str,
    format: str = Query("png", description="Overlay format (png)")
):
    """
    Renders visual diagnostic overlay on the original uploaded image.
    Draws all bounding boxes with confidence, OCR tags, and flagged potential defects.
    """
    repo = get_repo()
    circuit = repo.get(circuit_id)
    if not circuit:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_id}' not found")

    image_bytes = repo.get_image(circuit.image_id or circuit_id)
    png_bytes = generate_circuit_overlay(circuit, image_bytes)
    return Response(content=png_bytes, media_type="image/png")

# ---------------------------------------------------------------------------
# Diagnostics & LLM Explanation Endpoints
# ---------------------------------------------------------------------------

@router.get("/circuits/{circuit_id}/validate", response_model=ValidationResult, tags=["Validation"])
@router.post("/circuits/{circuit_id}/validate", response_model=ValidationResult, tags=["Validation"])
def run_validation(
    circuit_id: str,
    target: Optional[str] = Query(None, description="Optional target project name, e.g. 'led-blink-pin13'")
):
    """
    Runs deterministic validation engine against the saved circuit document.
    Reads from the single source of truth.
    """
    repo = get_repo()
    circuit = repo.get(circuit_id)
    if not circuit:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_id}' not found")
    return validate_circuit(circuit, target_name=target)

class DiagnoseRequest(BaseModel):
    circuit_id: str
    question: Optional[str] = None
    mode: Literal["summary", "troubleshoot", "assembly"] = "troubleshoot"

@router.post("/diagnose", tags=["Diagnostics"])
async def diagnose_circuit(payload: DiagnoseRequest):
    """
    Featherless hardware diagnostic agent.
    Generates plain-language summaries, component overviews, pinout checks,
    and step-by-step diagnostic/assembly instructions with component highlighting bboxes.
    """
    repo = get_repo()
    circuit = repo.get(payload.circuit_id)
    if not circuit:
        raise HTTPException(status_code=404, detail=f"Circuit '{payload.circuit_id}' not found")

    val_result = validate_circuit(circuit)
    res = await run_diagnostic_agent(
        circuit=circuit,
        val_result=val_result,
        question=payload.question,
        mode=payload.mode
    )
    return res

@router.post("/circuits/{circuit_id}/explain", tags=["Validation"])
async def explain_validation(
    circuit_id: str,
    target: Optional[str] = Query(None, description="Optional target project name")
):
    """
    Backward-compatible explanation endpoint.
    Uses diagnostic agent to explain findings without altering verdicts.
    """
    repo = get_repo()
    circuit = repo.get(circuit_id)
    if not circuit:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_id}' not found")

    val_result = validate_circuit(circuit, target_name=target)
    diag = await run_diagnostic_agent(circuit=circuit, val_result=val_result, mode="troubleshoot")

    return {
        "verdict": val_result.status,
        "findings": val_result.findings,
        "explanation": diag.get("summary_plain_language", ""),
        "diagnostic_details": diag,
        "llm_powered": diag.get("llm_available", False)
    }

@router.post("/explain", tags=["Validation"])
async def explain_general(payload: Dict[str, Any]):
    """Alias for POST /circuits/{id}/explain."""
    cid = payload.get("circuit_id")
    if not cid:
        raise HTTPException(status_code=400, detail="Missing circuit_id")
    return await explain_validation(circuit_id=cid)

# ---------------------------------------------------------------------------
# Live Frame Endpoint
# ---------------------------------------------------------------------------

@router.post("/circuits/{circuit_id}/live-frame", tags=["Live Tracking"])
async def process_live_frame(
    circuit_id: str,
    file: UploadFile = File(...)
):
    """
    Ingests live camera frame, tracks bounding boxes, promotes components appearing in N of M frames.
    Returns live tracks and normalized bounding boxes for frontend AR overlays.
    """
    repo = get_repo()
    saved = repo.get(circuit_id)
    if not saved:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_id}' not found")

    frame_bytes = await file.read()
    active_tracks, newly_promoted = component_detector.analyze_frame(frame_bytes)

    if newly_promoted:
        existing_part_ids = {p.id for p in saved.parts}
        for item in newly_promoted:
            label = item["label"]
            vg_type, wokwi_type = map_detector_label(label)
            prefix = "r" if vg_type == "resistor" else ("led" if vg_type == "led" else "part")
            cand_id = f"{prefix}_{item['track_id']}"
            if cand_id not in existing_part_ids:
                new_part = Part(
                    id=cand_id,
                    type=vg_type,
                    bbox=item["bbox"],
                    source="live",
                    conf=item["conf"],
                    confirmed=False,
                    track_id=item["track_id"],
                    wokwi_type=wokwi_type
                )
                saved.parts.append(new_part)
                existing_part_ids.add(cand_id)

        saved, _ = apply_proposals(saved, use_template=True, use_proximity=True, use_llm=False)
        saved = repo.save(saved)

    validation = validate_circuit(saved)

    return {
        "circuit": saved,
        "active_tracks": active_tracks,
        "promoted_count": len(newly_promoted),
        "validation": validation
    }

# ---------------------------------------------------------------------------
# Export Endpoints (All read saved document by id)
# ---------------------------------------------------------------------------

@router.get("/circuits/{circuit_id}/export/json", tags=["Exports"])
def export_json(circuit_id: str):
    """Exports the canonical Circuit JSON document exactly as saved."""
    circuit = get_repo().get(circuit_id)
    if not circuit:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_id}' not found")
    return circuit.model_dump(by_alias=True)

@router.get("/circuits/{circuit_id}/export/svg", tags=["Exports"])
def export_svg(circuit_id: str):
    """Exports clean, readable schematic-style SVG."""
    circuit = get_repo().get(circuit_id)
    if not circuit:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_id}' not found")
    svg_content = generate_circuit_svg(circuit)
    return Response(content=svg_content, media_type="image/svg+xml")

@router.get("/circuits/{circuit_id}/export/png", tags=["Exports"])
def export_png(circuit_id: str):
    """Exports PNG rasterized via cairosvg (guarded, returns graceful error if cairosvg unavailable)."""
    circuit = get_repo().get(circuit_id)
    if not circuit:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_id}' not found")

    png_bytes, err = render_circuit_png(circuit)
    if err or not png_bytes:
        return JSONResponse(
            status_code=501,
            content={
                "error": "PNG rasterization not available",
                "detail": err or "cairosvg not installed or failed",
                "suggestion": f"Use SVG export at /circuits/{circuit_id}/export/svg instead"
            }
        )
    return Response(content=png_bytes, media_type="image/png")

@router.get("/circuits/{circuit_id}/export/wokwi", tags=["Exports"])
def export_wokwi_route(circuit_id: str):
    """Exports Wokwi diagram.json, netlist_text, unsupported_parts, and proposed_wires."""
    circuit = get_repo().get(circuit_id)
    if not circuit:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_id}' not found")
    return export_wokwi(circuit)

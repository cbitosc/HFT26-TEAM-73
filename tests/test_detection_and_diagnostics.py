import os
from PIL import Image
import pytest

from visionguard.config import settings
from visionguard.models.circuit import Circuit, Detection, Defect, HealthSummary, Part, Connection
from visionguard.models.finding import Finding
from visionguard.vision.open_vocab_detector import OpenVocabDetector
from visionguard.vision.merge import calculate_iou, merge_layer_detections
from visionguard.knowledge.pinouts import PinoutKnowledgeBase
from visionguard.vision.defects import compute_health_summary
from visionguard.exports.overlay import generate_circuit_overlay
from visionguard.diagnostics.agent import assemble_diagnostic_context, generate_deterministic_fallback_diagnose
from visionguard.validation.engine import ValidationResult


def test_vocabulary_loading():
    """Verify component_vocab.yaml loads correctly with required components."""
    detector = OpenVocabDetector()
    vocab = detector.vocabulary
    assert len(vocab) > 10, "Vocabulary should contain at least 10 component definitions"

    names = {c["name"] for c in vocab}
    required = {"microcontroller", "resistor", "capacitor", "led", "arduino_board", "usb_port", "push_button"}
    assert required.issubset(names), f"Missing required components: {required - names}"

    for c in vocab:
        assert "prompts" in c and len(c["prompts"]) > 0
        assert "category" in c
        assert "mapped_wiring_type" in c


def test_nms_merge_across_layers():
    """Verify class-aware IoU merging and multi-layer confidence calibration."""
    # Two overlapping detections (IoU > 0.45) from PCB model and Open-Vocab
    layer1 = [{
        "name": "resistor",
        "category": "passive",
        "bbox_px": [100.0, 100.0, 50.0, 30.0],
        "bbox_norm": [0.1, 0.1, 0.05, 0.03],
        "conf": 0.80,
        "source": "pcb_model"
    }]
    layer2 = [{
        "name": "resistor",
        "category": "passive",
        "bbox_px": [102.0, 98.0, 48.0, 32.0],
        "bbox_norm": [0.102, 0.098, 0.048, 0.032],
        "conf": 0.85,
        "source": "open_vocab"
    }]

    merged = merge_layer_detections({"pcb_model": layer1, "open_vocab": layer2}, iou_threshold=0.45)
    assert len(merged) == 1, "Overlapping detections should be merged into one"
    m = merged[0]
    assert set(m["layers_matched"]) == {"pcb_model", "open_vocab"}
    assert m["conf"] >= 0.85, "Multi-model agreement should boost or preserve max confidence"


def test_bbox_px_and_norm_conversion():
    """Verify coordinate normalization and IoU calculations."""
    img_w, img_h = 1000.0, 500.0
    px_box = [100.0, 50.0, 200.0, 100.0] # [x, y, w, h]
    norm_box = [px_box[0] / img_w, px_box[1] / img_h, px_box[2] / img_w, px_box[3] / img_h]

    assert norm_box == [0.1, 0.1, 0.2, 0.2]
    # IoU of identical boxes is 1.0
    assert calculate_iou(px_box, px_box) == 1.0
    # Disjoint boxes have IoU 0.0
    assert calculate_iou(px_box, [400.0, 400.0, 50.0, 50.0]) == 0.0


def test_pinout_lookup_returns_null_for_unknown():
    """Verify pinout KB returns verified data for known parts and strictly null for unknown parts."""
    kb = PinoutKnowledgeBase()

    # Known part
    uno_entry, msg = kb.lookup("arduino-uno")
    assert uno_entry is not None
    assert uno_entry["verified"] is True
    assert "pins" in uno_entry
    assert len(uno_entry["pins"]) > 10

    # Known IC
    ne555, _ = kb.lookup("NE555")
    assert ne555 is not None
    assert ne555["verified"] is True
    assert any(p["name"] == "VCC" for p in ne555["pins"])

    # Unknown fake chip
    fake_entry, fake_msg = kb.lookup("XYZ-FAKE-CHIP-9999")
    assert fake_entry is None, "Pinout must NEVER be invented for unknown components"
    assert "no verified pinout available" in fake_msg


def test_health_summary_logic():
    """Verify deterministic health summary never claims safe and tracks skipped checks."""
    detections = [
        {"name": "led", "category": "semiconductor", "conf": 0.90, "weak": False},
        {"name": "resistor", "category": "passive", "conf": 0.35, "weak": True}
    ]
    defects = [
        {"id": "def1", "type": "missing_component", "severity": "likely", "conf": 0.95}
    ]
    findings = [Finding(rule_id="R009", severity="error", message="LED needs resistor", refs=["led1"])]
    skipped = ["pcb_defect_model"]

    summary = compute_health_summary(detections, defects, findings, skipped)

    assert summary["overall_status"] == "issues"
    assert summary["overall_status"] != "safe"
    assert summary["weak_detections_count"] == 1
    assert summary["defects_by_severity"]["likely"] == 1
    assert "pcb_defect_model" in summary["checks_not_run"]


def test_stub_flagging():
    """Verify stub mode behavior flags all wrappers and detections."""
    from visionguard.vision.board_detector import BoardDetector
    from visionguard.vision.open_vocab_detector import OpenVocabDetector

    bd = BoardDetector()
    st = bd.get_status()
    # In real mode (default), if weights missing, state is 'unavailable', available is False
    assert st["state"] in ("available", "unavailable", "stub")
    assert "status" in st


def test_overlay_drawing():
    """Verify overlay drawing returns a valid PNG image."""
    c = Circuit(
        version=1,
        id="c_test_overlay",
        board={"template": "arduino-uno"},
        detections=[
            Detection(
                id="det_1",
                name="arduino_board",
                category="board",
                bbox_px=[50.0, 50.0, 400.0, 300.0],
                bbox_norm=[0.1, 0.1, 0.8, 0.6],
                conf=0.92,
                source="open_vocab"
            ),
            Detection(
                id="det_2",
                name="resistor",
                category="passive",
                bbox_px=[100.0, 100.0, 60.0, 30.0],
                bbox_norm=[0.2, 0.2, 0.12, 0.06],
                conf=0.40,
                source="open_vocab",
                weak=True
            )
        ],
        defects=[
            Defect(id="def_1", type="image_blurry", bbox_px=[0.0, 0.0, 500.0, 400.0], severity="possible")
        ]
    )

    png_bytes = generate_circuit_overlay(c)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "Overlay must be valid PNG format"


def test_diagnose_prompt_builder_allowed_fields_only():
    """Verify that context assembled for the diagnostic agent contains only allowed structured fields."""
    c = Circuit(
        version=1,
        id="c_diag_test",
        board={"template": "arduino-uno"},
        detections=[
            Detection(
                id="det_1",
                name="atmega328p",
                category="ic",
                bbox_px=[100.0, 100.0, 80.0, 40.0],
                bbox_norm=[0.2, 0.2, 0.16, 0.08],
                conf=0.95,
                source="open_vocab",
                part_number="ATMEGA328P"
            )
        ]
    )
    val = ValidationResult(status="pass", findings=[])

    context = assemble_diagnostic_context(c, val)

    # Allowed keys only
    allowed_keys = {
        "circuit_id", "board_template", "detections", "matched_pinouts",
        "connections", "rule_verdict", "rule_findings", "defects",
        "health_summary", "checks_skipped"
    }
    assert set(context.keys()) == allowed_keys
    # Images must never be in the LLM context
    assert "image" not in context
    assert "image_bytes" not in context

    # Test deterministic fallback
    fb = generate_deterministic_fallback_diagnose(context)
    req_keys = {
        "summary_plain_language", "component_overview", "pinout_notes",
        "suspected_issues", "step_by_step_guide", "assumptions",
        "not_checked", "disclaimer", "llm_available"
    }
    assert req_keys.issubset(fb.keys())
    assert fb["llm_available"] is False
    assert "safe" not in fb["summary_plain_language"].lower()


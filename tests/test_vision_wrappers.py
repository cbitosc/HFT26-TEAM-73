import io
from PIL import Image
from visionguard.config import settings
from visionguard.vision.board_detector import BoardDetector
from visionguard.vision.component_detector import ComponentDetector
from visionguard.vision.slot_inspector import SlotInspector

def create_dummy_image_bytes():
    img = Image.new("RGB", (640, 480), color=(50, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_board_detector_stub():
    old_mode = settings.VISION_BACKEND
    try:
        settings.VISION_BACKEND = "stub"
        detector = BoardDetector()
        status = detector.get_status()
        assert status["state"] == "stub"
        assert status["available"] is False

        img_bytes = create_dummy_image_bytes()
        res = detector.detect_board(img_bytes)
        assert res["found"] is True
        assert res["crop"] is not None
        assert res["crop_bytes"] is not None
        assert res["crop_image"] is not None
        assert res["bbox"] is not None
        assert res["conf"] > 0.8
        assert "deskew" in res
        assert "deskew_params" in res
        assert "homography" in res

        # Test passing PIL Image directly
        pil_img = Image.new("RGB", (640, 480), color=(50, 100, 50))
        res_pil = detector.detect_board(pil_img)
        assert res_pil["found"] is True
        assert res_pil["crop"] is not None
    finally:
        settings.VISION_BACKEND = old_mode

def test_component_detector_stub():
    old_mode = settings.VISION_BACKEND
    try:
        settings.VISION_BACKEND = "stub"
        detector = ComponentDetector()
        status = detector.get_status()
        assert status["state"] == "stub"
        assert status["available"] is False

        img_bytes = create_dummy_image_bytes()
        dets = detector.detect_components(img_bytes)
        assert len(dets) >= 2
        labels = [d["label"] for d in dets]
        assert "led" in labels
        assert "resistor" in labels
        for d in dets:
            assert "bbox" in d
            assert "conf" in d
            assert d.get("source") == "stub"

        # Test passing PIL image directly
        pil_img = Image.new("RGB", (640, 480), color=(50, 100, 50))
        dets_pil = detector.detect_components(pil_img)
        assert len(dets_pil) >= 2
    finally:
        settings.VISION_BACKEND = old_mode

def test_slot_inspector_stub():
    old_mode = settings.VISION_BACKEND
    try:
        settings.VISION_BACKEND = "stub"
        inspector = SlotInspector()
        status = inspector.get_status()
        assert status["state"] == "stub"
        assert status["available"] is False

        img_bytes = create_dummy_image_bytes()
        slots = inspector.inspect_slots(img_bytes)
        assert len(slots) >= 3
        slot_names = [s["slot"] for s in slots]
        assert "slot_mcu" in slot_names
        assert "slot_regulator" in slot_names
        for s in slots:
            assert s["state"] in ("present", "missing", "skipped")
            assert 0.0 <= s["conf"] <= 1.0

        # Test passing PIL image and custom dict slots
        pil_img = Image.new("RGB", (640, 480), color=(50, 100, 50))
        custom_slots = [{"name": "custom_slot_1", "bbox": [0.1, 0.1, 0.2, 0.2]}]
        res_custom = inspector.inspect_slots(pil_img, template_slots=custom_slots)
        assert len(res_custom) == 1
    finally:
        settings.VISION_BACKEND = old_mode
    assert len(res_custom) == 1
    assert res_custom[0]["slot"] == "custom_slot_1"


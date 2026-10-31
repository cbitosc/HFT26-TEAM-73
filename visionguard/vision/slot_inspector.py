import io
import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from visionguard.vision.base import VisionWrapper
from visionguard.templates.boards import ARDUINO_UNO_SLOTS, TemplateSlot
from visionguard.config import settings

logger = logging.getLogger("visionguard.slot_inspector")

# Preprocessing constants from NVIDIA TAO model card
NET_SCALE_FACTOR = 0.01735207357279195
OFFSETS = np.array([123.675, 116.28, 103.53], dtype=np.float32) # RGB offsets
CLASSES = ["Missing", "Present"]

class SlotInspector(VisionWrapper):
    """
    SlotInspector wrapper for NVIDIA TAO GCViT-xxTiny PCB defect classification.
    Binary classifier on component crops: Missing vs Present.
    Crops directly from TEMPLATE slot positions (pad to square, resize to 224x224 RGB).
    """
    def __init__(self):
        self._session = None
        self._loaded = False
        self._load_error: Optional[str] = None

    def available(self) -> Tuple[bool, str]:
        if settings.VISION_BACKEND == "stub":
            return (False, "Stub mode active (canned demo data only)")

        try:
            import onnxruntime
        except ImportError:
            return (False, "Missing 'onnxruntime' package (run: pip install onnxruntime)")

        weights = settings.SLOT_INSPECTOR_WEIGHTS
        if not weights or not os.path.exists(weights):
            return (False, f"SLOT_INSPECTOR_WEIGHTS file not found: '{weights}'. Set SLOT_INSPECTOR_WEIGHTS to NVIDIA TAO GCViT ONNX model path.")

        return (True, "ONNXRuntime and slot inspector weights available")

    def load(self) -> bool:
        if self._loaded:
            return True

        if settings.VISION_BACKEND == "stub":
            self._loaded = True
            return True

        avail, reason = self.available()
        if not avail:
            self._load_error = reason
            logger.warning(f"SlotInspector failed to load: {reason}")
            return False

        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(settings.SLOT_INSPECTOR_WEIGHTS)
            self._loaded = True
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to initialize SlotInspector ONNX session: {e}")
            return False

    def inspect_slots(
        self,
        board_crop: Any,
        template_slots: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Inspects each expected slot on the board crop.
        Returns: [{ "slot": str, "state": "present"|"missing"|"skipped", "conf": float }]
        """
        slots = template_slots or ARDUINO_UNO_SLOTS

        if settings.VISION_BACKEND == "stub":
            return self._inspect_stub(slots)

        if not self.load():
            logger.warning(f"SlotInspector falling back to skipped: {self._load_error}")
            return [
                {
                    "slot": s.name if hasattr(s, "name") else s.get("name", s.get("slot", "unknown")),
                    "state": "skipped",
                    "conf": 0.0
                }
                for s in slots
            ]

        results: List[Dict[str, Any]] = []

        try:
            if isinstance(board_crop, Image.Image):
                pil_board = board_crop.convert("RGB")
            elif isinstance(board_crop, bytes):
                pil_board = Image.open(io.BytesIO(board_crop)).convert("RGB")
            elif hasattr(board_crop, "__array__"):
                pil_board = Image.fromarray(board_crop).convert("RGB")
            else:
                pil_board = Image.open(io.BytesIO(bytes(board_crop))).convert("RGB")
            bw, bh = pil_board.size

            for slot in slots:
                s_name = slot.name if hasattr(slot, "name") else slot.get("name", slot.get("slot", "unknown"))
                s_bbox = slot.bbox if hasattr(slot, "bbox") else slot.get("bbox", [0, 0, 1, 1])

                try:
                    # Convert normalized slot bbox [x, y, w, h] to pixel coordinates
                    sx, sy, sw, sh = s_bbox
                    px1 = max(0, int(sx * bw))
                    py1 = max(0, int(sy * bh))
                    px2 = min(bw, int((sx + sw) * bw))
                    py2 = min(bh, int((sy + sh) * bh))

                    if px2 <= px1 or py2 <= py1:
                        results.append({"slot": s_name, "state": "skipped", "conf": 0.0})
                        continue

                    slot_crop = pil_board.crop((px1, py1, px2, py2))

                    # Pad to square
                    cw, ch = slot_crop.size
                    max_dim = max(cw, ch)
                    square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
                    offset_x = (max_dim - cw) // 2
                    offset_y = (max_dim - ch) // 2
                    square_img.paste(slot_crop, (offset_x, offset_y))

                    # Resize to 224x224
                    resized = square_img.resize((224, 224), Image.Resampling.BILINEAR)

                    # Preprocess for TAO model: (pixel - offset) * scale_factor, NCHW RGB
                    np_img = np.array(resized, dtype=np.float32) # (224, 224, 3)
                    np_img = (np_img - OFFSETS) * NET_SCALE_FACTOR
                    # Transpose to NCHW: (1, 3, 224, 224)
                    input_tensor = np.transpose(np_img, (2, 0, 1))
                    input_tensor = np.expand_dims(input_tensor, axis=0)

                    input_name = self._session.get_inputs()[0].name
                    output = self._session.run(None, {input_name: input_tensor})[0]

                    # Softmax output: [prob_missing, prob_present]
                    probs = np.exp(output - np.max(output))
                    probs = probs / np.sum(probs)

                    prob_missing = float(probs[0][0])
                    prob_present = float(probs[0][1])

                    if prob_present >= 0.5:
                        state = "present"
                        conf = prob_present
                    else:
                        state = "missing"
                        conf = prob_missing

                    results.append({
                        "slot": s_name,
                        "state": state,
                        "conf": round(conf, 2)
                    })

                except Exception as slot_exc:
                    logger.error(f"Error inspecting slot {s_name}: {slot_exc}")
                    results.append({"slot": s_name, "state": "skipped", "conf": 0.0})

        except Exception as exc:
            logger.error(f"SlotInspector top-level error: {exc}")
            return [
                {
                    "slot": s.name if hasattr(s, "name") else s.get("name", s.get("slot", "unknown")),
                    "state": "skipped",
                    "conf": 0.0
                }
                for s in slots
            ]

        return results

    def _inspect_stub(self, slots: List[Any]) -> List[Dict[str, Any]]:
        """Stub slot inspection returning realistic present states."""
        stub_results = []
        for s in slots:
            s_name = s.name if hasattr(s, "name") else s.get("name", s.get("slot", "unknown"))
            stub_results.append({
                "slot": s_name,
                "state": "present",
                "conf": 0.96
            })
        return stub_results


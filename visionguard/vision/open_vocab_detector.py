import io
import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import yaml
from PIL import Image

from visionguard.vision.base import VisionWrapper
from visionguard.config import settings

logger = logging.getLogger("visionguard.open_vocab")

DEFAULT_VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "component_vocab.yaml"
)

class OpenVocabDetector(VisionWrapper):
    """
    Open-Vocabulary Zero-Shot Detector layer.
    Text-prompted with vocabulary from config/component_vocab.yaml.
    Supports OWLv2 (HuggingFace transformers) and YOLO-World (ultralytics).
    """
    def __init__(self, vocab_path: str = DEFAULT_VOCAB_PATH):
        self.vocab_path = vocab_path
        self._vocab: List[Dict[str, Any]] = []
        self._prompt_to_comp: Dict[str, Dict[str, Any]] = {}
        self._prompts: List[str] = []
        self._model = None
        self._processor = None
        self._backend_type: Optional[str] = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._load_vocab()

    def _load_vocab(self) -> None:
        if os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    self._vocab = data.get("components", [])
            except Exception as e:
                logger.error(f"Error loading vocabulary from {self.vocab_path}: {e}")
                self._vocab = []
        else:
            logger.warning(f"Vocabulary file {self.vocab_path} not found.")
            self._vocab = []

        self._prompts = []
        self._prompt_to_comp = {}
        for c in self._vocab:
            prompts = c.get("prompts", [c.get("name")])
            for p in prompts:
                p_clean = p.strip()
                self._prompts.append(p_clean)
                self._prompt_to_comp[p_clean.lower()] = c

    @property
    def vocabulary(self) -> List[Dict[str, Any]]:
        return self._vocab

    def available(self) -> Tuple[bool, str]:
        if settings.VISION_BACKEND == "stub":
            return (False, "Stub mode active (canned demo data only)")

        # Check ultralytics (for YOLO-World)
        try:
            import ultralytics # noqa: F401
            return (True, "Ultralytics available for YOLO-World")
        except ImportError:
            pass

        # Check transformers (for OWLv2)
        try:
            import transformers # noqa: F401
            import torch # noqa: F401
            return (True, "Transformers and PyTorch available for OWLv2")
        except ImportError:
            pass

        return (False, "Missing 'transformers' or 'ultralytics' (install requirements-vision.txt)")

    def load(self) -> bool:
        if self._loaded:
            return True

        if settings.VISION_BACKEND == "stub":
            self._loaded = True
            return True

        avail, reason = self.available()
        if not avail:
            self._load_error = reason
            return False

        # Attempt 1: YOLO-World via ultralytics
        try:
            from ultralytics import YOLO
            # If OPEN_VOCAB_MODEL points to a YOLO model or default
            model_name = settings.OPEN_VOCAB_MODEL
            if "yolo" in model_name.lower():
                self._model = YOLO(model_name)
                # Set custom text vocabulary
                if self._prompts:
                    self._model.set_classes(self._prompts[:64]) # Max prompt batch
                self._backend_type = "yolo_world"
                self._loaded = True
                return True
        except Exception as e:
            logger.info(f"YOLO-World load skipped: {e}")

        # Attempt 2: OWLv2 via transformers
        try:
            import torch
            from transformers import Owlv2Processor, Owlv2ForObjectDetection

            model_id = settings.OPEN_VOCAB_MODEL if "owl" in settings.OPEN_VOCAB_MODEL.lower() else "google/owlv2-base-patch16-ensemble"
            device = settings.OPEN_VOCAB_DEVICE if torch.cuda.is_available() and settings.OPEN_VOCAB_DEVICE == "cuda" else "cpu"

            self._processor = Owlv2Processor.from_pretrained(model_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)
            self._backend_type = "owlv2"
            self._loaded = True
            return True
        except Exception as exc:
            self._load_error = f"Failed to load open-vocab model: {exc}"
            logger.warning(self._load_error)
            return False

    def detect(
        self,
        image_input: Any,
        conf_threshold: Optional[float] = None,
        max_image_dim: int = 1024
    ) -> List[Dict[str, Any]]:
        """
        Runs open-vocabulary text-prompted detection on image.
        Returns:
        [{
            "name": str,
            "category": str,
            "bbox_px": [x, y, w, h],
            "bbox_norm": [x, y, w, h],
            "conf": float,
            "source": "open_vocab",
            "mapped_wiring_type": Optional[str],
            "weak": bool
        }]
        """
        conf_thresh = conf_threshold if conf_threshold is not None else settings.DETECT_CONF
        weak_thresh = settings.WEAK_CONF

        try:
            if isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
            elif isinstance(image_input, bytes):
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif hasattr(image_input, "__array__"):
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                pil_img = Image.open(io.BytesIO(bytes(image_input))).convert("RGB")
        except Exception as err:
            logger.error(f"Failed to read image in OpenVocabDetector: {err}")
            return []

        orig_w, orig_h = pil_img.size

        # In stub mode, produce rich multi-component detections
        if settings.VISION_BACKEND == "stub":
            return self._detect_stub(orig_w, orig_h)

        if not self.load():
            return []

        # Downscale for open-vocabulary model if image is very large
        scale = 1.0
        proc_img = pil_img
        if orig_w > max_image_dim or orig_h > max_image_dim:
            scale = max_image_dim / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            proc_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        detections: List[Dict[str, Any]] = []

        try:
            if self._backend_type == "owlv2" and self._processor and self._model:
                import torch
                # Group prompts in batches if needed
                queries = [self._prompts]
                inputs = self._processor(text=queries, images=proc_img, return_tensors="pt")
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model(**inputs)

                target_sizes = torch.tensor([[proc_img.height, proc_img.width]], device=device)
                results = self._processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=conf_thresh
                )

                if results and len(results) > 0:
                    r = results[0]
                    boxes = r["boxes"].cpu().numpy()
                    scores = r["scores"].cpu().numpy()
                    labels = r["labels"].cpu().numpy()

                    for box, score, label_idx in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        # Rescale to original image coordinates
                        if scale != 1.0:
                            x1 /= scale
                            y1 /= scale
                            x2 /= scale
                            y2 /= scale

                        bw = x2 - x1
                        bh = y2 - y1
                        conf = float(score)

                        prompt_str = self._prompts[int(label_idx)] if int(label_idx) < len(self._prompts) else "component"
                        comp_info = self._prompt_to_comp.get(prompt_str.lower(), {})

                        detections.append({
                            "name": comp_info.get("name", prompt_str),
                            "category": comp_info.get("category", "component"),
                            "bbox_px": [round(x1, 1), round(y1, 1), round(bw, 1), round(bh, 1)],
                            "bbox_norm": [round(x1 / orig_w, 4), round(y1 / orig_h, 4), round(bw / orig_w, 4), round(bh / orig_h, 4)],
                            "conf": round(conf, 3),
                            "source": "open_vocab",
                            "mapped_wiring_type": comp_info.get("mapped_wiring_type"),
                            "weak": conf < weak_thresh
                        })

            elif self._backend_type == "yolo_world" and self._model:
                results = self._model.predict(proc_img, conf=conf_thresh, verbose=False)
                if results and len(results[0].boxes) > 0:
                    for b in results[0].boxes:
                        conf = float(b.conf[0])
                        cls_idx = int(b.cls[0])
                        xyxy = b.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = float(xyxy[0]) / scale, float(xyxy[1]) / scale, float(xyxy[2]) / scale, float(xyxy[3]) / scale
                        bw, bh = x2 - x1, y2 - y1

                        label_name = results[0].names.get(cls_idx, "component")
                        comp_info = self._prompt_to_comp.get(label_name.lower(), {})

                        detections.append({
                            "name": comp_info.get("name", label_name),
                            "category": comp_info.get("category", "component"),
                            "bbox_px": [round(x1, 1), round(y1, 1), round(bw, 1), round(bh, 1)],
                            "bbox_norm": [round(x1 / orig_w, 4), round(y1 / orig_h, 4), round(bw / orig_w, 4), round(bh / orig_h, 4)],
                            "conf": round(conf, 3),
                            "source": "open_vocab",
                            "mapped_wiring_type": comp_info.get("mapped_wiring_type"),
                            "weak": conf < weak_thresh
                        })

        except Exception as e:
            logger.error(f"Inference error in OpenVocabDetector: {e}")

        return detections

    def _detect_stub(self, w: int, h: int) -> List[Dict[str, Any]]:
        """Stub zero-shot detections for comprehensive component demonstration."""
        return [
            {
                "name": "arduino_board",
                "category": "board",
                "bbox_px": [round(w * 0.15, 1), round(h * 0.15, 1), round(w * 0.70, 1), round(h * 0.70, 1)],
                "bbox_norm": [0.15, 0.15, 0.70, 0.70],
                "conf": 0.94,
                "source": "stub",
                "mapped_wiring_type": "arduino-uno",
                "weak": False
            },
            {
                "name": "microcontroller",
                "category": "ic",
                "bbox_px": [round(w * 0.35, 1), round(h * 0.50, 1), round(w * 0.22, 1), round(h * 0.10, 1)],
                "bbox_norm": [0.35, 0.50, 0.22, 0.10],
                "conf": 0.91,
                "source": "stub",
                "mapped_wiring_type": None,
                "weak": False
            },
            {
                "name": "voltage_regulator",
                "category": "ic",
                "bbox_px": [round(w * 0.20, 1), round(h * 0.25, 1), round(w * 0.08, 1), round(h * 0.08, 1)],
                "bbox_norm": [0.20, 0.25, 0.08, 0.08],
                "conf": 0.82,
                "source": "stub",
                "mapped_wiring_type": None,
                "weak": False
            },
            {
                "name": "usb_port",
                "category": "connector",
                "bbox_px": [round(w * 0.12, 1), round(h * 0.20, 1), round(w * 0.12, 1), round(h * 0.15, 1)],
                "bbox_norm": [0.12, 0.20, 0.12, 0.15],
                "conf": 0.89,
                "source": "stub",
                "mapped_wiring_type": None,
                "weak": False
            },
            {
                "name": "gpio_header",
                "category": "connector",
                "bbox_px": [round(w * 0.25, 1), round(h * 0.16, 1), round(w * 0.45, 1), round(h * 0.06, 1)],
                "bbox_norm": [0.25, 0.16, 0.45, 0.06],
                "conf": 0.85,
                "source": "stub",
                "mapped_wiring_type": None,
                "weak": False
            },
            {
                "name": "led",
                "category": "semiconductor",
                "bbox_px": [round(w * 0.55, 1), round(h * 0.35, 1), round(w * 0.08, 1), round(h * 0.08, 1)],
                "bbox_norm": [0.55, 0.35, 0.08, 0.08],
                "conf": 0.88,
                "source": "stub",
                "mapped_wiring_type": "led",
                "weak": False
            },
            {
                "name": "resistor",
                "category": "passive",
                "bbox_px": [round(w * 0.65, 1), round(h * 0.50, 1), round(w * 0.06, 1), round(h * 0.04, 1)],
                "bbox_norm": [0.65, 0.50, 0.06, 0.04],
                "conf": 0.76,
                "source": "stub",
                "mapped_wiring_type": "resistor",
                "weak": False
            },
            {
                "name": "push_button",
                "category": "switch",
                "bbox_px": [round(w * 0.20, 1), round(h * 0.70, 1), round(w * 0.08, 1), round(h * 0.08, 1)],
                "bbox_norm": [0.20, 0.70, 0.08, 0.08],
                "conf": 0.72,
                "source": "stub",
                "mapped_wiring_type": "pushbutton",
                "weak": False
            }
        ]

default_open_vocab_detector = OpenVocabDetector()


import os
import io
import logging
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from visionguard.vision.base import VisionWrapper
from visionguard.config import settings

logger = logging.getLogger("visionguard.board_detector")

class BoardDetector(VisionWrapper):
    """
    BoardDetector wrapper for SanderGi / PCB-Detection.
    Locates, crops (with 10-15% padding), and deskews the PCB in the image.
    Supports lazy loading and live caching every N frames.
    """
    def __init__(self):
        self._model = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._cached_result: Optional[Dict[str, Any]] = None
        self._last_checked_frame: int = -1

    def available(self) -> Tuple[bool, str]:
        if settings.VISION_BACKEND == "stub":
            return (False, "Stub mode active (canned demo data only)")

        try:
            # pyrefly: ignore[missing-import]
            import ultralytics
        except ImportError:
            return (False, "Missing 'ultralytics' package (run: pip install ultralytics)")

        weights = settings.BOARD_DETECTOR_WEIGHTS
        if not weights or not os.path.exists(weights):
            return (False, f"BOARD_DETECTOR_WEIGHTS file not found: '{weights}'. Set BOARD_DETECTOR_WEIGHTS to SanderGi/PCB-Detection model path.")

        return (True, "Ultralytics and board weights available")

    def load(self) -> bool:
        if self._loaded:
            return True

        if settings.VISION_BACKEND == "stub":
            self._loaded = True
            return True

        avail, reason = self.available()
        if not avail:
            self._load_error = reason
            logger.warning(f"BoardDetector failed to load: {reason}")
            return False

        try:
            # pyrefly: ignore[missing-import]
            from ultralytics import YOLO
            self._model = YOLO(settings.BOARD_DETECTOR_WEIGHTS)
            self._loaded = True
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load board detector model: {e}")
            return False

    def detect_board(
        self,
        image_input: Any,
        frame_index: int = 0,
        pad_fraction: float = 0.12
    ) -> Dict[str, Any]:
        """
        Detects PCB, returns:
        {
          "found": bool,
          "crop": bytes,
          "crop_bytes": bytes,
          "crop_image": PIL.Image,
          "crop_bbox": [x1, y1, x2, y2],
          "bbox": [x, y, w, h],
          "conf": float,
          "deskew": dict,
          "deskew_params": dict,
          "homography": list,
          "crop_to_orig_offset": (ox, oy)
        }
        """
        # Live caching: re-check only every N frames
        if (
            self._cached_result is not None
            and frame_index > 0
            and (frame_index % settings.BOARD_RECHECK_FRAMES != 0)
        ):
            return self._cached_result

        # Convert image_input to PIL image and bytes
        try:
            if isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                image_bytes = buf.getvalue()
            elif isinstance(image_input, bytes):
                image_bytes = image_input
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            elif hasattr(image_input, "__array__"):
                pil_img = Image.fromarray(image_input).convert("RGB")
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                image_bytes = buf.getvalue()
            else:
                image_bytes = bytes(image_input)
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as img_err:
            logger.error(f"Failed to parse input image in BoardDetector: {img_err}")
            return {
                "found": False, "crop": None, "crop_bytes": None, "crop_image": None,
                "bbox": None, "conf": 0.0, "deskew": None, "deskew_params": {}, "homography": None
            }

        # Handle stub backend
        if settings.VISION_BACKEND == "stub":
            res = self._detect_stub(pil_img, pad_fraction)
            self._cached_result = res
            return res

        # Real backend execution
        if not self.load():
            logger.warning(f"BoardDetector falling back to found=False: {self._load_error}")
            return {
                "found": False, "crop": None, "crop_bytes": None, "crop_image": None,
                "bbox": None, "conf": 0.0, "deskew": None, "deskew_params": {}, "homography": None
            }

        try:
            img_w, img_h = pil_img.size

            results = self._model(pil_img, verbose=False)
            if not results or len(results[0].boxes) == 0:
                return {
                    "found": False, "crop": None, "crop_bytes": None, "crop_image": None,
                    "bbox": None, "conf": 0.0, "deskew": None, "deskew_params": {}, "homography": None
                }

            # Pick highest confidence box for PCB
            best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            conf = float(best_box.conf[0])
            xyxy = best_box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]

            bx1, by1, bx2, by2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            bw = bx2 - bx1
            bh = by2 - by1

            # Pad the crop (about 10-15%)
            pad_w = bw * pad_fraction
            pad_h = bh * pad_fraction

            crop_x1 = max(0, int(bx1 - pad_w))
            crop_y1 = max(0, int(by1 - pad_h))
            crop_x2 = min(img_w, int(bx2 + pad_w))
            crop_y2 = min(img_h, int(by2 + pad_h))

            cropped = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG")
            crop_bytes = buf.getvalue()

            deskew_dict = {
                "angle_deg": 0.0,
                "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            }

            res = {
                "found": True,
                "crop": crop_bytes,
                "crop_bytes": crop_bytes,
                "crop_image": cropped,
                "crop_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
                "bbox": [bx1, by1, bw, bh],
                "conf": conf,
                "deskew": deskew_dict,
                "deskew_params": deskew_dict,
                "homography": deskew_dict["homography"],
                "crop_to_orig_offset": (crop_x1, crop_y1)
            }
            self._cached_result = res
            return res

        except Exception as exc:
            logger.error(f"Error in BoardDetector inference: {exc}")
            return {
                "found": False, "crop": None, "crop_bytes": None, "crop_image": None,
                "bbox": None, "conf": 0.0, "deskew": None, "deskew_params": {}, "homography": None
            }

    def _detect_stub(self, pil_img: Image.Image, pad_fraction: float) -> Dict[str, Any]:
        """Stub detection that generates a valid board crop without requiring GPU or weights."""
        img_w, img_h = pil_img.size
        # Simulate board filling about 70% of frame
        bx1 = int(img_w * 0.15)
        by1 = int(img_h * 0.15)
        bx2 = int(img_w * 0.85)
        by2 = int(img_h * 0.85)
        bw = bx2 - bx1
        bh = by2 - by1

        # Pad crop by 12%
        pad_w = int(bw * pad_fraction)
        pad_h = int(bh * pad_fraction)
        cx1 = max(0, bx1 - pad_w)
        cy1 = max(0, by1 - pad_h)
        cx2 = min(img_w, bx2 + pad_w)
        cy2 = min(img_h, by2 + pad_h)

        crop = pil_img.crop((cx1, cy1, cx2, cy2))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG")
        crop_bytes = buf.getvalue()

        deskew_dict = {
            "angle_deg": 0.0,
            "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        }

        return {
            "found": True,
            "crop": crop_bytes,
            "crop_bytes": crop_bytes,
            "crop_image": crop,
            "crop_bbox": [cx1, cy1, cx2, cy2],
            "bbox": [bx1, by1, bw, bh],
            "conf": 0.95,
            "deskew": deskew_dict,
            "deskew_params": deskew_dict,
            "homography": deskew_dict["homography"],
            "crop_to_orig_offset": (cx1, cy1)
        }


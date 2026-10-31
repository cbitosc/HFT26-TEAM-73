import io
import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from visionguard.vision.base import VisionWrapper
from visionguard.vision.tiling import tile_and_detect
from visionguard.vision.tracker import IoUTracker, Track
from visionguard.config import settings

logger = logging.getLogger("visionguard.component_detector")

# Non-negotiable: Print startup warning about research license
print(
    "[VisionGuard WARNING] ParsaVictor/YOLO26 component detection weights may be subject to "
    "research-use only licensing terms. Confirm license terms before any commercial use.",
    file=sys.stderr
)

class ComponentDetector(VisionWrapper):
    """
    ComponentDetector wrapper for ParsaVictor / YOLO26.
    Detects hardware components on a board crop.
    - Uses tiling at 1024px with overlap & NMS on large crops
    - Keeps raw confidences (starts confirmed=False)
    - Supports live frame tracking with IoUTracker & promotion (N of M frames)
    """
    def __init__(self):
        self._model = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self.tracker = IoUTracker()

    def available(self) -> Tuple[bool, str]:
        if settings.VISION_BACKEND == "stub":
            return (False, "Stub mode active (canned demo data only)")

        try:
            import ultralytics
        except ImportError:
            return (False, "Missing 'ultralytics' package (run: pip install ultralytics)")

        weights = settings.COMPONENT_DETECTOR_WEIGHTS
        if not weights or not os.path.exists(weights):
            return (False, f"COMPONENT_DETECTOR_WEIGHTS file not found: '{weights}'. Set COMPONENT_DETECTOR_WEIGHTS to ParsaVictor/YOLO26 model path.")

        return (True, "Ultralytics and component weights available")

    def load(self) -> bool:
        if self._loaded:
            return True

        if settings.VISION_BACKEND == "stub":
            self._loaded = True
            return True

        avail, reason = self.available()
        if not avail:
            self._load_error = reason
            logger.warning(f"ComponentDetector failed to load: {reason}")
            return False

        try:
            from ultralytics import YOLO
            self._model = YOLO(settings.COMPONENT_DETECTOR_WEIGHTS)
            self._loaded = True
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load component detector model: {e}")
            return False

    def detect_components(self, board_crop: Any) -> List[Dict[str, Any]]:
        """
        Runs component detection on board crop.
        Returns list of detections in board crop coordinates:
        [{ 'label': str, 'bbox': [x, y, w, h], 'conf': float, 'source': str }]
        """
        # Convert board_crop to PIL image
        try:
            if isinstance(board_crop, Image.Image):
                pil_img = board_crop.convert("RGB")
            elif isinstance(board_crop, bytes):
                pil_img = Image.open(io.BytesIO(board_crop)).convert("RGB")
            elif hasattr(board_crop, "__array__"):
                pil_img = Image.fromarray(board_crop).convert("RGB")
            else:
                pil_img = Image.open(io.BytesIO(bytes(board_crop))).convert("RGB")
        except Exception as err:
            logger.error(f"Failed to read board_crop in ComponentDetector: {err}")
            pil_img = Image.new("RGB", (800, 600), (30, 80, 40))

        if settings.VISION_BACKEND == "stub":
            return self._detect_stub(pil_img)

        if not self.load():
            logger.warning(f"ComponentDetector falling back to empty list: {self._load_error}")
            return []

        try:
            img_w, img_h = pil_img.size

            # If image exceeds tile_size (1024px), use tiling with overlap and NMS
            if img_w > settings.TILE_SIZE or img_h > settings.TILE_SIZE:
                def tile_detect_fn(tx: int, ty: int, tw: int, th: int) -> List[Dict]:
                    tile_crop = pil_img.crop((tx, ty, tx + tw, ty + th))
                    return self._infer_single_image(tile_crop)

                return tile_and_detect(
                    img_w,
                    img_h,
                    tile_detect_fn,
                    tile_size=settings.TILE_SIZE,
                    overlap=settings.TILE_OVERLAP,
                    nms_iou_threshold=settings.NMS_IOU_THRESHOLD
                )
            else:
                return self._infer_single_image(pil_img)

        except Exception as exc:
            logger.error(f"Error during component detection: {exc}")
            return []

    def _infer_single_image(self, pil_img: Image.Image) -> List[Dict[str, Any]]:
        results = self._model(pil_img, verbose=False)
        detections: List[Dict[str, Any]] = []
        if not results:
            return detections

        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label = results[0].names.get(cls_idx, "unknown")
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            w = x2 - x1
            h = y2 - y1
            detections.append({
                "label": label,
                "bbox": [x1, y1, w, h],
                "conf": conf,
                "source": "vision"
            })
        return detections

    def _detect_stub(self, pil_img: Image.Image) -> List[Dict[str, Any]]:
        """Stub component detection producing realistic components for Arduino Uno sample project."""
        w, h = pil_img.size

        # Realistic board-relative coordinates
        return [
            {
                "label": "led",
                "bbox": [w * 0.55, h * 0.35, w * 0.08, h * 0.08],
                "conf": 0.88,
                "color": "red",
                "source": "stub"
            },
            {
                "label": "resistor",
                "bbox": [w * 0.42, h * 0.36, w * 0.10, h * 0.06],
                "conf": 0.84,
                "value": "220",
                "source": "stub"
            },
            {
                "label": "pushbutton",
                "bbox": [w * 0.70, h * 0.50, w * 0.10, h * 0.10],
                "conf": 0.79,
                "source": "stub"
            }
        ]

    def analyze_frame(self, frame_bytes: bytes) -> Tuple[List[Dict], List[Dict]]:
        """
        Live video frame processing:
        - Downsizes frame to 640-1024px
        - Detects components
        - Updates IoUTracker
        - Returns (active_tracks, newly_promoted_components)
        """
        try:
            pil_img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            w, h = pil_img.size
            max_dim = max(w, h)
            if max_dim > 1024:
                scale = 1024.0 / max_dim
                pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            elif max_dim < 640:
                scale = 640.0 / max_dim
                pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            downsized_bytes = buf.getvalue()
        except Exception:
            downsized_bytes = frame_bytes

        # Detect raw components on downsized frame
        raw_dets = self.detect_components(downsized_bytes)

        # Update tracker
        all_tracks, newly_promoted = self.tracker.update(raw_dets)

        active_track_dicts = [
            {"track_id": t.track_id, "label": t.label, "bbox": t.bbox, "conf": t.conf, "promoted": t.promoted}
            for t in all_tracks
        ]

        promoted_components = [
            {
                "track_id": t.track_id,
                "label": t.label,
                "bbox": t.bbox,
                "conf": t.conf,
                "source": "live"
            }
            for t in newly_promoted
        ]

        return active_track_dicts, promoted_components


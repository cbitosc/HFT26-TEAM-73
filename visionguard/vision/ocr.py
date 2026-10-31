import io
import re
import logging
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from visionguard.vision.base import VisionWrapper
from visionguard.config import settings

logger = logging.getLogger("visionguard.ocr")

PART_NUMBER_REGEX = re.compile(r"\b([A-Z0-9]{2,10}[-_]?[A-Z0-9]{2,10})\b", re.IGNORECASE)

class OCREngine(VisionWrapper):
    """
    Guarded optical character recognition for IC chip and component crops.
    Extracts surface markings and derives clean candidate part numbers.
    Supports EasyOCR and pytesseract, degrading gracefully when not installed.
    """
    def __init__(self):
        self._engine_type: Optional[str] = None
        self._reader = None
        self._loaded = False
        self._load_error: Optional[str] = None

    def available(self) -> Tuple[bool, str]:
        if not settings.OCR_ENABLED:
            return (False, "OCR disabled by config (OCR_ENABLED=false)")

        if settings.VISION_BACKEND == "stub":
            return (False, "Stub mode active (canned demo data only)")

        # Try easyocr first
        try:
            import easyocr # noqa: F401
            return (True, "EasyOCR available")
        except ImportError:
            pass

        # Try pytesseract second
        try:
            import pytesseract # noqa: F401
            return (True, "pytesseract available")
        except ImportError:
            pass

        return (False, "No OCR library installed (install easyocr or pytesseract in requirements-vision.txt)")

    def load(self) -> bool:
        if self._loaded:
            return True

        avail, reason = self.available()
        if not avail:
            self._load_error = reason
            return False

        try:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=(settings.OPEN_VOCAB_DEVICE == "cuda"), verbose=False)
            self._engine_type = "easyocr"
            self._loaded = True
            return True
        except Exception:
            pass

        try:
            import pytesseract
            self._engine_type = "pytesseract"
            self._loaded = True
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def extract_text(self, crop_input: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        Runs OCR on component crop.
        Returns: (ocr_text: Optional[str], candidate_part_number: Optional[str])
        """
        if settings.VISION_BACKEND == "stub":
            return self._stub_ocr()

        if not self.load():
            return None, None

        try:
            if isinstance(crop_input, Image.Image):
                pil_img = crop_input.convert("RGB")
            elif isinstance(crop_input, bytes):
                pil_img = Image.open(io.BytesIO(crop_input)).convert("RGB")
            elif hasattr(crop_input, "__array__"):
                pil_img = Image.fromarray(crop_input).convert("RGB")
            else:
                pil_img = Image.open(io.BytesIO(bytes(crop_input))).convert("RGB")

            raw_text = ""
            if self._engine_type == "easyocr" and self._reader:
                import numpy as np
                img_np = np.array(pil_img)
                res = self._reader.readtext(img_np, detail=0)
                raw_text = " ".join(res).strip()
            elif self._engine_type == "pytesseract":
                import pytesseract
                raw_text = pytesseract.image_to_string(pil_img).strip()

            if not raw_text:
                return None, None

            candidate = self._clean_part_number(raw_text)
            return raw_text, candidate

        except Exception as exc:
            logger.warning(f"OCR execution warning: {exc}")
            return None, None

    def _clean_part_number(self, text: str) -> Optional[str]:
        """Filters OCR text to find standard semiconductor part number pattern."""
        matches = PART_NUMBER_REGEX.findall(text)
        if matches:
            # Pick longest match
            best = max(matches, key=len)
            return best.upper()
        return None

    def _stub_ocr(self) -> Tuple[Optional[str], Optional[str]]:
        """Stub OCR returning typical ATmega328P microcontroller chip marking."""
        return "ATMEGA328P-PU 2314", "ATMEGA328P"

default_ocr_engine = OCREngine()


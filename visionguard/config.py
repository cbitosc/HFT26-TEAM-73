import os
from typing import Dict

class Settings:
    # Runtime backend mode: "stub" (default when weights not configured) or "real"
    VISION_BACKEND: str = os.getenv("VISION_BACKEND", "stub").lower()

    # Multimodal LLM settings
    VISIONGUARD_MODEL: str = os.getenv("VISIONGUARD_MODEL", "gemini-2.5-flash")
    GEMINI_API_KEY: str = os.getenv("VISIONGUARD_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

    # Featherless LLM settings for diagnostic assistant
    FEATHERLESS_BASE_URL: str = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
    FEATHERLESS_MODEL: str = os.getenv("FEATHERLESS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    FEATHERLESS_API_KEY: str = os.getenv("FEATHERLESS_API_KEY", "")

    # Pretrained weights paths (configured via env, never hardcoded)
    BOARD_DETECTOR_WEIGHTS: str = os.getenv("BOARD_DETECTOR_WEIGHTS", "")
    COMPONENT_DETECTOR_WEIGHTS: str = os.getenv("COMPONENT_DETECTOR_WEIGHTS", "")
    SLOT_INSPECTOR_WEIGHTS: str = os.getenv("SLOT_INSPECTOR_WEIGHTS", "")
    DEFECT_MODEL_PATH: str = os.getenv("DEFECT_MODEL_PATH", "")

    # Open-Vocabulary Detector settings
    OPEN_VOCAB_MODEL: str = os.getenv("OPEN_VOCAB_MODEL", "google/owlv2-base-patch16-ensemble")
    OPEN_VOCAB_DEVICE: str = os.getenv("OPEN_VOCAB_DEVICE", "cpu").lower()
    DETECT_CONF: float = float(os.getenv("DETECT_CONF", "0.25"))
    WEAK_CONF: float = float(os.getenv("WEAK_CONF", "0.45"))
    OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "true").lower() in ("true", "1", "yes")
    LLM_VISION_ENABLED: bool = os.getenv("LLM_VISION_ENABLED", "false").lower() in ("true", "1", "yes")

    # Live streaming parameters
    BOARD_RECHECK_FRAMES: int = int(os.getenv("BOARD_RECHECK_FRAMES", "15"))
    LIVE_PROMOTE_N: int = int(os.getenv("LIVE_PROMOTE_N", "5"))
    LIVE_PROMOTE_M: int = int(os.getenv("LIVE_PROMOTE_M", "8"))

    # Tiling and NMS parameters
    TILE_SIZE: int = int(os.getenv("TILE_SIZE", "1024"))
    TILE_OVERLAP: float = float(os.getenv("TILE_OVERLAP", "0.20"))
    NMS_IOU_THRESHOLD: float = float(os.getenv("NMS_IOU_THRESHOLD", "0.45"))

    # Per-label confidence thresholds for component classification (R012 triggers if below)
    CONFIDENCE_THRESHOLDS: Dict[str, float] = {
        "led": float(os.getenv("CONF_THRESH_LED", "0.60")),
        "resistor": float(os.getenv("CONF_THRESH_RESISTOR", "0.55")),
        "pushbutton": float(os.getenv("CONF_THRESH_PUSHBUTTON", "0.50")),
        "arduino-uno": float(os.getenv("CONF_THRESH_ARDUINO", "0.70")),
        "default": float(os.getenv("CONF_THRESH_DEFAULT", "0.50")),
    }

settings = Settings()


from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

class VisionWrapper(ABC):
    """
    Base wrapper interface for all pretrained vision models.
    - Loads lazily
    - Exposes available() with human-readable reason
    - Reads paths and parameters from environment variables
    - NEVER raises uncaught exceptions into the API
    """

    @abstractmethod
    def available(self) -> Tuple[bool, str]:
        """Returns (is_available, status_or_reason)."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """Lazily loads weights and initializes the model. Returns True if successful."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """
        Returns structured status dict:
        { "available": bool, "state": "available"|"stub"|"unavailable", "status": str, "install_hint": str|None }
        """
        is_avail, reason = self.available()
        from visionguard.config import settings
        if settings.VISION_BACKEND == "stub":
            return {
                "available": False,
                "state": "stub",
                "status": "Stub mode active (canned data only)",
                "install_hint": None
            }
        if is_avail:
            return {
                "available": True,
                "state": "available",
                "status": reason,
                "install_hint": None
            }
        return {
            "available": False,
            "state": "unavailable",
            "status": reason,
            "install_hint": reason
        }


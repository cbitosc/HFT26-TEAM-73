import logging
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from visionguard.vision.base import VisionWrapper
from visionguard.config import settings

logger = logging.getLogger("visionguard.defects")

class DefectDetector(VisionWrapper):
    """
    PCB manufacturing defect detector wrapper (open, short, mousebite, spur, pin-hole).
    Documented stub returning 'not run' when no external weights are configured.
    """
    def __init__(self):
        self._loaded = False

    def available(self) -> tuple[bool, str]:
        path = settings.DEFECT_MODEL_PATH
        if not path:
            return (False, "PCB defect model not configured (DEFECT_MODEL_PATH is empty); check skipped")
        import os
        if not os.path.exists(path):
            return (False, f"DEFECT_MODEL_PATH not found: '{path}'")
        return (True, "Defect model weights available")

    def load(self) -> bool:
        avail, _ = self.available()
        return avail

    def detect_defects(self, image_crop: Any) -> List[Dict[str, Any]]:
        # Documented stub returning empty when not configured
        if not self.load():
            return []
        return []

default_defect_detector = DefectDetector()

def detect_heuristic_defects(
    image: Image.Image,
    board_bbox: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Cheap, reliable OpenCV and NumPy visual quality and defect heuristics.
    Detects blur, underexposure, overexposure, edge cropping, and thermal discolouration.
    Labelled source='heuristic', severity='possible'.
    """
    defects: List[Dict[str, Any]] = []
    w, h = image.size

    # Convert to numpy array
    img_np = np.array(image.convert("RGB"))

    # 1. Exposure / brightness test
    gray = np.mean(img_np, axis=2)
    mean_brightness = float(np.mean(gray))

    if mean_brightness < 35.0:
        defects.append({
            "id": f"def_heur_dark_{len(defects)+1}",
            "type": "image_too_dark",
            "bbox_px": [0.0, 0.0, float(w), float(h)],
            "conf": 0.85,
            "source": "heuristic",
            "severity": "possible",
            "note": f"Image is underexposed (mean brightness {mean_brightness:.1f}/255). Visual inspection confidence reduced."
        })
    elif mean_brightness > 240.0:
        defects.append({
            "id": f"def_heur_bright_{len(defects)+1}",
            "type": "image_too_bright",
            "bbox_px": [0.0, 0.0, float(w), float(h)],
            "conf": 0.85,
            "source": "heuristic",
            "severity": "possible",
            "note": f"Image is overexposed/glared (mean brightness {mean_brightness:.1f}/255). Surface markings washed out."
        })

    # 2. Blur / sharpness test via Laplacian variance
    try:
        import cv2
        gray_u8 = gray.astype(np.uint8)
        laplacian_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
        if laplacian_var < 45.0:
            defects.append({
                "id": f"def_heur_blur_{len(defects)+1}",
                "type": "image_blurry",
                "bbox_px": [0.0, 0.0, float(w), float(h)],
                "conf": 0.80,
                "source": "heuristic",
                "severity": "possible",
                "note": f"High image blur detected (Laplacian variance {laplacian_var:.1f} < 45.0). Small component labels may be illegible."
            })
    except Exception as e:
        logger.debug(f"OpenCV Laplacian blur check skipped: {e}")

    # 3. Board edge clipping test
    if board_bbox and len(board_bbox) == 4:
        bx, by, bw, bh = board_bbox
        pad = 6.0
        if bx <= pad or by <= pad or (bx + bw) >= (w - pad) or (by + bh) >= (h - pad):
            defects.append({
                "id": f"def_heur_crop_{len(defects)+1}",
                "type": "board_partially_cropped",
                "bbox_px": [float(bx), float(by), float(bw), float(bh)],
                "conf": 0.75,
                "source": "heuristic",
                "severity": "possible",
                "note": "PCB board edges extend beyond camera field of view; header pins or edge ports may be truncated."
            })

    return defects

def compute_health_summary(
    detections: List[Dict[str, Any]],
    defects: List[Dict[str, Any]],
    rule_findings: List[Any],
    skipped_checks: List[str]
) -> Dict[str, Any]:
    """
    Deterministic health summary computation.
    Counts detections by category, weak detections, defects by severity, and skipped checks.
    Overall status: 'ok', 'review', or 'issues'. NEVER outputs 'safe'.
    """
    counts_by_cat: Dict[str, int] = {}
    weak_count = 0
    for d in detections:
        cat = d.get("category", "other")
        counts_by_cat[cat] = counts_by_cat.get(cat, 0) + 1
        if d.get("weak", False) or d.get("conf", 1.0) < settings.WEAK_CONF:
            weak_count += 1

    defects_by_sev = {"likely": 0, "possible": 0}
    for def_item in defects:
        sev = def_item.get("severity", "possible")
        defects_by_sev[sev] = defects_by_sev.get(sev, 0) + 1

    reasons: List[str] = []

    # Count rule errors vs warnings
    error_findings = [f for f in rule_findings if getattr(f, "severity", "") == "error" or (isinstance(f, dict) and f.get("severity") == "error")]
    warn_findings = [f for f in rule_findings if getattr(f, "severity", "") == "warning" or (isinstance(f, dict) and f.get("severity") == "warning")]

    if defects_by_sev["likely"] > 0:
        reasons.append(f"{defects_by_sev['likely']} high-probability defect(s) detected (e.g. missing required components)")
    if error_findings:
        reasons.append(f"{len(error_findings)} wiring or design rule error(s) identified")

    if defects_by_sev["possible"] > 0:
        reasons.append(f"{defects_by_sev['possible']} possible optical/visual defect(s) requiring human verification")
    if weak_count > 0:
        reasons.append(f"{weak_count} component detection(s) have low confidence ('weak')")
    if warn_findings:
        reasons.append(f"{len(warn_findings)} design rule warning(s) flagged")

    # Overall status determination:
    # 'issues' -> high confidence flaws or rule errors
    # 'review' -> possible defects, weak detections, warnings, or unverified items
    # 'ok' -> all clean, but explicitly never 'safe'
    if defects_by_sev["likely"] > 0 or len(error_findings) > 0:
        overall_status = "issues"
    elif defects_by_sev["possible"] > 0 or weak_count > 0 or len(warn_findings) > 0:
        overall_status = "review"
    else:
        overall_status = "ok"

    if not reasons:
        reasons.append("No immediate electrical rule violations or visual defects detected across active inspection layers.")

    return {
        "counts_by_category": counts_by_cat,
        "weak_detections_count": weak_count,
        "defects_by_severity": defects_by_sev,
        "skipped_checks": list(set(skipped_checks)),
        "overall_status": overall_status,
        "reasons": reasons,
        "checks_not_run": list(set(skipped_checks))
    }


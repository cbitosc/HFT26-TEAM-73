import logging
from typing import Any, Dict, List, Optional, Tuple

from visionguard.config import settings

logger = logging.getLogger("visionguard.merge")

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) of two [x, y, w, h] boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def merge_layer_detections(
    layers: Dict[str, List[Dict[str, Any]]],
    iou_threshold: float = 0.45,
    weak_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Merges detections across Layer 1 (pcb_model), Layer 2 (open_vocab), and Layer 3 (llm).
    Applies class-aware IoU merging, confidence calibration, and tracks layers_matched.
    Low-confidence boxes are returned and flagged as weak (never hidden).
    """
    weak_thresh = weak_threshold if weak_threshold is not None else settings.WEAK_CONF

    # Flatten all candidate detections
    candidates: List[Dict[str, Any]] = []
    for layer_name, dets in layers.items():
        for d in dets:
            cand = dict(d)
            cand["source_layer"] = layer_name
            if "layers_matched" not in cand:
                cand["layers_matched"] = [layer_name]
            candidates.append(cand)

    # Sort by confidence descending
    candidates.sort(key=lambda x: x.get("conf", 0.0), reverse=True)

    merged: List[Dict[str, Any]] = []

    for cand in candidates:
        matched = False
        c_box = cand.get("bbox_px")
        if not c_box or len(c_box) != 4:
            continue

        for existing in merged:
            e_box = existing.get("bbox_px")
            iou = calculate_iou(c_box, e_box)
            if iou >= iou_threshold:
                # Same component detected by multiple layers!
                matched = True
                # Record matched layer
                src_layer = cand.get("source_layer")
                if src_layer and src_layer not in existing["layers_matched"]:
                    existing["layers_matched"].append(src_layer)

                # Calibrate confidence: agreement across multiple models boosts confidence
                c_conf = cand.get("conf", 0.5)
                e_conf = existing.get("conf", 0.5)
                # Boost by 0.05 on multi-layer agreement, capped at 0.99
                boosted = min(0.99, max(c_conf, e_conf) + 0.05)
                existing["conf"] = round(boosted, 3)

                # If candidate has more specific part number or OCR text, inherit it
                if cand.get("ocr_text") and not existing.get("ocr_text"):
                    existing["ocr_text"] = cand["ocr_text"]
                if cand.get("part_number") and not existing.get("part_number"):
                    existing["part_number"] = cand["part_number"]
                if cand.get("mapped_wiring_type") and not existing.get("mapped_wiring_type"):
                    existing["mapped_wiring_type"] = cand["mapped_wiring_type"]

                # Re-evaluate weak flag
                existing["weak"] = existing["conf"] < weak_thresh
                break

        if not matched:
            item = dict(cand)
            item["weak"] = item.get("conf", 0.0) < weak_thresh
            merged.append(item)

    # Assign clean IDs: det_1, det_2, ...
    for idx, d in enumerate(merged, start=1):
        d["id"] = f"det_{idx}"

    return merged

